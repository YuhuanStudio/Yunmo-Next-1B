# Instruction tuning, decoding and chat evaluation

This report covers Yunmo-Next-1B-Instruct: the supervised fine-tuning run, the inference engineering
done for it (CUDA-graph and continuous-batched decoding, `transformers` remote code), how it was
evaluated in chat mode against five 2026 open instruction-tuned models, and what the results say about
the model. The short version: fine-tuning taught the chat protocol, effort control and the shape of
step-by-step reasoning, but not the knowledge or reliability that the base model lacks. The model is a
research checkpoint.

## 1. Chat protocol

Conversations are serialized with 256 reserved protocol IDs (IDs 65,280–65,535 of the 65,536-entry
vocabulary). Every message is

```text
<|start|>{role}<|channel|>{channel}<|message|>{content}{terminal}
```

with roles `system`, `developer`, `user`, `assistant`, `tool`; channels `analysis` (reasoning or
context) and `final`; terminals `<|end|>` (message ends), `<|return|>` (assistant's final answer ends
the turn) and `<|call|>` (tool call, with a `<|recipient|>` field). A conversation starts with
`<|bos|>`. Message text must encode to lexical IDs only: the runtime rejects text that would encode
to a protocol ID (for example a literal `<|end|>`) instead of letting it act as structure.

Generation is grammar-constrained. After the prompt the runtime forces `<|start|>assistant<|channel|>`,
then allows only legal continuations: `final` (or `analysis` when reasoning is enabled), lexical tokens
inside a message, and a legal terminal. A per-message token cap forces the terminal when reached.

**Reasoning effort** is a line in the system message, `推理強度：無 / 低 / 中 / 高` or
`Reasoning: none / low / medium / high` (Simplified Chinese prompts use `推理强度`).

## 2. Supervised fine-tuning

### 2.1 Data

The data is 1,861,379 conversations from 35 public datasets packed into 107,210 training rows of 16,384
tokens (815,984,577 supervised target tokens) and 388 validation rows. The largest sources by supervised tokens:
AM-Thinking-v1 distillation (20.7%), UltraData-SFT (12.3%), GX-Chinese-Instruct (11.3%), smoltalk-chinese
(8.1%), Tulu 3 SFT mixture (8.1%), a Qwen3-235B Chinese reasoning distillation (7.5%), synthetic
long-context QA/summarization (6.8%), AM-DeepSeek-R1 distillation (6.0%), Chinese DeepSeek-R1
distillation (4.2%) and Toucan tool-use trajectories (3.5%). The full table with stated licenses is in the
[model card](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Instruct#data).

Processing that matters for interpreting the results:

- **Script.** Simplified-Chinese sources were converted to Traditional Chinese with OpenCC `s2twp`,
  protecting code, URLs and other machine text.
- **Effort tags.** Every row with reasoning gets an effort level from its reasoning length: low below
  1,500 characters, medium below 5,000, high above. Of the rows without reasoning, 40% (by a hash of the
  row ID) get an explicit `none` tag and the rest no tag, so an untagged prompt means "answer directly".
  The tags were introduced after a 30M-token pilot run that was probed with reasoning allowed: it opened a
  reasoning message on 28 of 33 prompts and closed none of them within 2,048 tokens, because nothing
  told the model when to reason.
- **Supervision.** Loss on assistant messages only, including their headers and terminals. Context-only
  assistant turns in multi-turn data are not supervised.
- **Packing.** Conversations are packed whole into 16K rows with document isolation (attention masks,
  KDA state and positions reset at every conversation boundary); conversations longer than 16K were
  excluded.
- **Replay.** One micro-batch in every 32 is replaced by pretraining text from the base model's last
  16K pack, with a full language-modeling loss.

### 2.2 Recipe

| | |
|---|---|
| Initialization | Yunmo-Next-1B-Base, 19.761B-token checkpoint |
| Updates | 26,802 × 65,536 packed tokens (1.757B), one pass over the data |
| Optimizer | as in pretraining: Muon variants for matrices, AdamW for embeddings, norms and scalars |
| Learning rate | 3×10⁻⁵ for both optimizers; 3% linear warmup; cosine decay to 10% |
| Regularization | weight decay 0.1, gradient clipping 1.0 |
| Precision | BF16 weights and activations with FP16 Kahan compensation, as in pretraining |
| Hardware | one RTX 4070 Ti 12 GB, 3,830 tokens/s, 28.9% MFU, 9.65 GiB peak |

The run crashed once, at 14:27 on its first day, with a CUDA "unknown error" while an unrelated
CPU-only process that imported Triton was running on the same machine; it was restarted from scratch.
After that no other process importing PyTorch, Triton or flash-linear-attention was allowed to run during
training.

### 2.3 Training curves and a mid-run check

Response-only validation bits per byte fell to 0.5010, 0.5008 and 0.5006 at updates 25,500, 26,000 and
26,500, flat over the last thousand updates.

At update 14,500 the run was stopped for a planned check before continuing:

- **Effort control.** Of 8 fixed prompts with reasoning enabled, the reasoning message closed by itself
  on 5 (a 30M-token diagnostic run of the same recipe: 0 of 8); all 8 prompts with effort `none` were
  answered directly.
- **Retention.** Held-out pretraining BPB rose 0.76% at 8K and 0.18% at 16K against the base model;
  code strata rose 1.6–2.5%, the largest drop, which suggests more code in replay next time.

## 3. Decoding

### 3.1 Why decoding was slow

Cached decoding through the regular modules ran at 9 tokens/s on an RTX 4070 Ti (111 ms/token) although
the GPU was busy for about 11 ms of each step. A decode step issues about 5,200 kernel launches across
60 layers (KDA kernels, short convolutions, depth-router reads, gates, norms), so decoding was bound by
kernel launches rather than the GPU.

### 3.2 CUDA-graph static decoding

`yunmo_next/static_decode.py` replays a whole decode step as one CUDA graph:

- GQA keys and values live in preallocated buffers: global layers use a buffer of the length bucket
  (4K, 16K, 64K or 256K), local layers a 2,048-slot ring. Keys are stored after RoPE, so slot order does
  not matter; a mask computed from a position tensor selects valid slots.
- KDA convolution caches and recurrent states are flash-linear-attention's own buffers, updated in place
  by its kernels (`inplace_final_state=True`).
- The depth router is the model's own cached code; only each layer's mixer call is redirected while the
  graph is captured.

Prefill still runs through the regular cached forward, and its caches are copied into the static buffers.

### 3.3 Continuous batching

A 1B FP32 model reads 4 GB of weights per decode step, and the step also pays for thousands of small
kernels; both costs barely depend on how many sequences share the step. The static decoder therefore
takes a batch: every row is an independent conversation with its own position and state, a finished row
is refilled at once with the next conversation's prefill, and parked rows rewrite a single slot.
`YunmoNextInference.chat_batch` drives one grammar-constrained loop per row (the same generator that
`chat` uses) through one batched graph.

Two kernels limited scaling and were replaced:

- **GQA decode attention.** SDPA with `enable_gqa` materialized the 2 KV heads expanded to 8 query heads
  in FP32 (about 2 ms per layer at batch 16, 4K slots), and SDPA on a folded shape picked a slow FP32
  kernel. With one query token, each KV head's four query heads are folded into the query-length axis
  and attention is two batched matmuls that read each KV buffer once (0.17–0.32 ms per layer).
- **Token selection.** Masking the 65,536-way logits and taking an argmax cost a few kernels and two
  synchronizations per row. The graph now also computes, per row, the lexical argmax and its value, the
  256 protocol logits and a NaN flag; greedy rows select on the host from that summary (the same token,
  ties to the lowest ID), with a fallback to the full row when the grammar names specific text tokens.

Step time of the captured graph, FP32, RTX 4070 Ti:

| Batch | Before the two changes | After | Ceiling (tokens/s) |
|---:|---:|---:|---:|
| 1 | 14.6 ms | 14.1 ms | 69 |
| 8 | 35.8 ms | 18.5 ms | 419 |
| 16 | 56.9 ms | 21.3 ms | 720 |
| 32 | 1,836 ms (memory spill) | 26.6 ms | 1,136 |

At batch 16, 12 ms of the 21 ms step is reading the FP32 weights (about two thirds of the GPU's peak
bandwidth); going further means lower-precision weights.

End to end on chat benchmarks (64 conversations, greedy, FP32): 66 tokens/s one at a time, 437 at batch 16,
485 at batch 32. Over the full evaluation (thousands of conversations, so few idle rows) it averaged
about 750 tokens/s.

### 3.4 Exactness

Every optimization was accepted only with identical outputs:

- Single-row static decoding against regular decoding on forced token sequences, prompts of 5 to 4,502
  tokens (the local ring wraps beyond 2,048): maximum logit difference ≤ 2.7×10⁻⁵, 127/127 greedy
  agreement per prompt.
- `chat_batch` against one-at-a-time `chat`: identical final text and token counts for 64/64
  conversations at batch 16 and 32 (4K bucket) and 8/8 at the 8K bucket.

### 3.5 `transformers`

`configuration_yunmo_next.py` and `modeling_yunmo_next.py` wrap the package model as remote code. The
checkpoint is stored in BF16 (`model.safetensors`); every released FP32 value is an exact BF16 upcast,
which the exporter checks bit by bit, so the BF16 file loses nothing. FP32 logits through `transformers`
are bit-identical to the package runtime's; right-padded batches match unpadded rows to 7×10⁻⁵.

## 4. Chat-mode evaluation

### 4.1 Protocol

| Task | Items | Scoring |
|---|---:|---|
| IFEval | 541 | lm-evaluation-harness checker, prompt-level strict accuracy |
| GSM8K | 1,319 | the reply's boxed or `####` number, else its last number |
| HumanEval | 164 | completed function executed against the tests, 10 s limit, isolated process |
| MMLU | 1,140 (20 per subject) | option letter the reply commits to |
| TMMLU+ | 990 (15 per subject, 66 subjects) | same |

- Every model answers through its own chat format with greedy decoding.
- Yunmo-Next runs in FP32 through its runtime at all four effort levels, with budgets of 1,024 / 2,048 /
  4,096 / 8,192 tokens for none / low / medium / high.
- Comparison models run in BF16 with vLLM 0.31: thinking disabled with a 1,024-token budget ("direct"),
  and thinking enabled with an 8,192-token budget.
- A reply whose reasoning does not close within the budget has an empty final answer.
- Yunmo-Next's medium and high levels and the comparison models' thinking mode run on every fourth item
  (1,040 items).

**Scoring corrections made during this evaluation.**

- **Option letters.** The first version took the first standalone A–D in a reply. It counted the English
  article "A" ("A cyclic group is …") and letters inside words ("ABC公司") as answers: 863 of 1,140
  Yunmo-Next MMLU replies were scored "A", for 24.5%.
  The extractor now prefers, in order:
  1. a reply that starts with a letter;
  2. a last line that is only a letter;
  3. explicit answers (`\boxed{C}`, `答案是C`, `Answer: B`), excluding options named to reject them
     (`選項D錯誤`);
  4. bold letters;
  5. a reply quoting exactly one option's text;
  6. a guarded lone letter in the last line.

  A reply that never commits scores as unanswered. Two manual audits of 30 random replies each found 2
  and 1 errors, both fixed, and those cases became unit tests.
- **Missing subject.** lm-evaluation-harness lists 67 TMMLU+ subjects; the dataset has 66 (no linear
  algebra).
- **Peer scoring writes.** A file-handling bug corrupted one comparison model's scored output; it was
  rescored from the saved generations.

All saved replies were rescored with the final scorer.

### 4.2 Results: answering directly (all items)

| Model | Setting | IFEval | GSM8K | HumanEval | MMLU | TMMLU+ |
|---|---|---:|---:|---:|---:|---:|
| **Yunmo-Next 1B** | effort none | 44.7 | 13.9 | 15.9 | 29.0 | 23.0 |
| Qwen3.5 0.8B | direct | 51.9 | 55.6 | 31.7 | 53.5 | 32.2 |
| LFM2.5 1.2B | direct | 80.8 | 76.1 | 51.8 | 51.6 | 30.4 |
| MiniCPM5 1B | direct | 63.6 | 57.8 | 61.6 | 53.3 | 26.1 |
| Qwen3.5 2B | direct | 67.5 | 77.6 | 53.0 | 62.1 | 40.8 |
| MiniCPM5 2B | direct | 85.0 | 87.1 | 82.3 | 61.6 | 40.3 |

Yunmo-Next is last on every task. The gap is smallest on IFEval (7 points behind Qwen3.5-0.8B) and TMMLU+
(3 points behind MiniCPM5-1B) and largest on GSM8K (42–73 points), HumanEval (16–66) and MMLU (23–33),
consistent with a base model pretrained on about a thousandth of the comparison models' tokens.

### 4.3 Results: reasoning effort (every fourth item)

| Model | Setting | IFEval | GSM8K | HumanEval | MMLU | TMMLU+ |
|---|---|---:|---:|---:|---:|---:|
| **Yunmo-Next 1B** | effort none | 49.3 | 17.3 | 12.2 | 26.7 | 23.8 |
| **Yunmo-Next 1B** | effort low | 17.6 | 18.2 | 9.8 | 29.8 | 22.2 |
| **Yunmo-Next 1B** | effort medium | 19.1 | 15.8 | 9.8 | 24.9 | 12.5 |
| **Yunmo-Next 1B** | effort high | 11.0 | 13.0 | 2.4 | 19.6 | 10.5 |
| Qwen3.5 0.8B | direct | 53.7 | 54.8 | 39.0 | 52.6 | 30.6 |
| Qwen3.5 0.8B | thinking, greedy | 0.7 | 0.0 | 19.5 | 9.1 | 2.0 |
| Qwen3.5 0.8B | thinking, recommended sampling¹ | 53.7 | 12.1 | 31.7 | 47.0 | 35.1 |
| LFM2.5 1.2B | direct | 83.1 | 76.4 | 53.7 | 53.0 | 31.0 |
| MiniCPM5 1B | direct | 69.9 | 57.6 | 68.3 | 52.3 | 24.6 |
| MiniCPM5 1B | thinking, greedy | 75.7 | 75.8 | 68.3 | 61.4 | 22.2 |
| Qwen3.5 2B | direct | 72.8 | 78.5 | 61.0 | 63.9 | 40.7 |
| Qwen3.5 2B | thinking, greedy | 8.1 | 19.4 | 48.8 | 38.2 | 24.2 |
| Qwen3.5 2B | thinking, recommended sampling¹ | 74.3 | 62.7 | 58.5 | 65.6 | 52.8 |
| MiniCPM5 2B | direct | 86.8 | 87.6 | 82.9 | 61.1 | 38.7 |
| MiniCPM5 2B | thinking, greedy | 83.1 | 80.9 | 68.3 | 80.0 | 51.2 |

¹ Qwen3.5 only, with Qwen's recommended text-thinking sampling (temperature 1.0, top-p 0.95, top-k 20,
presence penalty 1.5; one sample per item). With greedy decoding, Qwen3.5's thinking does not finish within
8,192 tokens on most items (empty answers on 66–100% of items per task for 0.8B, 37–92% for 2B), so its
greedy rows understate it; Qwen advises against greedy decoding in thinking mode. Even with the recommended
sampling, Qwen3.5-0.8B exceeds the 8,192-token budget on 88% of GSM8K items (2B: 36%); Qwen's own
evaluations allow much longer reasoning. Every other row is greedy.

## 5. Analysis

### 5.1 Knowledge comes from pretraining

The base model scores at chance on letter-scored MMLU (22.9%) and TMMLU+ (25.0%). In chat mode the
fine-tuned model reaches 29.0% and 23.0%. Supervised fine-tuning on 0.8B target tokens did not add the
knowledge that 19.76B pretraining tokens did not provide. The comparison models were pretrained on
roughly a thousand times more tokens.

### 5.2 Reasoning effort hurts

Generation behavior at each effort level (every fourth item, all tasks pooled):

| Effort | Budget (tokens) | Reasoning hit its cap | Empty final answer | Final answer ends in a loop | Mean generated tokens |
|---|---:|---:|---:|---:|---:|
| none | 1,024 | — | 0.0% | 5.0% | 237 |
| low | 2,048 | 18.8% | 11.6% | 0.7% | 833 |
| medium | 4,096 | 35.7% | 22.3% | 0.2% | 1,906 |
| high | 8,192 | 60.2% | 26.6% | 0.0% | 5,188 |

Effort control works mechanically: tags change how long the model reasons. Longer reasoning does not make
answers better: on all items `low` beats `none` only on GSM8K, by 1.1 points (within noise), and `medium`
and `high` score lower than `none` on every task. Reading the reasoning shows why:

- The model reproduces the structure of the distilled traces: option-by-option analysis, "wait, let me
  double-check", a boxed answer.
- It does not reproduce their correctness. Arithmetic slips ("35 + 60 = 95") are copied through the
  re-check, and conclusions contradict the analysis they follow.
- When unsure, it repeats a sentence or an equation until the budget runs out. That accounts for most
  capped reasoning, and therefore for the empty answers at high effort.

This matches what was known when the data was designed: small models trained directly on long
chain-of-thought from much larger teachers can fall behind short chain-of-thought training (arXiv
2502.12143). About 40% of the supervised tokens come from reasoning-distillation sets (AM-Thinking-v1,
DeepSeek-R1, Qwen3-235B, GLM-4.7 and Claude teachers), all 32B parameters or larger.

Greedy decoding is not the main cause. On a 64-item sample at low effort, sampling at T = 0.7,
top-p 0.9 reduced capped reasoning from 17 to 6 items and repetition from 10 to 6, but answered fewer
items correctly (13 vs 19).

### 5.3 Direct answers

Direct answers (`none`) are the model's best setting. IFEval at 44.7% means it follows simple format
instructions about half the time. On IFEval, however, 19.8% of replies end in a repetition loop and
32% hit the 1,024-token cap, mostly on prompts asking for long outputs.

## 6. Weight-only quantization

Held-out bits per byte on the 16-stratum pretraining evaluation set at 8K context (8 rows per stratum), FP32 compute on quantize-dequantize weights, so the difference is quantization error only. Affine schemes follow MLX (`w ≈ scale·q + bias` per group of input channels). Group-64/32 schemes leave the 60 MLP down projections in full precision because their input widths (5,472 / 4,560 / 2,736 / 1,824) are not multiples of 64 or 32 for every layer; INT8 per-channel covers every linear layer. Norms, convolutions and router parameters are never quantized.

| Scheme | Quantized parameters | Held-out BPB | vs FP32 |
|---|---:|---:|---:|
| FP32 (reference) | — | 0.6716 | — |
| INT8, symmetric per output channel | 940M (93%) | 0.6716 | +0.01% |
| 8-bit affine, group 64 | 716M (71%) | 0.6716 | +0.00% |
| 4-bit affine, group 64 | 716M (71%) | 0.6764 | +0.72% |
| 4-bit affine, group 32 | 828M (82%) | 0.6756 | +0.61% |
| 8-bit affine, group 64, + tied embedding | 783M (78%) | 0.6716 | +0.01% |
| 4-bit affine, group 64, + tied embedding | 783M (78%) | 0.6875 | +2.37% |

8-bit weights are lossless within this measurement. 4-bit weights cost 0.6–0.7% BPB; quantizing the
tied embedding (which is also the output projection) to 4 bits as well roughly triples the loss, so keep it
at 8 bits or more.

## 7. What would fix this

In order of expected effect:

1. **Pretraining tokens.** Knowledge and basic reasoning are pretraining properties, and the model is
   about three orders of magnitude short of its comparison models.
2. **Short reasoning for small models.** Mostly short traces, with long traces only as a minority (a
   20:80 long:short mix gave 7–8 points in the cited study), and explicit truncated-reasoning examples.
3. **Anti-repetition training.** Preference optimization against looping and non-terminating
   generations, or unlikelihood-style penalties on repeated spans.
4. **More code in replay**, given the 1.6–2.5% code BPB regression.

## 8. Reproducibility

- **Scripts.** Training: `trainer/train_sft.py`. Release export: `scripts/export_model.py` and
  `scripts/export_hf.py`. Chat benchmarks: `research/training/evaluation/chat_benchmarks.py` (items and
  scoring), `yunmo_next_chat_eval.py` (Yunmo-Next), `peer_vllm_generate.py` + `peer_chat_eval.py`
  (comparison models), `chat_eval_table.py` (tables). All are in
  [YuhuanStudio/Yunmo-Next](https://github.com/YuhuanStudio/Yunmo-Next).
- **Data.** Every reply and score is in `data/chat-eval/` of this repository.
