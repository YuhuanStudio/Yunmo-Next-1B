# Evaluating Yunmo-Next-1B-Base

*Technical report · Yunmo-Next-1B · October 2026*

## Summary

We compare Yunmo-Next-1B-Base with five open base models released in 2026, on 13 tasks run through one
harness with identical settings. Yunmo-Next was trained on 19.76B tokens; the one comparison model that
states its budget (LFM2.5-1.2B) was trained on 28T, about 1,400 times more.

- On English reasoning and knowledge tasks Yunmo-Next trails the comparison models, typically by 5–25
  points; the exception is OpenBookQA, where it is within noise of the 1B models (34.4 vs 33.8–39.0).
- On letter-scored multiple-choice knowledge tests (MMLU, TMMLU+) it is at chance level.
- On three Chinese cloze tasks it is within 1–4 points of the 0.8–1.2B models, ahead of LFM2.5-1.2B on two
  of them, and 4–10 points behind the 2B models.

We also report held-out bits per byte, answer likelihoods on reading comprehension and math, and long-context
probes, all of which are internal measurements.

## 1. Comparison with 2026 open base models

### 1.1 Setup

| Setting | Value |
|---|---|
| Harness | lm-evaluation-harness 0.4.13, default task definitions |
| Shots | 0-shot, except MMLU 5-shot |
| Seed | 1234 for sampling few-shot examples |
| Metric | normalized accuracy (`acc_norm`) for ARC, HellaSwag, PIQA, OpenBookQA and SciQ; accuracy otherwise |
| Comparison models | Hugging Face `transformers` backend, BF16, batch size 4 |
| Yunmo-Next | a small harness adapter over the release package, FP32 |

The adapter scores exactly what completion mode would see: `<|bos|>` followed by context and continuation,
using the harness's own context/continuation split. Batches are right-padded to a multiple of 256 tokens;
because every layer is causal, padding after a sequence cannot change its scores, which we verified by
changing the pad token.

Yunmo-Next is evaluated in FP32, the release precision, rather than BF16. In BF16 the log-likelihood of a
short answer moved by up to 0.15 nats when only the padded batch width changed, while in FP32 it moved by
0.007. This numerical sensitivity is a property of the model in BF16 and is discussed in §4.

TMMLU+ runs 0-shot because some of its subjects have only four development examples, too few for 5-shot.

### 1.2 Models

| Model | Released | Parameters | Pretraining tokens |
|---|---|---:|---|
| **Yunmo-Next-1B-Base** | 2026-10 | 1.01B | 0.0198T |
| Qwen3.5-0.8B-Base | 2026-02 | 0.87B | not stated |
| LFM2.5-1.2B-Base | 2026-01 | 1.17B | 28T |
| MiniCPM5-1B-Base | 2026-05 | 1.08B | not stated |
| Qwen3.5-2B-Base | 2026-02 | 2.27B | not stated |
| MiniCPM5-2B-Base | 2026-08 | 2.52B | not stated |

Parameter counts are the checkpoints' totals. The Qwen3.5 checkpoints are vision-language models trained
on multimodal tokens and were scored text-only. AI2's `Dense_1b_130B`, which would have been the closest
comparison by token budget (130B), could not be evaluated: its custom model code targets transformers 4.57
and fails to load under transformers 5.

### 1.3 Results

| Task | Yunmo-Next 1B | Qwen3.5 0.8B | LFM2.5 1.2B | MiniCPM5 1B | Qwen3.5 2B | MiniCPM5 2B |
|---|---:|---:|---:|---:|---:|---:|
| ARC-Easy | 50.3 | 67.2 | 77.7 | 65.6 | 71.4 | 74.7 |
| ARC-Challenge | 29.2 | 40.1 | 51.3 | 37.3 | 45.9 | 50.5 |
| HellaSwag | 41.3 | 54.9 | 59.2 | 56.8 | 66.0 | 67.3 |
| PIQA | 65.9 | 71.8 | 73.3 | 72.7 | 74.8 | 77.0 |
| Winogrande | 56.1 | 60.2 | 58.7 | 58.2 | 64.6 | 63.5 |
| OpenBookQA | 34.4 | 35.4 | 39.0 | 33.8 | 37.6 | 41.8 |
| SciQ | 85.2 | 91.7 | 93.3 | 92.3 | 93.2 | 93.9 |
| LAMBADA (OpenAI) | 44.5 | 50.9 | 46.6 | 55.4 | 58.4 | 61.2 |
| MMLU (5-shot) | 22.9 | 48.4 | 55.0 | 43.5 | 54.0 | 65.3 |
| TMMLU+ | 25.0 | 31.2 | 31.2 | 28.4 | 42.1 | 44.1 |
| XStoryCloze-zh | 57.5 | 58.3 | 58.5 | 58.0 | 62.3 | 63.6 |
| XCOPA-zh | 61.8 | 63.4 | 60.8 | 62.4 | 67.8 | 67.0 |
| XWinograd-zh | 67.5 | 68.8 | 63.1 | 71.6 | 77.4 | 74.6 |

Standard errors are about 1 point on ARC-Easy and larger on the smaller sets (OpenBookQA 500 items,
XCOPA-zh 500, XWinograd-zh 504, XStoryCloze-zh 1,511); differences of 2–3 points on those sets are within
noise. LAMBADA perplexity: Yunmo-Next 36.9, Qwen3.5-0.8B 11.7, LFM2.5 13.5, MiniCPM5-1B 8.7, Qwen3.5-2B 7.2,
MiniCPM5-2B 6.2.

### 1.4 Reading the results

- **Token budget dominates.** With roughly three orders of magnitude fewer training tokens than LFM2.5, the
  gap on English commonsense and science tasks is the expected outcome, not evidence about the architecture
  either way. The comparison shows where this checkpoint stands, not what the architecture would reach at
  equal compute.
- **Multiple-choice knowledge.** MMLU and TMMLU+ ask the model to put probability on the letter of the right
  answer. Yunmo-Next scores at chance on both, in this harness and in the internal 5-shot evaluation
  (TMMLU+ 25.1%, MMLU 24.2%). The comparison models score 28–65%.
- **Chinese cloze.** On XStoryCloze, XCOPA and XWinograd (Simplified Chinese), Yunmo-Next is 0.8–1.6
  points below Qwen3.5-0.8B, ahead of LFM2.5-1.2B on XCOPA (+1.0) and XWinograd (+4.4), and 4.1 points behind
  MiniCPM5-1B on XWinograd. These tasks score whole-sentence plausibility, which a model can do from language
  modeling alone. Yunmo-Next's training data is mostly Traditional Chinese; these sets are Simplified.
- **No Traditional Chinese cloze benchmark** of comparable standing was available in the harness, so the
  model's main language is measured here only through TMMLU+ (letter-scored) and the internal evaluations
  below.

## 2. Internal evaluations

### 2.1 Held-out bits per byte

125 documents from 16 strata (Traditional and converted Chinese, Chinese knowledge, English, mathematics, and
11 programming-language strata), never trained on, scored at 8K and 16K context:

| Context | Bits per byte |
|---|---:|
| 8K | 0.671 |
| 16K | 0.637 |

Bits per byte is comparable across tokenizers, but these documents are drawn from the same source families
as the training data, so the numbers are not comparable with other models' published perplexities.

### 2.2 Knowledge and reading (5-shot letter and answer likelihood)

| Benchmark | Protocol | Items | Result |
|---|---|---:|---:|
| TMMLU+ | 5-shot, next-token letter, subject macro-average | 20,160 | 25.1% |
| MMLU | same | 14,042 | 24.2% |
| DRCD | bits per byte of the gold answer, teacher-forced | 3,524 | 0.445 |
| GSM8K | bits per byte of the gold solution, teacher-forced | 1,319 | 0.633 |

DRCD and GSM8K here measure how likely the model finds the reference answer, not whether it generates it.
For reference, the project's previous model (Yunmo v1) scored 0.704 and 1.022 on the same two measurements,
with a different tokenizer and evaluation code.

### 2.3 Long context

| Probe | 8K | 16K | 32K |
|---|---:|---:|---:|
| Held-out BPB | 0.671 | 0.637 | — |
| MQAR (synthetic associative recall) accuracy | 40.6% | 6.3% | 0% |
| Multi-needle retrieval, 4 needles | 25% | 0% | 0% |
| Multi-needle retrieval, 16 needles | 0% | 6.3% | 0% |

Retrieval is weak at 8K and largely fails beyond it, despite training at 16K and the 32K stage described in
the training report. InfiniteBench answer likelihoods were measured from 64K to 256K tokens (24 items, 59
variants; macro bits per byte 1.84) as a diagnostic of the cached inference path; they do not indicate
usable capability at those lengths.

## 3. Generation

Base-model completions are fluent Traditional Chinese but unreliable. Given `臺灣最高的山是`, the released
package produced:

```text
玉山，臺灣最低的山是阿里山。
臺灣最高的山是玉山，臺灣最低的山是阿里山。
臺灣最高的山是玉山，臺灣最低的山是阿里山。…
```

The first clause is correct; the continuation is a false statement repeated until the token limit. This
pattern — correct start, invented continuation, repetition — is typical of this checkpoint.

## 4. Numerical precision at inference

The release package defaults to FP16 inference. During evaluation we found that, in BF16, the
log-likelihood of a three-token answer ranged from −2.38 to −2.53 nats as the padded batch width changed
from 11 to 1,024 tokens, with identical scores whenever the width was fixed and only the padding content
changed. In FP32 the same range was −2.506 to −2.513. The sensitivity is numerical, not a masking error,
but it is large enough to flip close multiple-choice decisions. FP16 inference was not measured; until it
is, FP32 is the safer choice when scores matter.

## 5. Limitations

- One checkpoint, one evaluation run per model; no seed variation for few-shot sampling.
- Comparison models in BF16 and Yunmo-Next in FP32 (§1.1).
- No Traditional Chinese cloze-style benchmark in the comparison.
- Internal held-out and long-context sets were used repeatedly during development, including for choosing
  this checkpoint; they are development measurements, not untouched test sets.
- No generation-based benchmarks (exact match, code execution) for the base model.
- Exact-match decontamination only; benchmark paraphrases in web data cannot be ruled out for any model.

## Reproducibility

Raw harness outputs for all six models and the adapter (`yunmo_next_lm_eval.py`) are kept with the project's
evaluation artifacts; the table is regenerated by `scripts/lm_eval_table.py`.
