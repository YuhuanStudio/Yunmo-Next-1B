# Yunmo-Next Architecture

*Technical report · Yunmo-Next-1B · October 2026*

## Summary

Yunmo-Next-1B is a 60-layer, 1,024-wide decoder built from a repeating four-layer cell: one grouped-query
attention (GQA) layer followed by three Kimi Delta Attention (KDA) layers. Three further design choices
matter:

- **Low-rank Block Attention Residuals.** Each sublayer reads its input through a softmax over earlier
  block outputs instead of a residual sum. Keys use 64 of the 1,024 channels; values stay full width.
- **Cell-Echo.** One zero-initialized scalar per cell lets the cell's last layer up-weight the block that
  holds that cell's own GQA output.
- **Cell-phase SwiGLU widths.** Feed-forward width follows a cosine across the four cell positions, with
  the cell total equal to four uniform layers, so the parameter budget is unchanged and depth can grow by
  whole cells without reshaping any tensor.

Every choice was made with short paired training runs on a 24-layer model (4–8M tokens per arm, one to
three seeds) on the same RTX 4070 Ti used for training. This report gives the design, each experiment
behind it, the variants that were rejected, and what that evidence does and does not establish.

## 1. Design constraints

The model had to be trained on one 12 GB GPU and run at long context on consumer hardware. That pushes
toward few attention layers with a KV cache, a fixed-size recurrent state elsewhere, and memory-efficient
depth routing. KDA, Attention Residuals, gated attention, GQA, partial RoPE, RMSNorm, SwiGLU, Muon and
staged depth growth are prior work; the specific combination, Cell-Echo, the cell-phase width schedule and
the low-rank two-phase routing implementation are particular to this model.

## 2. The hybrid cell

```text
cell = GQA → KDA → KDA → KDA          × 15 cells = 60 layers
```

**GQA layers (15).** 8 query heads and 2 key/value heads of dimension 128. Queries and keys are
RMS-normalized per head before rotary embedding. Rotary embedding covers the first 32 of 128 dimensions
(θ = 10⁷); the rest carry no position. The query projection also produces an elementwise output gate:
`o_proj(attention ⊙ σ(gate))`. Ten GQA layers use a 2,048-token causal sliding window; every third GQA
layer (layers 8, 20, 32, 44, 56) attends globally.

**KDA layers (45).** The gated delta-rule recurrence from Kimi Linear, via flash-linear-attention:

```text
S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ,     o_t = S_tᵀ q_t
```

with 8 heads of key and value dimension 128, a channel-wise decay `α_t`, a width-4 short convolution and a
gated output norm. KDA has no position encoding; order enters through the recurrence.

**Inference memory.** Each KDA head carries a 128 × 128 state regardless of sequence length. Only the 15
GQA layers keep a KV cache, and 10 of those keep at most 2,048 positions. Per-token cache growth beyond
2K therefore comes from 5 layers × 2 KV heads × 128 dims × 2 (K and V).

### 2.1 Experiments behind the cell

All runs below use the 24-layer configuration (about 440M parameters), 2,048-token sequences, identical
initial weights for shared parameters, and report the difference in final validation loss (nats per token,
candidate minus control; negative favors the candidate).

| Question | Candidate vs control | Tokens/arm | Seeds | Δ validation loss | Throughput | Outcome |
|---|---|---:|---:|---|---:|---|
| KDA heads | 8 heads × 128 vs 16 heads × 128 | 4.2M | 17, 29 | −0.0100, −0.0097 | 1.21× | 8 heads adopted (also 5.2 vs 6.8 GiB peak) |
| KDA heads at 60 layers | 7 heads vs 8 heads | 4.2M | 17, 29 | −0.0048, +0.0057 | — | 8 heads kept |
| Attention ratio | 2:1 (GQA every 3rd layer) vs 3:1 | 4.2M | 17 | +0.0101 | 1.01× | 3:1 kept |
| GQA position | first in cell vs last | 4.2M | 17 | −0.0026 | 1.00× | below adoption threshold alone; see Cell-Echo |
| GQA position | second in cell vs first | 4.2M | 17 | +0.0003 | 1.00× | rejected |
| Split mixer | layers 9/21 replaced by 4 KDA heads + small GQA in one layer | 4.2M | 17 | +0.0098 | 1.00× | rejected |

The 8-head KDA change is the clearest result here: two seeds agree, it is faster and it uses less memory.
The others are single-seed differences of about 0.01 nats or less and are best read as "no evidence that the
alternative is better."

## 3. Depth routing: low-rank Block Attention Residuals

### 3.1 Mechanism

Sublayer outputs are grouped into blocks of six. Before each sublayer (both the mixer and the MLP of every
layer have one), a learned query `q ∈ ℝ⁶⁴` attends over the available sources:

- the token embedding,
- the summed output of each completed block, and
- the running sum of the current block so far.

```text
logit(s) = q · RMSNorm(s[−64:]) + b · 1[s is the Cell-Echo source]     (b exists on 15 reads, §4)
input    = Σ_s softmax(logit)_s · s
```

Keys use only the last 64 channels; values are the full 1,024-dimensional sources. Two implementation
details make this affordable:

- **Two-phase evaluation.** Completed-block sources do not change within a block, so their logits for all
  six reads in the block are computed in one batched pass; the running partial block is merged afterwards
  with an online-softmax update.
- **Blockwise recomputation.** Source keys are materialized once per block in the forward pass and
  recomputed per block in the backward pass instead of being stored for every read.

Query initialization for the first 24 layers uses a fixed-seed orthogonal bank scaled by 0.02·√64; the
first read has a single source, so its query is zero.

### 3.2 Experiments

| Comparison | Tokens/arm | Seed | Candidate loss | Control loss | Δ | Throughput |
|---|---:|---:|---:|---:|---:|---:|
| Rank-64 two-phase AttnRes vs previous block-skip residual | 4.2M | 17 | 6.0985 | 6.0994 | −0.0009 | 0.95× |
| same | 4.2M | 29 | 6.0210 | 6.0560 | −0.0350 | 0.95× |
| same | 8.4M | 41 | 5.5780 | 5.7886 | −0.2106 | 0.95× |
| Full-rank Block AttnRes vs previous block-skip residual (separate run) | 8.4M | 41 | 5.5683 | 5.7816 | −0.2133 | 0.85× |

The control is the project's earlier cumulative block-skip residual, not a plain additive residual. The
effect grows with training length in these runs, but the three seeds also differ in length, so this is not
a clean dose–response measurement. Across separate seed-41 runs, the rank-64 version reaches nearly the
same loss as full-rank keys at a smaller throughput cost (0.95× vs 0.85×). Peak memory rose from 6.1 to
6.9 GiB in the 24-layer test.

Rejected alternatives:

| Variant | Result | Outcome |
|---|---|---|
| Dual AttnRes (two routing paths) | 0.22× throughput in the cost screen | stopped before quality testing |
| Softmax-plus-one depth routing | +0.0009 | rejected |

## 4. Cell-Echo

In each four-layer cell, the router that feeds the last layer's mixer gets one zero-initialized scalar `b`,
added to the logit of the completed block that contains the cell's GQA output. The model has 15 such
scalars, one per cell. Each changes a single routing logit; nothing else in the computation changes.

| Comparison | Seeds | Δ validation loss | Throughput |
|---|---|---|---:|
| GQA-first cell **with** Cell-Echo vs GQA-last cell without | 17, 29 | −0.0032, −0.0081 | 1.00× |
| Centered contrast `a' = a + γ(a − mean)` added on top of Cell-Echo | 17 | −0.0030 | 0.99× |
| Source-count rescaled bias | 17 | −0.0024 | 0.99× |
| Tanh compatibility curve | 17 | +0.0055 | 1.01× |

The adopted change is a bundle: moving GQA to the first cell position and adding Cell-Echo, compared with
the previous GQA-last cell. The GQA-first move alone (−0.0026, §2.1) did not clear the adoption threshold;
the bundle improved on both seeds. This evidence does not separate the two parts. The three refinements of
Cell-Echo were single-seed; two of them improved slightly but did not meet the preregistered adoption
criteria, and the scalar version was kept.

## 5. Cell-phase SwiGLU widths

For position `r ∈ {0,1,2,3}` within a cell:

```text
w_r = round_16( d · (1 + ½ cos(π r / 3)) )
d = 3,648  →  [5472, 4560, 2736, 1824]
```

The widths sum to exactly `4d`, so each cell has the parameters and MLP FLOPs of four uniform layers. The
widest MLP sits on the GQA layer. Growth adds 12 layers (three whole cells), and `(ℓ + 12) mod 4 = ℓ mod 4`,
so every existing layer keeps its phase and tensor shapes when the model grows.

| Comparison (24 layers, d = 3,584 at the time) | Tokens/arm | Seeds | Δ validation loss | Throughput |
|---|---:|---:|---|---:|
| Cosine widths [5376, 4480, 2688, 1792] vs uniform 3,584 | 4.2M | 17, 29 | −0.0044, −0.0097 | 1.01×, 0.99× |

These trials used base width 3,584; the released model uses 3,648, a product decision that was not
separately re-tested. The schedule was adopted after the two paired trials, a growth check and an
integration check.

## 6. Other decisions

| Decision | Evidence | Outcome |
|---|---|---|
| Remove hashed lexical memory | removal improved loss by 0.025 and 0.030 (seeds 17, 29) | removed |
| Partial RoPE vs YaRN / randomized position encoding | RPE and YaRN+RPE passed a short non-inferiority screen but were worse on 4K/8K synthetic recall (MQAR query loss +0.060 to +0.126) | partial RoPE kept |
| Inference-time normalization removal | recorded as negative on CUDA; the raw result file was not located for this report | rejected |
| Mixture of experts | not tested at this scale | not pursued for a 12 GB dense design |

The position result comes from about 4M tokens per arm with near-zero MQAR accuracy in both arms; it does
not show that YaRN would fail when extending a mature checkpoint.

## 7. Parameter accounting

| Stage | Layers | Parameters |
|---|---:|---:|
| K24 | 24 | 443,489,302 |
| K36 | 36 | 631,678,945 |
| K48 | 48 | 819,868,588 |
| K60 (released) | 60 | 1,008,058,231 |

Each growth step adds 188,189,643 parameters (three cells). The embedding matrix (65,536 × 1,024 =
67,108,864) is shared with the output head.

## 8. What this evidence establishes

- Each component was chosen on short, mostly single-seed runs at 24 layers and about 4M tokens. Differences
  of 0.01 nats or less at that scale are directional, not reliable estimates of the effect at 60 layers and
  20B tokens.
- No full-scale ablation removes any component from the 1B model. The released model shows the combination
  trains stably and reaches the results in the evaluation report; it does not isolate why.
- The comparisons report seed-to-seed differences, not confidence intervals.
- The 3,648 base width, the 60-layer depth and the 16K context of the release were not part of these
  comparisons.

## Reproducibility

Configuration: [`config.json`](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Base/blob/main/config.json).
The model code ships with the weights under `code/yunmo_next/` (`model.py`, `attention.py`, `attn_res.py`,
`kda_runtime.py`).
