# Yunmo-Next-1B

Technical reports for **Yunmo-Next-1B**, a 1.008B-parameter Traditional-Chinese-first language model with a
hybrid Kimi Delta Attention / grouped-query attention architecture, depth-grown from 24 to 60 layers and
pretrained on 19.76B tokens on one GPU at a time.

| Report | Contents |
|---|---|
| [Architecture](reports/architecture.md) | hybrid cell, low-rank Block Attention Residuals, Cell-Echo, cell-phase SwiGLU, and the experiments behind each |
| [Training](reports/training.md) | depth-grown route, optimizer, schedule, precision and memory on a 12 GB GPU, data repetition |
| [Data](reports/data.md) | sources and licenses, Traditional Chinese processing, stage mixtures, hygiene limits |
| [Tokenizer](reports/tokenizer.md) | exposure-allocated corpus, two-phase BPE, selection, compression |
| [Evaluation](reports/evaluation.md) | comparison with five 2026 open base models, held-out BPB, long-context probes, inference precision |

Weights: [yuhuanstudio/Yunmo-Next-1B-Base](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Base) ·
Tokenizer: [yuhuanstudio/Yunmo-Next-Tokenizer](https://huggingface.co/yuhuanstudio/Yunmo-Next-Tokenizer)

Figures are generated from `data/` by `scripts/make_figures.py`.

## License

Apache-2.0 ([LICENSE](LICENSE)). Benchmark results of third-party models in `data/lm-eval/` were produced with lm-evaluation-harness.
