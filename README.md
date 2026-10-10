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
| [Instruction tuning](reports/instruct.md) | fine-tuning data and recipe, CUDA-graph and continuous-batched decoding, chat-mode evaluation against five 2026 instruction-tuned models, reasoning effort, quantization |

Weights: [yuhuanstudio/Yunmo-Next-1B-Base](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Base) ·
[yuhuanstudio/Yunmo-Next-1B-Instruct](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Instruct) ·
Tokenizer: [yuhuanstudio/Yunmo-Next-Tokenizer](https://huggingface.co/yuhuanstudio/Yunmo-Next-Tokenizer)

Figures are generated from `data/` by `scripts/make_figures.py`. `data/chat-eval/` holds every chat-mode reply
(gzipped JSONL per model, task and setting) with its score, `summary.json` and the item list;
`data/instruct/` holds the decoding parity checks and the quantization results.

## License

Apache-2.0 ([LICENSE](LICENSE)). Benchmark results of third-party models in `data/lm-eval/` were produced with lm-evaluation-harness.
