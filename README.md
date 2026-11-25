# Yunmo-Next-1B

Technical reports for **Yunmo-Next-1B**, a 1.008B-parameter Traditional-Chinese-first language model with a
hybrid Kimi Delta Attention / grouped-query attention architecture, depth-grown from 24 to 60 layers and
pretrained on 19.76B tokens on one GPU at a time.

The **[technical report (PDF)](paper/yunmo-next-1b.pdf)** covers the whole model in one document. Start with the **[technical overview](reports/overview.md)** ([繁體中文](reports/overview.zh-TW.md)): highlights,
models, architecture, training, evaluation, quickstart and limitations on one page.

| Report | Contents |
|---|---|
| [Architecture](reports/architecture.md) | hybrid cell, low-rank Block Attention Residuals, Cell-Echo, cell-phase SwiGLU, and the experiments behind each |
| [Training](reports/training.md) | depth-grown route, optimizer, schedule, precision and memory on a 12 GB GPU, data repetition |
| [Data](reports/data.md) | sources and licenses, Traditional Chinese processing, stage mixtures, hygiene limits |
| [Tokenizer](reports/tokenizer.md) | exposure-allocated corpus, two-phase BPE, selection, compression |
| [Evaluation](reports/evaluation.md) | 2026 open base models; equal-compute comparison with Pythia, DataDecide and OLMo 2; cloze-form MMLU/TMMLU+ with continuous metrics; new Traditional Chinese Wikipedia text with an overlap audit, and non-Wikipedia text; long context; quantized weights |
| [Instruction tuning](reports/instruct.md) | fine-tuning data and recipe, CUDA-graph and continuous-batched decoding, chat-mode evaluation against five 2026 instruction-tuned models, reasoning effort, quantization |

Models on Hugging Face:

| | |
|---|---|
| Base | [Yunmo-Next-1B-Base](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Base) · [bnb-4bit](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Base-bnb-4bit) · MLX [bf16](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Base-MLX-bf16) / [8bit](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Base-MLX-8bit) / [4bit](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Base-MLX-4bit) |
| Instruct | [Yunmo-Next-1B-Instruct](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Instruct) · [INT8](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Instruct-INT8) · [FP8](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Instruct-FP8) · [NVFP4](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Instruct-NVFP4) · [bnb-4bit](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Instruct-bnb-4bit) · MLX [bf16](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Instruct-MLX-bf16) / [8bit](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Instruct-MLX-8bit) / [4bit](https://huggingface.co/yuhuanstudio/Yunmo-Next-1B-Instruct-MLX-4bit) |
| Tokenizer | [Yunmo-Next-Tokenizer](https://huggingface.co/yuhuanstudio/Yunmo-Next-Tokenizer) |

Figures are generated from `data/` by `scripts/make_paper_figures.py` (plots) and `figures/src/architecture.tex` (architecture diagram, built with tectonic). `data/chat-eval/` holds every chat-mode reply
(gzipped JSONL per model, task and setting) with its score, `summary.json` and the item list;
`data/instruct/` holds the decoding parity checks; the other folders hold the raw outputs behind each
evaluation table (see the Reproducibility section of the evaluation report).

## License

Apache-2.0 ([LICENSE](LICENSE)). Scores of third-party models in `data/` were produced by us with the code linked from each report; the models themselves are under their own licenses.
