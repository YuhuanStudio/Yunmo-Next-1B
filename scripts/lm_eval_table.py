"""Build the lm-eval comparison table from raw lm-evaluation-harness outputs."""
import json, sys
from pathlib import Path

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "data" / "lm-eval"
ORDER = ["Yunmo-Next-1B-Base", "Qwen3.5-0.8B-Base", "LFM2.5-1.2B-Base", "MiniCPM5-1B-Base", "Qwen3.5-2B-Base", "MiniCPM5-2B-Base"]
METRIC = {"arc_easy": "acc_norm,none", "arc_challenge": "acc_norm,none", "hellaswag": "acc_norm,none",
          "piqa": "acc_norm,none", "winogrande": "acc,none", "openbookqa": "acc_norm,none", "sciq": "acc_norm,none",
          "lambada_openai": "acc,none", "mmlu": "acc,none", "tmmluplus": "acc,none",
          "xstorycloze_zh": "acc,none", "xcopa_zh": "acc,none", "xwinograd_zh": "acc,none"}
rows = {}
for name in ORDER:
    f = SRC / f"{name}.json"
    if not f.exists():
        continue
    tasks = json.loads(f.read_text())["tasks"]
    rows[name] = {t: tasks[t]["results"][t][m] for t, m in METRIC.items() if t in tasks}
print("| Task | " + " | ".join(rows) + " |")
print("|---|" + "---:|" * len(rows))
for t in METRIC:
    print(f"| {t} | " + " | ".join(f"{100 * rows[n][t]:.1f}" if t in rows[n] else "—" for n in rows) + " |")
