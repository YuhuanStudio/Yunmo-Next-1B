"""Paper-style figures for the overview and reports (vector PDF plus 300-dpi PNG).

    uv run --with matplotlib python scripts/make_paper_figures.py
    tectonic figures/src/architecture.tex   # the architecture diagram (TikZ)

Style: serif text (STIX), 8-9 pt, single-column 3.4 in or full-width 7 in, Okabe-Ito colours.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGURES = ROOT / "data", ROOT / "figures"

BLUE, ORANGE, GREEN, VERMILION, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#8C8C8C"
PURPLE = "#CC79A7"
# One colour per model across all figures.
MODEL_COLOR = {"Yunmo-Next-1B": VERMILION, "Qwen3.5-0.8B": BLUE, "MiniCPM5-1B": GREEN, "LFM2.5-1.2B": ORANGE,
               "Barbet-1B": PURPLE, "Qwen3.5-2B": SKY}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def save(fig, name):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(FIGURES / f"{name}.{ext}", **kw)
    plt.close(fig)
    print(FIGURES / f"{name}.png")


def lm_eval(path):
    data = json.loads(Path(path).read_text())["tasks"]
    norm = {"arc_easy", "arc_challenge", "hellaswag", "piqa", "openbookqa", "sciq"}
    return {t: 100 * v["results"][t]["acc_norm,none" if t in norm else "acc,none"] for t, v in data.items()}


ENGLISH = ("arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "openbookqa", "sciq", "lambada_openai")


def training_curve():
    """Held-out bits per byte at the end of each training stage, one point per stage (categorical x)."""
    rows = [r for r in csv.DictReader(open(DATA / "heldout_bpb_trajectory.csv")) if "20.16B" not in r["stage"]]
    names = {"K24": "24 layers\n8K", "K36": "36 layers\n8K", "K48": "48 layers\n8K", "K48-32K": "48 layers\n32K",
             "K60": "60 layers\n16K", "K60-capability": "60 layers\nnew mixture", "K60-extension": "60 layers\nextension",
             "K60-anneal-19.76B (released)": "60 layers\nanneal"}
    xs = range(len(rows))
    ys = [float(r["heldout_bpb_8k"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    ax.plot(xs, ys, color=BLUE, lw=1.3, marker="o", ms=3.5, zorder=3)
    for x, y, r in zip(xs, ys, rows):
        ax.annotate(f"{y:.3f}", (x, y), xytext=(0, -11 if r["stage"] in ("K48", "K60") else 6), textcoords="offset points",
                    ha="center", fontsize=6.5)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"{names[r['stage']]}\n{int(r['cumulative_tokens']) / 1e9:.1f}B" for r in rows], fontsize=6.5)
    ax.set_xlim(-0.4, len(rows) - 0.6)
    ax.set_ylim(0.65, 0.86)
    ax.set_ylabel("Held-out bits per byte ($\\downarrow$)")
    ax.grid(axis="y", color="#E6E6E6", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "training_curve")


def pretraining_loss():
    """Training loss across the route. Hatched spans: training happened but its loss log was not kept."""
    rows = list(csv.DictReader(open(DATA / "pretraining_loss.csv")))
    stages = {}
    for r in rows:
        stages.setdefault(r["stage"], []).append((int(r["tokens_seen"]) / 1e9, float(r["train_loss"])))
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    for points in stages.values():
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        k = 25
        smooth = [sorted(ys[max(0, i - k):i + k + 1])[len(ys[max(0, i - k):i + k + 1]) // 2] for i in range(len(ys))]
        ax.plot(xs, ys, color="#CFCFCF", lw=0.3, zorder=1)
        ax.plot(xs, smooth, color=BLUE, lw=1.1, zorder=2)
    for a, b in ((0, 1.2), (4.0, 6.5), (10.57, 12.0)):
        ax.axvspan(a, b, facecolor="none", edgecolor="#DDDDDD", hatch="////", lw=0, zorder=0)
        ax.text((a + b) / 2, 2.27, "log not\nkept", ha="center", va="bottom", fontsize=5.8, color="#666666")
    bounds = [(0, "24L 8K"), (2.0, "36L 8K"), (4.0, "48L 8K"), (6.5, "48L 32K"), (7.0, "60L 16K"),
              (12.0, "new mixture"), (14.4, "extension"), (18.56, "anneal")]
    for x, label in bounds:
        if x:
            ax.axvline(x, color="#888888", lw=0.5, ls=":", zorder=0)
        ax.text(x + 0.1, 3.27 if label == "48L 32K" else 3.36, label, fontsize=5.8, color="#444444", va="top")
    ax.axvline(19.761, color=VERMILION, lw=0.8)
    ax.text(19.85, 3.0, "released\n19.76B", fontsize=5.8, color=VERMILION, va="center")
    ax.set_xlim(0, 21.2)
    ax.set_ylim(2.2, 3.4)
    ax.set_xlabel("Training tokens (B)")
    ax.set_ylabel("Training loss (nats / token)")
    save(fig, "pretraining_loss")


def tokenizer_domains():
    """Bytes per token of 18 tokenizers on 12 domains; Yunmo-Next highlighted, its rank at right."""
    rows = list(csv.DictReader(open(DATA / "tokenizer_bytes_per_token.csv")))
    domains = [("hant-web", "Traditional Chinese web"), ("hant-wiki", "Traditional Chinese Wikipedia"),
               ("tw-law", "Taiwan statutes"), ("tw-judgment", "Taiwan court judgments"), ("tw-patent", "Taiwan patents"),
               ("hant-classical", "Classical Chinese"), ("zhuyin-bopomofo", "Zhuyin"),
               ("hans-web", "Simplified Chinese web"), ("hans-general", "Simplified Chinese, general"),
               ("english-edu", "English, educational"), ("math", "Mathematics"), ("code", "Code")]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for i, (key, label) in enumerate(domains):
        y = len(domains) - 1 - i
        values = [(float(r[key]), r["tokenizer"]) for r in rows]
        others = [v for v, n in values if n != "yunmo-next"]
        ours = next(v for v, n in values if n == "yunmo-next")
        ax.scatter(others, [y] * len(others), s=7, color=GREY, alpha=0.7, lw=0, zorder=2)
        ax.scatter([ours], [y], s=18, color=VERMILION, zorder=3)
        rank = 1 + sum(v > ours for v in others)
        ax.text(5.55, y, f"#{rank}", fontsize=6.3, va="center", ha="right")
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([d[1] for d in reversed(domains)], fontsize=6.8)
    ax.set_xlim(1.2, 5.6)
    ax.set_xlabel("UTF-8 bytes per token ($\\uparrow$)")
    ax.grid(axis="x", color="#E6E6E6", lw=0.5)
    ax.set_axisbelow(True)
    save(fig, "tokenizer_compression")


def fresh_wiki_bpb():
    models = [("Yunmo-Next-1B-Base", "Yunmo-Next-1B (ours)"), ("MiniCPM5-2B-Base", "MiniCPM5-2B"),
              ("Qwen3.5-2B-Base", "Qwen3.5-2B"), ("Qwen3.5-0.8B-Base", "Qwen3.5-0.8B"),
              ("MiniCPM5-1B-Base", "MiniCPM5-1B"), ("barbet-1b-structured-sft", "Barbet-1B"),
              ("LFM2.5-1.2B-Base", "LFM2.5-1.2B")]
    get = lambda f, v: json.loads((DATA / "fresh-wiki-bpb" / f"{f}.{v}.json").read_text())["bits_per_byte"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    h = 0.38
    for i, (f, label) in enumerate(models):
        y = len(models) - 1 - i
        tw, cn = get(f, "tw"), get(f, "cn")
        ax.barh(y + h / 2, tw, h, color=BLUE, label="Traditional" if i == 0 else None)
        ax.barh(y - h / 2, cn, h, color=ORANGE, label="Simplified" if i == 0 else None)
        ax.text(tw + 0.008, y + h / 2, f"{tw:.3f}", va="center", fontsize=6.3)
        ax.text(cn + 0.008, y - h / 2, f"{cn:.3f}", va="center", fontsize=6.3, color="#555555")
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m[1] for m in reversed(models)])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlim(0.8, 1.45)
    ax.set_xlabel("Bits per byte ($\\downarrow$)")
    ax.legend(loc="upper right")
    save(fig, "fresh_wiki_bpb")


def compute_scatter():
    tokens_per_step = 2_097_152
    avg = lambda r: sum(r[t] for t in ENGLISH) / len(ENGLISH)
    pythia_dir = DATA / "lm-eval/compute-matched-pythia"
    sizes = {"70m": 70_426_624, "160m": 162_322_944, "410m": 405_334_016, "1b": 1_011_781_632}
    steps = {"70m": 118_000, "160m": 51_000, "410m": 21_000, "1b": 8_000}
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    # Pythia: one line per size, the matched intermediate checkpoint and the final (step 143,000).
    for name, n in sizes.items():
        xs = [6 * n * steps[name] * tokens_per_step, 6 * n * 143_000 * tokens_per_step]
        ys = [avg(lm_eval(pythia_dir / f"pythia-{name}@step{steps[name]}.json")), avg(lm_eval(pythia_dir / f"pythia-{name}.json"))]
        ax.plot(xs, ys, color=GREY, lw=0.7, ls=":", zorder=2)
        ax.scatter(xs[:1], ys[:1], s=14, color=GREY, zorder=3, label="Pythia (2023), mid-training" if name == "70m" else None)
        ax.scatter(xs[1:], ys[1:], s=14, facecolor="white", edgecolor=GREY, lw=0.8, zorder=3,
                   label="Pythia (2023), final" if name == "70m" else None)
        ax.annotate(name.upper(), (xs[1], ys[1]), xytext=(4, 0), textcoords="offset points", fontsize=5.8,
                    color="#555555", va="center")
    # DataDecide (2025): fully annealed small models, two data recipes.
    dd = DATA / "lm-eval/compute-matched-2025"
    for size, n, d in (("300M", 371.5e6, 30e9), ("530M", 597.7e6, 53e9)):
        for recipe, marker in (("dclm-baseline", "^"), ("fineweb-edu", "v")):
            y = avg(lm_eval(dd / f"DataDecide-{recipe}-{size}.json"))
            ax.scatter([6 * n * d], [y], s=18, marker=marker, color=BLUE, zorder=3,
                       label=f"DataDecide (2025), {'DCLM' if recipe == 'dclm-baseline' else 'FineWeb-Edu'}"
                       if size == "300M" else None)
        ax.annotate(size, (6 * n * d, y), xytext=(0, -10) if size == "300M" else (5, -1),
                    textcoords="offset points", fontsize=5.8, color="#555555", va="center",
                    ha="center" if size == "300M" else "left")
    olmo = avg(lm_eval(dd / "OLMo-2-0425-1B-early-training@stage1-step6000-tokens13B.json"))
    ax.scatter([6 * 1.485e9 * 13e9], [olmo], s=18, marker="s", color=SKY, zorder=3, label="OLMo 2 1B (2025), 13B tokens")
    yunmo = avg(lm_eval(DATA / "lm-eval/Yunmo-Next-1B-Base.json"))
    ax.scatter([1.05e20], [yunmo], s=70, color=VERMILION, marker="*", zorder=4, label="Yunmo-Next-1B (ours)")
    ax.set_xscale("log")
    ax.set_xlabel("Pretraining compute (FLOPs, $6ND$)")
    ax.set_ylabel("Average accuracy, 8 English tasks (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=5.8, handletextpad=0.3,
              labelspacing=0.3, columnspacing=1.0)
    ax.set_ylim(30, 58)
    ax.set_xlim(5e19, 3e21)
    save(fig, "compute_scatter")


def _cloze(label):
    return json.loads((DATA / "cloze-continuous" / f"{label}.json").read_text())["tasks"]


def cloze_continuous():
    """TMMLU+ in cloze form: accuracy is near chance for every 1B-class model, bits per byte of the correct
    answer is not."""
    rows = [("Yunmo-Next-1B-Base", "Yunmo-Next-1B (ours)"), ("Qwen3.5-0.8B-Base", "Qwen3.5-0.8B"),
            ("MiniCPM5-1B-Base", "MiniCPM5-1B"), ("barbet-1b-structured-sft", "Barbet-1B (SFT)"),
            ("LFM2.5-1.2B-Base", "LFM2.5-1.2B"), ("pythia-1b", "Pythia-1B, final"),
            ("DataDecide-fineweb-edu-530M", "DataDecide-530M, FW-Edu"),
            ("DataDecide-dclm-baseline-530M", "DataDecide-530M, DCLM"),
            ("OLMo-2-0425-1B-early-training@stage1-step6000-tokens13B", "OLMo 2 1B, 13B tokens"),
            ("Qwen3.5-2B-Base", "Qwen3.5-2B (2$\\times$ size)")]
    data = []
    for f, label in rows:
        t = _cloze(f)["tmmluplus_cloze"]
        acc = t["results"]["tmmluplus_cloze"]
        data.append((label, 100 * t["continuous"]["correct_probability"], t["continuous"]["correct_bits_per_byte"]))
    data.sort(key=lambda r: -r[2])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True, gridspec_kw={"wspace": 0.08})
    for i, (label, acc, bpb) in enumerate(data):
        color = VERMILION if "ours" in label else (GREY if any(k in label for k in ("Pythia", "DataDecide", "OLMo")) else SKY)
        a1.barh(i, acc, 0.65, color=color)
        a1.text(acc + 0.15, i, f"{acc:.1f}", va="center", fontsize=6.3, bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        a2.barh(i, bpb, 0.65, color=color)
        a2.text(bpb + 0.03, i, f"{bpb:.2f}", va="center", fontsize=6.3)
    a1.axvline(25, color="#444444", lw=0.6, ls="--")
    a1.text(25.2, -0.75, "chance", fontsize=6.3, color="#444444")
    a1.set_xlim(20, 30)
    a1.set_yticks(range(len(data)))
    a1.set_yticklabels([d[0] for d in data])
    a1.set_xlabel("(a) Probability of the correct choice (%)")
    a2.set_xlim(1.0, 3.3)
    a2.set_xlabel("(b) Bits per byte of the correct answer ($\\downarrow$)")
    for lab in a1.get_yticklabels():
        if "ours" in lab.get_text():
            lab.set_fontweight("bold")
    save(fig, "tmmluplus_cloze")


def heldout_text():
    """Bits per byte on five Traditional Chinese test sets: new Wikipedia articles (all, and those with no
    25-character overlap with our Wikipedia training data) and three non-Wikipedia OpenFormosa corpora."""
    models = [("Yunmo-Next-1B-Base", "Yunmo-Next-1B"), ("Qwen3.5-0.8B-Base", "Qwen3.5-0.8B"),
              ("MiniCPM5-1B-Base", "MiniCPM5-1B"), ("LFM2.5-1.2B-Base", "LFM2.5-1.2B"),
              ("barbet-1b-structured-sft", "Barbet-1B")]
    overlap = {r["id"]: r["covered_share"] for r in
               json.loads((DATA / "fresh-wiki-bpb/overlap-tw.json").read_text())["per_document"]}

    def wiki(f, keep):
        docs = json.loads((DATA / "fresh-wiki-bpb" / f"{f}.tw.json").read_text())["per_document"]
        docs = [d for d in docs if keep(d)]
        return sum(d["nll_nats"] for d in docs) / math.log(2) / sum(d["bytes"] for d in docs)

    nonwiki = lambda f, d: json.loads((DATA / "heldout-nonwiki/scores" / f"{f}.{d}.json").read_text())["bits_per_byte"]
    panels = [("New Wikipedia\n(all 3,754)", lambda f: wiki(f, lambda d: True)),
              ("New Wikipedia\n(no overlap, 3,077)", lambda f: wiki(f, lambda d: overlap[d["id"]] == 0)),
              ("Legislative Yuan\ntranscripts", lambda f: nonwiki(f, "of-parliament")),
              ("Common Voice\nsentences", lambda f: nonwiki(f, "of-common-voice")),
              ("Localization\nreferences", lambda f: nonwiki(f, "of-tw-localization"))]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.0, 1.9), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (title, get) in zip(axes, panels):
        values = [get(f) for f, _ in models]
        for i, ((_, label), v) in enumerate(zip(models, values)):
            y = len(models) - 1 - i
            ax.barh(y, v, 0.65, color=MODEL_COLOR[label])
            ax.text(v * 1.01, y, f"{v:.2f}", va="center", fontsize=5.8)
        lo, hi = min(values), max(values)
        ax.set_xlim(lo - (hi - lo) * 0.6, hi + (hi - lo) * 0.35)
        ax.set_title(title, fontsize=7.2)
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels([m[1] for m in reversed(models)])
    fig.supxlabel("Bits per byte ($\\downarrow$)", fontsize=8, y=-0.08)
    save(fig, "heldout_text")


def long_context():
    models = [("Yunmo-Next-1B-Base", "Yunmo-Next-1B"), ("Qwen3.5-0.8B-Base", "Qwen3.5-0.8B"),
              ("MiniCPM5-1B-Base", "MiniCPM5-1B"), ("LFM2.5-1.2B-Base", "LFM2.5-1.2B")]
    lc = DATA / "long-context"
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.1), gridspec_kw={"wspace": 0.42})
    for f, label in models:
        style = dict(color=MODEL_COLOR[label], lw=1.5 if "Yunmo" in label else 1.0, marker="o", ms=2.5)
        rows = [r for r in json.loads((lc / f"position-{f}.json").read_text())["rows"] if r["bits_per_byte"]]
        a.plot([(r["chars_from"] + r["chars_to"]) / 2 / 1000 for r in rows], [r["bits_per_byte"] for r in rows],
               label=label, **style)
        eff = json.loads((lc / f"efficiency-{f}.json").read_text())["rows"]
        ctx = [r["context_tokens"] / 1024 for r in eff]
        b.plot(ctx, [r["state_bytes_after_prefill"] / 2**20 for r in eff], **style)
        c.plot(ctx, [r["peak_allocated_bytes"] / 2**30 for r in eff], **style)
    a.set_xscale("log")
    a.set_xticks([0.25, 1, 4, 16])
    a.set_xticklabels(["0.25", "1", "4", "16"])
    a.set_xlabel("Position in document (K characters)")
    a.set_ylabel("Bits per byte ($\\downarrow$)")
    a.set_title("(a) Loss by position", fontsize=7.5)
    a.legend(fontsize=5.8, loc="upper right", handlelength=1.4)
    for ax in (b, c):
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 4, 8, 16, 32])
        ax.set_xticklabels(["1K", "4K", "8K", "16K", "32K"])
        ax.set_xlabel("Context length (tokens)")
    b.set_ylabel("MiB")
    b.set_title("(b) Cache / state after prefill", fontsize=7.5)
    c.set_ylabel("GiB")
    c.set_title("(c) Peak GPU memory", fontsize=7.5)
    c.set_ylim(0, 5.5)
    save(fig, "long_context")


def quantization():
    """Six weight formats of the Instruct model: memory, loss increase, decode latency (aligned rows)."""
    q, lc = DATA / "quantization", DATA / "long-context"
    rows = [("BF16", "bf16-Instruct.tw.json", None), ("INT8 (torchao)", "quant-Instruct-torchao-int8.tw.json", "INT8"),
            ("FP8 (torchao)", "quant-Instruct-fp8.tw.json", "FP8"),
            ("LLM.int8 (bitsandbytes)", "quant-Instruct-int8.tw.json", "bnb-int8"),
            ("NVFP4 (torchao)", "quant-Instruct-nvfp4.tw.json", "NVFP4"),
            ("NF4 (bitsandbytes)", "quant-Instruct-nf4.tw.json", "bnb-nf4")]
    base = json.loads((q / "bf16-Instruct.tw.json").read_text())["bits_per_byte"]
    data = []
    for label, score, eff in rows:
        bpb = json.loads((q / score).read_text())["bits_per_byte"]
        e = json.loads((lc / (f"efficiency-Yunmo-Next-1B-Instruct-{eff}.json" if eff else
                              "efficiency-Yunmo-Next-1B-Base.json")).read_text())
        size = e["weights_bytes"] / 1e9
        ms = e["rows"][0].get("static_decode_ms_per_token") or e["rows"][0]["decode_ms_per_token"]
        color = VERMILION if label == "BF16" else (BLUE if "torchao" in label else GREY)
        data.append((label, size, bpb - base, ms, color))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9), sharey=True, gridspec_kw={"wspace": 0.12})
    titles = ("(a) Weights in GPU memory (GB)", "(b) $\\Delta$ bits per byte vs BF16", "(c) ms per token, 1K context")
    for k, (ax, title) in enumerate(zip(axes, titles)):
        for i, (label, size, delta, ms, color) in enumerate(data):
            y = len(data) - 1 - i
            v = (size, delta, ms)[k]
            ax.barh(y, v, 0.62, color=color)
            text = (f"{v:.2f}", "—" if label == "BF16" else f"+{v:.4f}", f"{v:.1f}")[k]
            ax.text(v * (1.12 if k == 2 else 1.0) + (0 if k == 2 else (0.04, 0.0002)[k]), y, text, va="center",
                    fontsize=6)
        ax.set_title(title, fontsize=7.2)
        ax.tick_params(axis="x", labelsize=6)
    axes[0].set_xlim(0, 2.6)
    axes[1].set_xlim(0, 0.0125)
    axes[1].set_xticks([0, 0.005, 0.01])
    axes[2].set_xscale("log")
    axes[2].set_xlim(5, 1800)
    axes[0].set_yticks(range(len(data)))
    axes[0].set_yticklabels([d[0] for d in reversed(data)], fontsize=7)
    save(fig, "quantization")


def tokenizer_ood():
    data = json.loads((DATA / "tokenizer-ood/results.json").read_text())
    rows = sorted(((v["domains"]["fresh-wiki-tw"]["bytes_per_token"], k, v["vocab_size"]) for k, v in data.items() if not k.startswith("Barbet")),
                  reverse=True)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for i, (bpt, name, vocab) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, bpt, 0.65, color=VERMILION if name == "Yunmo-Next" else SKY)
        ax.text(bpt + 0.03, y, f"{bpt:.2f}", va="center", fontsize=6.3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{n} ({v // 1000}K)" for _, n, v in reversed(rows)])
    ax.set_xlabel("UTF-8 bytes per token on new Traditional Chinese Wikipedia ($\\uparrow$)")
    ax.set_xlim(2.0, 4.0)
    save(fig, "tokenizer_ood")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    training_curve()
    fresh_wiki_bpb()
    compute_scatter()
    tokenizer_ood()
    cloze_continuous()
    heldout_text()
    long_context()
    quantization()
    pretraining_loss()
    tokenizer_domains()
