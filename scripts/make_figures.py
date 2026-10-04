"""Render the report figures from the CSV files in ../data.

    python scripts/make_figures.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA, FIGURES = ROOT / "data", ROOT / "figures"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#b9b8b2"
SERIES_1 = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "text.color": TEXT, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def read(name):
    with open(DATA / name, newline="") as handle:
        return list(csv.DictReader(handle))


def heldout_trajectory():
    rows = [r for r in read("heldout_bpb_trajectory.csv") if "20.16B" not in r["stage"]]
    x = [int(r["cumulative_tokens"]) / 1e9 for r in rows]
    y = [float(r["heldout_bpb_8k"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(x, y, color=SERIES_1, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
    labels = {"K24": (0, 8), "K36": (-14, -16), "K48": (0, -16), "K48-32K": (0, 8), "K60": (0, 8),
              "K60-capability": (0, -16), "K60-extension": (-10, 8),
              "K60-anneal-19.76B (released)": (-28, -14)}
    for r, xi, yi in zip(rows, x, y):
        name = r["stage"].replace(" (released)", "")
        dx, dy = labels[r["stage"]]
        ax.annotate(f"{name}\n{r['layers']}L, {int(r['context']) // 1024}K", (xi, yi), textcoords="offset points",
                    xytext=(dx, dy), ha="center", va="bottom" if dy > 0 else "top", fontsize=8, color=TEXT_2)
    ax.set_xlabel("Cumulative training tokens (billions)")
    ax.set_ylabel("Held-out bits per byte (8K context, lower is better)")
    ax.set_title("Held-out BPB across the depth-grown training route", loc="left", fontsize=11)
    ax.set_xlim(0, 21)
    ax.set_ylim(0.62, 0.86)
    fig.tight_layout()
    fig.savefig(FIGURES / "heldout_bpb_trajectory.png", dpi=200)
    plt.close(fig)


DOMAIN_LABELS = {
    "hant-web": "Trad. Chinese web", "hant-wiki": "Trad. Chinese Wikipedia", "tw-law": "Taiwan statutes",
    "tw-judgment": "Taiwan court judgments", "tw-patent": "Taiwan patents", "hant-classical": "Classical Chinese",
    "zhuyin-bopomofo": "Zhuyin", "hans-web": "Simp. Chinese web", "hans-general": "Simp. Chinese general",
    "english-edu": "English (educational)", "math": "Mathematics", "code": "Code",
}


def tokenizer_strip():
    rows = read("tokenizer_bytes_per_token.csv")
    domains = list(DOMAIN_LABELS)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for i, domain in enumerate(domains):
        others = [float(r[domain]) for r in rows if r["tokenizer"] != "yunmo-next"]
        ours = next(float(r[domain]) for r in rows if r["tokenizer"] == "yunmo-next")
        rank = 1 + sum(value > ours for value in others)
        ax.scatter(others, [i] * len(others), s=36, color=NEUTRAL, edgecolors=SURFACE, linewidths=1, zorder=2)
        ax.scatter([ours], [i], s=70, color=SERIES_1, edgecolors=SURFACE, linewidths=2, zorder=3)
        ax.annotate(f"#{rank}", (1.0, i), xycoords=("axes fraction", "data"), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=TEXT_2)
    ax.set_yticks(range(len(domains)), [DOMAIN_LABELS[d] for d in domains])
    ax.invert_yaxis()
    ax.set_xlabel("UTF-8 bytes per token (higher = fewer tokens)")
    ax.set_title("Yunmo-Next (blue) vs 17 other tokenizers (gray); rank at right", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "tokenizer_compression.png", dpi=200)
    plt.close(fig)


def pretraining_loss():
    rows = read("pretraining_loss.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    segments, current, last = [], [], None
    for r in rows:
        x, y = int(r["tokens_seen"]) / 1e9, float(r["train_loss"])
        if last is not None and (r["stage"] != last[0] or x - last[1] > 0.05):
            segments.append(current)
            current = []
        current.append((x, y))
        last = (r["stage"], x)
    segments.append(current)
    for seg in segments:
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        k = 25  # centered rolling median over logged windows
        smooth = [sorted(ys[max(0, i - k):i + k + 1])[len(ys[max(0, i - k):i + k + 1]) // 2] for i in range(len(ys))]
        ax.plot(xs, ys, color=NEUTRAL, linewidth=0.5, alpha=0.6, zorder=1)
        ax.plot(xs, smooth, color=SERIES_1, linewidth=2, zorder=2)
    for start, end, label in ((0, 1.2, "not exported"), (4.0, 6.5, "K48 8K\nnot exported"), (10.57, 12.0, "not\nexported")):
        ax.axvspan(start, end, color=GRID, alpha=0.5, zorder=0)
        ax.text((start + end) / 2, 3.55, label, ha="center", va="top", fontsize=7.5, color=TEXT_2)
    for x, label, y in ((2.0, "K36", 2.12), (6.5, "K48 32K", 2.26), (7.0, "K60", 2.12), (12.0, "capability", 2.12),
                        (14.4, "extension", 2.12), (18.56, "anneal", 2.12)):
        ax.axvline(x, color=TEXT_2, linewidth=0.6, linestyle=":", zorder=0)
        ax.text(x + 0.12, y, label, fontsize=7.5, color=TEXT_2, va="bottom")
    ax.axvline(19.761, color=SERIES_1, linewidth=1, zorder=0)
    ax.text(19.6, 3.5, "released\n19.76B", fontsize=7.5, color=SERIES_1, ha="right", va="top")
    ax.set_xlim(0, 20.5)
    ax.set_ylim(2.1, 3.6)
    ax.set_xlabel("Cumulative training tokens (billions)")
    ax.set_ylabel("Training loss (nats/token)")
    ax.set_title("Pretraining loss (gray: logged windows, blue: rolling median)", loc="left", fontsize=11)
    ax.text(0.0, -0.17, "Loss levels are not comparable across stages: data mixture and context length change at each boundary.",
            transform=ax.transAxes, fontsize=7.5, color=TEXT_2)
    fig.tight_layout()
    fig.savefig(FIGURES / "pretraining_loss.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    heldout_trajectory()
    tokenizer_strip()
    pretraining_loss()
