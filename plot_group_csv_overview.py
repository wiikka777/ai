import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


GROUP_ORDER = ["ai_consensus", "ai_human_conflict", "unknown_related", "both_human"]
GROUP_LABEL = {
    "ai_consensus": "AI Consensus",
    "ai_human_conflict": "AI/Human Conflict",
    "unknown_related": "Unknown Related",
    "both_human": "Both Human",
}
CLASS_ORDER = ["AI", "Human", "Unknown"]


def load_all_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "sample_idx": int(row["sample_idx"]),
                    "detectgpt": row["detectgpt"].strip(),
                    "gptzero": row["gptzero"].strip(),
                    "group": row["group"].strip(),
                }
            )
    return rows


def summarize(rows):
    group_counts = Counter(r["group"] for r in rows)
    detect_by_group = defaultdict(Counter)
    gpt_by_group = defaultdict(Counter)

    for r in rows:
        detect_by_group[r["group"]][r["detectgpt"]] += 1
        gpt_by_group[r["group"]][r["gptzero"]] += 1

    return group_counts, detect_by_group, gpt_by_group


def make_plot(rows, out_png: Path):
    group_counts, detect_by_group, gpt_by_group = summarize(rows)
    total = len(rows)

    groups = [g for g in GROUP_ORDER if g in group_counts]
    group_labels = [GROUP_LABEL.get(g, g) for g in groups]
    group_vals = [group_counts[g] for g in groups]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))

    # Left panel: group distribution
    ax = axes[0]
    x = np.arange(len(groups))
    colors = ["#C44E52", "#DD8452", "#8172B3", "#55A868"]
    bars = ax.bar(x, group_vals, color=colors[: len(groups)], edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=18, ha="right")
    ax.set_ylabel("Sample Count")
    ax.set_title("Group Size Distribution")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    for i, b in enumerate(bars):
        v = int(group_vals[i])
        pct = (v / total * 100.0) if total else 0.0
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v}\n{pct:.1f}%", ha="center", va="bottom", fontsize=9)

    # Right panel: model decisions inside each group
    ax = axes[1]
    positions = np.arange(len(groups))
    width = 0.36

    for cls_i, cls in enumerate(CLASS_ORDER):
        det_vals = [detect_by_group[g][cls] for g in groups]
        gpt_vals = [gpt_by_group[g][cls] for g in groups]

        det_bottom = np.sum(
            [[detect_by_group[g][c] for g in groups] for c in CLASS_ORDER[:cls_i]], axis=0
        ) if cls_i > 0 else np.zeros(len(groups), dtype=int)
        gpt_bottom = np.sum(
            [[gpt_by_group[g][c] for g in groups] for c in CLASS_ORDER[:cls_i]], axis=0
        ) if cls_i > 0 else np.zeros(len(groups), dtype=int)

        ax.bar(
            positions - width / 2,
            det_vals,
            width=width,
            bottom=det_bottom,
            label=f"DetectGPT {cls}",
            alpha=0.9,
        )
        ax.bar(
            positions + width / 2,
            gpt_vals,
            width=width,
            bottom=gpt_bottom,
            label=f"GPTZero {cls}",
            alpha=0.55,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(group_labels, rotation=18, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Per-Group Decision Composition")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)

    fig.suptitle(f"Scheme 2 Group CSV Overview (n={total})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize group CSV exports from scheme 2.")
    parser.add_argument(
        "--all-csv",
        default="/user/zhuohang.yu/u24922/exam/scheme2_post2024_groups_default_all.csv",
    )
    parser.add_argument(
        "--out",
        default="/user/zhuohang.yu/u24922/exam/scheme2_post2024_groups_default_overview.png",
    )
    args = parser.parse_args()

    rows = load_all_csv(Path(args.all_csv))
    make_plot(rows, Path(args.out))
    print(f"Saved overview plot: {args.out}")


if __name__ == "__main__":
    main()
