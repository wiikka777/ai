import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


CLASSES = ["AI", "Human", "Unknown"]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_scheme1_fpr_matrix(
    report_path: Path,
    out_path: Path,
    threshold_kind: str,
    title_prefix: str = "Scheme 1: Subset A FPR Analysis Matrix",
) -> None:
    report = load_json(report_path)
    key = f"class_counts_{threshold_kind}_threshold"
    counts = report["three_class_summary"][key]
    values = np.array([[int(counts.get(c, 0)) for c in CLASSES]], dtype=float)
    total = int(values.sum())
    norm = values / total if total > 0 else values

    if threshold_kind == "recommended":
        fpr = float(report["calibration"]["recommended_fpr"])
        th = float(report["calibration"]["recommended_threshold"])
    else:
        fpr = float(report["calibration"]["default_fpr"])
        th = float(report["calibration"]["default_threshold"])

    fig, ax = plt.subplots(figsize=(8.8, 2.9))
    im = ax.imshow(norm, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([f"Pred: {c}" for c in CLASSES])
    ax.set_yticks([0])
    ax.set_yticklabels(["Actual: Human"])
    ax.set_title(
        f"{title_prefix}\n{report.get('method', 'method')} / {threshold_kind} threshold={th:.6f} / FPR={fpr:.2%}",
        fontsize=11,
        fontweight="bold",
    )

    for j in range(3):
        cnt = int(values[0, j])
        pct = norm[0, j] * 100.0 if total > 0 else 0.0
        color = "white" if norm[0, j] > 0.45 else "black"
        ax.text(j, 0, f"{cnt}\n{pct:.1f}%", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Proportion")

    plt.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def read_scores(path: Path, column: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["sample_idx"])
            pred = row[column].strip()
            if pred not in CLASSES:
                pred = "Unknown"
            out[idx] = pred
    return out


def build_consistency_matrix(
    detect_map: Dict[int, str], gpt_map: Dict[int, str]
) -> Tuple[np.ndarray, List[int], int, int]:
    shared_ids = sorted(set(detect_map.keys()) & set(gpt_map.keys()))
    only_detect = len(set(detect_map.keys()) - set(gpt_map.keys()))
    only_gpt = len(set(gpt_map.keys()) - set(detect_map.keys()))

    mat = np.zeros((3, 3), dtype=int)
    to_i = {c: i for i, c in enumerate(CLASSES)}
    for sid in shared_ids:
        d = detect_map[sid]  # col
        g = gpt_map[sid]  # row
        mat[to_i[g], to_i[d]] += 1
    return mat, shared_ids, only_detect, only_gpt


def save_matrix_csv(path: Path, mat: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gptzero\\detectgpt", *CLASSES])
        for i, row_name in enumerate(CLASSES):
            writer.writerow([row_name, *mat[i].tolist()])


def plot_scheme2_consistency_matrix(
    detect_scores_csv: Path,
    gpt_scores_csv: Path,
    detect_col: str,
    gpt_col: str,
    out_png: Path,
    out_csv: Path,
) -> Tuple[Dict[int, str], Dict[int, str], List[int]]:
    detect_map = read_scores(detect_scores_csv, detect_col)
    gpt_map = read_scores(gpt_scores_csv, gpt_col)
    mat, shared_ids, only_detect, only_gpt = build_consistency_matrix(detect_map, gpt_map)
    save_matrix_csv(out_csv, mat)

    total = int(mat.sum())
    norm = mat / total if total > 0 else mat.astype(float)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=max(float(norm.max()), 0.01))
    ax.set_xticks(np.arange(3))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels([f"DetectGPT: {c}" for c in CLASSES], rotation=20, ha="right")
    ax.set_yticklabels([f"GPTZero: {c}" for c in CLASSES])
    ax.set_title(
        "Scheme 2: Cross-Model Consistency Matrix\n"
        f"(shared={len(shared_ids)}, detect_only={only_detect}, gptzero_only={only_gpt})",
        fontsize=11,
        fontweight="bold",
    )

    for i in range(3):
        for j in range(3):
            cnt = int(mat[i, j])
            pct = norm[i, j] * 100.0 if total > 0 else 0.0
            color = "white" if norm[i, j] > 0.45 * max(float(norm.max()), 0.01) else "black"
            ax.text(j, i, f"{cnt}\n{pct:.1f}%", ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_xlabel("DetectGPT decision")
    ax.set_ylabel("GPTZero decision")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.02)
    cbar.set_label("Proportion")
    plt.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return detect_map, gpt_map, shared_ids


def export_consistency_groups(
    detect_map: Dict[int, str],
    gpt_map: Dict[int, str],
    shared_ids: List[int],
    out_prefix: Path,
) -> None:
    all_rows = []
    ai_consensus = []
    ai_human_conflict = []
    unknown_related = []

    for sid in shared_ids:
        d = detect_map[sid]
        g = gpt_map[sid]
        if d == "AI" and g == "AI":
            group = "ai_consensus"
            ai_consensus.append((sid, d, g))
        elif (d == "AI" and g == "Human") or (d == "Human" and g == "AI"):
            group = "ai_human_conflict"
            ai_human_conflict.append((sid, d, g))
        elif d == "Unknown" or g == "Unknown":
            group = "unknown_related"
            unknown_related.append((sid, d, g))
        else:
            group = "both_human"

        all_rows.append((sid, d, g, group))

    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    def write_rows(path: Path, rows: List[Tuple], with_group: bool = False) -> None:
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if with_group:
                w.writerow(["sample_idx", "detectgpt", "gptzero", "group"])
            else:
                w.writerow(["sample_idx", "detectgpt", "gptzero"])
            for r in rows:
                w.writerow(r)

    write_rows(Path(f"{out_prefix}_all.csv"), all_rows, with_group=True)
    write_rows(Path(f"{out_prefix}_ai_consensus.csv"), ai_consensus)
    write_rows(Path(f"{out_prefix}_ai_human_conflict.csv"), ai_human_conflict)
    write_rows(Path(f"{out_prefix}_unknown_related.csv"), unknown_related)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Scheme 1 FPR matrix and Scheme 2 consistency matrix.")
    parser.add_argument(
        "--pre2022-report-detectgpt",
        default="/user/zhuohang.yu/u24922/exam/fpr_calibration_report_detectgpt.json",
    )
    parser.add_argument(
        "--scheme1-threshold-kind",
        choices=["default", "recommended"],
        default="recommended",
    )
    parser.add_argument(
        "--scheme1-out",
        default="/user/zhuohang.yu/u24922/exam/scheme1_subsetA_fpr_matrix.png",
    )

    parser.add_argument(
        "--post2024-detectgpt-scores",
        default="/user/zhuohang.yu/u24922/exam/post2024_fpr_calibration_scores_detectgpt.csv",
    )
    parser.add_argument(
        "--post2024-gptzero-scores",
        default="/user/zhuohang.yu/u24922/exam/post2024_fpr_calibration_scores_gptzero.csv",
    )
    parser.add_argument(
        "--scheme2-detectgpt-col",
        choices=["class_default", "class_recommended"],
        default="class_recommended",
    )
    parser.add_argument(
        "--scheme2-gptzero-col",
        choices=["class_default", "class_recommended"],
        default="class_recommended",
    )
    parser.add_argument(
        "--scheme2-out",
        default="/user/zhuohang.yu/u24922/exam/scheme2_post2024_consistency_matrix.png",
    )
    parser.add_argument(
        "--scheme2-matrix-csv",
        default="/user/zhuohang.yu/u24922/exam/scheme2_post2024_consistency_matrix.csv",
    )
    parser.add_argument(
        "--export-groups-csv",
        action="store_true",
        help="Export aligned sample lists for AI consensus / AI-Human conflict / Unknown-related groups.",
    )
    parser.add_argument(
        "--groups-out-prefix",
        default="/user/zhuohang.yu/u24922/exam/scheme2_post2024_groups",
    )
    args = parser.parse_args()

    plot_scheme1_fpr_matrix(
        report_path=Path(args.pre2022_report_detectgpt),
        out_path=Path(args.scheme1_out),
        threshold_kind=args.scheme1_threshold_kind,
    )
    print(f"Saved scheme 1: {args.scheme1_out}")

    detect_map, gpt_map, shared_ids = plot_scheme2_consistency_matrix(
        detect_scores_csv=Path(args.post2024_detectgpt_scores),
        gpt_scores_csv=Path(args.post2024_gptzero_scores),
        detect_col=args.scheme2_detectgpt_col,
        gpt_col=args.scheme2_gptzero_col,
        out_png=Path(args.scheme2_out),
        out_csv=Path(args.scheme2_matrix_csv),
    )
    print(f"Saved scheme 2: {args.scheme2_out}")
    print(f"Saved scheme 2 matrix csv: {args.scheme2_matrix_csv}")

    if args.export_groups_csv:
        export_consistency_groups(
            detect_map=detect_map,
            gpt_map=gpt_map,
            shared_ids=shared_ids,
            out_prefix=Path(args.groups_out_prefix),
        )
        print(f"Saved groups csv prefix: {args.groups_out_prefix}_*.csv")


if __name__ == "__main__":
    main()
