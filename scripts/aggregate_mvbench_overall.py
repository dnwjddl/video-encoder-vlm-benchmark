#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from vlmevalbench.data import read_jsonl


DISPLAY_NAMES = {
    "clip-vit-l-14-336": "CLIP-L/14",
    "siglip-so400m": "SigLIP",
    "siglip2-so400m": "SigLIP2",
    "internvit-300m": "InternViT",
    "dinov2-vitl14": "DINOv2-L",
    "videomaev2-base": "VideoMAEv2",
    "vjepa2-vith-256": "V-JEPA2",
    "internvideo2-clip-s": "InternVideo2-S",
}

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate plain MVBench original-video overall accuracy.")
    parser.add_argument("--manifest", default="data/benchmarks/mvbench_all.jsonl")
    parser.add_argument("--eval-root", default="outputs/mvbench/projector_eval")
    parser.add_argument("--out", default="outputs/mvbench/analysis/overall_accuracy.csv")
    parser.add_argument("--plot-prefix", default=None, help="Output prefix for PNG/PDF overall accuracy figure.")
    parser.add_argument("--encoders", default=None, help="Comma-separated encoder list. Defaults to discovered eval dirs.")
    return parser.parse_args()


def discover_encoders(eval_root: Path, encoders_csv: str | None) -> list[str]:
    if encoders_csv:
        return [item.strip() for item in encoders_csv.split(",") if item.strip()]
    if not eval_root.exists():
        return []
    return sorted(path.name for path in eval_root.iterdir() if path.is_dir())


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return {str(row["id"]): row for row in read_jsonl(path)}


def display_name(encoder: str) -> str:
    return DISPLAY_NAMES.get(encoder, encoder)


def plot_overall_accuracy(rows: list[dict[str, Any]], out_prefix: Path) -> None:
    if not rows:
        return
    labels = [display_name(str(row["encoder"])) for row in rows]
    values = [float(row["overall_accuracy"]) for row in rows]
    complete = [bool(row["complete"]) for row in rows]

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.15), 6), constrained_layout=True)
    bars = ax.bar(
        range(len(rows)),
        values,
        color=[PALETTE[idx % len(PALETTE)] if complete[idx] else "#BAB0AC" for idx in range(len(rows))],
        linewidth=0,
    )
    for idx, (bar, value, row) in enumerate(zip(bars, values, rows)):
        label = f"{value * 100:.1f}%"
        if not complete[idx]:
            label += "*"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.015,
            f"{row['correct']}/{row['manifest_examples']}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="white",
            rotation=90 if value < 0.12 else 0,
        )

    ax.set_title("MVBench Overall Accuracy", loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, min(max(max(values) + 0.12, 0.45), 1.05))
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0,
        -0.22,
        "* incomplete: original-video predictions do not cover every manifest item.",
        transform=ax.transAxes,
        fontsize=8,
        va="top",
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote MVBench overall accuracy figure to {png_path}")
    print(f"Wrote MVBench overall accuracy figure to {pdf_path}")


def main() -> None:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    manifest_ids = {str(row["id"]) for row in manifest_rows}
    manifest_n = len(manifest_ids)
    eval_root = Path(args.eval_root)
    rows = []

    for encoder in discover_encoders(eval_root, args.encoders):
        pred_path = eval_root / encoder / "original" / "predictions.jsonl"
        predictions = load_predictions(pred_path)
        evaluated_ids = manifest_ids & set(predictions)
        extra_ids = set(predictions) - manifest_ids
        correct = sum(bool(predictions[item_id].get("correct")) for item_id in evaluated_ids)
        evaluated_n = len(evaluated_ids)
        missing_n = manifest_n - evaluated_n
        rows.append(
            {
                "encoder": encoder,
                "manifest_examples": manifest_n,
                "evaluated_examples": evaluated_n,
                "missing_predictions": missing_n,
                "extra_predictions_ignored": len(extra_ids),
                "correct": correct,
                "overall_accuracy": correct / max(manifest_n, 1),
                "accuracy_on_evaluated": correct / max(evaluated_n, 1),
                "complete": evaluated_n == manifest_n,
                "prediction_file": str(pred_path),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "encoder",
            "manifest_examples",
            "evaluated_examples",
            "missing_predictions",
            "extra_predictions_ignored",
            "correct",
            "overall_accuracy",
            "accuracy_on_evaluated",
            "complete",
            "prediction_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_prefix = Path(args.plot_prefix) if args.plot_prefix else out_path.with_suffix("")
    plot_overall_accuracy(rows, plot_prefix)

    print(f"Wrote MVBench overall accuracy to {out_path}")
    for row in rows:
        print(
            f"{row['encoder']}: overall={row['overall_accuracy']:.4f} "
            f"correct={row['correct']}/{row['manifest_examples']} "
            f"evaluated={row['evaluated_examples']} complete={row['complete']}"
        )


if __name__ == "__main__":
    main()
