#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vlmevalbench.data import read_jsonl, write_jsonl


MODES = ("original", "single", "reverse", "shuffle")
SHORTCUT_MODES = ("single", "reverse", "shuffle")
PALETTE = {
    "text_only": "#4C78A8",
    "single": "#F58518",
    "reverse_shuffle": "#E45756",
    "hard": "#54A24B",
    "all_accuracy": "#72B7B2",
    "hard_accuracy": "#B279A2",
    "shared_hard_accuracy": "#9D755D",
}


def save_figure(fig: plt.Figure, out_prefix: Path) -> None:
    fig.savefig(out_prefix.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate MVBench text-only and perturbation filters.")
    parser.add_argument("--manifest", default="data/benchmarks/mvbench_all.jsonl")
    parser.add_argument("--text-predictions", default="outputs/mvbench/text_only/predictions.jsonl")
    parser.add_argument("--eval-root", default="outputs/mvbench/projector_eval")
    parser.add_argument("--out-dir", default="outputs/mvbench/analysis")
    parser.add_argument("--encoders", default=None, help="Comma-separated encoder list. Defaults to discovered eval dirs.")
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Skip encoders that do not yet have original/single/reverse/shuffle predictions.",
    )
    return parser.parse_args()


def load_by_id(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def discover_encoders(eval_root: Path, encoders_csv: str | None) -> list[str]:
    if encoders_csv:
        return [item.strip() for item in encoders_csv.split(",") if item.strip()]
    return sorted(path.name for path in eval_root.iterdir() if path.is_dir())


def load_predictions(eval_root: Path, encoder: str) -> dict[str, dict[str, dict[str, Any]]]:
    out = {}
    for mode in MODES:
        path = eval_root / encoder / mode / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing {mode} predictions for {encoder}: {path}")
        out[mode] = load_by_id(path)
    return out


def missing_prediction_paths(eval_root: Path, encoder: str) -> list[Path]:
    missing = []
    for mode in MODES:
        path = eval_root / encoder / mode / "predictions.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            missing.append(path)
    return missing


def correct(row: dict[str, Any] | None) -> bool:
    return bool(row and row.get("correct"))


def row_task(row: dict[str, Any], manifest_map: dict[str, dict[str, Any]]) -> str:
    record = manifest_map.get(str(row["id"]), {})
    return str(row.get("task_type") or row.get("benchmark") or record.get("task_type") or record.get("benchmark") or "unknown")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_rows(
    *,
    encoder: str,
    preds: dict[str, dict[str, dict[str, Any]]],
    text_by_id: dict[str, dict[str, Any]],
    manifest_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    common_mode_ids = set(preds["original"].keys())
    for mode in MODES[1:]:
        common_mode_ids &= set(preds[mode].keys())
    original_ids = sorted(set(manifest_map.keys()) & common_mode_ids)
    per_item = []
    for item_id in original_ids:
        text_correct = correct(text_by_id.get(item_id))
        single_correct = correct(preds["single"].get(item_id))
        reverse_correct = correct(preds["reverse"].get(item_id))
        shuffle_correct = correct(preds["shuffle"].get(item_id))
        reverse_or_shuffle = reverse_correct or shuffle_correct
        shortcut_any = text_correct or single_correct or reverse_or_shuffle
        original_correct = correct(preds["original"].get(item_id))
        original_row = preds["original"][item_id]
        per_item.append(
            {
                "id": item_id,
                "encoder": encoder,
                "benchmark": original_row.get("benchmark", "unknown"),
                "task_type": row_task(original_row, manifest_map),
                "text_only_correct": text_correct,
                "single_correct": single_correct,
                "reverse_correct": reverse_correct,
                "shuffle_correct": shuffle_correct,
                "shortcut_any_correct": shortcut_any,
                "hard_after_filters": not shortcut_any,
                "original_correct": original_correct,
                "original_prediction": original_row.get("prediction"),
                "answer": original_row.get("answer"),
            }
        )

    groups: dict[str, list[dict[str, Any]]] = {"ALL": per_item}
    for row in per_item:
        groups.setdefault(str(row["task_type"]), []).append(row)

    summary = []
    for group, items in sorted(groups.items()):
        n = len(items)
        text_n = sum(row["text_only_correct"] for row in items)
        single_n = sum((not row["text_only_correct"]) and row["single_correct"] for row in items)
        reverse_shuffle_n = sum(
            (not row["text_only_correct"]) and (not row["single_correct"]) and (row["reverse_correct"] or row["shuffle_correct"])
            for row in items
        )
        hard_n = sum(row["hard_after_filters"] for row in items)
        original_correct_n = sum(row["original_correct"] for row in items)
        hard_correct_n = sum(row["hard_after_filters"] and row["original_correct"] for row in items)
        summary.append(
            {
                "encoder": encoder,
                "group": group,
                "num_examples": n,
                "text_only_correct": text_n,
                "single_frame_shortcut": single_n,
                "reverse_or_shuffle_shortcut": reverse_shuffle_n,
                "hard_after_filters": hard_n,
                "original_correct": original_correct_n,
                "hard_original_correct": hard_correct_n,
                "text_only_rate": text_n / max(n, 1),
                "single_frame_shortcut_rate": single_n / max(n, 1),
                "reverse_or_shuffle_shortcut_rate": reverse_shuffle_n / max(n, 1),
                "hard_rate": hard_n / max(n, 1),
                "original_accuracy_all": original_correct_n / max(n, 1),
                "original_accuracy_hard": hard_correct_n / max(hard_n, 1),
            }
        )
    return per_item, summary


def add_shared_hard_accuracy(
    summary_rows: list[dict[str, Any]],
    per_encoder_items: dict[str, list[dict[str, Any]]],
    shared_hard_ids: set[str],
) -> None:
    for row in summary_rows:
        if row["group"] != "ALL":
            row["shared_hard_examples"] = ""
            row["shared_hard_correct"] = ""
            row["shared_hard_accuracy"] = ""
            continue
        items = [item for item in per_encoder_items[row["encoder"]] if item["id"] in shared_hard_ids]
        correct_n = sum(item["original_correct"] for item in items)
        row["shared_hard_examples"] = len(items)
        row["shared_hard_correct"] = correct_n
        row["shared_hard_accuracy"] = correct_n / max(len(items), 1)


def plot_overview(summary_df: pd.DataFrame, out_prefix: Path) -> None:
    all_df = summary_df[summary_df["group"] == "ALL"].copy()
    if all_df.empty:
        return
    encoders = all_df["encoder"].astype(str).tolist()
    x = np.arange(len(encoders))

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)

    ax = fig.add_subplot(grid[0, 0])
    bottoms = np.zeros(len(all_df))
    stack_cols = [
        ("text_only_correct", "text-only correct", PALETTE["text_only"]),
        ("single_frame_shortcut", "single-frame shortcut", PALETTE["single"]),
        ("reverse_or_shuffle_shortcut", "reverse/shuffle shortcut", PALETTE["reverse_shuffle"]),
        ("hard_after_filters", "hard after filters", PALETTE["hard"]),
    ]
    for col, label, color in stack_cols:
        values = pd.to_numeric(all_df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottoms, label=label, color=color, linewidth=0)
        bottoms += values
    ax.set_title("MVBench Filter Distribution", loc="left", fontweight="bold")
    ax.set_ylabel("questions")
    ax.set_xticks(x)
    ax.set_xticklabels(encoders, rotation=35, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(grid[0, 1])
    rate_cols = [
        ("text_only_rate", "text-only"),
        ("single_frame_shortcut_rate", "single-frame"),
        ("reverse_or_shuffle_shortcut_rate", "reverse/shuffle"),
        ("hard_rate", "hard"),
    ]
    width = 0.18
    for idx, (col, label) in enumerate(rate_cols):
        values = pd.to_numeric(all_df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.bar(x + (idx - 1.5) * width, values, width=width, label=label, linewidth=0)
    ax.set_title("Filter Rates", loc="left", fontweight="bold")
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(encoders, rotation=35, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(grid[1, 0])
    acc_cols = [
        ("original_accuracy_all", "original all", PALETTE["all_accuracy"]),
        ("original_accuracy_hard", "original hard", PALETTE["hard_accuracy"]),
        ("shared_hard_accuracy", "shared hard", PALETTE["shared_hard_accuracy"]),
    ]
    width = 0.24
    for idx, (col, label, color) in enumerate(acc_cols):
        values = pd.to_numeric(all_df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.bar(x + (idx - 1) * width, values, width=width, label=label, color=color, linewidth=0)
    ax.set_title("Accuracy After Filtering", loc="left", fontweight="bold")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(encoders, rotation=35, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    ax = fig.add_subplot(grid[1, 1])
    hard_counts = pd.to_numeric(all_df["hard_after_filters"], errors="coerce").fillna(0).to_numpy(dtype=float)
    shared_counts = pd.to_numeric(all_df["shared_hard_examples"], errors="coerce").fillna(0).to_numpy(dtype=float)
    ax.plot(x, hard_counts, marker="o", label="per-encoder hard", color=PALETTE["hard"])
    ax.plot(x, shared_counts, marker="o", label="shared hard", color=PALETTE["shared_hard_accuracy"])
    ax.set_title("Hard Questions Left After Filtering", loc="left", fontweight="bold")
    ax.set_ylabel("questions")
    ax.set_xticks(x)
    ax.set_xticklabels(encoders, rotation=35, ha="right")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("MVBench Shortcut Filtering and Projector Evaluation", x=0.01, ha="left", fontweight="bold")
    save_figure(fig, out_prefix)


def plot_filter_distribution(all_df: pd.DataFrame, out_prefix: Path) -> None:
    encoders = all_df["encoder"].astype(str).tolist()
    y = np.arange(len(encoders))
    fig, ax = plt.subplots(figsize=(12, max(5, len(encoders) * 0.5)), constrained_layout=True)
    left = np.zeros(len(all_df))
    stack_cols = [
        ("text_only_correct", "text-only correct", PALETTE["text_only"]),
        ("single_frame_shortcut", "single-frame shortcut", PALETTE["single"]),
        ("reverse_or_shuffle_shortcut", "reverse/shuffle shortcut", PALETTE["reverse_shuffle"]),
        ("hard_after_filters", "hard after filters", PALETTE["hard"]),
    ]
    for col, label, color in stack_cols:
        values = pd.to_numeric(all_df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.barh(y, values, left=left, label=label, color=color, linewidth=0)
        left += values
    ax.set_title("MVBench Filter Distribution", loc="left", fontweight="bold")
    ax.set_xlabel("questions")
    ax.set_yticks(y)
    ax.set_yticklabels(encoders)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save_figure(fig, out_prefix)


def plot_filter_rates(all_df: pd.DataFrame, out_prefix: Path) -> None:
    encoders = all_df["encoder"].astype(str).tolist()
    y = np.arange(len(encoders))
    fig, ax = plt.subplots(figsize=(12, max(5, len(encoders) * 0.55)), constrained_layout=True)
    rate_cols = [
        ("text_only_rate", "text-only"),
        ("single_frame_shortcut_rate", "single-frame"),
        ("reverse_or_shuffle_shortcut_rate", "reverse/shuffle"),
        ("hard_rate", "hard"),
    ]
    height = 0.18
    for idx, (col, label) in enumerate(rate_cols):
        values = pd.to_numeric(all_df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.barh(y + (idx - 1.5) * height, values, height=height, label=label, linewidth=0)
    ax.set_title("MVBench Filter Rates", loc="left", fontweight="bold")
    ax.set_xlabel("rate")
    ax.set_xlim(0, 1.05)
    ax.set_yticks(y)
    ax.set_yticklabels(encoders)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    save_figure(fig, out_prefix)


def plot_accuracy_after_filtering(all_df: pd.DataFrame, out_prefix: Path) -> None:
    encoders = all_df["encoder"].astype(str).tolist()
    y = np.arange(len(encoders))
    fig, ax = plt.subplots(figsize=(12, max(5, len(encoders) * 0.55)), constrained_layout=True)
    acc_cols = [
        ("original_accuracy_all", "original all", PALETTE["all_accuracy"]),
        ("original_accuracy_hard", "original hard", PALETTE["hard_accuracy"]),
        ("shared_hard_accuracy", "shared hard", PALETTE["shared_hard_accuracy"]),
    ]
    height = 0.22
    for idx, (col, label, color) in enumerate(acc_cols):
        values = pd.to_numeric(all_df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        bar_positions = y + (idx - 1) * height
        ax.barh(bar_positions, values, height=height, label=label, color=color, linewidth=0)
        for y_pos, value in zip(bar_positions, values):
            if value <= 0:
                continue
            inside = value >= 0.12
            x_pos = value - 0.015 if inside else value + 0.015
            ax.text(
                x_pos,
                y_pos,
                f"{value:.3f}",
                va="center",
                ha="right" if inside else "left",
                fontsize=7,
                color="white" if inside else "black",
                clip_on=False,
            )
    ax.set_title("MVBench Accuracy After Filtering", loc="left", fontweight="bold")
    ax.set_xlabel("accuracy")
    ax.set_xlim(0, 1.12)
    ax.set_yticks(y)
    ax.set_yticklabels(encoders)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    save_figure(fig, out_prefix)


def plot_remaining_question_counts(all_df: pd.DataFrame, out_prefix: Path) -> None:
    encoders = all_df["encoder"].astype(str).tolist()
    y = np.arange(len(encoders))
    fig, ax = plt.subplots(figsize=(12, max(5, len(encoders) * 0.5)), constrained_layout=True)
    count_cols = [
        ("hard_after_filters", "per-encoder hard", PALETTE["hard"]),
        ("shared_hard_examples", "shared hard", PALETTE["shared_hard_accuracy"]),
    ]
    height = 0.28
    for idx, (col, label, color) in enumerate(count_cols):
        values = pd.to_numeric(all_df[col], errors="coerce").fillna(0).to_numpy(dtype=float)
        ax.barh(y + (idx - 0.5) * height, values, height=height, label=label, color=color, linewidth=0)
    ax.set_title("MVBench Hard Questions Left After Filtering", loc="left", fontweight="bold")
    ax.set_xlabel("questions")
    ax.set_yticks(y)
    ax.set_yticklabels(encoders)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save_figure(fig, out_prefix)


def plot_separate_overview_panels(summary_df: pd.DataFrame, out_dir: Path) -> None:
    all_df = summary_df[summary_df["group"] == "ALL"].copy()
    if all_df.empty:
        return
    plot_filter_distribution(all_df, out_dir / "mvbench_filter_distribution")
    plot_filter_rates(all_df, out_dir / "mvbench_filter_rates")
    plot_accuracy_after_filtering(all_df, out_dir / "mvbench_accuracy_after_filtering")
    plot_remaining_question_counts(all_df, out_dir / "mvbench_remaining_question_counts")


def plot_task_heatmap(summary_df: pd.DataFrame, out_prefix: Path) -> None:
    task_df = summary_df[summary_df["group"] != "ALL"].copy()
    if task_df.empty:
        return
    pivot = task_df.pivot_table(index="group", columns="encoder", values="hard_rate", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(14, max(6, len(pivot) * 0.35)), constrained_layout=True)
    matrix = pivot.fillna(0.0).to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Hard-After-Filter Rate by MVBench Task", loc="left", fontweight="bold")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for row_idx, task in enumerate(pivot.index):
        for col_idx, encoder in enumerate(pivot.columns):
            value = pivot.loc[task, encoder]
            if pd.isna(value):
                label = ""
            else:
                label = f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=7, color="white" if matrix[row_idx, col_idx] > 0.55 else "black")
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("hard rate")
    save_figure(fig, out_prefix)


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    manifest_map = {str(row["id"]): row for row in manifest}
    text_by_id = load_by_id(Path(args.text_predictions))
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    encoders = discover_encoders(eval_root, args.encoders)
    per_encoder_items = {}
    summary_rows = []
    skipped = []

    for encoder in encoders:
        missing = missing_prediction_paths(eval_root, encoder)
        if missing:
            message = f"Skipping incomplete encoder {encoder}; missing: {', '.join(str(path) for path in missing)}"
            if not args.skip_incomplete:
                raise FileNotFoundError(message)
            print(f"Warning: {message}")
            skipped.append(
                {
                    "encoder": encoder,
                    "missing_files": ";".join(str(path) for path in missing),
                }
            )
            continue
        preds = load_predictions(eval_root, encoder)
        per_item, encoder_summary = build_rows(
            encoder=encoder,
            preds=preds,
            text_by_id=text_by_id,
            manifest_map=manifest_map,
        )
        per_encoder_items[encoder] = per_item
        summary_rows.extend(encoder_summary)
        hard_rows = [manifest_map[item["id"]] | {"filter_encoder": encoder} for item in per_item if item["hard_after_filters"] and item["id"] in manifest_map]
        write_jsonl(out_dir / "hard_ids_per_encoder" / f"{encoder}.jsonl", hard_rows)

    if skipped:
        write_csv(out_dir / "skipped_incomplete_encoders.csv", skipped)
    if not per_encoder_items:
        raise RuntimeError("No complete encoders were available for MVBench analysis.")

    common_ids: set[str] | None = None
    for items in per_encoder_items.values():
        encoder_ids = {item["id"] for item in items}
        common_ids = encoder_ids if common_ids is None else common_ids & encoder_ids
    common_ids = common_ids or set()
    shortcut_by_id = {item_id: correct(text_by_id.get(item_id)) for item_id in common_ids}
    for items in per_encoder_items.values():
        for item in items:
            if item["id"] in shortcut_by_id:
                shortcut_by_id[item["id"]] = shortcut_by_id[item["id"]] or item["single_correct"] or item["reverse_correct"] or item["shuffle_correct"]
    shared_hard_ids = {item_id for item_id, shortcut in shortcut_by_id.items() if not shortcut}
    shared_hard_rows = [manifest_map[item_id] for item_id in sorted(shared_hard_ids) if item_id in manifest_map]
    write_jsonl(out_dir / "shared_hard_ids.jsonl", shared_hard_rows)

    add_shared_hard_accuracy(summary_rows, per_encoder_items, shared_hard_ids)
    write_csv(out_dir / "filter_summary.csv", summary_rows)

    item_rows = [item for items in per_encoder_items.values() for item in items]
    write_csv(out_dir / "per_item_filters.csv", item_rows)

    summary_df = pd.DataFrame(summary_rows)
    plot_overview(summary_df, out_dir / "mvbench_filter_overview")
    plot_separate_overview_panels(summary_df, out_dir)
    plot_task_heatmap(summary_df, out_dir / "mvbench_task_hardness_heatmap")

    print(f"Wrote summary to {out_dir / 'filter_summary.csv'}")
    print(f"Wrote per-item filters to {out_dir / 'per_item_filters.csv'}")
    print(f"Wrote shared hard subset to {out_dir / 'shared_hard_ids.jsonl'}")
    print(f"Wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
