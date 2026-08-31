#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from vlmevalbench.data import read_jsonl


DEFAULT_ENCODERS = (
    "clip-vit-l-14-336",
    "siglip-so400m",
    "siglip2-so400m",
    "dinov2-vitl14",
    "internvit-300m",
    "videomaev2-base",
    "vjepa2-vith-256",
    "internvideo2-clip-s",
)
DEFAULT_MODES = ("original", "single", "reverse", "shuffle")
METRICS = ("text_score", "video_score", "group_score")
METRIC_LABELS = {
    "text_score": "Text score",
    "video_score": "Video score",
    "group_score": "Group score",
}
COLORS = {
    "text_score": "#4C78A8",
    "video_score": "#F58518",
    "group_score": "#54A24B",
    "original": "#4C78A8",
    "single": "#F58518",
    "reverse": "#E45756",
    "shuffle": "#72B7B2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate official and diagnostic Vinoground scores.")
    parser.add_argument("--manifest", default="data/benchmarks/vinoground_all.jsonl")
    parser.add_argument(
        "--text-variant-manifest",
        default="data/benchmarks/vinoground_text_variants.jsonl",
    )
    parser.add_argument(
        "--text-predictions",
        default="outputs/vinoground/text_only/predictions.jsonl",
    )
    parser.add_argument("--eval-root", default="outputs/vinoground/projector_eval")
    parser.add_argument("--out-dir", default="outputs/vinoground/analysis")
    parser.add_argument("--encoders", default=",".join(DEFAULT_ENCODERS))
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--robust-text-rate", type=float, default=0.8)
    parser.add_argument("--skip-incomplete", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def load_plotting() -> None:
    global np, plt
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_module
    import numpy as np_module

    plt = plt_module
    np = np_module


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prediction_map(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in read_jsonl(path)}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
    radius /= denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def build_groups(manifest: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    groups: dict[int, dict[str, Any]] = {}
    for row in manifest:
        group_id = int(row["group_id"])
        group = groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                "text_ids": [],
                "video_ids": [],
                "major_category": row.get("major_category", "unknown"),
                "minor_categories": list(row.get("minor_categories") or []),
                "categories": list(row.get("categories") or []),
            },
        )
        score_type = str(row["score_type"])
        group[f"{score_type}_ids"].append(str(row["id"]))

    malformed = [
        group_id
        for group_id, group in groups.items()
        if len(group["text_ids"]) != 2 or len(group["video_ids"]) != 2
    ]
    if malformed:
        raise RuntimeError(f"Malformed Vinoground groups: {malformed[:10]}")
    return groups


def score_prediction_groups(
    predictions: dict[str, dict[str, Any]],
    groups: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_id, group in sorted(groups.items()):
        required_ids = [*group["text_ids"], *group["video_ids"]]
        if any(record_id not in predictions for record_id in required_ids):
            continue
        text_correct = all(bool(predictions[record_id].get("correct")) for record_id in group["text_ids"])
        video_correct = all(bool(predictions[record_id].get("correct")) for record_id in group["video_ids"])
        rows.append(
            {
                "group_id": group_id,
                "major_category": group["major_category"],
                "minor_categories": ";".join(group["minor_categories"]),
                "categories": list(group["categories"]),
                "text_correct": text_correct,
                "video_correct": video_correct,
                "group_correct": text_correct and video_correct,
            }
        )
    return rows


def summarize_group_rows(
    rows: list[dict[str, Any]],
    *,
    encoder: str,
    mode: str,
    expected_groups: int,
) -> dict[str, Any]:
    total = len(rows)
    output: dict[str, Any] = {
        "encoder": encoder,
        "mode": mode,
        "expected_groups": expected_groups,
        "evaluated_groups": total,
        "missing_groups": expected_groups - total,
        "complete": total == expected_groups,
    }
    for metric in METRICS:
        key = metric.replace("_score", "_correct")
        successes = sum(bool(row[key]) for row in rows)
        low, high = wilson_interval(successes, total)
        output[f"{metric}_correct"] = successes
        output[metric] = successes / max(total, 1)
        output[f"{metric}_ci_low"] = low
        output[f"{metric}_ci_high"] = high
    return output


def category_summaries(
    rows: list[dict[str, Any]],
    *,
    encoder: str,
    mode: str,
) -> list[dict[str, Any]]:
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category_rows["ALL"].append(row)
        for category in row["categories"]:
            category_rows[str(category)].append(row)

    summaries: list[dict[str, Any]] = []
    for category, items in sorted(category_rows.items()):
        output: dict[str, Any] = {
            "encoder": encoder,
            "mode": mode,
            "category": category,
            "num_groups": len(items),
        }
        for metric in METRICS:
            key = metric.replace("_score", "_correct")
            output[metric] = sum(bool(item[key]) for item in items) / max(len(items), 1)
        summaries.append(output)
    return summaries


def analyze_text_only(
    *,
    manifest: list[dict[str, Any]],
    variant_manifest_path: Path,
    prediction_path: Path,
    groups: dict[int, dict[str, Any]],
    robust_rate: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not variant_manifest_path.is_file() or not prediction_path.is_file():
        return [], [], []

    base_map = {str(row["id"]): row for row in manifest}
    variants = read_jsonl(variant_manifest_path)
    predictions = prediction_map(prediction_path)
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for variant in variants:
        by_base[str(variant["base_id"])].append(variant)

    base_rows: list[dict[str, Any]] = []
    base_status: dict[str, dict[str, Any]] = {}
    for base_id, expected_variants in sorted(by_base.items()):
        evaluated = [variant for variant in expected_variants if str(variant["id"]) in predictions]
        correct_count = sum(bool(predictions[str(variant["id"])].get("correct")) for variant in evaluated)
        threshold = math.ceil(len(expected_variants) * robust_rate)
        canonical = next(
            (
                variant
                for variant in expected_variants
                if int(variant["prompt_variant"]) == 0 and variant["option_order"] == "identity"
            ),
            None,
        )
        canonical_correct = bool(
            canonical is not None
            and str(canonical["id"]) in predictions
            and predictions[str(canonical["id"])].get("correct")
        )

        semantic_counts: Counter[str] = Counter()
        for variant in evaluated:
            prediction = str(predictions[str(variant["id"])].get("prediction") or "")
            if len(prediction) != 1 or prediction not in "AB":
                continue
            choices = [str(choice) for choice in variant.get("choices") or []]
            semantic_counts[choices[ord(prediction) - ord("A")]] += 1
        stable_count = max(semantic_counts.values(), default=0)
        complete = len(evaluated) == len(expected_variants)
        robust_correct = complete and correct_count >= threshold
        base = base_map[base_id]
        row = {
            "base_id": base_id,
            "group_id": int(base["group_id"]),
            "score_type": base["score_type"],
            "pair_role": base["pair_role"],
            "expected_variants": len(expected_variants),
            "evaluated_variants": len(evaluated),
            "correct_variants": correct_count,
            "correct_rate": correct_count / max(len(evaluated), 1),
            "robust_threshold": threshold,
            "canonical_correct": canonical_correct,
            "robust_correct": robust_correct,
            "semantic_stability_rate": stable_count / max(len(evaluated), 1),
            "complete": complete,
        }
        base_rows.append(row)
        base_status[base_id] = row

    text_group_rows: list[dict[str, Any]] = []
    for group_id, group in sorted(groups.items()):
        required = [*group["text_ids"], *group["video_ids"]]
        if any(record_id not in base_status for record_id in required):
            continue
        canonical_text = all(base_status[record_id]["canonical_correct"] for record_id in group["text_ids"])
        canonical_video = all(base_status[record_id]["canonical_correct"] for record_id in group["video_ids"])
        robust_text = all(base_status[record_id]["robust_correct"] for record_id in group["text_ids"])
        robust_video = all(base_status[record_id]["robust_correct"] for record_id in group["video_ids"])
        text_group_rows.append(
            {
                "group_id": group_id,
                "canonical_text_correct": canonical_text,
                "canonical_video_correct": canonical_video,
                "canonical_group_correct": canonical_text and canonical_video,
                "robust_text_correct": robust_text,
                "robust_video_correct": robust_video,
                "robust_group_correct": robust_text and robust_video,
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for condition in ("canonical", "robust"):
        output: dict[str, Any] = {
            "condition": condition,
            "num_groups": len(text_group_rows),
        }
        for metric in METRICS:
            key = f"{condition}_{metric.replace('_score', '_correct')}"
            output[metric] = sum(bool(row[key]) for row in text_group_rows) / max(len(text_group_rows), 1)
        summary_rows.append(output)
    return base_rows, text_group_rows, summary_rows


def add_bar_labels(ax: plt.Axes, bars: Iterable[Any], *, fontsize: int = 7) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1.0,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def save_figure(fig: plt.Figure, out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_prefix.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_official_scores(rows: list[dict[str, Any]], out_dir: Path, encoders: list[str]) -> None:
    by_encoder = {row["encoder"]: row for row in rows if row["mode"] == "original"}
    labels = [encoder for encoder in encoders if encoder in by_encoder]
    if not labels:
        return
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(12, 1.45 * len(labels)), 6.5))
    for idx, metric in enumerate(METRICS):
        values = [100.0 * float(by_encoder[label][metric]) for label in labels]
        bars = ax.bar(x + (idx - 1) * width, values, width, label=METRIC_LABELS[metric], color=COLORS[metric])
        add_bar_labels(ax, bars)
    ax.axhline(25.0, color="#777777", linestyle="--", linewidth=1, label="Random text/video: 25%")
    ax.axhline(100.0 / 6.0, color="#AAAAAA", linestyle=":", linewidth=1, label="Random group: 16.7%")
    ax.set_title("Vinoground Official Scores", loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, max(40.0, ax.get_ylim()[1] + 8.0))
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, out_dir / "vinoground_official_scores")


def plot_perturbations(rows: list[dict[str, Any]], out_dir: Path, encoders: list[str], modes: list[str]) -> None:
    lookup = {(row["encoder"], row["mode"]): row for row in rows}
    labels = [encoder for encoder in encoders if any((encoder, mode) in lookup for mode in modes)]
    available_modes = [mode for mode in modes if any((encoder, mode) in lookup for encoder in labels)]
    if not labels or not available_modes:
        return
    x = np.arange(len(labels))
    width = min(0.8 / max(len(available_modes), 1), 0.22)
    fig, ax = plt.subplots(figsize=(max(12, 1.45 * len(labels)), 6.5))
    offsets = np.arange(len(available_modes)) - (len(available_modes) - 1) / 2.0
    for offset, mode in zip(offsets, available_modes):
        values = [100.0 * float(lookup.get((encoder, mode), {}).get("group_score", 0.0)) for encoder in labels]
        bars = ax.bar(x + offset * width, values, width, label=mode, color=COLORS.get(mode))
        add_bar_labels(ax, bars, fontsize=6)
    ax.axhline(100.0 / 6.0, color="#888888", linestyle="--", linewidth=1, label="Random group: 16.7%")
    ax.set_title("Vinoground Group Score Under Frame Perturbations", loc="left", fontsize=14, fontweight="bold")
    ax.text(
        0.0,
        1.01,
        "Non-original modes retain the original labels and are sensitivity diagnostics, not new ground truth.",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    ax.set_ylabel("Original-label group score (%)")
    ax.set_ylim(0, max(40.0, ax.get_ylim()[1] + 8.0))
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend(ncol=max(2, len(available_modes)), fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, out_dir / "vinoground_perturbation_group_scores")


def plot_text_only(rows: list[dict[str, Any]], out_dir: Path) -> None:
    if not rows:
        return
    conditions = [str(row["condition"]) for row in rows]
    lookup = {str(row["condition"]): row for row in rows}
    x = np.arange(len(METRICS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    offsets = np.arange(len(conditions)) - (len(conditions) - 1) / 2.0
    for offset, condition in zip(offsets, conditions):
        values = [100.0 * float(lookup[condition][metric]) for metric in METRICS]
        bars = ax.bar(x + offset * width, values, width, label=condition)
        add_bar_labels(ax, bars, fontsize=8)
    ax.axhline(25.0, color="#777777", linestyle="--", linewidth=1, label="Random text/video: 25%")
    ax.axhline(100.0 / 6.0, color="#AAAAAA", linestyle=":", linewidth=1, label="Random group: 16.7%")
    ax.set_title("Vinoground Query-Only Shortcut Audit", loc="left", fontsize=14, fontweight="bold")
    ax.set_ylabel("Pair score (%)")
    ax.set_ylim(0, max(35.0, ax.get_ylim()[1] + 8.0))
    ax.set_xticks(x, [METRIC_LABELS[metric] for metric in METRICS])
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save_figure(fig, out_dir / "vinoground_query_only_scores")


def plot_category_heatmap(
    category_rows: list[dict[str, Any]],
    out_dir: Path,
    encoders: list[str],
) -> None:
    original = [row for row in category_rows if row["mode"] == "original" and row["category"] != "ALL"]
    if not original:
        return
    categories = sorted({str(row["category"]) for row in original})
    labels = [encoder for encoder in encoders if any(row["encoder"] == encoder for row in original)]
    lookup = {(row["encoder"], row["category"]): float(row["group_score"]) for row in original}
    values = np.full((len(labels), len(categories)), np.nan, dtype=float)
    for row_idx, encoder in enumerate(labels):
        for col_idx, category in enumerate(categories):
            if (encoder, category) in lookup:
                values[row_idx, col_idx] = 100.0 * lookup[(encoder, category)]
    fig, ax = plt.subplots(figsize=(max(10, 1.15 * len(categories)), max(5, 0.65 * len(labels) + 2)))
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=max(35.0, float(np.nanmax(values))))
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            if np.isfinite(values[row_idx, col_idx]):
                ax.text(col_idx, row_idx, f"{values[row_idx, col_idx]:.1f}", ha="center", va="center", fontsize=7, color="white")
    ax.set_title("Vinoground Original Group Score by Category", loc="left", fontsize=14, fontweight="bold")
    ax.set_xticks(np.arange(len(categories)), categories, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    fig.colorbar(image, ax=ax, label="Group score (%)")
    fig.tight_layout()
    save_figure(fig, out_dir / "vinoground_category_group_heatmap")


def main() -> None:
    args = parse_args()
    manifest = read_jsonl(args.manifest)
    groups = build_groups(manifest)
    expected_groups = len(groups)
    encoders = [value.strip() for value in args.encoders.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    eval_root = Path(args.eval_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    score_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    per_group_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for encoder in encoders:
        for mode in modes:
            path = eval_root / encoder / mode / "predictions.jsonl"
            if not path.is_file():
                skipped.append({"encoder": encoder, "mode": mode, "reason": "missing predictions"})
                if not args.skip_incomplete:
                    raise FileNotFoundError(f"Missing Vinoground predictions: {path}")
                continue
            group_rows = score_prediction_groups(prediction_map(path), groups)
            summary = summarize_group_rows(
                group_rows,
                encoder=encoder,
                mode=mode,
                expected_groups=expected_groups,
            )
            if not summary["complete"] and not args.skip_incomplete:
                raise RuntimeError(
                    f"Incomplete Vinoground predictions for {encoder}/{mode}: "
                    f"{summary['evaluated_groups']}/{expected_groups} groups"
                )
            score_rows.append(summary)
            category_rows.extend(category_summaries(group_rows, encoder=encoder, mode=mode))
            for row in group_rows:
                per_group_rows.append(
                    {
                        "encoder": encoder,
                        "mode": mode,
                        **{key: value for key, value in row.items() if key != "categories"},
                    }
                )

    base_text_rows, text_group_rows, text_summary_rows = analyze_text_only(
        manifest=manifest,
        variant_manifest_path=Path(args.text_variant_manifest),
        prediction_path=Path(args.text_predictions),
        groups=groups,
        robust_rate=args.robust_text_rate,
    )

    score_fields = [
        "encoder",
        "mode",
        "expected_groups",
        "evaluated_groups",
        "missing_groups",
        "complete",
        *[
            field
            for metric in METRICS
            for field in (
                f"{metric}_correct",
                metric,
                f"{metric}_ci_low",
                f"{metric}_ci_high",
            )
        ],
    ]
    write_csv(out_dir / "model_scores.csv", score_rows, score_fields)
    write_csv(
        out_dir / "category_scores.csv",
        category_rows,
        ["encoder", "mode", "category", "num_groups", *METRICS],
    )
    write_csv(
        out_dir / "per_group_scores.csv",
        per_group_rows,
        [
            "encoder",
            "mode",
            "group_id",
            "major_category",
            "minor_categories",
            "text_correct",
            "video_correct",
            "group_correct",
        ],
    )
    write_csv(out_dir / "skipped_incomplete.csv", skipped, ["encoder", "mode", "reason"])

    if base_text_rows:
        write_csv(
            out_dir / "text_only_base_records.csv",
            base_text_rows,
            [
                "base_id",
                "group_id",
                "score_type",
                "pair_role",
                "expected_variants",
                "evaluated_variants",
                "correct_variants",
                "correct_rate",
                "robust_threshold",
                "canonical_correct",
                "robust_correct",
                "semantic_stability_rate",
                "complete",
            ],
        )
        write_csv(
            out_dir / "text_only_per_group.csv",
            text_group_rows,
            [
                "group_id",
                "canonical_text_correct",
                "canonical_video_correct",
                "canonical_group_correct",
                "robust_text_correct",
                "robust_video_correct",
                "robust_group_correct",
            ],
        )
        write_csv(
            out_dir / "text_only_summary.csv",
            text_summary_rows,
            ["condition", "num_groups", *METRICS],
        )

    metadata = {
        "num_manifest_records": len(manifest),
        "num_groups": expected_groups,
        "encoders": encoders,
        "modes": modes,
        "robust_text_rate": args.robust_text_rate,
        "official_chance": {"text_score": 0.25, "video_score": 0.25, "group_score": 1.0 / 6.0},
        "perturbation_note": (
            "single/reverse/shuffle retain original labels. They measure shortcut retention or sensitivity; "
            "only original mode is an official Vinoground score."
        ),
    }
    (out_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if not args.no_plots:
        load_plotting()
        plot_official_scores(score_rows, out_dir, encoders)
        plot_perturbations(score_rows, out_dir, encoders, modes)
        plot_text_only(text_summary_rows, out_dir)
        plot_category_heatmap(category_rows, out_dir, encoders)

    print(f"Wrote Vinoground analysis to {out_dir}")
    for row in score_rows:
        if row["mode"] == "original":
            print(
                f"{row['encoder']}: text={100 * row['text_score']:.2f}% "
                f"video={100 * row['video_score']:.2f}% group={100 * row['group_score']:.2f}% "
                f"groups={row['evaluated_groups']} complete={row['complete']}"
            )


if __name__ == "__main__":
    main()
