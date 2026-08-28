#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TEMPORAL_METRICS = [
    "order_distance",
    "shuffle_distance",
    "cycle_shift_distance",
    "half_swap_distance",
    "stride_distance",
]
SEGMENT_METRICS = [
    "segment_diversity",
    "segment_adjacent_distance",
    "segment_far_distance",
    "segment_temporal_margin",
    "segment_distance_correlation",
]
TOKEN_METRICS = [
    "token_effective_rank",
    "token_rank_ratio",
    "token_top1_energy_ratio",
    "token_mean_pairwise_distance",
]
KNN_METRICS = ["knn_top1", "knn_top5"]
ALL_METRICS = TEMPORAL_METRICS + SEGMENT_METRICS + TOKEN_METRICS + KNN_METRICS

DISPLAY_NAMES = {
    "clip-vit-l-14-336": "CLIP-L/14",
    "siglip-so400m": "SigLIP",
    "siglip2-so400m": "SigLIP2",
    "internvit-300m": "InternViT",
    "dinov2-vitl14": "DINOv2-L",
    "videomaev2-base": "VideoMAEv2",
    "vjepa2-vith-256": "V-JEPA2",
    "vjepa2-with-256": "V-JEPA2",
    "internvideo2-clip-s": "InternVideo2-S",
}

METRIC_LABELS = {
    "order_distance": "reverse",
    "shuffle_distance": "shuffle",
    "cycle_shift_distance": "cycle",
    "half_swap_distance": "half swap",
    "stride_distance": "stride",
    "segment_diversity": "segment diversity",
    "segment_adjacent_distance": "adjacent dist.",
    "segment_far_distance": "far dist.",
    "segment_temporal_margin": "temporal margin",
    "segment_distance_correlation": "distance corr.",
    "token_effective_rank": "effective rank",
    "token_rank_ratio": "rank ratio",
    "token_top1_energy_ratio": "top1 energy",
    "token_mean_pairwise_distance": "token pairwise",
    "knn_top1": "KNN@1",
    "knn_top5": "KNN@5",
}

ENCODER_ORDER = [
    "clip-vit-l-14-336",
    "siglip-so400m",
    "siglip2-so400m",
    "internvit-300m",
    "dinov2-vitl14",
    "videomaev2-base",
    "vjepa2-vith-256",
    "vjepa2-with-256",
    "internvideo2-clip-s",
]

PALETTE = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot no-training encoder diagnostic metrics.")
    parser.add_argument("--input", default="outputs/no_train_diagnostics_table.csv")
    parser.add_argument("--group", default="ALL", help="Summary group to plot, usually ALL.")
    parser.add_argument("--out-prefix", default="outputs/figures/no_train_diagnostics_overview")
    parser.add_argument("--title", default="Training-free Encoder Diagnostics")
    return parser.parse_args()


def display_name(encoder: str) -> str:
    return DISPLAY_NAMES.get(encoder, encoder)


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def load_summary(path: str | Path, group: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "encoder" not in df.columns:
        raise ValueError("Input CSV must contain an 'encoder' column. Run scripts/aggregate_diagnostics.py first.")
    if "group" in df.columns and group:
        df = df[df["group"].astype(str) == str(group)].copy()
    if df.empty:
        raise ValueError(f"No rows found for group={group!r} in {path}")

    available_metrics = [metric for metric in ALL_METRICS if metric in df.columns]
    missing = sorted(set(ALL_METRICS) - set(available_metrics))
    if missing:
        print(f"Warning: missing metrics will be omitted: {', '.join(missing)}")

    order = {name: idx for idx, name in enumerate(ENCODER_ORDER)}
    df["_order"] = df["encoder"].map(lambda x: order.get(str(x), len(order)))
    df = df.sort_values(["_order", "encoder"]).drop(columns=["_order"])
    for metric in available_metrics:
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def normalized(values: pd.DataFrame, *, invert: set[str] | None = None) -> pd.DataFrame:
    invert = invert or set()
    out = values.copy()
    for col in out.columns:
        series = out[col].astype(float)
        vmin = np.nanmin(series.to_numpy()) if np.isfinite(series.to_numpy()).any() else 0.0
        vmax = np.nanmax(series.to_numpy()) if np.isfinite(series.to_numpy()).any() else 0.0
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-12:
            norm = pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
        else:
            norm = (series - vmin) / (vmax - vmin)
        if col in invert:
            norm = 1.0 - norm
        out[col] = norm
    return out


def add_grouped_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    metrics: list[str],
    *,
    title: str,
    ylabel: str,
) -> None:
    metrics = [metric for metric in metrics if metric in df.columns]
    x = np.arange(len(df))
    width = min(0.75 / max(len(metrics), 1), 0.18)
    for idx, metric in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2) * width
        ax.bar(
            x + offset,
            df[metric].fillna(0.0).to_numpy(dtype=float),
            width=width,
            label=metric_label(metric),
            color=PALETTE[idx % len(PALETTE)],
            linewidth=0,
        )
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(str(x)) for x in df["encoder"]], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(fontsize=8, ncol=2, frameon=False)


def add_normalized_token_bars(ax: plt.Axes, df: pd.DataFrame) -> None:
    metrics = [metric for metric in TOKEN_METRICS if metric in df.columns]
    values = df[metrics].copy()
    norm = normalized(values, invert={"token_top1_energy_ratio"})
    x = np.arange(len(df))
    width = min(0.75 / max(len(metrics), 1), 0.18)
    for idx, metric in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2) * width
        ax.bar(
            x + offset,
            norm[metric].fillna(0.0).to_numpy(dtype=float),
            width=width,
            label=metric_label(metric),
            color=PALETTE[(idx + 3) % len(PALETTE)],
            linewidth=0,
        )
    ax.set_title("Token Compression / Diversity", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("min-max normalized")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(str(x)) for x in df["encoder"]], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(fontsize=8, ncol=2, frameon=False)


def add_knn_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    metrics = [metric for metric in KNN_METRICS if metric in df.columns]
    if not metrics:
        ax.axis("off")
        return
    x = np.arange(len(df))
    width = 0.32
    for idx, metric in enumerate(metrics):
        offset = (idx - (len(metrics) - 1) / 2) * width
        ax.bar(
            x + offset,
            df[metric].fillna(0.0).to_numpy(dtype=float),
            width=width,
            label=metric_label(metric),
            color=PALETTE[(idx + 5) % len(PALETTE)],
            linewidth=0,
        )
    ax.set_title("KNN Sanity Check", loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels([display_name(str(x)) for x in df["encoder"]], rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(fontsize=8, frameon=False)


def add_heatmap(ax: plt.Axes, df: pd.DataFrame, metrics: list[str]) -> None:
    raw = df.set_index("encoder")[metrics]
    heat = normalized(raw, invert={"token_top1_energy_ratio"})
    matrix = heat.T.to_numpy(dtype=float)
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("All Metrics Heatmap", loc="left", fontsize=12, fontweight="bold")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels([metric_label(metric) for metric in metrics], fontsize=8)
    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels([display_name(str(x)) for x in df["encoder"]], rotation=35, ha="right")
    ax.tick_params(axis="both", length=0)
    for row_idx, metric in enumerate(metrics):
        for col_idx, encoder in enumerate(df["encoder"]):
            value = raw.loc[encoder, metric]
            if pd.isna(value):
                label = ""
            elif abs(float(value)) < 0.001 and float(value) != 0.0:
                label = f"{float(value):.1e}"
            else:
                label = f"{float(value):.2f}" if abs(float(value)) >= 1 else f"{float(value):.3f}"
            color = "white" if matrix[row_idx, col_idx] > 0.55 else "black"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=6.5, color=color)
    cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label("normalized score", fontsize=8)
    cbar.ax.tick_params(labelsize=8)


def compute_rank_table(df: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame({"encoder": df["encoder"]})

    available_temporal = [metric for metric in TEMPORAL_METRICS if metric in df.columns]
    available_segment = [metric for metric in SEGMENT_METRICS if metric in df.columns]
    available_token = [metric for metric in TOKEN_METRICS if metric in df.columns]

    if available_temporal:
        scores["temporal_sensitivity_score"] = normalized(df[available_temporal]).mean(axis=1)
    if available_segment:
        scores["segment_structure_score"] = normalized(df[available_segment]).mean(axis=1)
    if available_token:
        token_norm = normalized(df[available_token], invert={"token_top1_energy_ratio"})
        token_raw = df[available_token].fillna(0.0)
        single_vector = token_raw.abs().sum(axis=1) == 0.0
        scores["token_diversity_score"] = token_norm.mean(axis=1).mask(single_vector)
    if set(KNN_METRICS).issubset(df.columns):
        scores["knn_score"] = df[KNN_METRICS].mean(axis=1)

    score_cols = [col for col in scores.columns if col.endswith("_score")]
    if score_cols:
        scores["overall_normalized_mean"] = scores[score_cols].mean(axis=1, skipna=True)
    return scores.sort_values("overall_normalized_mean", ascending=False, na_position="last")


def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    df = load_summary(args.input, args.group)
    metrics = [metric for metric in ALL_METRICS if metric in df.columns]

    rank_table = compute_rank_table(df)
    rank_table.to_csv(out_prefix.with_name(out_prefix.name + "_ranked.csv"), index=False)

    num_examples = int(pd.to_numeric(df.get("num_examples", pd.Series([0])), errors="coerce").max())
    fig = plt.figure(figsize=(18, 13), constrained_layout=True)
    grid = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.35])

    add_grouped_bars(
        fig.add_subplot(grid[0, :2]),
        df,
        TEMPORAL_METRICS,
        title="Temporal Perturbation Sensitivity",
        ylabel="cosine distance",
    )
    add_knn_panel(fig.add_subplot(grid[0, 2]), df)
    add_grouped_bars(
        fig.add_subplot(grid[1, :2]),
        df,
        SEGMENT_METRICS,
        title="Segment-level Video Structure",
        ylabel="cosine distance / correlation",
    )
    add_normalized_token_bars(fig.add_subplot(grid[1, 2]), df)
    add_heatmap(fig.add_subplot(grid[2, :]), df, metrics)

    fig.suptitle(
        f"{args.title} | group={args.group} | n={num_examples}",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.01,
        0.01,
        "Note: heatmap values are min-max normalized per metric; token_top1_energy is inverted for diversity scoring. "
        "Raw values are annotated in each heatmap cell.",
        fontsize=9,
    )

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path)
    print(f"Wrote {png_path}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {out_prefix.with_name(out_prefix.name + '_ranked.csv')}")


if __name__ == "__main__":
    main()
