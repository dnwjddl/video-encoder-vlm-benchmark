#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from vlmevalbench.configs import resolve_encoder
from vlmevalbench.data import read_jsonl, write_jsonl
from vlmevalbench.encoders import FrozenEncoder
from vlmevalbench.utils import get_dtype, set_seed
from vlmevalbench.video_io import load_media_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "No-training diagnostics for frozen visual encoders. "
            "Measures temporal order sensitivity, shuffle sensitivity, and segment diversity."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--encoder-config", default="configs/encoders.yaml")
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--knn-k", type=int, default=5)
    parser.add_argument("--knn-max-examples", type=int, default=5000)
    parser.add_argument("--skip-token-metrics", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-media", action="store_true")
    parser.add_argument(
        "--strict-media",
        action="store_true",
        help="Raise on unreadable/corrupt media instead of skipping it.",
    )
    return parser.parse_args()


def pooled_vector(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError(f"Expected token tensor [N,D], got {tuple(tokens.shape)}")
    vec = tokens.mean(dim=0)
    return F.normalize(vec.float(), dim=0)


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((1.0 - F.cosine_similarity(a, b, dim=0)).item())


def mean_pairwise_distance(vectors: list[torch.Tensor]) -> float:
    if len(vectors) < 2:
        return 0.0
    distances = [cosine_distance(a, b) for a, b in itertools.combinations(vectors, 2)]
    return sum(distances) / len(distances)


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.float32)
    x = x - x.mean()
    y = y - y.mean()
    denom = x.norm() * y.norm()
    if float(denom) == 0.0:
        return 0.0
    return float((x @ y / denom).item())


def adjacent_far_stats(vectors: list[torch.Tensor]) -> tuple[float, float, float]:
    if len(vectors) < 2:
        return 0.0, 0.0, 0.0
    adjacent = [cosine_distance(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
    far = []
    for i in range(len(vectors)):
        for j in range(i + 2, len(vectors)):
            far.append(cosine_distance(vectors[i], vectors[j]))
    adjacent_mean = sum(adjacent) / len(adjacent)
    far_mean = sum(far) / len(far) if far else adjacent_mean
    return adjacent_mean, far_mean, far_mean - adjacent_mean


def temporal_distance_correlation(vectors: list[torch.Tensor]) -> float:
    distances: list[float] = []
    gaps: list[float] = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distances.append(cosine_distance(vectors[i], vectors[j]))
            gaps.append(float(j - i))
    return pearson(gaps, distances)


def resize_frames(frames: list, target_count: int) -> list:
    if len(frames) == target_count:
        return frames
    if len(frames) == 0:
        raise ValueError("Cannot resize an empty frame list.")
    if target_count <= 1:
        return [frames[len(frames) // 2]]
    indices = torch.linspace(0, len(frames) - 1, target_count).round().long().tolist()
    return [frames[idx] for idx in indices]


def split_segments(frames: list, count: int, target_frames: int) -> list[list]:
    if count <= 1:
        return [resize_frames(frames, target_frames)]
    segments = []
    total = len(frames)
    for seg_idx in range(count):
        start = round(seg_idx * total / count)
        end = round((seg_idx + 1) * total / count)
        chunk = frames[start:max(end, start + 1)]
        segments.append(resize_frames(chunk, target_frames))
    return segments


def shifted_frames(frames: list) -> list:
    if len(frames) <= 1:
        return frames
    offset = len(frames) // 2
    return frames[offset:] + frames[:offset]


def swapped_halves(frames: list) -> list:
    if len(frames) <= 1:
        return frames
    mid = len(frames) // 2
    return frames[mid:] + frames[:mid]


def strided_frames(frames: list, target_count: int) -> list:
    if len(frames) <= 2:
        return resize_frames(frames, target_count)
    return resize_frames(frames[::2], target_count)


def encode_clip_vector(encoder: FrozenEncoder, family: str, frames: list) -> torch.Tensor:
    if family != "image":
        return pooled_vector(encoder.encode_frames(frames))

    # Frame encoders do not model temporal order by themselves. Average frame-level
    # vectors explicitly so reverse/shuffle metrics do not come from token-pooling artifacts.
    frame_vectors = [pooled_vector(encoder.encode_frames([frame])) for frame in frames]
    return F.normalize(torch.stack(frame_vectors, dim=0).mean(dim=0), dim=0)


def token_compression_metrics(tokens: torch.Tensor, max_pairwise_tokens: int = 128) -> dict[str, float]:
    if tokens.ndim != 2 or tokens.shape[0] < 2:
        return {
            "token_effective_rank": 0.0,
            "token_rank_ratio": 0.0,
            "token_top1_energy_ratio": 0.0,
            "token_mean_pairwise_distance": 0.0,
        }
    x = F.normalize(tokens.float(), dim=-1)
    centered = x - x.mean(dim=0, keepdim=True)
    gram = centered @ centered.T
    eigvals = torch.linalg.eigvalsh(gram).clamp_min(0)
    total = eigvals.sum().clamp_min(1e-12)
    probs = eigvals / total
    entropy = -(probs * (probs + 1e-12).log()).sum()
    effective_rank = float(torch.exp(entropy).item())
    rank_ratio = effective_rank / max(float(min(tokens.shape[0], tokens.shape[1])), 1.0)
    top1 = float((eigvals.max() / total).item())

    if x.shape[0] > max_pairwise_tokens:
        idx = torch.linspace(0, x.shape[0] - 1, max_pairwise_tokens).round().long()
        x_pair = x[idx]
    else:
        x_pair = x
    sims = x_pair @ x_pair.T
    n = sims.shape[0]
    tri = torch.triu_indices(n, n, offset=1)
    distances = 1.0 - sims[tri[0], tri[1]]
    mean_pairwise = float(distances.mean().item()) if distances.numel() else 0.0
    return {
        "token_effective_rank": effective_rank,
        "token_rank_ratio": rank_ratio,
        "token_top1_energy_ratio": top1,
        "token_mean_pairwise_distance": mean_pairwise,
    }


def label_for_knn(record: dict) -> str | None:
    label = record.get("label")
    if label not in (None, ""):
        return str(label)
    task = str(record.get("task", "")).lower()
    if task in {"classification", "class"} and record.get("answer") not in (None, ""):
        return str(record["answer"])
    return None


def knn_metrics(
    vectors: list[torch.Tensor],
    labels: list[str | None],
    *,
    k: int,
    max_examples: int,
    seed: int,
) -> dict[str, float]:
    valid = [(vec, label) for vec, label in zip(vectors, labels) if label]
    label_counts = Counter(label for _, label in valid)
    valid = [(vec, label) for vec, label in valid if label_counts[label] >= 2]
    if len(valid) < 2:
        return {"knn_num_examples": 0, "knn_top1": 0.0, f"knn_top{k}": 0.0}

    if max_examples > 0 and len(valid) > max_examples:
        rng = random.Random(seed)
        valid = rng.sample(valid, max_examples)

    mat = F.normalize(torch.stack([vec for vec, _ in valid]).float(), dim=-1)
    label_list = [label for _, label in valid]
    top1_correct = 0
    topk_correct = 0
    batch_size = 512
    k_eff = min(k, len(valid) - 1)

    for start in range(0, len(valid), batch_size):
        end = min(start + batch_size, len(valid))
        sims = mat[start:end] @ mat.T
        row_ids = torch.arange(start, end)
        sims[torch.arange(end - start), row_ids] = -float("inf")
        top_idx = sims.topk(k_eff, dim=1).indices.tolist()
        for local_idx, neighbors in enumerate(top_idx):
            gold = label_list[start + local_idx]
            neighbor_labels = [label_list[idx] for idx in neighbors]
            if neighbor_labels and neighbor_labels[0] == gold:
                top1_correct += 1
            if gold in neighbor_labels:
                topk_correct += 1

    denom = len(valid)
    return {
        "knn_num_examples": float(denom),
        "knn_top1": top1_correct / denom,
        f"knn_top{k}": topk_correct / denom,
    }


def summarize(rows: list[dict], global_metrics: dict[str, float] | None = None) -> list[dict]:
    metric_keys = [
        "order_distance",
        "shuffle_distance",
        "cycle_shift_distance",
        "half_swap_distance",
        "stride_distance",
        "segment_diversity",
        "segment_adjacent_distance",
        "segment_far_distance",
        "segment_temporal_margin",
        "segment_distance_correlation",
        "token_effective_rank",
        "token_rank_ratio",
        "token_top1_energy_ratio",
        "token_mean_pairwise_distance",
    ]
    groups: dict[str, list[dict]] = {"ALL": rows}
    for row in rows:
        groups.setdefault(str(row.get("source", "unknown")), []).append(row)

    summary = []
    for group, items in sorted(groups.items()):
        out = {"group": group, "num_examples": len(items)}
        for key in metric_keys:
            vals = [float(item[key]) for item in items if key in item]
            out[key] = sum(vals) / len(vals) if vals else 0.0
        if group == "ALL" and global_metrics:
            out.update(global_metrics)
        summary.append(out)
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = resolve_encoder(
        args.encoder,
        args.encoder_config,
        overrides={
            "model_id": args.model_id,
            "num_frames": args.num_frames,
            "max_tokens": args.max_tokens,
        },
    )
    records = read_jsonl(args.manifest)
    if args.limit:
        records = records[: args.limit]

    encoder = FrozenEncoder(cfg, device=args.device, dtype=get_dtype(args.dtype))
    rng = random.Random(args.seed)
    rows = []
    pooled_vectors: list[torch.Tensor] = []
    labels: list[str | None] = []
    skipped = 0
    skipped_bad_media = 0

    for record in tqdm(records, desc=f"diagnose:{args.encoder}"):
        media_path = record.get("media_path")
        if not media_path or not Path(media_path).exists():
            if args.allow_missing_media:
                skipped += 1
                continue
            raise FileNotFoundError(f"Missing media for id={record.get('id')}: {media_path}")

        try:
            frames = load_media_frames(media_path, record.get("media_type", "video"), cfg.num_frames)
        except Exception as exc:
            if args.strict_media:
                raise
            skipped_bad_media += 1
            print(
                f"Warning: skipping unreadable media id={record.get('id')} "
                f"path={media_path}: {type(exc).__name__}: {exc}"
            )
            continue
        reversed_frames = list(reversed(frames))
        shuffled_frames = list(frames)
        rng.shuffle(shuffled_frames)
        cycle_frames = shifted_frames(frames)
        swapped_frames = swapped_halves(frames)
        stride_frames = strided_frames(frames, cfg.num_frames)

        orig_tokens = encoder.encode_frames(frames)
        orig_vec = encode_clip_vector(encoder, cfg.family, frames)
        rev_vec = encode_clip_vector(encoder, cfg.family, reversed_frames)
        shuf_vec = encode_clip_vector(encoder, cfg.family, shuffled_frames)
        cycle_vec = encode_clip_vector(encoder, cfg.family, cycle_frames)
        swap_vec = encode_clip_vector(encoder, cfg.family, swapped_frames)
        stride_vec = encode_clip_vector(encoder, cfg.family, stride_frames)

        segment_vectors = [
            encode_clip_vector(encoder, cfg.family, segment)
            for segment in split_segments(frames, args.segments, cfg.num_frames)
        ]
        adjacent, far, margin = adjacent_far_stats(segment_vectors)
        compression = {} if args.skip_token_metrics else token_compression_metrics(orig_tokens)
        pooled_vectors.append(orig_vec.cpu())
        labels.append(label_for_knn(record))

        row = {
            "id": record["id"],
            "source": record.get("source", "unknown"),
            "benchmark": record.get("benchmark", record.get("source", "unknown")),
            "encoder": args.encoder,
            "family": cfg.family,
            "label": label_for_knn(record),
            "order_distance": cosine_distance(orig_vec, rev_vec),
            "shuffle_distance": cosine_distance(orig_vec, shuf_vec),
            "cycle_shift_distance": cosine_distance(orig_vec, cycle_vec),
            "half_swap_distance": cosine_distance(orig_vec, swap_vec),
            "stride_distance": cosine_distance(orig_vec, stride_vec),
            "segment_diversity": mean_pairwise_distance(segment_vectors),
            "segment_adjacent_distance": adjacent,
            "segment_far_distance": far,
            "segment_temporal_margin": margin,
            "segment_distance_correlation": temporal_distance_correlation(segment_vectors),
        }
        row.update(compression)
        rows.append(row)

    write_jsonl(args.out_jsonl, rows)
    global_metrics = knn_metrics(
        pooled_vectors,
        labels,
        k=args.knn_k,
        max_examples=args.knn_max_examples,
        seed=args.seed,
    )
    summary = summarize(rows, global_metrics)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["group", "num_examples"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"Wrote per-example diagnostics to {args.out_jsonl}")
    print(f"Wrote summary diagnostics to {args.out_csv}")
    if skipped:
        print(f"Skipped {skipped} rows with missing media.")
    if skipped_bad_media:
        print(f"Skipped {skipped_bad_media} rows with unreadable/corrupt media.")


if __name__ == "__main__":
    main()
