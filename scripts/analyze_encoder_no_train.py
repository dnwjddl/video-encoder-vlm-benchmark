#!/usr/bin/env python
from __future__ import annotations

import argparse
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-media", action="store_true")
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


def encode_clip_vector(encoder: FrozenEncoder, family: str, frames: list) -> torch.Tensor:
    if family != "image":
        return pooled_vector(encoder.encode_frames(frames))

    # Frame encoders do not model temporal order by themselves. Average frame-level
    # vectors explicitly so reverse/shuffle metrics do not come from token-pooling artifacts.
    frame_vectors = [pooled_vector(encoder.encode_frames([frame])) for frame in frames]
    return F.normalize(torch.stack(frame_vectors, dim=0).mean(dim=0), dim=0)


def summarize(rows: list[dict]) -> list[dict]:
    metric_keys = [
        "order_distance",
        "shuffle_distance",
        "segment_diversity",
        "segment_adjacent_distance",
        "segment_far_distance",
        "segment_temporal_margin",
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
    skipped = 0

    for record in tqdm(records, desc=f"diagnose:{args.encoder}"):
        media_path = record.get("media_path")
        if not media_path or not Path(media_path).exists():
            if args.allow_missing_media:
                skipped += 1
                continue
            raise FileNotFoundError(f"Missing media for id={record.get('id')}: {media_path}")

        frames = load_media_frames(media_path, record.get("media_type", "video"), cfg.num_frames)
        reversed_frames = list(reversed(frames))
        shuffled_frames = list(frames)
        rng.shuffle(shuffled_frames)

        orig_vec = encode_clip_vector(encoder, cfg.family, frames)
        rev_vec = encode_clip_vector(encoder, cfg.family, reversed_frames)
        shuf_vec = encode_clip_vector(encoder, cfg.family, shuffled_frames)

        segment_vectors = [
            encode_clip_vector(encoder, cfg.family, segment)
            for segment in split_segments(frames, args.segments, cfg.num_frames)
        ]
        adjacent, far, margin = adjacent_far_stats(segment_vectors)

        rows.append(
            {
                "id": record["id"],
                "source": record.get("source", "unknown"),
                "benchmark": record.get("benchmark", record.get("source", "unknown")),
                "encoder": args.encoder,
                "family": cfg.family,
                "order_distance": cosine_distance(orig_vec, rev_vec),
                "shuffle_distance": cosine_distance(orig_vec, shuf_vec),
                "segment_diversity": mean_pairwise_distance(segment_vectors),
                "segment_adjacent_distance": adjacent,
                "segment_far_distance": far,
                "segment_temporal_margin": margin,
            }
        )

    write_jsonl(args.out_jsonl, rows)
    summary = summarize(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["group", "num_examples"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"Wrote per-example diagnostics to {args.out_jsonl}")
    print(f"Wrote summary diagnostics to {args.out_csv}")
    if skipped:
        print(f"Skipped {skipped} rows with missing media.")


if __name__ == "__main__":
    main()
