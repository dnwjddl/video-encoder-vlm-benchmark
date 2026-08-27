#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from vlmevalbench.data import safe_id, write_jsonl
from vlmevalbench.video_io import load_media_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download video files that are hosted directly inside a Hugging Face dataset repo "
            "and write a unified manifest."
        )
    )
    parser.add_argument("--dataset-id", default="VLM2Vec/Kinetics-700")
    parser.add_argument("--split", default="test")
    parser.add_argument("--video-path-column", default="video_path")
    parser.add_argument("--label-column", default="pos_text")
    parser.add_argument("--id-column", default="video_id")
    parser.add_argument("--video-dir", default="/mnt/disks/data/vlm_encoder_benchmark/videos/kinetics700_1k")
    parser.add_argument("--out", default="data/manifests/kinetics700_1k.jsonl")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep dataset order before applying --max-samples.",
    )
    parser.add_argument("--validate", action="store_true", help="Decode frames before keeping a row.")
    parser.add_argument("--num-frames", type=int, default=8)
    return parser.parse_args()


def get_value(row: dict[str, Any], key: str) -> Any:
    value = row.get(key)
    if value is None:
        raise KeyError(f"Column '{key}' not found in row. Available columns: {sorted(row)}")
    return value


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset_id, split=args.split)
    rows = [dict(row) for row in ds]
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)
    if args.max_samples and len(rows) > args.max_samples:
        rows = rows[: args.max_samples]

    manifest = []
    skipped = 0
    for row in tqdm(rows, desc=f"download:{args.dataset_id}:{args.split}"):
        video_path = str(get_value(row, args.video_path_column))
        label = str(get_value(row, args.label_column))
        video_id = str(row.get(args.id_column) or Path(video_path).stem)

        try:
            downloaded = hf_hub_download(
                repo_id=args.dataset_id,
                repo_type="dataset",
                filename=video_path,
                local_dir=video_dir,
            )
        except Exception as exc:
            skipped += 1
            print(f"Warning: failed to download {video_path}: {type(exc).__name__}: {exc}")
            continue

        media_path = Path(downloaded)
        if args.validate:
            try:
                load_media_frames(media_path, "video", args.num_frames)
            except Exception as exc:
                skipped += 1
                print(f"Warning: skipping unreadable video {media_path}: {type(exc).__name__}: {exc}")
                continue

        manifest.append(
            {
                "id": f"{safe_id(args.dataset_id)}_{safe_id(video_id)}",
                "source": args.dataset_id,
                "benchmark": f"{safe_id(args.dataset_id)}_diagnostic",
                "task": "diagnostic",
                "media_type": "video",
                "media_path": str(media_path),
                "question": "Describe the video.",
                "answer": "",
                "choices": None,
                "label": label,
                "video_id": video_id,
                "original_video_path": video_path,
            }
        )

    write_jsonl(args.out, manifest)
    print(f"Wrote {len(manifest)} rows to {args.out}")
    if skipped:
        print(f"Skipped {skipped} rows")


if __name__ == "__main__":
    main()
