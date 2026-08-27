#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from vlmevalbench.data import read_jsonl, write_jsonl
from vlmevalbench.video_io import load_media_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a manifest to rows whose media files can be decoded.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--min-bytes", type=int, default=1024 * 128)
    parser.add_argument("--allow-missing-media", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.input)
    kept = []
    missing = 0
    too_small = 0
    bad = 0

    for record in tqdm(records, desc="filter_media"):
        media_path = record.get("media_path")
        if not media_path or not Path(media_path).exists():
            missing += 1
            if args.allow_missing_media:
                kept.append(record)
            continue

        path = Path(media_path)
        if args.min_bytes > 0 and path.stat().st_size < args.min_bytes:
            too_small += 1
            continue

        try:
            load_media_frames(path, record.get("media_type", "video"), args.num_frames)
        except Exception:
            bad += 1
            continue

        kept.append(record)

    write_jsonl(args.out, kept)
    print(f"Wrote {len(kept)} valid rows to {args.out}")
    print(f"Skipped missing={missing}, too_small={too_small}, bad_decode={bad}")


if __name__ == "__main__":
    main()
