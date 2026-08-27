#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from tqdm import tqdm

from vlmevalbench.data import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a small ActivityNet MP4 subset with direct video URLs and "
            "write a manifest that can be used by no-training diagnostics."
        )
    )
    parser.add_argument("--dataset-id", default="TornadoLabs/activitynet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--video-dir", default="/data/videos/activitynet")
    parser.add_argument("--out", default="data/manifests/activitynet_debug.jsonl")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=180.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def pick(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def download_file(url: str, path: Path, timeout: int) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp_path.open("wb") as f:
            progress = tqdm(
                total=total if total > 0 else None,
                unit="B",
                unit_scale=True,
                leave=False,
                desc=path.name,
            )
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    progress.update(len(chunk))
            progress.close()
    tmp_path.replace(path)
    return True


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset_id, split=args.split)
    rows = [dict(row) for row in ds]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    manifest = []
    failures = 0
    seen = set()

    for row in tqdm(rows, desc="download_activitynet"):
        if len(manifest) >= args.max_samples:
            break
        duration = pick(row, ["duration"])
        if duration is not None and args.max_duration > 0 and float(duration) > args.max_duration:
            continue

        video_id = pick(row, ["video_id", "youtube_id", "id"])
        video_url = pick(row, ["video_url", "url"])
        if not video_id or not video_url or str(video_id) in seen:
            continue

        path = video_dir / f"{video_id}.mp4"
        try:
            if not path.exists() or not args.skip_existing:
                download_file(str(video_url), path, args.timeout)
        except Exception as exc:
            failures += 1
            print(f"Warning: failed to download {video_id}: {type(exc).__name__}: {exc}")
            continue

        seen.add(str(video_id))
        manifest.append(
            {
                "id": f"activitynet_{video_id}",
                "source": "activitynet",
                "benchmark": "activitynet_diagnostic",
                "task": "diagnostic",
                "media_type": "video",
                "media_path": str(path),
                "question": "Describe the video.",
                "answer": "",
                "choices": None,
                "duration": duration,
                "label": pick(row, ["label"]),
                "youtube_url": pick(row, ["youtube_url"]),
            }
        )

    write_jsonl(args.out, manifest)
    print(f"Wrote {len(manifest)} downloaded-video records to {args.out}")
    if failures:
        print(f"Download failures: {failures}")


if __name__ == "__main__":
    main()
