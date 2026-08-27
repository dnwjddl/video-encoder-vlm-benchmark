#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from tqdm import tqdm

from vlmevalbench.data import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a small ActivityNet MP4 subset with direct video URLs and "
            "write a manifest that can be used by no-training diagnostics."
        )
    )
    parser.add_argument("--dataset-id", default="TornadoLabs/activitynet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--video-dir", default="/mnt/disks/data/vlm_encoder_benchmark/videos/activitynet")
    parser.add_argument("--out", default="data/manifests/activitynet_debug.jsonl")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration", type=float, default=180.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--sleep-between-downloads",
        type=float,
        default=1.0,
        help="Seconds to sleep after each successful download. Use this to reduce rate limits.",
    )
    parser.add_argument(
        "--retry-sleep",
        type=float,
        default=300.0,
        help="Seconds to sleep before stopping after HTTP 429 unless --continue-on-429 is set.",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=25,
        help="Stop after this many consecutive failed downloads. Use 0 to disable.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Rewrite output manifest after this many new successes.",
    )
    parser.add_argument(
        "--continue-on-429",
        action="store_true",
        help="Keep scanning after HTTP 429. Default behavior stops to avoid hammering YouTube.",
    )
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


def make_manifest_record(row: dict[str, Any], video_id: str, path: Path) -> dict[str, Any]:
    return {
        "id": f"activitynet_{video_id}",
        "source": "activitynet",
        "benchmark": "activitynet_diagnostic",
        "task": "diagnostic",
        "media_type": "video",
        "media_path": str(path),
        "question": "Describe the video.",
        "answer": "",
        "choices": None,
        "duration": pick(row, ["duration"]),
        "label": pick(row, ["label"]),
        "video_id": video_id,
        "youtube_url": pick(row, ["youtube_url"]),
    }


def existing_video_ids(manifest: list[dict[str, Any]]) -> set[str]:
    ids = set()
    for record in manifest:
        video_id = record.get("video_id")
        if video_id:
            ids.add(str(video_id))
            continue
        record_id = str(record.get("id", ""))
        if record_id.startswith("activitynet_"):
            ids.add(record_id[len("activitynet_") :])
    return ids


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset_id, split=args.split)
    rows = [dict(row) for row in ds]
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    manifest = read_jsonl(out_path) if out_path.exists() else []
    failures = 0
    consecutive_failures = 0
    successes_since_flush = 0
    seen = existing_video_ids(manifest)
    print(f"Loaded {len(manifest)} existing manifest rows from {out_path}")

    try:
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
                    if args.sleep_between_downloads > 0:
                        time.sleep(args.sleep_between_downloads)
            except Exception as exc:
                failures += 1
                consecutive_failures += 1
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                print(f"Warning: failed to download {video_id}: {type(exc).__name__}: {exc}")
                if status_code == 429 and not args.continue_on_429:
                    write_jsonl(out_path, manifest)
                    print(f"HTTP 429 rate-limit hit. Wrote {len(manifest)} rows to {out_path}.")
                    print(f"Sleeping {args.retry_sleep:.0f}s, then stopping. Resume later with the same command.")
                    if args.retry_sleep > 0:
                        time.sleep(args.retry_sleep)
                    break
                if args.max_consecutive_failures > 0 and consecutive_failures >= args.max_consecutive_failures:
                    write_jsonl(out_path, manifest)
                    print(
                        f"Stopping after {consecutive_failures} consecutive failures. "
                        f"Wrote {len(manifest)} rows to {out_path}."
                    )
                    break
                continue

            consecutive_failures = 0
            seen.add(str(video_id))
            manifest.append(make_manifest_record(row, str(video_id), path))
            successes_since_flush += 1
            if args.flush_every > 0 and successes_since_flush >= args.flush_every:
                write_jsonl(out_path, manifest)
                successes_since_flush = 0
    except KeyboardInterrupt:
        print("Interrupted by user. Writing partial manifest before exit.")
    finally:
        write_jsonl(out_path, manifest)

    print(f"Wrote {len(manifest)} downloaded-video records to {out_path}")
    if failures:
        print(f"Download failures: {failures}")


if __name__ == "__main__":
    main()
