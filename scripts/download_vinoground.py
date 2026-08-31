#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download


REQUIRED_FILES = (
    "vinoground.csv",
    "vinoground_textscore.json",
    "vinoground_videoscore.json",
    "vinoground_videos.zip",
    "vinoground_videos_concated.zip",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract the official Vinoground dataset.")
    parser.add_argument(
        "--out-dir",
        default="/mnt/disks/data/vlm_encoder_benchmark/datasets/vinoground",
    )
    parser.add_argument("--repo-id", default="HanSolo9682/Vinoground")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise RuntimeError(f"Unsafe path in {archive}: {member.filename}")
        handle.extractall(destination)


def media_counts(root: Path) -> dict[str, int]:
    return {
        "individual_videos": len(list((root / "vinoground_videos").glob("*.mp4"))),
        "concatenated_videos": len(list((root / "vinoground_videos_concated").glob("*.mp4"))),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_missing = [name for name in REQUIRED_FILES if not (out_dir / name).is_file()]
    existing_counts = media_counts(out_dir)
    if (
        not args.force_download
        and not existing_missing
        and existing_counts["individual_videos"] >= 1000
        and existing_counts["concatenated_videos"] >= 1000
    ):
        print(f"SKIP download; complete Vinoground data already exists in {out_dir}")
        print(json.dumps(existing_counts, indent=2))
        return

    print(f"Downloading {args.repo_id} to {out_dir}")
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=list(REQUIRED_FILES) + ["README.md"],
        force_download=args.force_download,
        local_files_only=args.local_files_only,
    )

    missing = [name for name in REQUIRED_FILES if not (out_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Vinoground download is incomplete; missing: {', '.join(missing)}")

    archives = (
        (out_dir / "vinoground_videos.zip", out_dir / "vinoground_videos", "individual_videos"),
        (
            out_dir / "vinoground_videos_concated.zip",
            out_dir / "vinoground_videos_concated",
            "concatenated_videos",
        ),
    )
    for archive, extracted_dir, count_key in archives:
        if media_counts(out_dir)[count_key] >= 1000:
            print(f"SKIP extraction; found videos in {extracted_dir}")
            continue
        print(f"Extracting {archive.name}")
        safe_extract(archive, out_dir)

    counts = media_counts(out_dir)
    if counts["individual_videos"] < 1000:
        raise RuntimeError(
            f"Expected at least 1000 individual videos, found {counts['individual_videos']} in {out_dir}"
        )
    if counts["concatenated_videos"] < 1000:
        raise RuntimeError(
            "Expected at least 1000 concatenated videos, "
            f"found {counts['concatenated_videos']} in {out_dir}"
        )

    summary = {
        "repo_id": args.repo_id,
        "out_dir": str(out_dir),
        "hf_home": os.environ.get("HF_HOME"),
        **counts,
    }
    summary_path = out_dir / "download_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
