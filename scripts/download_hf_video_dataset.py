#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from vlmevalbench.data import safe_id, write_jsonl
from vlmevalbench.video_io import load_media_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a unified video manifest from a Hugging Face dataset. "
            "Supports datasets with a real Video column and metadata-only datasets "
            "that point to files inside the dataset repo."
        )
    )
    parser.add_argument("--dataset-id", default="VLM2Vec/mvbench-FunQA_test")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--source-mode",
        choices=["video-column", "path-column"],
        default="video-column",
        help="Use video-column for HF Video features; path-column for repo-hosted file paths.",
    )
    parser.add_argument("--video-column", default="video")
    parser.add_argument("--video-path-column", default="video_path")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--id-column", default=None)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--choices-column", default="candidates")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument(
        "--video-dir",
        default="/mnt/disks/data/vlm_encoder_benchmark/videos/hf_video_debug",
    )
    parser.add_argument("--out", default="data/manifests/hf_video_debug.jsonl")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer-size", type=int, default=1000)
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Keep dataset order before applying --max-samples.",
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable HF streaming. Useful when you want a fully materialized local dataset.",
    )
    parser.add_argument("--validate", action="store_true", help="Decode frames before keeping a row.")
    parser.add_argument("--num-frames", type=int, default=8)
    return parser.parse_args()


def optional_value(row: dict[str, Any], key: str | None) -> Any:
    if not key:
        return None
    return row.get(key)


def feature_for_column(dataset: Any, column: str | None) -> Any:
    if not column:
        return None
    features = getattr(dataset, "features", None)
    if not features:
        return None
    try:
        return features[column]
    except Exception:
        return None


def label_to_text(dataset: Any, column: str | None, value: Any) -> str | None:
    if value in (None, ""):
        return None
    feature = feature_for_column(dataset, column)
    if feature is not None and hasattr(feature, "int2str"):
        try:
            return str(feature.int2str(int(value)))
        except Exception:
            pass
    return str(value)


def disable_video_decoding(dataset: Any, video_column: str) -> Any:
    if hasattr(dataset, "decode"):
        try:
            return dataset.decode(False)
        except Exception:
            pass

    try:
        from datasets import Video

        return dataset.cast_column(video_column, Video(decode=False))
    except Exception as exc:
        raise RuntimeError(
            "Could not disable video decoding. Upgrade `datasets`, or install the "
            "video backend required by Hugging Face datasets."
        ) from exc


def maybe_shuffle(dataset: Any, args: argparse.Namespace) -> Any:
    if args.no_shuffle:
        return dataset
    if hasattr(dataset, "shuffle"):
        try:
            return dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)
        except TypeError:
            return dataset.shuffle(seed=args.seed)
        except Exception:
            return dataset
    return dataset


def iter_rows(dataset: Any, args: argparse.Namespace) -> Iterable[tuple[int, dict[str, Any]]]:
    if not args.no_streaming and hasattr(dataset, "take"):
        for idx, row in enumerate(dataset):
            yield idx, dict(row)
        return

    rows = [dict(row) for row in dataset]
    if not args.no_shuffle:
        random.Random(args.seed).shuffle(rows)
    for idx, row in enumerate(rows):
        yield idx, row


def suffix_from_path(path: str | Path | None, fallback: str = ".mp4") -> str:
    if not path:
        return fallback
    suffix = Path(str(path)).suffix
    return suffix if suffix else fallback


def write_video_bytes(video_value: dict[str, Any], out_path: Path) -> Path:
    data = video_value.get("bytes")
    if data is None:
        raise ValueError("Video value has no bytes.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(data)
    return out_path


def copy_video_file(source: str | Path, out_path: Path) -> Path:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    shutil.copy2(source_path, out_path)
    return out_path


def materialize_video_column(row: dict[str, Any], args: argparse.Namespace, video_dir: Path, row_id: str) -> Path:
    value = row.get(args.video_column)
    if value is None:
        raise KeyError(f"Column '{args.video_column}' not found. Available columns: {sorted(row)}")

    if isinstance(value, dict):
        source_path = value.get("path")
        suffix = suffix_from_path(source_path)
        out_path = video_dir / f"{safe_id(row_id)}{suffix}"
        if value.get("bytes") is not None:
            return write_video_bytes(value, out_path)
        if source_path:
            return copy_video_file(source_path, out_path)
        raise ValueError(f"Video column '{args.video_column}' has neither bytes nor path.")

    if isinstance(value, (str, Path)):
        suffix = suffix_from_path(value)
        return copy_video_file(value, video_dir / f"{safe_id(row_id)}{suffix}")

    raise TypeError(
        f"Unsupported video value type from column '{args.video_column}': {type(value).__name__}. "
        "Use --source-mode path-column for metadata paths."
    )


def materialize_path_column(row: dict[str, Any], args: argparse.Namespace, video_dir: Path) -> tuple[Path, str]:
    video_path = optional_value(row, args.video_path_column)
    if not video_path:
        raise KeyError(f"Column '{args.video_path_column}' not found. Available columns: {sorted(row)}")
    video_path = str(video_path)
    downloaded = hf_hub_download(
        repo_id=args.dataset_id,
        repo_type="dataset",
        filename=video_path,
        local_dir=video_dir,
    )
    return Path(downloaded), video_path


def row_identifier(row: dict[str, Any], args: argparse.Namespace, row_idx: int) -> str:
    value = optional_value(row, args.id_column)
    if value not in (None, ""):
        return str(value)

    if args.source_mode == "path-column":
        video_path = optional_value(row, args.video_path_column)
        if video_path:
            return Path(str(video_path)).stem

    video_value = row.get(args.video_column)
    if isinstance(video_value, dict) and video_value.get("path"):
        return Path(str(video_value["path"])).stem
    if isinstance(video_value, (str, Path)):
        return Path(str(video_value)).stem
    return f"row_{row_idx:06d}"


def manifest_record(
    *,
    row: dict[str, Any],
    args: argparse.Namespace,
    dataset: Any,
    row_idx: int,
    row_id: str,
    media_path: Path,
    original_video_path: str | None,
) -> dict[str, Any]:
    label = label_to_text(dataset, args.label_column, optional_value(row, args.label_column))
    question = optional_value(row, args.question_column)
    choices = optional_value(row, args.choices_column)
    answer = optional_value(row, args.answer_column)

    if choices is not None:
        choices = [str(choice) for choice in choices]
    task = "mcq" if choices and answer not in (None, "") else "diagnostic"

    return {
        "id": f"{safe_id(args.dataset_id)}_{safe_id(row_id)}",
        "source": args.dataset_id,
        "benchmark": f"{safe_id(args.dataset_id)}_diagnostic",
        "task": task,
        "media_type": "video",
        "media_path": str(media_path),
        "question": str(question or "Describe the video."),
        "answer": str(answer or ""),
        "choices": choices,
        "label": label,
        "video_id": row_id,
        "row_index": row_idx,
        "original_video_path": original_video_path,
    }


def main() -> None:
    args = parse_args()
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(
        args.dataset_id,
        split=args.split,
        streaming=not args.no_streaming,
    )
    if args.source_mode == "video-column":
        dataset = disable_video_decoding(dataset, args.video_column)
    dataset = maybe_shuffle(dataset, args)

    manifest = []
    skipped = 0
    pbar = tqdm(desc=f"download:{args.dataset_id}:{args.split}", total=args.max_samples or None)
    for row_idx, row in iter_rows(dataset, args):
        if args.max_samples and len(manifest) >= args.max_samples:
            break

        row_id = row_identifier(row, args, row_idx)
        try:
            if args.source_mode == "video-column":
                media_path = materialize_video_column(row, args, video_dir, row_id)
                original_video_path = None
            else:
                media_path, original_video_path = materialize_path_column(row, args, video_dir)
        except Exception as exc:
            skipped += 1
            print(f"Warning: failed to materialize row {row_idx} ({row_id}): {type(exc).__name__}: {exc}")
            pbar.update(1)
            continue

        if args.validate:
            try:
                load_media_frames(media_path, "video", args.num_frames)
            except Exception as exc:
                skipped += 1
                print(f"Warning: skipping unreadable video {media_path}: {type(exc).__name__}: {exc}")
                pbar.update(1)
                continue

        manifest.append(
            manifest_record(
                row=row,
                args=args,
                dataset=dataset,
                row_idx=row_idx,
                row_id=row_id,
                media_path=media_path,
                original_video_path=original_video_path,
            )
        )
        pbar.update(1)

    pbar.close()
    write_jsonl(args.out, manifest)
    print(f"Wrote {len(manifest)} rows to {args.out}")
    if skipped:
        print(f"Skipped {skipped} rows")


if __name__ == "__main__":
    main()
