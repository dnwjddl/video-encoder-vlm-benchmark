#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from vlmevalbench.data import normalize_choice_answer, safe_id, stable_id, write_jsonl
from vlmevalbench.video_io import load_media_frames


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
JSON_SKIP_NAMES = {"manifest.json", "metadata.json"}
QUESTION_KEYS = ("question", "query", "prompt", "problem")
CHOICE_KEYS = ("choices", "candidates", "options", "answer_options")
ANSWER_KEYS = ("answer", "gt_answer", "correct_answer", "label", "target")
VIDEO_KEYS = ("video", "video_path", "video_name", "filename", "file", "media_path", "vid", "video_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified MCQ JSONL manifest from an MVBench directory.")
    parser.add_argument("--mvbench-root", default="/home/woojunghan_google_com/hf_cache/mvbench_video")
    parser.add_argument("--out", default="data/benchmarks/mvbench_all.jsonl")
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--skip-missing-media", action="store_true")
    parser.add_argument("--validate-media", action="store_true")
    parser.add_argument("--validate-frames", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def first_value(item: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    lower_map = {str(k).lower(): v for k, v in item.items()}
    for key in keys:
        value = lower_map.get(key.lower())
        if value not in (None, ""):
            return value
    return None


def choice_to_text(choice: Any) -> str:
    if isinstance(choice, dict):
        for key in ("text", "value", "answer", "caption", "label", "name"):
            if key in choice and choice[key] not in (None, ""):
                return str(choice[key]).strip()
        return json.dumps(choice, ensure_ascii=False, sort_keys=True)
    return str(choice).strip()


def normalize_choices(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        keys = sorted(raw.keys(), key=lambda x: str(x))
        return [choice_to_text(raw[key]) for key in keys]
    if isinstance(raw, (list, tuple)):
        return [choice_to_text(choice) for choice in raw if choice_to_text(choice)]
    return []


def normalize_answer(raw: Any, choices: list[str]) -> str:
    if isinstance(raw, dict):
        raw = first_value(raw, ("answer", "text", "value", "label"))
    if isinstance(raw, int):
        if 0 <= raw < len(choices):
            return chr(ord("A") + raw)
        if 1 <= raw <= len(choices):
            return chr(ord("A") + raw - 1)
    return normalize_choice_answer(raw, choices)


def looks_like_mcq(item: dict[str, Any]) -> bool:
    return (
        first_value(item, QUESTION_KEYS) not in (None, "")
        and first_value(item, CHOICE_KEYS) is not None
        and first_value(item, ANSWER_KEYS) not in (None, "")
    )


def iter_mcq_items(obj: Any) -> Iterable[dict[str, Any]]:
    if isinstance(obj, list):
        for value in obj:
            yield from iter_mcq_items(value)
        return
    if not isinstance(obj, dict):
        return
    if looks_like_mcq(obj):
        yield obj
        return
    for value in obj.values():
        if isinstance(value, (list, dict)):
            yield from iter_mcq_items(value)


def build_media_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        rel = path.relative_to(root).as_posix()
        keys = {
            str(path),
            rel,
            path.name,
            path.stem,
            rel.removeprefix("video/"),
            rel.removeprefix("videos/"),
        }
        parts = path.relative_to(root).parts
        for start in range(len(parts)):
            suffix = Path(*parts[start:]).as_posix()
            keys.add(suffix)
            keys.add(Path(suffix).with_suffix("").as_posix())
        for key in keys:
            if key:
                index[key].append(path)
    return index


def resolve_media_path(root: Path, media_index: dict[str, list[Path]], raw: Any) -> str | None:
    if raw in (None, ""):
        return None
    raw_text = str(raw).strip()
    candidates = [raw_text, raw_text.lstrip("./"), raw_text.replace("\\", "/").lstrip("./")]
    raw_path = Path(raw_text)
    if raw_path.is_absolute() and raw_path.exists():
        return str(raw_path)
    for candidate in candidates:
        direct = root / candidate
        if direct.exists():
            return str(direct)
        for prefix in ("video", "videos", "mvbench_video"):
            prefixed = root / prefix / candidate
            if prefixed.exists():
                return str(prefixed)
        keys = [candidate, Path(candidate).name, Path(candidate).stem]
        if not Path(candidate).suffix:
            keys.extend(f"{candidate}{ext}" for ext in VIDEO_EXTS)
            keys.extend(f"{Path(candidate).name}{ext}" for ext in VIDEO_EXTS)
        for key in keys:
            matches = media_index.get(key)
            if matches:
                return str(matches[0])
    return None


def task_name_from_json(root: Path, json_path: Path) -> str:
    rel = json_path.relative_to(root)
    if rel.parent == Path("."):
        return json_path.stem
    return "/".join((*rel.parent.parts, json_path.stem))


def main() -> None:
    args = parse_args()
    root = Path(args.mvbench_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"MVBench root does not exist: {root}")

    media_index = build_media_index(root)
    rows = []
    counters = Counter()
    task_counts = Counter()
    seen_ids: set[str] = set()

    json_paths = [
        path
        for path in sorted(root.rglob("*.json"))
        if path.name not in JSON_SKIP_NAMES and ".ipynb_checkpoints" not in path.parts
    ]
    for json_path in tqdm(json_paths, desc="mvbench_json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: failed to parse {json_path}: {type(exc).__name__}: {exc}")
            continue

        task_name = task_name_from_json(root, json_path)
        for idx, item in enumerate(iter_mcq_items(data)):
            counters["raw_mcq"] += 1
            question = str(first_value(item, QUESTION_KEYS) or "").strip()
            choices = normalize_choices(first_value(item, CHOICE_KEYS))
            answer = normalize_answer(first_value(item, ANSWER_KEYS), choices)
            if not question or not choices or not answer:
                counters["invalid_mcq"] += 1
                continue

            raw_video = first_value(item, VIDEO_KEYS)
            media_path = resolve_media_path(root, media_index, raw_video)
            if not media_path:
                counters["missing_media"] += 1
                if args.skip_missing_media:
                    continue

            if media_path and args.validate_media:
                try:
                    load_media_frames(media_path, "video", args.validate_frames)
                except Exception as exc:
                    counters["bad_media"] += 1
                    if args.skip_missing_media:
                        continue
                    print(f"Warning: bad media {media_path}: {type(exc).__name__}: {exc}")

            raw_id = first_value(item, ("id", "question_id", "qid", "uid"))
            base_row_id = (
                f"mvbench_{safe_id(task_name)}_{safe_id(str(raw_id))}"
                if raw_id not in (None, "")
                else f"mvbench_{stable_id(task_name, str(idx), question)}"
            )
            row_id = base_row_id
            if row_id in seen_ids:
                counters["duplicate_ids"] += 1
                row_id = f"{base_row_id}_{stable_id(str(json_path), str(idx), question)}"
            seen_ids.add(row_id)
            row = {
                "id": row_id,
                "source": "MVBench",
                "benchmark": task_name,
                "task_type": task_name,
                "task": "mcq",
                "media_type": "video",
                "media_path": media_path,
                "question": question,
                "choices": choices,
                "answer": answer,
                "answer_text": choices[ord(answer) - ord("A")] if len(answer) == 1 and answer.isalpha() and ord(answer) - ord("A") < len(choices) else str(first_value(item, ANSWER_KEYS)),
                "mvbench_video": raw_video,
                "mvbench_json": str(json_path),
            }
            rows.append(row)
            task_counts[task_name] += 1
            counters["kept"] += 1
            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    write_jsonl(args.out, rows)
    summary = {
        "mvbench_root": str(root),
        "out": args.out,
        "num_json_files": len(json_paths),
        "num_media_files_indexed": sum(1 for _ in root.rglob("*") if _.is_file() and _.suffix.lower() in VIDEO_EXTS),
        "counts": dict(counters),
        "task_counts": dict(sorted(task_counts.items())),
    }
    summary_out = Path(args.summary_out) if args.summary_out else Path(args.out).with_suffix(".summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} MVBench rows to {args.out}")
    print(f"Wrote summary to {summary_out}")
    print(f"Counts: {dict(counters)}")


if __name__ == "__main__":
    main()
