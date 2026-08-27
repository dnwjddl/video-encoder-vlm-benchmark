#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable

from datasets import load_dataset
from tqdm import tqdm

from vlmevalbench.data import sample_records, stable_id, write_jsonl

@dataclass(frozen=True)
class HFDatasetSpec:
    dataset_id: str
    split: str
    source: str
    config: str | None = None


LLAVA_VIDEO_CONFIGS = [
    "0_30_s_academic_v0_1",
    "0_30_s_youtube_v0_1",
    "30_60_s_academic_v0_1",
    "30_60_s_youtube_v0_1",
    "1_2_m_academic_v0_1",
    "1_2_m_youtube_v0_1",
    "2_3_m_academic_v0_1",
    "2_3_m_youtube_v0_1",
]


DEFAULT_HF_DATASETS = {
    "videoinstruct100k": [
        HFDatasetSpec("MBZUAI/VideoInstruct-100K", "train", "videoinstruct100k")
    ],
    "llava-video-178k": [
        HFDatasetSpec("lmms-lab/LLaVA-Video-178K", split, f"llava-video-178k_{config}", config)
        for config in LLAVA_VIDEO_CONFIGS
        for split in ("caption", "open_ended", "multi_choice")
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a unified training manifest from public video instruction annotations. "
            "This script downloads/streams annotations; videos may still need separate source-specific download."
        )
    )
    parser.add_argument("--out", required=True, help="Output unified JSONL manifest.")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["videoinstruct100k", "llava-video-178k"],
        help="Dataset aliases from defaults or explicit HF ids.",
    )
    parser.add_argument("--local-json", nargs="*", default=[], help="Optional local JSON/JSONL annotation files.")
    parser.add_argument("--video-root", default="", help="Root directory containing downloaded videos.")
    parser.add_argument(
        "--video-extension",
        default=".mp4",
        help="Append this extension when a media id/path has no suffix. Use '' to disable.",
    )
    parser.add_argument("--caption-count", type=int, default=100_000)
    parser.add_argument("--qa-count", type=int, default=100_000)
    parser.add_argument("--mcq-count", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--streaming", action="store_true", help="Stream HF datasets instead of full download.")
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument(
        "--exclude-source-keywords",
        nargs="*",
        default=["nextqa", "perceptiontest", "activitynet", "activitynetqa"],
        help="Drop records whose source/id/path contains these keywords to reduce eval leakage.",
    )
    return parser.parse_args()


def pick(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return None


def iter_local_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("data", "annotations", "records"):
                if key in data and isinstance(data[key], list):
                    yield from data[key]
                    return
            yield data
        else:
            yield from data


def extract_conversation(row: dict[str, Any]) -> tuple[str | None, str | None]:
    conversations = pick(row, ["conversations", "conversation", "messages"])
    if not isinstance(conversations, list):
        return None, None

    question = None
    answer = None
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        role = str(pick(msg, ["from", "role", "speaker"]) or "").lower()
        text = pick(msg, ["value", "content", "text"])
        if text is None:
            continue
        if role in {"human", "user"} and question is None:
            question = str(text).replace("<video>", "").replace("<image>", "").strip()
        elif role in {"gpt", "assistant"} and answer is None:
            answer = str(text).strip()
    return question, answer


def normalize_media_path(raw: Any, video_root: str, video_extension: str) -> str:
    if raw is None:
        return ""
    if isinstance(raw, dict):
        raw = pick(raw, ["path", "filename", "video", "video_path"])
    raw_text = str(raw)
    if raw_text.startswith("http://") or raw_text.startswith("https://"):
        return raw_text
    path = Path(raw_text)
    if not path.suffix and video_extension:
        path = path.with_suffix(video_extension)
    if path.is_absolute() or not video_root:
        return str(path)
    return str(Path(video_root) / path)


def normalize_row(
    row: dict[str, Any],
    source: str,
    video_root: str,
    video_extension: str,
) -> dict[str, Any] | None:
    question, answer = extract_conversation(row)
    question = question or pick(row, ["question", "query", "prompt", "instruction", "q"])
    answer = answer or pick(row, ["answer", "response", "output", "caption", "text", "a"])
    caption = pick(row, ["caption", "dense_caption", "description", "cap"])
    choices = pick(row, ["choices", "options", "candidates"])

    if isinstance(choices, str):
        try:
            choices = json.loads(choices)
        except Exception:
            choices = None
    if choices is not None:
        choices = [str(choice) for choice in choices]

    if question is None and caption is not None:
        question = "Describe the video."
        answer = caption
    if answer is None:
        return None

    media_raw = pick(
        row,
        [
            "video",
            "video_path",
            "video_name",
            "video_id",
            "videoid",
            "image",
            "image_path",
        ],
    )
    media_path = normalize_media_path(media_raw, video_root, video_extension)
    media_type = "image" if pick(row, ["image", "image_path"]) is not None else "video"

    task = "mcq" if choices else ("caption" if str(question).lower().startswith("describe") else "qa")
    raw_id = pick(row, ["id", "uid", "qid", "question_id", "video_id", "videoid"]) or stable_id(
        source, str(question), str(answer)
    )
    record_id = f"{source}_{raw_id}"

    normalized = {
        "id": record_id,
        "source": source,
        "benchmark": source,
        "task": task,
        "media_type": media_type,
        "media_path": media_path,
        "question": str(question).strip(),
        "answer": str(answer).strip(),
        "choices": choices,
    }
    return normalized


def should_exclude(record: dict[str, Any], keywords: list[str]) -> bool:
    text = " ".join(str(record.get(key, "")) for key in ("id", "source", "media_path")).lower()
    return any(keyword.lower() in text for keyword in keywords)


def expand_hf_specs(alias_or_id: str) -> list[HFDatasetSpec]:
    if alias_or_id in DEFAULT_HF_DATASETS:
        return DEFAULT_HF_DATASETS[alias_or_id]

    # Explicit form:
    #   owner/name
    #   owner/name:split
    #   owner/name:config:split
    parts = alias_or_id.split(":")
    dataset_id = parts[0]
    if len(parts) == 1:
        config = None
        split = "train"
    elif len(parts) == 2:
        config = None
        split = parts[1]
    elif len(parts) == 3:
        config = parts[1]
        split = parts[2]
    else:
        raise ValueError(f"Invalid HF dataset spec: {alias_or_id}")
    source = dataset_id.split("/")[-1].lower() if config is None else f"{dataset_id.split('/')[-1].lower()}_{config}"
    return [HFDatasetSpec(dataset_id, split, source, config)]


def load_hf_records(spec: HFDatasetSpec, streaming: bool, max_rows: int | None) -> list[dict[str, Any]]:
    if spec.config:
        ds = load_dataset(spec.dataset_id, spec.config, split=spec.split, streaming=streaming)
    else:
        ds = load_dataset(spec.dataset_id, split=spec.split, streaming=streaming)
    rows = []
    desc = f"load:{spec.source}:{spec.split}"
    for idx, row in enumerate(tqdm(ds, desc=desc)):
        rows.append(dict(row) | {"_source_alias": spec.source, "_split": spec.split})
        if max_rows is not None and idx + 1 >= max_rows:
            break
    return rows


def main() -> None:
    args = parse_args()
    normalized: list[dict[str, Any]] = []

    for item in args.include:
        for spec in expand_hf_specs(item):
            try:
                rows = load_hf_records(spec, args.streaming, args.max_rows_per_dataset)
            except Exception as exc:
                print(f"Warning: failed to load HF dataset '{spec}': {exc}")
                continue
            for row in rows:
                source = row.pop("_source_alias")
                record = normalize_row(row, source, args.video_root, args.video_extension)
                if record and not should_exclude(record, args.exclude_source_keywords):
                    normalized.append(record)

    for local_path in args.local_json:
        path = Path(local_path)
        source = path.stem.lower()
        for row in tqdm(iter_local_records(path), desc=f"load:{source}"):
            record = normalize_row(row, source, args.video_root, args.video_extension)
            if record and not should_exclude(record, args.exclude_source_keywords):
                normalized.append(record)

    captions = [r for r in normalized if r["task"] == "caption"]
    qas = [r for r in normalized if r["task"] == "qa"]
    mcqs = [r for r in normalized if r["task"] == "mcq"]

    selected = []
    selected.extend(sample_records(captions, args.caption_count, args.seed))
    selected.extend(sample_records(qas, args.qa_count, args.seed + 1))
    selected.extend(sample_records(mcqs, args.mcq_count, args.seed + 2))

    write_jsonl(args.out, selected)
    print(
        f"Wrote {len(selected)} records to {args.out} "
        f"(caption={len([r for r in selected if r['task']=='caption'])}, "
        f"qa={len([r for r in selected if r['task']=='qa'])}, "
        f"mcq={len([r for r in selected if r['task']=='mcq'])})."
    )
    print("Note: media_path is only valid if --video-root matches where you store the videos.")


if __name__ == "__main__":
    main()
