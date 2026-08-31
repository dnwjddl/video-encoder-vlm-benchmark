#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from vlmevalbench.data import write_jsonl


CHOICE_RE = re.compile(r"^([A-Z])\.\s*(.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a unified manifest from the official Vinoground files.")
    parser.add_argument("--vinoground-root", required=True)
    parser.add_argument("--out", default="data/benchmarks/vinoground_all.jsonl")
    parser.add_argument("--skip-missing-media", action="store_true")
    return parser.parse_args()


def parse_official_question(text: str) -> tuple[str, list[str]]:
    question_lines: list[str] = []
    choices: list[str] = []
    for raw_line in str(text).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = CHOICE_RE.match(line)
        if match:
            choices.append(match.group(2).strip())
            continue
        if line.lower().startswith("answer with the option"):
            continue
        question_lines.append(line)
    if len(choices) != 2:
        raise ValueError(f"Expected two Vinoground choices, found {len(choices)} in: {text!r}")
    return "\n".join(question_lines), choices


def load_categories(csv_path: Path) -> dict[int, dict[str, Any]]:
    categories: dict[int, dict[str, Any]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            group_id = int(row["index"])
            major = str(row.get("major") or "unknown").strip() or "unknown"
            minor = [value.strip() for value in str(row.get("minor") or "").split(";") if value.strip()]
            categories[group_id] = {
                "major_category": major,
                "minor_categories": minor,
                "categories": [major, *minor],
            }
    return categories


def load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list in {path}")
    return data


def build_record(
    *,
    item: dict[str, Any],
    score_type: str,
    root: Path,
    categories: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    official_idx = str(item["idx"])
    group_text, pair_role = official_idx.rsplit("_", 1)
    group_id = int(group_text)
    question, choices = parse_official_question(str(item["question"]))
    category = categories[group_id]
    media_path = (root / str(item["video_name"])).resolve()
    return {
        "id": f"vinoground:{score_type}:{official_idx}",
        "source": "Vinoground",
        "benchmark": f"Vinoground-{score_type}",
        "task_type": category["major_category"],
        "group_id": group_id,
        "official_idx": official_idx,
        "score_type": score_type,
        "pair_role": pair_role,
        **category,
        "media_type": "video",
        "media_path": str(media_path),
        "question": question,
        "choices": choices,
        "answer": str(item["GT"]).strip().upper(),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.vinoground_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Vinoground root does not exist: {root}")

    categories = load_categories(root / "vinoground.csv")
    rows: list[dict[str, Any]] = []
    missing_media: list[str] = []
    for score_type, filename in (
        ("text", "vinoground_textscore.json"),
        ("video", "vinoground_videoscore.json"),
    ):
        for item in load_json(root / filename):
            row = build_record(
                item=item,
                score_type=score_type,
                root=root,
                categories=categories,
            )
            if not Path(row["media_path"]).is_file():
                missing_media.append(row["media_path"])
                if args.skip_missing_media:
                    continue
            rows.append(row)

    if missing_media and not args.skip_missing_media:
        preview = "\n".join(missing_media[:10])
        raise FileNotFoundError(
            f"Vinoground is missing {len(missing_media)} referenced videos. First missing paths:\n{preview}"
        )

    group_counts = Counter((row["group_id"], row["score_type"]) for row in rows)
    malformed = [key for key, count in group_counts.items() if count != 2]
    if malformed:
        raise RuntimeError(f"Expected two records per group and score type; malformed groups: {malformed[:10]}")

    rows.sort(key=lambda row: (int(row["group_id"]), str(row["score_type"]), str(row["pair_role"])))
    write_jsonl(args.out, rows)
    summary = {
        "vinoground_root": str(root),
        "num_groups": len({int(row["group_id"]) for row in rows}),
        "num_records": len(rows),
        "text_records": sum(row["score_type"] == "text" for row in rows),
        "video_records": sum(row["score_type"] == "video" for row in rows),
        "missing_media": len(missing_media),
    }
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} Vinoground records to {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
