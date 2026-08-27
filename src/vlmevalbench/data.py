from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

VISUAL_PLACEHOLDER = "<VISUAL>"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def stable_id(*parts: str) -> str:
    raw = "::".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def safe_id(record_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in record_id)


def sample_records(
    records: list[dict[str, Any]],
    n: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if n is None or n <= 0 or len(records) <= n:
        return records
    rng = random.Random(seed)
    return rng.sample(records, n)


def normalize_choice_answer(answer: Any, choices: list[str] | None) -> str:
    if answer is None:
        return ""
    text = str(answer).strip()
    if not choices:
        return text
    if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return text.upper()
    for idx, choice in enumerate(choices):
        if text == choice or text.lower() == choice.lower():
            return chr(ord("A") + idx)
    return text


def format_choices(choices: list[str] | None) -> str:
    if not choices:
        return ""
    lines = []
    for idx, choice in enumerate(choices):
        label = chr(ord("A") + idx)
        lines.append(f"{label}. {choice}")
    return "\n".join(lines)


def make_prompt(record: dict[str, Any], *, for_mcq: bool = False) -> tuple[str, str]:
    question = str(record.get("question") or "Describe the visual content.").strip()
    choices = record.get("choices")
    answer = normalize_choice_answer(record.get("answer"), choices)
    media_word = "video" if record.get("media_type", "video") == "video" else "image"

    if choices:
        choice_text = format_choices(choices)
        instruction = (
            f"You are given a {media_word}. Answer the question using only the option letter.\n"
            f"{VISUAL_PLACEHOLDER}\n"
            f"Question: {question}\n"
            f"Options:\n{choice_text}\n"
            "Answer:"
        )
    elif for_mcq:
        instruction = (
            f"You are given a {media_word}.\n"
            f"{VISUAL_PLACEHOLDER}\n"
            f"Question: {question}\n"
            "Answer:"
        )
    else:
        instruction = (
            f"You are given a {media_word}. Answer concisely and accurately.\n"
            f"{VISUAL_PLACEHOLDER}\n"
            f"Question: {question}\n"
            "Answer:"
        )
    return instruction, str(answer)


def split_prompt_on_visual(prompt: str) -> tuple[str, str]:
    if VISUAL_PLACEHOLDER not in prompt:
        return prompt + "\n", ""
    prefix, suffix = prompt.split(VISUAL_PLACEHOLDER, 1)
    return prefix, suffix


def records_by_id(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        out[str(record["id"])] = record
    return out
