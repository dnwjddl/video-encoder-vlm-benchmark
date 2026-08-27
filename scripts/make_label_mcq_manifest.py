#!/usr/bin/env python
from __future__ import annotations

import argparse
import random
from pathlib import Path

from vlmevalbench.data import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a multiple-choice manifest from a labeled video manifest. "
            "Useful for zero-shot perturbation MCQ smoke tests."
        )
    )
    parser.add_argument("--input", required=True, help="Input manifest with a label field.")
    parser.add_argument("--out", required=True, help="Output MCQ JSONL manifest.")
    parser.add_argument("--num-choices", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmark-name", default="activitynet_label_mcq")
    parser.add_argument(
        "--question",
        default="Which activity is shown in the video?",
        help="Question string used for every generated MCQ row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_choices < 2:
        raise ValueError("--num-choices must be >= 2")

    records = read_jsonl(args.input)
    labeled = [r for r in records if r.get("label") not in (None, "")]
    labels = sorted({str(r["label"]) for r in labeled})
    if len(labels) < args.num_choices:
        raise ValueError(
            f"Need at least {args.num_choices} unique labels, found {len(labels)} in {args.input}"
        )

    rng = random.Random(args.seed)
    out_rows = []
    for record in labeled:
        gold = str(record["label"])
        distractors = [label for label in labels if label != gold]
        sampled = rng.sample(distractors, args.num_choices - 1)
        choices = sampled + [gold]
        rng.shuffle(choices)
        answer_idx = choices.index(gold)
        answer = chr(ord("A") + answer_idx)
        out_rows.append(
            {
                "id": f"{record['id']}_label_mcq",
                "source": record.get("source", "unknown"),
                "benchmark": args.benchmark_name,
                "task": "mcq",
                "media_type": record.get("media_type", "video"),
                "media_path": record["media_path"],
                "question": args.question,
                "answer": answer,
                "choices": choices,
                "label": gold,
                "original_id": record["id"],
            }
        )

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, out_rows)
    print(f"Wrote {len(out_rows)} MCQ rows to {args.out}")
    print(f"Unique labels: {len(labels)}")


if __name__ == "__main__":
    main()
