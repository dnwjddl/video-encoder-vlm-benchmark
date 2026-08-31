#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from vlmevalbench.data import normalize_choice_answer, read_jsonl, write_jsonl


QUESTION_TEMPLATES = (
    "{question}",
    "Select the correct option for the following task.\n{question}",
    "Determine which of the two choices correctly answers this question.\n{question}",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic option-order and prompt variants for Vinoground text-only auditing."
    )
    parser.add_argument("--input", default="data/benchmarks/vinoground_all.jsonl")
    parser.add_argument("--out", default="data/benchmarks/vinoground_text_variants.jsonl")
    return parser.parse_args()


def answer_index(record: dict[str, Any], choices: list[str]) -> int:
    answer = normalize_choice_answer(record.get("answer"), choices)
    if len(answer) != 1 or answer not in "AB":
        raise ValueError(f"Expected a binary A/B answer for {record.get('id')}, got {answer!r}")
    return ord(answer) - ord("A")


def build_variants(base_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for record in base_rows:
        choices = [str(choice) for choice in record.get("choices") or []]
        if len(choices) != 2:
            raise ValueError(f"Vinoground record {record.get('id')} does not have exactly two choices")
        gold_idx = answer_index(record, choices)
        for prompt_idx, template in enumerate(QUESTION_TEMPLATES):
            for order_name, permutation in (("identity", (0, 1)), ("swapped", (1, 0))):
                permuted_choices = [choices[idx] for idx in permutation]
                remapped_gold = permutation.index(gold_idx)
                variant = dict(record)
                variant.update(
                    {
                        "id": f"{record['id']}::q{prompt_idx}::{order_name}",
                        "base_id": str(record["id"]),
                        "prompt_variant": prompt_idx,
                        "option_order": order_name,
                        "option_permutation": list(permutation),
                        "question": template.format(question=str(record.get("question") or "").strip()),
                        "choices": permuted_choices,
                        "answer": chr(ord("A") + remapped_gold),
                        "correct_choice_text": choices[gold_idx],
                    }
                )
                variants.append(variant)
    return variants


def main() -> None:
    args = parse_args()
    base_rows = read_jsonl(args.input)
    variants = build_variants(base_rows)

    write_jsonl(args.out, variants)
    summary = {
        "base_records": len(base_rows),
        "variants": len(variants),
        "variants_per_record": len(QUESTION_TEMPLATES) * 2,
        "robust_threshold_at_80_percent": math.ceil(len(QUESTION_TEMPLATES) * 2 * 0.8),
    }
    summary_path = Path(args.out).with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(variants)} text-only variants to {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
