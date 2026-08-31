#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from vlmevalbench.data import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate plain MVBench original-video overall accuracy.")
    parser.add_argument("--manifest", default="data/benchmarks/mvbench_all.jsonl")
    parser.add_argument("--eval-root", default="outputs/mvbench/projector_eval")
    parser.add_argument("--out", default="outputs/mvbench/analysis/overall_accuracy.csv")
    parser.add_argument("--encoders", default=None, help="Comma-separated encoder list. Defaults to discovered eval dirs.")
    return parser.parse_args()


def discover_encoders(eval_root: Path, encoders_csv: str | None) -> list[str]:
    if encoders_csv:
        return [item.strip() for item in encoders_csv.split(",") if item.strip()]
    if not eval_root.exists():
        return []
    return sorted(path.name for path in eval_root.iterdir() if path.is_dir())


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return {str(row["id"]): row for row in read_jsonl(path)}


def main() -> None:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    manifest_ids = {str(row["id"]) for row in manifest_rows}
    manifest_n = len(manifest_ids)
    eval_root = Path(args.eval_root)
    rows = []

    for encoder in discover_encoders(eval_root, args.encoders):
        pred_path = eval_root / encoder / "original" / "predictions.jsonl"
        predictions = load_predictions(pred_path)
        evaluated_ids = manifest_ids & set(predictions)
        extra_ids = set(predictions) - manifest_ids
        correct = sum(bool(predictions[item_id].get("correct")) for item_id in evaluated_ids)
        evaluated_n = len(evaluated_ids)
        missing_n = manifest_n - evaluated_n
        rows.append(
            {
                "encoder": encoder,
                "manifest_examples": manifest_n,
                "evaluated_examples": evaluated_n,
                "missing_predictions": missing_n,
                "extra_predictions_ignored": len(extra_ids),
                "correct": correct,
                "overall_accuracy": correct / max(manifest_n, 1),
                "accuracy_on_evaluated": correct / max(evaluated_n, 1),
                "complete": evaluated_n == manifest_n,
                "prediction_file": str(pred_path),
            }
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "encoder",
            "manifest_examples",
            "evaluated_examples",
            "missing_predictions",
            "extra_predictions_ignored",
            "correct",
            "overall_accuracy",
            "accuracy_on_evaluated",
            "complete",
            "prediction_file",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote MVBench overall accuracy to {out_path}")
    for row in rows:
        print(
            f"{row['encoder']}: overall={row['overall_accuracy']:.4f} "
            f"correct={row['correct']}/{row['manifest_examples']} "
            f"evaluated={row['evaluated_examples']} complete={row['complete']}"
        )


if __name__ == "__main__":
    main()
