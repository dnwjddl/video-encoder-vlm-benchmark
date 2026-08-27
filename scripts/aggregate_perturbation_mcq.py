#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate zero-shot perturbation MCQ summary CSV files.")
    parser.add_argument("--root", default="outputs/zeroshot_perturbation_mcq")
    parser.add_argument("--out", default="outputs/zeroshot_perturbation_mcq_table.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in sorted(Path(args.root).glob("*/summary.csv")):
        encoder = summary_path.parent.name
        df = pd.read_csv(summary_path)
        df.insert(0, "encoder", encoder)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No summary.csv files found under {args.root}")
    out = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
