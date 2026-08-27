#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-encoder benchmark summary CSV files.")
    parser.add_argument("--eval-root", default="outputs/eval")
    parser.add_argument("--out", default="outputs/benchmark_table.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for summary_path in sorted(Path(args.eval_root).glob("*/summary.csv")):
        encoder = summary_path.parent.name
        df = pd.read_csv(summary_path)
        df.insert(0, "encoder", encoder)
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No summary.csv files found under {args.eval_root}")
    out = pd.concat(rows, ignore_index=True)
    pivot = out.pivot_table(index="encoder", columns="benchmark", values="accuracy", aggfunc="mean")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
