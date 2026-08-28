#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that a JSONL file exists and has enough rows.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--min-rows", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    if not path.exists() or path.stat().st_size == 0:
        print(f"Missing or empty JSONL: {path}")
        raise SystemExit(1)
    with path.open("r", encoding="utf-8") as f:
        rows = sum(1 for _ in f)
    if rows < args.min_rows:
        print(f"JSONL has {rows} rows, expected at least {args.min_rows}: {path}")
        raise SystemExit(1)
    print(f"OK {path}: {rows} rows")


if __name__ == "__main__":
    main()
