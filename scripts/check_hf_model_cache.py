#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check that a Hugging Face model is present in the local cache.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ")
    if len(message) > 240:
        message = message[:237] + "..."
    return f"{type(exc).__name__}: {message}"


def fail(message: str) -> None:
    raise SystemExit(f"FAIL {message}")


def check_weight_files(snapshot: Path) -> None:
    index_files = sorted(snapshot.glob("*.safetensors.index.json")) + sorted(snapshot.glob("pytorch_model*.bin.index.json"))
    if index_files:
        for index_file in index_files:
            data = json.loads(index_file.read_text())
            missing = sorted({name for name in data.get("weight_map", {}).values() if not (snapshot / name).exists()})
            if missing:
                fail(f"missing weight shards listed in {index_file.name}: {missing[:5]}")
        return

    weights = (
        sorted(snapshot.glob("*.safetensors"))
        + sorted(snapshot.glob("pytorch_model*.bin"))
        + sorted(snapshot.glob("*.pth"))
        + sorted(snapshot.glob("*.pt"))
    )
    if not weights:
        fail(f"no model weight files found in {snapshot}")


def main() -> None:
    args = parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=args.model_id,
                repo_type=args.repo_type,
                local_files_only=True,
            )
        )
    except Exception as exc:
        fail(f"snapshot not found for {args.model_id}: {short_error(exc)}")

    try:
        AutoConfig.from_pretrained(
            snapshot_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=True,
        )
    except Exception as exc:
        fail(f"config not loadable for {args.model_id}: {short_error(exc)}")

    try:
        AutoTokenizer.from_pretrained(
            snapshot_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=True,
            local_files_only=True,
        )
    except Exception as exc:
        fail(f"tokenizer not loadable for {args.model_id}: {short_error(exc)}")

    check_weight_files(snapshot_path)
    print(f"OK {args.model_id} cached at {snapshot_path}")


if __name__ == "__main__":
    main()
