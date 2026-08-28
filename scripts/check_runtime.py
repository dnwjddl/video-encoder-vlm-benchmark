#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check runtime wiring for cached Hugging Face model loading.")
    parser.add_argument("--model-id", default="OpenGVLab/InternVideo2-CLIP-1B-224p-f8")
    parser.add_argument("--repo-type", default="model")
    return parser.parse_args()


def print_status(name: str, ok: bool, detail: str = "") -> None:
    prefix = "OK" if ok else "FAIL"
    if detail:
        print(f"{prefix} {name}: {detail}")
    else:
        print(f"{prefix} {name}")


def main() -> None:
    args = parse_args()

    print(f"HF_HOME={os.environ.get('HF_HOME', '')}")
    print(f"VLMEB_LOCAL_FILES_ONLY={os.environ.get('VLMEB_LOCAL_FILES_ONLY', '')}")

    import torch
    import vlmevalbench.encoders as encoders
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor

    print(f"torch={torch.__version__}")
    print(f"torch_cuda={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
    print(f"vlmevalbench.encoders={encoders.__file__}")

    source = inspect.getsource(encoders.FrozenEncoder._load_processor)
    print_status("local-only code", "local_files_only" in source)

    if not os.environ.get("HF_HOME"):
        print_status("HF_HOME", False, "not set")
    elif Path(os.environ["HF_HOME"]).exists():
        print_status("HF_HOME exists", True, os.environ["HF_HOME"])
    else:
        print_status("HF_HOME exists", False, os.environ["HF_HOME"])

    try:
        snapshot_path = snapshot_download(
            repo_id=args.model_id,
            repo_type=args.repo_type,
            local_files_only=True,
        )
        print_status("snapshot cache", True, snapshot_path)
    except Exception as exc:
        print_status("snapshot cache", False, f"{type(exc).__name__}: {exc}")
        return

    try:
        AutoConfig.from_pretrained(args.model_id, trust_remote_code=True, local_files_only=True)
        print_status("AutoConfig local-only", True)
    except Exception as exc:
        print_status("AutoConfig local-only", False, f"{type(exc).__name__}: {exc}")

    try:
        AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, local_files_only=True)
        print_status("AutoProcessor local-only", True)
    except Exception as exc:
        print_status("AutoProcessor local-only", False, f"{type(exc).__name__}: {exc}")
        try:
            AutoImageProcessor.from_pretrained(args.model_id, trust_remote_code=True, local_files_only=True)
            print_status("AutoImageProcessor local-only", True)
        except Exception as image_exc:
            print_status("AutoImageProcessor local-only", False, f"{type(image_exc).__name__}: {image_exc}")


if __name__ == "__main__":
    main()
