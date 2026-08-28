#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check runtime wiring for cached Hugging Face model loading.")
    parser.add_argument("--model-id", default="OpenGVLab/InternVideo2_CLIP_S")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--load-model", action="store_true", help="Also instantiate AutoModel. This can use GPU/CPU memory.")
    return parser.parse_args()


def print_status(name: str, ok: bool, detail: str = "") -> None:
    prefix = "OK" if ok else "FAIL"
    if detail:
        print(f"{prefix} {name}: {detail}")
    else:
        print(f"{prefix} {name}")


def short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ")
    if len(message) > 260:
        message = message[:257] + "..."
    return f"{type(exc).__name__}: {message}"


def list_snapshot_files(snapshot_path: str) -> None:
    path = Path(snapshot_path)
    files = sorted(p.name for p in path.iterdir()) if path.exists() else []
    shown = ", ".join(files[:40])
    if len(files) > 40:
        shown += f", ... ({len(files)} total)"
    print(f"snapshot_files={shown}")
    for required in ("config.json", "preprocessor_config.json", "processor_config.json", "model.safetensors"):
        print_status(f"snapshot has {required}", (path / required).exists())


def check_transformers_loaders(source: str, label: str) -> None:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor
    from vlmevalbench.encoders import _disable_flash_attn_in_config

    config = None
    try:
        config = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
        _disable_flash_attn_in_config(config)
        print_status(f"AutoConfig local-only ({label})", True)
    except Exception as exc:
        print_status(f"AutoConfig local-only ({label})", False, short_error(exc))

    try:
        AutoProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)
        print_status(f"AutoProcessor local-only ({label})", True)
    except Exception as exc:
        print_status(f"AutoProcessor local-only ({label})", False, short_error(exc))
        try:
            AutoImageProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)
            print_status(f"AutoImageProcessor local-only ({label})", True)
        except Exception as image_exc:
            print_status(f"AutoImageProcessor local-only ({label})", False, short_error(image_exc))

    return config, AutoModel


def main() -> None:
    args = parse_args()

    print(f"HF_HOME={os.environ.get('HF_HOME', '')}")
    print(f"VLMEB_LOCAL_FILES_ONLY={os.environ.get('VLMEB_LOCAL_FILES_ONLY', '')}")

    import torch
    import vlmevalbench.encoders as encoders
    from huggingface_hub import snapshot_download

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
        list_snapshot_files(snapshot_path)
    except Exception as exc:
        print_status("snapshot cache", False, short_error(exc))
        return

    check_transformers_loaders(args.model_id, "repo-id")
    config, auto_model = check_transformers_loaders(snapshot_path, "snapshot-path")

    if args.load_model and config is not None:
        from vlmevalbench.encoders import _temporary_flash_attn_stub

        try:
            with _temporary_flash_attn_stub():
                model = auto_model.from_pretrained(
                    snapshot_path,
                    trust_remote_code=True,
                    local_files_only=True,
                    low_cpu_mem_usage=False,
                    config=config,
                )
            print_status("AutoModel local-only (snapshot-path)", True, type(model).__name__)
        except Exception as exc:
            print_status("AutoModel local-only (snapshot-path)", False, short_error(exc))


if __name__ == "__main__":
    main()
