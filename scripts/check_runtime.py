#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib
import inspect
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check runtime wiring for cached Hugging Face model loading.")
    parser.add_argument("--model-id", default="OpenGVLab/InternVideo2_CLIP_S")
    parser.add_argument("--repo-type", default="model")
    parser.add_argument("--load-model", action="store_true", help="Also instantiate AutoModel. This can use GPU/CPU memory.")
    parser.add_argument(
        "--allow-missing-processor",
        action="store_true",
        help="Treat missing AutoProcessor/AutoImageProcessor as expected for model-transform encoders.",
    )
    parser.add_argument(
        "--required-module",
        action="append",
        default=[],
        help="Python module that must import successfully. Can be passed multiple times.",
    )
    return parser.parse_args()


def print_status(name: str, ok: bool, detail: str = "") -> None:
    prefix = "OK" if ok else "FAIL"
    if detail:
        print(f"{prefix} {name}: {detail}")
    else:
        print(f"{prefix} {name}")


def print_skip(name: str, detail: str = "") -> None:
    if detail:
        print(f"SKIP {name}: {detail}")
    else:
        print(f"SKIP {name}")


def short_error(exc: Exception) -> str:
    message = str(exc).replace("\n", " ")
    if len(message) > 260:
        message = message[:257] + "..."
    return f"{type(exc).__name__}: {message}"


def list_snapshot_files(snapshot_path: str, *, allow_missing_processor: bool = False) -> bool:
    path = Path(snapshot_path)
    files = sorted(p.name for p in path.iterdir()) if path.exists() else []
    shown = ", ".join(files[:40])
    if len(files) > 40:
        shown += f", ... ({len(files)} total)"
    print(f"snapshot_files={shown}")
    ok = True
    for required in ("config.json", "model.safetensors"):
        exists = (path / required).exists()
        print_status(f"snapshot has {required}", exists)
        ok = ok and exists
    for optional_processor in ("preprocessor_config.json", "processor_config.json"):
        exists = (path / optional_processor).exists()
        if exists:
            print_status(f"snapshot has {optional_processor}", True)
        elif allow_missing_processor:
            print_skip(f"snapshot has {optional_processor}", "model uses bundled transform")
        else:
            print_status(f"snapshot has {optional_processor}", False)
            ok = False
    return ok


def check_transformers_loaders(source: str, label: str, *, allow_missing_processor: bool = False) -> tuple[object, object, bool]:
    from transformers import AutoConfig, AutoImageProcessor, AutoModel, AutoProcessor
    from vlmevalbench.encoders import _disable_flash_attn_in_config, _patch_transformers_tied_weights_compat

    _patch_transformers_tied_weights_compat()

    config = None
    ok = True
    try:
        config = AutoConfig.from_pretrained(source, trust_remote_code=True, local_files_only=True)
        _disable_flash_attn_in_config(config)
        print_status(f"AutoConfig local-only ({label})", True)
    except Exception as exc:
        print_status(f"AutoConfig local-only ({label})", False, short_error(exc))
        ok = False

    processor_ok = True
    try:
        AutoProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)
        print_status(f"AutoProcessor local-only ({label})", True)
    except Exception as exc:
        processor_ok = False
        if allow_missing_processor:
            print_skip(f"AutoProcessor local-only ({label})", short_error(exc))
        else:
            print_status(f"AutoProcessor local-only ({label})", False, short_error(exc))
        try:
            AutoImageProcessor.from_pretrained(source, trust_remote_code=True, local_files_only=True)
            print_status(f"AutoImageProcessor local-only ({label})", True)
            processor_ok = True
        except Exception as image_exc:
            if allow_missing_processor:
                print_skip(f"AutoImageProcessor local-only ({label})", short_error(image_exc))
            else:
                print_status(f"AutoImageProcessor local-only ({label})", False, short_error(image_exc))

    if not allow_missing_processor:
        ok = ok and processor_ok

    return config, AutoModel, ok


def check_python_module(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        print_status(f"python module {module_name}", True)
        return True
    except Exception as exc:
        print_status(f"python module {module_name}", False, short_error(exc))
        return False


def check_flash_attn_stub() -> bool:
    from vlmevalbench.encoders import _temporary_flash_attn_stub

    try:
        with _temporary_flash_attn_stub():
            from flash_attn.bert_padding import pad_input, unpad_input
            from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
            from flash_attn.modules.mlp import FusedMLP, Mlp
            from flash_attn.ops.rms_norm import DropoutAddRMSNorm

            assert callable(flash_attn_varlen_qkvpacked_func)
            assert callable(pad_input)
            assert callable(unpad_input)
            assert Mlp is not None
            assert FusedMLP is not None
            assert DropoutAddRMSNorm is not None
        print_status("flash-attn stub imports", True)
        return True
    except Exception as exc:
        print_status("flash-attn stub imports", False, short_error(exc))
        return False


def main() -> None:
    args = parse_args()
    failed = False

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
    local_only_code_ok = "local_files_only" in source
    print_status("local-only code", local_only_code_ok)
    failed = failed or not local_only_code_ok
    failed = failed or not check_flash_attn_stub()

    for module_name in args.required_module:
        failed = failed or not check_python_module(module_name)

    if not os.environ.get("HF_HOME"):
        print_status("HF_HOME", False, "not set")
        failed = True
    elif Path(os.environ["HF_HOME"]).exists():
        print_status("HF_HOME exists", True, os.environ["HF_HOME"])
    else:
        print_status("HF_HOME exists", False, os.environ["HF_HOME"])
        failed = True

    try:
        snapshot_path = snapshot_download(
            repo_id=args.model_id,
            repo_type=args.repo_type,
            local_files_only=True,
        )
        print_status("snapshot cache", True, snapshot_path)
        failed = failed or not list_snapshot_files(
            snapshot_path,
            allow_missing_processor=args.allow_missing_processor,
        )
    except Exception as exc:
        print_status("snapshot cache", False, short_error(exc))
        raise SystemExit(1) from exc

    _, _, repo_loaders_ok = check_transformers_loaders(
        args.model_id,
        "repo-id",
        allow_missing_processor=args.allow_missing_processor,
    )
    config, auto_model, snapshot_loaders_ok = check_transformers_loaders(
        snapshot_path,
        "snapshot-path",
        allow_missing_processor=args.allow_missing_processor,
    )
    failed = failed or not repo_loaders_ok or not snapshot_loaders_ok

    if args.load_model and config is not None:
        from vlmevalbench.encoders import _patch_transformers_tied_weights_compat, _temporary_flash_attn_stub

        try:
            _patch_transformers_tied_weights_compat()
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
            failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
