"""Folder-local encoder configuration and immutable snapshot resolution.

The parent benchmark exposes the actual ``FrozenEncoder`` implementation.  This
module keeps the diagnostic suite's stronger revision contract inside this
folder: an immutable Hub revision is resolved to a content-addressed local
snapshot before the unchanged parent encoder is constructed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EncoderConfig:
    name: str
    family: str
    model_id: str
    revision: str | None = None
    trust_remote_code: bool = False
    processor: str = "auto"
    feature_key: str = "last_hidden_state"
    drop_cls: bool = False
    input_layout: str | None = None
    disable_flash_attn: bool = False
    low_cpu_mem_usage: bool = True
    num_frames: int = 8
    max_tokens: int = 256
    note: str | None = None


def load_encoder_registry(path: str | Path) -> dict[str, EncoderConfig]:
    registry_path = Path(path)
    raw_registry = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw_registry, dict):
        raise ValueError(f"encoder registry must contain a mapping: {registry_path}")
    registry: dict[str, EncoderConfig] = {}
    for name, raw in raw_registry.items():
        if not isinstance(raw, dict):
            raise ValueError(f"encoder {name!r} configuration must be a mapping")
        registry[str(name)] = EncoderConfig(name=str(name), **raw)
    return registry


def resolve_encoder(
    name: str,
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> EncoderConfig:
    registry = load_encoder_registry(config_path)
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown encoder {name!r}. Available encoders: {available}")
    config = registry[name]
    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(config, key):
                raise KeyError(f"Unknown encoder override: {key}")
            setattr(config, key, value)
    return config


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _download_snapshot(
    *,
    repo_id: str,
    revision: str,
    local_files_only: bool,
) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_files_only=local_files_only,
    )


def prepare_encoder_runtime(
    config: EncoderConfig,
) -> tuple[EncoderConfig, dict[str, Any]]:
    """Return an unchanged scientific config and a loadable runtime config.

    Hub revisions are resolved before calling the parent ``FrozenEncoder`` so
    no modification to the repository's shared encoder implementation is
    required.  The declared identity retains the original repo ID and revision;
    only the runtime copy points at the resolved snapshot directory.
    """

    declared = asdict(config)
    runtime = replace(config)
    candidate = Path(config.model_id).expanduser()
    if candidate.exists():
        runtime.model_id = str(candidate.resolve())
        return runtime, declared
    if config.revision in (None, ""):
        return runtime, declared

    runtime.model_id = _download_snapshot(
        repo_id=config.model_id,
        revision=str(config.revision),
        local_files_only=_env_flag("VLMEB_LOCAL_FILES_ONLY"),
    )
    return runtime, declared


__all__ = [
    "EncoderConfig",
    "load_encoder_registry",
    "prepare_encoder_runtime",
    "resolve_encoder",
]
