from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EncoderConfig:
    name: str
    family: str
    model_id: str
    trust_remote_code: bool = False
    processor: str = "auto"
    feature_key: str = "last_hidden_state"
    drop_cls: bool = False
    input_layout: str | None = None
    num_frames: int = 8
    max_tokens: int = 256
    note: str | None = None


def load_encoder_registry(path: str | Path) -> dict[str, EncoderConfig]:
    path = Path(path)
    data = yaml.safe_load(path.read_text()) or {}
    registry: dict[str, EncoderConfig] = {}
    for name, raw in data.items():
        registry[name] = EncoderConfig(name=name, **raw)
    return registry


def resolve_encoder(
    name: str,
    config_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> EncoderConfig:
    registry = load_encoder_registry(config_path)
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown encoder '{name}'. Available encoders: {available}")
    cfg = registry[name]
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
    return cfg
