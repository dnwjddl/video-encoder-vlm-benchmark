from __future__ import annotations

from pathlib import Path
from typing import Any

from .clevrer import CLEVRERAdapter
from .common import AdapterError, BaseAdapter
from .egoschema import EgoSchemaAdapter
from .mvp import MVPAdapter
from .next_gqa import NExTGQAAdapter
from .perception_test import PerceptionTestAdapter
from .tempcompass import TempCompassAdapter
from .tvbench import TVBenchAdapter


ADAPTERS: dict[str, type[BaseAdapter]] = {
    "tempcompass": TempCompassAdapter,
    "tvbench": TVBenchAdapter,
    "perception_test": PerceptionTestAdapter,
    "next_gqa": NExTGQAAdapter,
    "clevrer": CLEVRERAdapter,
    "egoschema": EgoSchemaAdapter,
    "mvp": MVPAdapter,
}

ALIASES = {
    "perception-test": "perception_test",
    "next-gqa": "next_gqa",
    "nextgqa": "next_gqa",
    "ego_schema": "egoschema",
}


def available_adapters() -> tuple[str, ...]:
    return tuple(sorted(ADAPTERS))


def build_adapter(
    name: str,
    annotation_path: str | Path,
    media_root: str | Path,
    *,
    split: str = "eval",
    require_media: bool = True,
    **options: Any,
) -> BaseAdapter:
    canonical = ALIASES.get(str(name).strip().casefold(), str(name).strip().casefold())
    adapter_type = ADAPTERS.get(canonical)
    if adapter_type is None:
        raise AdapterError(
            f"unknown adapter {name!r}; available adapters: {', '.join(available_adapters())}"
        )
    return adapter_type(
        annotation_path,
        media_root,
        split=split,
        require_media=require_media,
        **options,
    )


def load_records(
    name: str,
    annotation_path: str | Path,
    media_root: str | Path,
    *,
    split: str = "eval",
    require_media: bool = True,
    **options: Any,
) -> list[dict[str, Any]]:
    return build_adapter(
        name,
        annotation_path,
        media_root,
        split=split,
        require_media=require_media,
        **options,
    ).load()
