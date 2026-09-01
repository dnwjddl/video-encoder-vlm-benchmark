from __future__ import annotations

import argparse
from collections.abc import Callable
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import yaml

from .io import sha256_file
from .protocol import (
    load_protocol,
    trial_build_protocol_sha256,
    validate_data_protocol,
    validate_frozen_model_protocol,
    validate_locked_projector_protocol,
)


PROJECTOR_LOCK_SCHEMA_VERSION = "information_upper_bound.projector_lock.v3"


def _load_json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return value


def merge_projector_lock(
    protocol: Mapping[str, Any],
    projector_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace only the explicitly late-bound projector section."""

    existing = protocol.get("projector")
    if not isinstance(existing, Mapping) or not existing:
        raise ValueError("protocol must contain a non-empty projector template section")
    schema_version = str(projector_lock.get("schema_version", ""))
    if schema_version != PROJECTOR_LOCK_SCHEMA_VERSION:
        raise ValueError(
            "projector lock schema mismatch: "
            f"expected={PROJECTOR_LOCK_SCHEMA_VERSION!r}, actual={schema_version!r}"
        )
    replacement = {
        str(key): deepcopy(value)
        for key, value in projector_lock.items()
        if str(key) != "schema_version"
    }
    existing_keys = {str(key) for key in existing}
    replacement_keys = set(replacement)
    if replacement_keys != existing_keys:
        raise ValueError(
            "projector lock fields differ from the protocol template: "
            f"missing={sorted(existing_keys - replacement_keys)}, "
            f"unexpected={sorted(replacement_keys - existing_keys)}"
        )

    before_hash = trial_build_protocol_sha256(protocol)
    finalized = deepcopy(dict(protocol))
    finalized["projector"] = replacement
    if trial_build_protocol_sha256(finalized) != before_hash:
        raise AssertionError("late-bound projector merge changed trial-build protocol")
    return finalized


def _write_yaml_atomic(
    path: Path,
    value: Mapping[str, Any],
    *,
    validate_candidate: Callable[[Path], None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                dict(value),
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        validate_candidate(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authenticate a trained projector lock and create a final scoring protocol "
            "without modifying its preregistered sections."
        )
    )
    parser.add_argument("--protocol", required=True, help="pre-projector protocol YAML")
    parser.add_argument("--projector-lock", required=True)
    parser.add_argument("--projector-ckpt", required=True)
    parser.add_argument("--projector-metadata", required=True)
    parser.add_argument("--out", required=True, help="new finalized protocol YAML")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    protocol_path = Path(args.protocol).expanduser().resolve()
    lock_path = Path(args.projector_lock).expanduser().resolve()
    checkpoint_path = Path(args.projector_ckpt).expanduser().resolve()
    metadata_path = Path(args.projector_metadata).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()
    inputs = {protocol_path, lock_path, checkpoint_path, metadata_path}
    if output_path in inputs:
        raise ValueError("finalized protocol output must not alias an input")
    if (output_path.exists() or output_path.is_symlink()) and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output_path}")
    for label, path in (
        ("protocol", protocol_path),
        ("projector lock", lock_path),
        ("projector checkpoint", checkpoint_path),
        ("projector metadata", metadata_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    protocol, _protocol_metadata = load_protocol(protocol_path)
    before_hash = trial_build_protocol_sha256(protocol)
    projector_lock = _load_json_mapping(lock_path, label="projector lock")
    projector_metadata = _load_json_mapping(metadata_path, label="projector metadata")
    finalized = merge_projector_lock(protocol, projector_lock)

    validate_data_protocol(finalized)
    validate_frozen_model_protocol(finalized)
    locked = validate_locked_projector_protocol(
        finalized,
        checkpoint_sha256=sha256_file(checkpoint_path),
        metadata_sha256=sha256_file(metadata_path),
        projector_metadata=projector_metadata,
    )

    def validate_candidate(candidate_path: Path) -> None:
        candidate, _candidate_metadata = load_protocol(candidate_path)
        if trial_build_protocol_sha256(candidate) != before_hash:
            raise AssertionError("written final protocol changed preregistered content")
        if candidate.get("projector") != finalized.get("projector"):
            raise AssertionError("written final protocol changed the projector lock")
        validate_data_protocol(candidate)
        validate_frozen_model_protocol(candidate)
        validate_locked_projector_protocol(
            candidate,
            checkpoint_sha256=sha256_file(checkpoint_path),
            metadata_sha256=sha256_file(metadata_path),
            projector_metadata=projector_metadata,
        )

    _write_yaml_atomic(
        output_path,
        finalized,
        validate_candidate=validate_candidate,
    )
    written, written_metadata = load_protocol(output_path)

    print(
        json.dumps(
            {
                "output": str(output_path),
                "protocol_sha256": written_metadata["sha256"],
                "trial_build_protocol_sha256": before_hash,
                "checkpoint_sha256": locked["checkpoint_sha256"],
                "metadata_sha256": locked["metadata_sha256"],
                "evaluation_trial_count": locked["evaluation_trial_count"],
                "projector_section_finalized": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
