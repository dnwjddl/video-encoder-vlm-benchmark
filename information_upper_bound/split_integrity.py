"""Leakage checks for projector-training and predeclared evaluation manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .io import iter_jsonl, sha256_file


def _diagnostic_value(record: Mapping[str, Any], name: str) -> Any:
    if record.get(name) not in (None, ""):
        return record[name]
    diagnostic = record.get("diagnostic")
    if isinstance(diagnostic, Mapping):
        return diagnostic.get(name)
    return None


def _media_path(record: Mapping[str, Any]) -> Path:
    value = record.get("media_path")
    if value in (None, ""):
        visual_spec = record.get("visual_spec")
        if isinstance(visual_spec, Mapping):
            value = visual_spec.get("media_path")
    if value in (None, ""):
        raise ValueError(
            f"record {record.get('id')!r} has no media_path for split leakage auditing"
        )
    path = Path(str(value)).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"record {record.get('id')!r} media does not exist for split audit: {path}"
        )
    return path


def _manifest_identities(
    path: str | Path,
    *,
    media_digest_cache: dict[Path, str],
) -> tuple[set[str], set[str], int]:
    units: set[str] = set()
    media_digests: set[str] = set()
    rows = 0
    for record in iter_jsonl(path):
        rows += 1
        record_id = str(record.get("id", ""))
        unit = str(_diagnostic_value(record, "resampling_unit_id") or "").strip()
        if not unit:
            raise ValueError(
                f"record {record_id!r} has no resampling_unit_id for split leakage auditing"
            )
        units.add(unit)
        media_path = _media_path(record)
        digest = media_digest_cache.get(media_path)
        if digest is None:
            digest = sha256_file(media_path)
            media_digest_cache[media_path] = digest
        media_digests.add(digest)
    if not rows:
        raise ValueError(f"split leakage manifest is empty: {path}")
    return units, media_digests, rows


def audit_projector_split_disjointness(
    training_manifest: str | Path,
    evaluation_manifest: str | Path,
) -> dict[str, Any]:
    """Fail if training and predeclared evaluation share a family or media bytes."""

    cache: dict[Path, str] = {}
    training_units, training_media, training_rows = _manifest_identities(
        training_manifest,
        media_digest_cache=cache,
    )
    evaluation_units, evaluation_media, evaluation_rows = _manifest_identities(
        evaluation_manifest,
        media_digest_cache=cache,
    )
    overlapping_units = sorted(training_units & evaluation_units)
    overlapping_media = sorted(training_media & evaluation_media)
    if overlapping_units or overlapping_media:
        messages: list[str] = []
        if overlapping_units:
            messages.append(
                f"{len(overlapping_units)} overlapping resampling_unit_id values "
                f"(first={overlapping_units[:5]})"
            )
        if overlapping_media:
            messages.append(
                f"{len(overlapping_media)} overlapping source-media SHA256 values "
                f"(first={overlapping_media[:5]})"
            )
        raise ValueError(
            "projector training/evaluation split leakage detected: "
            + "; ".join(messages)
        )
    return {
        "schema_version": "information_upper_bound.projector_split_audit.v1",
        "training_manifest_sha256": sha256_file(training_manifest),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest),
        "training_rows": training_rows,
        "evaluation_rows": evaluation_rows,
        "training_resampling_units": len(training_units),
        "evaluation_resampling_units": len(evaluation_units),
        "training_unique_media_sha256": len(training_media),
        "evaluation_unique_media_sha256": len(evaluation_media),
        "overlapping_resampling_units": 0,
        "overlapping_media_sha256": 0,
    }
