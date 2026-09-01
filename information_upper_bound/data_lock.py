"""Create and authenticate a portable, byte-bound official-data release.

The scientific identity in this file is deliberately different from a local
file checksum. Manifest and report paths are machine-local audit metadata;
the release identity binds semantic records, portable source artifacts, and
the bytes of every media file. Moving an otherwise identical release to a
different mount point therefore does not change ``data_release_sha256``.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence

from .integrity import canonical_sha256
from .io import read_jsonl, sha256_file, write_json
from .unit_sampling import validate_resampling_unit_selection
from .validate import validate_manifest


DATA_LOCK_SCHEMA_VERSION = "information_upper_bound.data_lock.v2"
DATA_RELEASE_SCHEMA_VERSION = "information_upper_bound.data_release.v1"
ADAPTER_REPORT_SCHEMA_VERSION = "information_upper_bound.adapter_report.v1"
ADAPTER_RUN_SCHEMA_VERSION = "information_upper_bound.adapter_run.v1"
RECORD_SET_SCHEMA_VERSION = "information_upper_bound.adapter_record_set.v1"
SEMANTIC_RECORD_SET_SCHEMA_VERSION = (
    "information_upper_bound.semantic_manifest_record_set.v1"
)
SOURCE_RELEASE_SCHEMA_VERSION = "information_upper_bound.source_release.v1"
MEDIA_BINDING_SCHEMA_VERSION = "information_upper_bound.media_binding_set.v1"
UNIT_MEMBERSHIP_SCHEMA_VERSION = "information_upper_bound.unit_membership.v1"
AUDIT_SCHEMA_VERSION = "information_upper_bound.data_lock_audit.v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ADAPTER_RUN_ID = re.compile(r"^adapter-run::[0-9a-f]{64}$")
_LOCAL_PATH_KEY = re.compile(
    r"(?:^raw_location$|(?:^|_)(?:path|paths|file|files|root|dir|directory|directories)$)",
    flags=re.IGNORECASE,
)


def _sha256(value: Any, *, field: str) -> str:
    digest = str(value).strip().lower()
    if _SHA256.fullmatch(digest) is None:
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return digest


def manifest_record_set_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return the adapter report's exact, row-order-independent identity."""

    leaves: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        record_id = str(row.get("id", "")).strip()
        if not record_id or record_id in seen:
            raise ValueError(
                f"manifest record-set identity has an empty/duplicate id: {record_id!r}"
            )
        seen.add(record_id)
        leaves.append({"id": record_id, "sha256": canonical_sha256(row)})
    if not leaves:
        raise ValueError("cannot lock an empty manifest record set")
    leaves.sort(key=lambda value: value["id"])
    return canonical_sha256(
        {"schema_version": RECORD_SET_SCHEMA_VERSION, "records": leaves}
    )


def _portable_provenance(value: Any) -> Any:
    """Remove only machine-local path fields from provenance recursively."""

    if isinstance(value, Mapping):
        return {
            str(key): _portable_provenance(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if _LOCAL_PATH_KEY.search(str(key)) is None
        }
    if isinstance(value, list):
        return [_portable_provenance(item) for item in value]
    return deepcopy(value)


def semantic_manifest_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project a record to scientific content, excluding local mount noise."""

    projected = deepcopy(dict(row))
    projected.pop("media_path", None)
    diagnostic = projected.get("diagnostic")
    if isinstance(diagnostic, Mapping):
        diagnostic = dict(diagnostic)
        if "provenance" in diagnostic:
            diagnostic["provenance"] = _portable_provenance(
                diagnostic.get("provenance")
            )
        projected["diagnostic"] = diagnostic
    return projected


def manifest_semantic_record_set_sha256(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Return a path- and row-order-independent semantic manifest identity."""

    leaves: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        record_id = str(row.get("id", "")).strip()
        if not record_id or record_id in seen:
            raise ValueError(
                f"semantic manifest identity has an empty/duplicate id: {record_id!r}"
            )
        seen.add(record_id)
        leaves.append(
            {
                "id": record_id,
                "sha256": canonical_sha256(semantic_manifest_record(row)),
            }
        )
    if not leaves:
        raise ValueError("cannot lock an empty semantic manifest record set")
    leaves.sort(key=lambda value: value["id"])
    return canonical_sha256(
        {
            "schema_version": SEMANTIC_RECORD_SET_SCHEMA_VERSION,
            "records": leaves,
        }
    )


def _load_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return value


def _dataset_name(row: Mapping[str, Any]) -> str:
    diagnostic = row.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        return ""
    return str(diagnostic.get("dataset", "")).strip()


def _adapter_run_id(row: Mapping[str, Any]) -> str:
    diagnostic = row.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        return ""
    return str(diagnostic.get("adapter_run_id", "")).strip()


def _row_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset = _dataset_name(row)
        run_id = _adapter_run_id(row)
        if not dataset:
            raise ValueError(
                f"manifest row {row.get('id')!r} has no diagnostic.dataset"
            )
        if _ADAPTER_RUN_ID.fullmatch(run_id) is None:
            raise ValueError(
                f"manifest row {row.get('id')!r} has no valid diagnostic.adapter_run_id"
            )
        groups[(dataset, run_id)].append(row)
    return dict(groups)


def _normal_source_artifacts(raw: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{label} must contain at least one portable source artifact")
    artifacts: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int]] = set()
    for index, item in enumerate(raw):
        path = f"{label}[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{path} must be an object")
        if set(item) != {"role", "relative_path", "sha256", "size_bytes"}:
            raise ValueError(
                f"{path} must contain exactly role, relative_path, sha256, size_bytes"
            )
        role = str(item.get("role", "")).strip()
        relative_path = str(item.get("relative_path", "")).strip().replace("\\", "/")
        if not role:
            raise ValueError(f"{path}.role must be non-empty")
        relative = PurePosixPath(relative_path)
        if (
            not relative_path
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError(f"{path}.relative_path must be a safe relative path")
        digest = _sha256(item.get("sha256"), field=f"{path}.sha256")
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{path}.size_bytes must be a non-negative integer")
        identity = (role, relative.as_posix(), digest, size)
        if identity in seen:
            raise ValueError(
                f"{label} contains a duplicate portable artifact: {identity}"
            )
        seen.add(identity)
        artifacts.append(
            {
                "role": role,
                "relative_path": relative.as_posix(),
                "sha256": digest,
                "size_bytes": size,
            }
        )
    artifacts.sort(
        key=lambda item: (
            item["role"],
            item["relative_path"],
            item["sha256"],
            item["size_bytes"],
        )
    )
    return artifacts


def source_artifact_root_sha256(artifacts: Sequence[Mapping[str, Any]]) -> str:
    normal = _normal_source_artifacts(list(artifacts), label="source_artifacts")
    # This intentionally matches the adapter-report contract. The global
    # release root below supplies its own schema/adapter-run namespace.
    return canonical_sha256(normal)


def _validate_source_checksums(
    report: Mapping[str, Any], *, report_path: Path
) -> dict[str, str]:
    raw = report.get("source_checksums_sha256")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"adapter report has no source checksums: {report_path}")
    checked: dict[str, str] = {}
    for raw_path, raw_digest in raw.items():
        source = Path(str(raw_path)).expanduser()
        expected = _sha256(
            raw_digest, field=f"adapter report source checksum for {source}"
        )
        if not source.is_file():
            raise FileNotFoundError(
                f"adapter source named by the report is unavailable: {source}"
            )
        actual = sha256_file(source)
        if actual != expected:
            raise ValueError(
                f"adapter source changed after manifest creation: {source}; "
                f"expected {expected}, got {actual}"
            )
        checked[str(source.resolve())] = actual
    return dict(sorted(checked.items()))


def _validate_report_source_artifacts(
    report: Mapping[str, Any],
    *,
    report_path: Path,
    checked_sources: Mapping[str, str],
) -> tuple[list[dict[str, Any]], str]:
    artifacts = _normal_source_artifacts(
        report.get("source_artifacts"),
        label=f"adapter report {report_path} source_artifacts",
    )
    actual_root = source_artifact_root_sha256(artifacts)
    expected_root = _sha256(
        report.get("source_artifact_root_sha256"),
        field=f"adapter report {report_path} source_artifact_root_sha256",
    )
    if actual_root != expected_root:
        raise ValueError(f"adapter report source artifact root mismatch: {report_path}")
    artifact_digests = {str(item["sha256"]) for item in artifacts}
    checked_digests = set(checked_sources.values())
    if artifact_digests != checked_digests:
        raise ValueError(
            f"adapter report portable artifacts and local source checksums disagree: {report_path}"
        )
    artifact_blobs = {
        (str(item["sha256"]), int(item["size_bytes"])) for item in artifacts
    }
    checked_blobs = {
        (digest, int(Path(path).stat().st_size))
        for path, digest in checked_sources.items()
    }
    if artifact_blobs != checked_blobs:
        raise ValueError(
            f"adapter report portable artifact sizes disagree with local sources: {report_path}"
        )
    return artifacts, actual_root


def _validate_portable_options(value: Any, *, path: str) -> None:
    """Prevent a future adapter option from silently reintroducing mount paths."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if _LOCAL_PATH_KEY.search(key_text) is not None:
                raise ValueError(f"{path}.{key_text} is a machine-local path option")
            _validate_portable_options(item, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_portable_options(item, path=f"{path}[{index}]")
    elif isinstance(value, str) and (
        Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise ValueError(f"{path} contains an absolute machine-local path")


def _adapter_run_identity(
    *,
    dataset: str,
    canonical_split: str,
    adapter_options: Mapping[str, Any],
    source_artifacts: Sequence[Mapping[str, Any]],
    record_ids: Sequence[str],
) -> str:
    return "adapter-run::" + canonical_sha256(
        {
            "schema_version": ADAPTER_RUN_SCHEMA_VERSION,
            "dataset": dataset,
            "canonical_split": canonical_split,
            "adapter_options": dict(adapter_options),
            "source_artifacts": list(source_artifacts),
            "record_ids": sorted(record_ids),
        }
    )


def _media_identity_by_record(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[str, int]], list[dict[str, Any]]]:
    """Hash media once per local path and bind every record to those bytes."""

    path_cache: dict[str, tuple[str, int]] = {}
    path_records: dict[str, list[str]] = defaultdict(list)
    record_media: dict[str, tuple[str, int]] = {}
    for row in rows:
        record_id = str(row.get("id", "")).strip()
        media_path = Path(str(row.get("media_path", ""))).expanduser().resolve()
        if not media_path.is_file():
            raise FileNotFoundError(
                f"media named by manifest row {record_id!r} is unavailable: {media_path}"
            )
        key = str(media_path)
        if key not in path_cache:
            path_cache[key] = (sha256_file(media_path), int(media_path.stat().st_size))
        record_media[record_id] = path_cache[key]
        path_records[key].append(record_id)
    audit_files = [
        {
            "path": path,
            "sha256": path_cache[path][0],
            "size_bytes": path_cache[path][1],
            "record_ids": sorted(path_records[path]),
        }
        for path in sorted(path_cache)
    ]
    return record_media, audit_files


def _media_bindings(
    record_media: Mapping[str, tuple[str, int]],
) -> list[dict[str, Any]]:
    by_blob: dict[tuple[str, int], list[str]] = defaultdict(list)
    for record_id, identity in record_media.items():
        by_blob[identity].append(record_id)
    bindings = [
        {
            "sha256": digest,
            "size_bytes": size,
            "record_ids": sorted(record_ids),
        }
        for (digest, size), record_ids in by_blob.items()
    ]
    bindings.sort(
        key=lambda item: (item["sha256"], item["size_bytes"], item["record_ids"])
    )
    return bindings


def _normal_media_bindings(
    raw: Any, *, expected_record_ids: Iterable[str]
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, int]]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("media_bindings must be a non-empty list")
    normal: list[dict[str, Any]] = []
    record_media: dict[str, tuple[str, int]] = {}
    blob_ids: set[tuple[str, int]] = set()
    for index, item in enumerate(raw):
        path = f"media_bindings[{index}]"
        if not isinstance(item, Mapping) or set(item) != {
            "sha256",
            "size_bytes",
            "record_ids",
        }:
            raise ValueError(
                f"{path} must contain exactly sha256, size_bytes, record_ids"
            )
        digest = _sha256(item.get("sha256"), field=f"{path}.sha256")
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{path}.size_bytes must be a non-negative integer")
        record_ids = item.get("record_ids")
        if (
            not isinstance(record_ids, list)
            or not record_ids
            or any(
                not isinstance(value, str) or not value.strip() for value in record_ids
            )
            or record_ids != sorted(record_ids)
            or len(set(record_ids)) != len(record_ids)
        ):
            raise ValueError(
                f"{path}.record_ids must be sorted, unique non-empty strings"
            )
        blob_id = (digest, size)
        if blob_id in blob_ids:
            raise ValueError(
                f"media_bindings contains duplicate blob identity {blob_id}"
            )
        blob_ids.add(blob_id)
        for record_id in record_ids:
            if record_id in record_media:
                raise ValueError(
                    f"media_bindings assigns record {record_id!r} to multiple media blobs"
                )
            record_media[record_id] = blob_id
        normal.append(
            {"sha256": digest, "size_bytes": size, "record_ids": list(record_ids)}
        )
    normal.sort(
        key=lambda item: (item["sha256"], item["size_bytes"], item["record_ids"])
    )
    if list(raw) != normal:
        raise ValueError("media_bindings must use canonical sorted representation")
    expected = set(expected_record_ids)
    actual = set(record_media)
    if actual != expected:
        raise ValueError(
            "media_bindings record coverage mismatch; "
            f"missing={sorted(expected - actual)[:10]}, extra={sorted(actual - expected)[:10]}"
        )
    return normal, record_media


def media_binding_root_sha256(bindings: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256(
        {
            "schema_version": MEDIA_BINDING_SCHEMA_VERSION,
            "bindings": list(bindings),
        }
    )


def _unit_summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for field in ("resampling_unit_id", "independent_unit_id", "pair_id"):
        memberships: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            diagnostic = row.get("diagnostic") or {}
            raw = diagnostic.get(field)
            if raw in (None, ""):
                continue
            namespace = (
                f"{_dataset_name(row)}::{_adapter_run_id(row)}::{str(raw).strip()}"
            )
            memberships[namespace].append(str(row["id"]))
        leaves = [
            {"namespaced_unit_id": unit, "record_ids": sorted(record_ids)}
            for unit, record_ids in memberships.items()
        ]
        leaves.sort(key=lambda item: item["namespaced_unit_id"])
        summaries[field] = {
            "unique_units": len(leaves),
            "membership_root_sha256": canonical_sha256(
                {
                    "schema_version": UNIT_MEMBERSHIP_SCHEMA_VERSION,
                    "unit_field": field,
                    "memberships": leaves,
                }
            ),
        }
    return summaries


def _source_release_root(adapter_runs: Sequence[Mapping[str, Any]]) -> str:
    leaves = [
        {
            "dataset": str(entry["dataset"]),
            "adapter_run_id": str(entry["adapter_run_id"]),
            "source_artifact_root_sha256": str(entry["source_artifact_root_sha256"]),
        }
        for entry in adapter_runs
    ]
    leaves.sort(key=lambda item: (item["dataset"], item["adapter_run_id"]))
    return canonical_sha256(
        {"schema_version": SOURCE_RELEASE_SCHEMA_VERSION, "adapter_runs": leaves}
    )


def _release_payload(lock: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "records",
        "datasets",
        "manifest_semantic_record_set_sha256",
        "adapter_runs",
        "source_artifact_root_sha256",
        "media_bindings",
        "media_binding_root_sha256",
        "unit_summaries",
    )
    return {
        "schema_version": DATA_RELEASE_SCHEMA_VERSION,
        **{field: lock.get(field) for field in fields},
    }


def _lock_payload(lock: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in lock.items() if key != "lock_payload_sha256"}


def create_data_lock(
    *, manifest_path: str | Path, adapter_report_paths: Sequence[str | Path]
) -> dict[str, Any]:
    """Validate reports and freeze a complete merged official-data release."""

    manifest_source = Path(manifest_path)
    rows = read_jsonl(manifest_source)
    validation = validate_manifest(rows, require_media=True, strict_diagnostic=True)
    if not validation.get("valid"):
        preview = (validation.get("issues") or [])[:10]
        raise ValueError(
            "confirmatory base manifest failed strict media/schema validation; "
            f"first issues: {preview}"
        )
    groups = _row_groups(rows)
    if not adapter_report_paths:
        raise ValueError("at least one --adapter-report is required")

    report_entries: list[dict[str, Any]] = []
    audit_reports: list[dict[str, Any]] = []
    seen_report_groups: set[tuple[str, str]] = set()
    for raw_report_path in adapter_report_paths:
        report_path = Path(raw_report_path)
        report = _load_object(report_path, label="adapter report")
        if report.get("schema_version") != ADAPTER_REPORT_SCHEMA_VERSION:
            raise ValueError(
                f"adapter report {report_path} has no supported report schema; regenerate it"
            )
        dataset = str(report.get("dataset", "")).strip()
        run_id = str(report.get("adapter_run_id", "")).strip()
        group_key = (dataset, run_id)
        if not dataset or _ADAPTER_RUN_ID.fullmatch(run_id) is None:
            raise ValueError(
                f"adapter report has an invalid dataset/adapter_run_id: {report_path}"
            )
        if group_key in seen_report_groups:
            raise ValueError(
                "duplicate/overlapping adapter reports for "
                f"dataset={dataset!r}, adapter_run_id={run_id!r}"
            )
        seen_report_groups.add(group_key)
        if group_key not in groups:
            raise ValueError(
                "adapter report has no exact manifest row group for "
                f"dataset={dataset!r}, adapter_run_id={run_id!r}"
            )
        if report.get("confirmatory_eligible") is not True:
            raise ValueError(
                f"adapter report {report_path} used a debug escape hatch and is not "
                "confirmatory-eligible"
            )
        eligibility_issues = report.get("confirmatory_eligibility_issues")
        if eligibility_issues is not None and (
            not isinstance(eligibility_issues, list) or eligibility_issues
        ):
            raise ValueError(
                f"adapter report {report_path} has confirmatory eligibility issues: "
                f"{eligibility_issues!r}"
            )
        if (
            report.get("limited") is not False
            or report.get("require_media") is not True
        ):
            raise ValueError(
                f"adapter report {report_path} must be unlimited and require all media"
            )
        debug_options = report.get("debug_options")
        if not isinstance(debug_options, Mapping) or any(
            value is not False for value in debug_options.values()
        ):
            raise ValueError(
                f"adapter report {report_path} has missing/enabled debug options"
            )
        report_validation = report.get("validation")
        if (
            not isinstance(report_validation, Mapping)
            or report_validation.get("valid") is not True
        ):
            raise ValueError(f"adapter report validation is not valid: {report_path}")

        group_rows = groups[group_key]
        expected_count = report.get("records")
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count != len(group_rows)
        ):
            raise ValueError(
                "adapter report row count does not match its exact manifest group for "
                f"{dataset}/{run_id}: report={expected_count}, manifest={len(group_rows)}"
            )
        exact_root = manifest_record_set_sha256(group_rows)
        if exact_root != _sha256(
            report.get("manifest_record_set_sha256"),
            field=f"adapter report {report_path} manifest_record_set_sha256",
        ):
            raise ValueError(
                f"adapter report record content does not match its exact manifest group: {report_path}"
            )
        checked_sources = _validate_source_checksums(report, report_path=report_path)
        source_artifacts, source_root = _validate_report_source_artifacts(
            report,
            report_path=report_path,
            checked_sources=checked_sources,
        )
        canonical_split = str(report.get("canonical_split", "")).strip()
        adapter_options = report.get("adapter_options")
        if not canonical_split or not isinstance(adapter_options, Mapping):
            raise ValueError(
                f"adapter report {report_path} must bind canonical_split and adapter_options"
            )
        _validate_portable_options(
            adapter_options,
            path=f"adapter report {report_path} adapter_options",
        )
        selection_options = adapter_options.get("resampling_unit_selection")
        selection_report = report.get("resampling_unit_selection")
        if selection_options is None and selection_report is not None:
            raise ValueError(
                f"adapter report {report_path} has an unbound resampling-unit selection"
            )
        if selection_options is not None:
            if not isinstance(selection_options, Mapping) or not isinstance(
                selection_report, Mapping
            ):
                raise ValueError(
                    f"adapter report {report_path} has incomplete resampling-unit "
                    "selection metadata"
                )
            try:
                validate_resampling_unit_selection(
                    report=selection_report,
                    options=selection_options,
                    selected_rows=group_rows,
                    dataset=dataset,
                    canonical_split=canonical_split,
                )
            except ValueError as exc:
                raise ValueError(
                    f"adapter report {report_path} has invalid resampling-unit "
                    f"selection metadata: {exc}"
                ) from exc
        expected_run_id = _adapter_run_identity(
            dataset=dataset,
            canonical_split=canonical_split,
            adapter_options=adapter_options,
            source_artifacts=source_artifacts,
            record_ids=[str(row["id"]) for row in group_rows],
        )
        if run_id != expected_run_id:
            raise ValueError(
                f"adapter report adapter_run_id identity mismatch: {report_path}"
            )
        report_entries.append(
            {
                "dataset": dataset,
                "adapter_run_id": run_id,
                "records": len(group_rows),
                "canonical_split": canonical_split,
                "adapter_options": dict(adapter_options),
                "manifest_semantic_record_set_sha256": (
                    manifest_semantic_record_set_sha256(group_rows)
                ),
                "source_artifacts": source_artifacts,
                "source_artifact_root_sha256": source_root,
                "media_binding_root_sha256": None,
            }
        )
        audit_reports.append(
            {
                "dataset": dataset,
                "adapter_run_id": run_id,
                "report_path": str(report_path.resolve()),
                "report_sha256": sha256_file(report_path),
                "manifest_record_set_sha256": exact_root,
                "source_checksums_sha256": checked_sources,
            }
        )

    missing_groups = sorted(set(groups) - seen_report_groups)
    if missing_groups:
        raise ValueError(
            "final manifest has exact adapter-run row groups with no strict report: "
            f"{missing_groups}"
        )

    record_media, audit_media_files = _media_identity_by_record(rows)
    bindings = _media_bindings(record_media)
    for entry in report_entries:
        group_rows = groups[(str(entry["dataset"]), str(entry["adapter_run_id"]))]
        group_record_media = {
            str(row["id"]): record_media[str(row["id"])] for row in group_rows
        }
        entry["media_binding_root_sha256"] = media_binding_root_sha256(
            _media_bindings(group_record_media)
        )
    report_entries.sort(key=lambda value: (value["dataset"], value["adapter_run_id"]))
    audit_reports.sort(key=lambda value: (value["dataset"], value["adapter_run_id"]))

    dataset_counts = Counter(_dataset_name(row) for row in rows)
    lock: dict[str, Any] = {
        "schema_version": DATA_LOCK_SCHEMA_VERSION,
        "records": len(rows),
        "datasets": dict(sorted(dataset_counts.items())),
        "manifest_semantic_record_set_sha256": (
            manifest_semantic_record_set_sha256(rows)
        ),
        "adapter_runs": report_entries,
        "source_artifact_root_sha256": _source_release_root(report_entries),
        "media_bindings": bindings,
        "media_binding_root_sha256": media_binding_root_sha256(bindings),
        "unit_summaries": _unit_summaries(rows),
        "audit": {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "manifest_path": str(manifest_source.resolve()),
            "manifest_sha256": sha256_file(manifest_source),
            "adapter_reports": audit_reports,
            "media_files": audit_media_files,
            "strict_validation_issue_counts": validation.get("issue_counts", {}),
        },
    }
    lock["data_release_sha256"] = canonical_sha256(_release_payload(lock))
    lock["lock_payload_sha256"] = canonical_sha256(_lock_payload(lock))
    return lock


def _validate_audit(lock: Mapping[str, Any]) -> Mapping[str, Any]:
    audit = lock.get("audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("schema_version") != AUDIT_SCHEMA_VERSION
    ):
        raise ValueError("data lock has malformed audit metadata")
    if not isinstance(audit.get("adapter_reports"), list):
        raise ValueError("data lock audit.adapter_reports must be a list")
    if not isinstance(audit.get("media_files"), list):
        raise ValueError("data lock audit.media_files must be a list")
    _sha256(audit.get("manifest_sha256"), field="audit.manifest_sha256")
    return audit


def _validate_audit_structure(
    audit: Mapping[str, Any],
    *,
    expected_groups: set[tuple[str, str]],
    expected_record_ids: set[str],
) -> None:
    manifest_path = audit.get("manifest_path")
    if not isinstance(manifest_path, str) or not manifest_path.strip():
        raise ValueError("data lock audit.manifest_path must be non-empty")
    report_keys: list[tuple[str, str]] = []
    for index, entry in enumerate(audit.get("adapter_reports") or []):
        path = f"audit.adapter_reports[{index}]"
        if not isinstance(entry, Mapping) or set(entry) != {
            "dataset",
            "adapter_run_id",
            "report_path",
            "report_sha256",
            "manifest_record_set_sha256",
            "source_checksums_sha256",
        }:
            raise ValueError(f"{path} has malformed fields")
        dataset = str(entry.get("dataset", "")).strip()
        run_id = str(entry.get("adapter_run_id", "")).strip()
        report_path = entry.get("report_path")
        if (
            not dataset
            or _ADAPTER_RUN_ID.fullmatch(run_id) is None
            or not isinstance(report_path, str)
            or not report_path.strip()
        ):
            raise ValueError(f"{path} has invalid dataset/run/path metadata")
        _sha256(entry.get("report_sha256"), field=f"{path}.report_sha256")
        _sha256(
            entry.get("manifest_record_set_sha256"),
            field=f"{path}.manifest_record_set_sha256",
        )
        checksums = entry.get("source_checksums_sha256")
        if not isinstance(checksums, Mapping) or not checksums:
            raise ValueError(f"{path}.source_checksums_sha256 must be non-empty")
        for source_path, digest in checksums.items():
            if not isinstance(source_path, str) or not source_path.strip():
                raise ValueError(f"{path} has an empty local source path")
            _sha256(digest, field=f"{path} source checksum")
        report_keys.append((dataset, run_id))
    if len(set(report_keys)) != len(report_keys) or set(report_keys) != expected_groups:
        raise ValueError(
            "audit adapter-report coverage mismatch; "
            f"expected={sorted(expected_groups)}, actual={sorted(report_keys)}"
        )

    audited_record_ids: set[str] = set()
    audited_paths: set[str] = set()
    for index, entry in enumerate(audit.get("media_files") or []):
        path = f"audit.media_files[{index}]"
        if not isinstance(entry, Mapping) or set(entry) != {
            "path",
            "sha256",
            "size_bytes",
            "record_ids",
        }:
            raise ValueError(f"{path} has malformed fields")
        local_path = entry.get("path")
        if not isinstance(local_path, str) or not local_path.strip():
            raise ValueError(f"{path}.path must be non-empty")
        if local_path in audited_paths:
            raise ValueError(f"audit.media_files duplicates local path {local_path!r}")
        audited_paths.add(local_path)
        _sha256(entry.get("sha256"), field=f"{path}.sha256")
        size = entry.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"{path}.size_bytes must be a non-negative integer")
        record_ids = entry.get("record_ids")
        if (
            not isinstance(record_ids, list)
            or not record_ids
            or record_ids != sorted(record_ids)
            or len(set(record_ids)) != len(record_ids)
        ):
            raise ValueError(f"{path}.record_ids must be sorted and unique")
        overlap = audited_record_ids.intersection(record_ids)
        if overlap:
            raise ValueError(
                f"audit.media_files assigns records to multiple paths: {sorted(overlap)}"
            )
        audited_record_ids.update(record_ids)
    if audited_record_ids != expected_record_ids:
        raise ValueError("audit media-file record coverage does not match the manifest")


def _validate_locked_adapter_runs(
    raw: Any,
    *,
    groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    record_media: Mapping[str, tuple[str, int]],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("adapter_runs must be a non-empty list")
    normal: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_entry in enumerate(raw):
        path = f"adapter_runs[{index}]"
        if not isinstance(raw_entry, Mapping) or set(raw_entry) != {
            "dataset",
            "adapter_run_id",
            "records",
            "canonical_split",
            "adapter_options",
            "manifest_semantic_record_set_sha256",
            "source_artifacts",
            "source_artifact_root_sha256",
            "media_binding_root_sha256",
        }:
            raise ValueError(f"{path} has malformed fields")
        dataset = str(raw_entry.get("dataset", "")).strip()
        run_id = str(raw_entry.get("adapter_run_id", "")).strip()
        key = (dataset, run_id)
        if key in seen:
            raise ValueError(f"adapter_runs contains duplicate/overlapping group {key}")
        seen.add(key)
        if key not in groups:
            raise ValueError(f"adapter_runs has no matching manifest row group: {key}")
        group_rows = list(groups[key])
        records = raw_entry.get("records")
        if (
            isinstance(records, bool)
            or not isinstance(records, int)
            or records != len(group_rows)
        ):
            raise ValueError(f"{path}.records does not match its manifest row group")
        semantic_root = manifest_semantic_record_set_sha256(group_rows)
        if semantic_root != _sha256(
            raw_entry.get("manifest_semantic_record_set_sha256"),
            field=f"{path}.manifest_semantic_record_set_sha256",
        ):
            raise ValueError(f"{path} semantic record root does not match the manifest")
        artifacts = _normal_source_artifacts(
            raw_entry.get("source_artifacts"), label=f"{path}.source_artifacts"
        )
        source_root = source_artifact_root_sha256(artifacts)
        if source_root != _sha256(
            raw_entry.get("source_artifact_root_sha256"),
            field=f"{path}.source_artifact_root_sha256",
        ):
            raise ValueError(f"{path} source artifact root mismatch")
        canonical_split = str(raw_entry.get("canonical_split", "")).strip()
        adapter_options = raw_entry.get("adapter_options")
        if not canonical_split or not isinstance(adapter_options, Mapping):
            raise ValueError(f"{path} has invalid canonical_split/adapter_options")
        _validate_portable_options(adapter_options, path=f"{path}.adapter_options")
        if run_id != _adapter_run_identity(
            dataset=dataset,
            canonical_split=canonical_split,
            adapter_options=adapter_options,
            source_artifacts=artifacts,
            record_ids=[str(row["id"]) for row in group_rows],
        ):
            raise ValueError(f"{path} adapter_run_id cannot be reproduced")
        group_media = {
            str(row["id"]): record_media[str(row["id"])] for row in group_rows
        }
        group_media_root = media_binding_root_sha256(_media_bindings(group_media))
        if group_media_root != _sha256(
            raw_entry.get("media_binding_root_sha256"),
            field=f"{path}.media_binding_root_sha256",
        ):
            raise ValueError(f"{path} media binding root mismatch")
        normal.append(
            {
                "dataset": dataset,
                "adapter_run_id": run_id,
                "records": records,
                "canonical_split": canonical_split,
                "adapter_options": dict(adapter_options),
                "manifest_semantic_record_set_sha256": semantic_root,
                "source_artifacts": artifacts,
                "source_artifact_root_sha256": source_root,
                "media_binding_root_sha256": group_media_root,
            }
        )
    normal.sort(key=lambda value: (value["dataset"], value["adapter_run_id"]))
    if list(raw) != normal:
        raise ValueError("adapter_runs must use canonical sorted representation")
    if seen != set(groups):
        raise ValueError(
            "adapter_runs coverage mismatch; "
            f"missing={sorted(set(groups) - seen)}, extra={sorted(seen - set(groups))}"
        )
    return normal


def validate_data_lock(
    lock_path: str | Path,
    *,
    manifest_path: str | Path,
    verify_sources: bool = False,
    verify_media: bool = False,
) -> dict[str, Any]:
    """Recompute release structure and authenticate it against a manifest.

    ``verify_sources`` re-hashes source paths captured in local audit metadata.
    ``verify_media`` re-hashes media at the current manifest paths, so an
    equivalent release can be checked after relocation.
    """

    lock_source = Path(lock_path)
    lock = _load_object(lock_source, label="data lock")
    allowed_fields = {
        "schema_version",
        "records",
        "datasets",
        "manifest_semantic_record_set_sha256",
        "adapter_runs",
        "source_artifact_root_sha256",
        "media_bindings",
        "media_binding_root_sha256",
        "unit_summaries",
        "audit",
        "data_release_sha256",
        "lock_payload_sha256",
    }
    if set(lock) != allowed_fields:
        raise ValueError(
            "data lock has missing/unknown fields: "
            f"missing={sorted(allowed_fields - set(lock))}, "
            f"unknown={sorted(set(lock) - allowed_fields)}"
        )
    if lock.get("schema_version") != DATA_LOCK_SCHEMA_VERSION:
        raise ValueError(f"unsupported data-lock schema: {lock_source}")
    expected_payload_digest = _sha256(
        lock.get("lock_payload_sha256"), field="lock_payload_sha256"
    )
    if canonical_sha256(_lock_payload(lock)) != expected_payload_digest:
        raise ValueError(f"data-lock payload digest mismatch: {lock_source}")
    audit = _validate_audit(lock)

    manifest_source = Path(manifest_path)
    rows = read_jsonl(manifest_source)
    validation = validate_manifest(
        rows, require_media=verify_media, strict_diagnostic=True
    )
    if not validation.get("valid"):
        preview = (validation.get("issues") or [])[:10]
        raise ValueError(f"base manifest failed validation; first issues: {preview}")
    groups = _row_groups(rows)
    _validate_audit_structure(
        audit,
        expected_groups=set(groups),
        expected_record_ids={str(row["id"]) for row in rows},
    )
    records = lock.get("records")
    if (
        isinstance(records, bool)
        or not isinstance(records, int)
        or records != len(rows)
    ):
        raise ValueError("base manifest record count does not match the data lock")
    datasets = dict(sorted(Counter(_dataset_name(row) for row in rows).items()))
    if lock.get("datasets") != datasets:
        raise ValueError("base manifest dataset counts do not match the data lock")
    semantic_root = manifest_semantic_record_set_sha256(rows)
    if lock.get("manifest_semantic_record_set_sha256") != semantic_root:
        raise ValueError(
            "base manifest semantic record content does not match the data lock"
        )

    bindings, locked_record_media = _normal_media_bindings(
        lock.get("media_bindings"),
        expected_record_ids=[str(row["id"]) for row in rows],
    )
    binding_root = media_binding_root_sha256(bindings)
    if lock.get("media_binding_root_sha256") != binding_root:
        raise ValueError("data-lock media binding root is not reproducible")
    if verify_media:
        actual_record_media, _audit_media = _media_identity_by_record(rows)
        if actual_record_media != locked_record_media:
            changed = sorted(
                record_id
                for record_id in set(actual_record_media) | set(locked_record_media)
                if actual_record_media.get(record_id)
                != locked_record_media.get(record_id)
            )
            raise ValueError(
                "media bytes do not match the locked data release; "
                f"first changed record IDs: {changed[:10]}"
            )

    adapter_runs = _validate_locked_adapter_runs(
        lock.get("adapter_runs"), groups=groups, record_media=locked_record_media
    )
    source_root = _source_release_root(adapter_runs)
    if lock.get("source_artifact_root_sha256") != source_root:
        raise ValueError("data-lock source artifact root is not reproducible")
    unit_summaries = _unit_summaries(rows)
    if lock.get("unit_summaries") != unit_summaries:
        raise ValueError(
            "data-lock namespaced unit summaries do not match the manifest"
        )

    expected_release_digest = _sha256(
        lock.get("data_release_sha256"), field="data_release_sha256"
    )
    if canonical_sha256(_release_payload(lock)) != expected_release_digest:
        raise ValueError("data-release digest mismatch")
    if verify_sources:
        run_by_key = {
            (str(entry["dataset"]), str(entry["adapter_run_id"])): entry
            for entry in adapter_runs
        }
        for entry in audit.get("adapter_reports") or []:
            if not isinstance(entry, Mapping):
                raise ValueError(
                    "data lock has a malformed audit.adapter_reports entry"
                )
            checked_sources = _validate_source_checksums(
                {"source_checksums_sha256": entry.get("source_checksums_sha256")},
                report_path=lock_source,
            )
            key = (str(entry["dataset"]), str(entry["adapter_run_id"]))
            locked_run = run_by_key[key]
            _validate_report_source_artifacts(
                {
                    "source_artifacts": locked_run["source_artifacts"],
                    "source_artifact_root_sha256": locked_run[
                        "source_artifact_root_sha256"
                    ],
                },
                report_path=lock_source,
                checked_sources=checked_sources,
            )

    return {
        "path": str(lock_source.resolve()),
        # Compatibility alias: downstream trial/protocol identity must use the
        # stable scientific release digest, never the local JSON file digest.
        "sha256": expected_release_digest,
        "data_release_sha256": expected_release_digest,
        "file_sha256": sha256_file(lock_source),
        "lock_payload_sha256": expected_payload_digest,
        "manifest_semantic_record_set_sha256": semantic_root,
        "records": len(rows),
        "datasets": datasets,
        "adapter_runs": adapter_runs,
        "source_artifact_root_sha256": source_root,
        "media_binding_root_sha256": binding_root,
        "unit_summaries": unit_summaries,
        "audit_manifest_path": audit.get("manifest_path"),
        "audit_manifest_sha256": audit.get("manifest_sha256"),
        "current_manifest_sha256": sha256_file(manifest_source),
    }


def validate_trial_media_lock(
    lock_path: str | Path,
    trial_rows_or_manifest: Iterable[Mapping[str, Any]] | str | Path,
) -> dict[str, Any]:
    """Authenticate locked media bytes using only an expanded trial manifest.

    Extraction receives condition-expanded trials rather than the base
    manifest. This helper collapses them by ``base_id``, requires exact locked
    record coverage, and hashes each unique current media path once.
    """

    lock_source = Path(lock_path)
    lock = _load_object(lock_source, label="data lock")
    allowed_fields = {
        "schema_version",
        "records",
        "datasets",
        "manifest_semantic_record_set_sha256",
        "adapter_runs",
        "source_artifact_root_sha256",
        "media_bindings",
        "media_binding_root_sha256",
        "unit_summaries",
        "audit",
        "data_release_sha256",
        "lock_payload_sha256",
    }
    if (
        set(lock) != allowed_fields
        or lock.get("schema_version") != DATA_LOCK_SCHEMA_VERSION
    ):
        raise ValueError("trial media check received a malformed/unsupported data lock")
    payload_digest = _sha256(
        lock.get("lock_payload_sha256"), field="lock_payload_sha256"
    )
    if canonical_sha256(_lock_payload(lock)) != payload_digest:
        raise ValueError("data-lock payload digest mismatch")
    release_digest = _sha256(
        lock.get("data_release_sha256"), field="data_release_sha256"
    )
    if canonical_sha256(_release_payload(lock)) != release_digest:
        raise ValueError("data-release digest mismatch")
    audit = _validate_audit(lock)

    raw_bindings = lock.get("media_bindings")
    if not isinstance(raw_bindings, list):
        raise ValueError("data lock has malformed media_bindings")
    locked_record_ids: list[str] = []
    for item in raw_bindings:
        if not isinstance(item, Mapping) or not isinstance(
            item.get("record_ids"), list
        ):
            raise ValueError("data lock has malformed media_bindings")
        locked_record_ids.extend(str(value) for value in item["record_ids"])
    bindings, locked_record_media = _normal_media_bindings(
        raw_bindings, expected_record_ids=locked_record_ids
    )
    media_root = media_binding_root_sha256(bindings)
    if lock.get("media_binding_root_sha256") != media_root:
        raise ValueError("data-lock media binding root is not reproducible")
    records = lock.get("records")
    if (
        isinstance(records, bool)
        or not isinstance(records, int)
        or records != len(locked_record_media)
    ):
        raise ValueError("data-lock records does not match locked media coverage")

    raw_runs = lock.get("adapter_runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("data lock has malformed adapter_runs")
    run_keys: set[tuple[str, str]] = set()
    run_record_counts: Counter[str] = Counter()
    normal_runs: list[dict[str, Any]] = []
    for index, entry in enumerate(raw_runs):
        path = f"adapter_runs[{index}]"
        if not isinstance(entry, Mapping) or set(entry) != {
            "dataset",
            "adapter_run_id",
            "records",
            "canonical_split",
            "adapter_options",
            "manifest_semantic_record_set_sha256",
            "source_artifacts",
            "source_artifact_root_sha256",
            "media_binding_root_sha256",
        }:
            raise ValueError(f"{path} has malformed fields")
        dataset = str(entry.get("dataset", "")).strip()
        run_id = str(entry.get("adapter_run_id", "")).strip()
        key = (dataset, run_id)
        if not dataset or _ADAPTER_RUN_ID.fullmatch(run_id) is None or key in run_keys:
            raise ValueError(f"{path} has a duplicate/invalid adapter-run identity")
        run_keys.add(key)
        run_records = entry.get("records")
        if (
            isinstance(run_records, bool)
            or not isinstance(run_records, int)
            or run_records < 1
        ):
            raise ValueError(f"{path}.records must be positive")
        run_record_counts[dataset] += run_records
        artifacts = _normal_source_artifacts(
            entry.get("source_artifacts"), label=f"{path}.source_artifacts"
        )
        source_root = source_artifact_root_sha256(artifacts)
        if entry.get("source_artifact_root_sha256") != source_root:
            raise ValueError(f"{path} source artifact root mismatch")
        _sha256(
            entry.get("manifest_semantic_record_set_sha256"),
            field=f"{path}.manifest_semantic_record_set_sha256",
        )
        _sha256(
            entry.get("media_binding_root_sha256"),
            field=f"{path}.media_binding_root_sha256",
        )
        canonical_split = str(entry.get("canonical_split", "")).strip()
        adapter_options = entry.get("adapter_options")
        if not canonical_split or not isinstance(adapter_options, Mapping):
            raise ValueError(f"{path} has invalid canonical_split/adapter_options")
        _validate_portable_options(adapter_options, path=f"{path}.adapter_options")
        normal_runs.append(dict(entry))
    normal_runs.sort(key=lambda value: (value["dataset"], value["adapter_run_id"]))
    if list(raw_runs) != normal_runs:
        raise ValueError("adapter_runs must use canonical sorted representation")
    if dict(sorted(run_record_counts.items())) != lock.get("datasets"):
        raise ValueError("adapter-run record counts do not reproduce locked datasets")
    if sum(run_record_counts.values()) != records:
        raise ValueError("adapter-run record counts do not reproduce locked records")
    if _source_release_root(normal_runs) != lock.get("source_artifact_root_sha256"):
        raise ValueError("data-lock source artifact root is not reproducible")
    _validate_audit_structure(
        audit,
        expected_groups=run_keys,
        expected_record_ids=set(locked_record_media),
    )

    if isinstance(trial_rows_or_manifest, (str, Path)):
        trial_rows = read_jsonl(trial_rows_or_manifest)
    else:
        trial_rows = [dict(row) for row in trial_rows_or_manifest]
    if not trial_rows:
        raise ValueError("trial media verification requires a non-empty trial manifest")
    base_paths: dict[str, str] = {}
    base_datasets: dict[str, str] = {}
    for trial in trial_rows:
        base_id = str(trial.get("base_id", "")).strip()
        if not base_id:
            raise ValueError("trial row has no base_id for locked media verification")
        trial_release = trial.get("data_release_sha256")
        if trial_release not in (None, "") and str(trial_release) != release_digest:
            raise ValueError(
                f"trial {trial.get('id')!r} names a different data release"
            )
        visual_spec = trial.get("visual_spec")
        media_value = (
            visual_spec.get("media_path")
            if isinstance(visual_spec, Mapping)
            else trial.get("media_path")
        )
        if media_value in (None, ""):
            raise ValueError(f"trial {trial.get('id')!r} has no current media path")
        media_path = str(Path(str(media_value)).expanduser().resolve())
        prior = base_paths.setdefault(base_id, media_path)
        if prior != media_path:
            raise ValueError(f"trial base_id {base_id!r} maps to multiple media paths")
        diagnostic = trial.get("diagnostic") or {}
        dataset = str(diagnostic.get("dataset", trial.get("dataset", ""))).strip()
        if dataset:
            prior_dataset = base_datasets.setdefault(base_id, dataset)
            if prior_dataset != dataset:
                raise ValueError(f"trial base_id {base_id!r} maps to multiple datasets")

    actual_ids = set(base_paths)
    locked_ids = set(locked_record_media)
    if actual_ids != locked_ids:
        raise ValueError(
            "trial base_id coverage does not match the locked release; "
            f"missing={sorted(locked_ids - actual_ids)[:10]}, "
            f"extra={sorted(actual_ids - locked_ids)[:10]}"
        )
    if len(base_datasets) == len(base_paths):
        trial_dataset_counts = dict(sorted(Counter(base_datasets.values()).items()))
        if trial_dataset_counts != lock.get("datasets"):
            raise ValueError(
                "trial base_id dataset counts do not match the locked release"
            )
    media_rows = [
        {"id": base_id, "media_path": media_path}
        for base_id, media_path in sorted(base_paths.items())
    ]
    actual_record_media, _media_audit = _media_identity_by_record(media_rows)
    if actual_record_media != locked_record_media:
        changed = sorted(
            record_id
            for record_id in locked_ids
            if actual_record_media.get(record_id) != locked_record_media.get(record_id)
        )
        raise ValueError(
            "trial media bytes do not match the locked data release; "
            f"first changed base IDs: {changed[:10]}"
        )
    return {
        "path": str(lock_source.resolve()),
        "data_release_sha256": release_digest,
        "file_sha256": sha256_file(lock_source),
        "media_binding_root_sha256": media_root,
        "records": records,
        "datasets": dict(lock["datasets"]),
        "adapter_runs": normal_runs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze a strict merged official-data manifest and its adapter provenance."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--adapter-report", action="append", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    target = Path(args.out)
    output_identity = target.expanduser().resolve()
    input_identities = [
        Path(value).expanduser().resolve()
        for value in [args.manifest, *args.adapter_report]
    ]
    if output_identity in input_identities:
        raise ValueError(
            "--out must not alias --manifest or any --adapter-report input"
        )
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {target}")
    lock = create_data_lock(
        manifest_path=args.manifest,
        adapter_report_paths=args.adapter_report,
    )
    write_json(target, lock)
    print(
        json.dumps(
            {**lock, "path": str(target.resolve())}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
