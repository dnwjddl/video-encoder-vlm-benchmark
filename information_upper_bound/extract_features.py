#!/usr/bin/env python
"""Extract frozen-encoder features for auditable diagnostic video views.

Unlike the repository's generic extractor, this script treats a view as a
first-class experimental object.  A cached artifact is addressed by the stable
``visual_id`` plus its complete ``ViewSpec``; a second hash incorporates the
encoder configuration so changing weights, frame count, or token compression
cannot silently reuse incompatible features.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import itertools
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch
from tqdm import tqdm

from vlmevalbench.utils import get_dtype, set_seed

from information_upper_bound.encoder_runtime import (
    prepare_encoder_runtime,
    resolve_encoder,
)
from information_upper_bound.io import iter_jsonl, sha256_file, write_json, write_jsonl
from information_upper_bound.attestation import validate_trial_build_attestation
from information_upper_bound.conditions import (
    DEFAULT_CONDITION_PATH,
    trial_content_sha256,
)
from information_upper_bound.integrity import (
    canonical_sha256,
    decoded_frames_identity,
    feature_artifact_root,
    resolved_pretrained_identity,
    tensor_identity,
    validate_locked_pretrained_revision,
)
from information_upper_bound.protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
    protocol_section,
    validate_data_protocol,
)
from information_upper_bound.trial_matrix import validate_trial_matrix_closure

try:
    from information_upper_bound.media import (
        EVIDENCE_REQUIRED_VIEWS,
        VALID_VIEWS,
        EvidenceSpanError,
        VideoDecodeError,
        ViewSamplingError,
        ViewSpec,
        load_video_view,
    )
except (
    ModuleNotFoundError
):  # Supports `python information_upper_bound/extract_features.py`.
    from media import (  # type: ignore[no-redef]
        EVIDENCE_REQUIRED_VIEWS,
        VALID_VIEWS,
        EvidenceSpanError,
        VideoDecodeError,
        ViewSamplingError,
        ViewSpec,
        load_video_view,
    )


SCHEMA_VERSION = "information_upper_bound.features.v4.matrix_closure"
DEFAULT_EVIDENCE_FIELDS = (
    "diagnostic.evidence_spans",
    "evidence_spans",
    "evidence_span",
)
MISSING = object()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen visual features for timestamp-aware diagnostic views."
    )
    parser.add_argument(
        "--manifest", required=True, help="Unified JSONL diagnostic manifest."
    )
    parser.add_argument(
        "--encoder", required=True, help="Encoder name in --encoder-config."
    )
    parser.add_argument(
        "--encoder-config",
        required=True,
        help="Explicit encoder registry YAML (required so wheel installs never depend on CWD).",
    )
    parser.add_argument(
        "--protocol-config",
        default=str(DEFAULT_PROTOCOL_PATH),
        help="locked protocol supplying the sampling seed",
    )
    parser.add_argument(
        "--conditions-config",
        default=str(DEFAULT_CONDITION_PATH),
        help=(
            "Exact condition matrix used to build the trials; confirmatory extraction "
            "deterministically regenerates the complete expansion from this file."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--data-lock",
        help="Release lock used to authenticate confirmatory trial media before extraction.",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Permit base/unlocked manifests and debug-only extraction options.",
    )
    parser.add_argument(
        "--view",
        action="append",
        choices=VALID_VIEWS,
        help=(
            "View to extract from a base manifest, or view filter for a trial manifest. "
            "Repeat for multiple views; base manifests default to full."
        ),
    )
    parser.add_argument("--model-id", default=None)
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Immutable Hugging Face commit SHA for the visual encoder.",
    )
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--force", action="store_true", help="Recompute matching cached features."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing index.jsonl/metadata.json; cached tensors remain reusable.",
    )
    parser.add_argument(
        "--media-sha256",
        action="store_true",
        help="Hash every unique video for confirmatory provenance (slower but content-safe).",
    )
    parser.add_argument("--allow-missing-media", action="store_true")
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "decord", "opencv"],
        help="Video decoder. auto tries decord before OpenCV.",
    )
    parser.add_argument(
        "--evidence-field",
        default=None,
        help=(
            "Dotted manifest field containing evidence spans. By default tries "
            + ", ".join(DEFAULT_EVIDENCE_FIELDS)
            + "."
        ),
    )
    parser.add_argument(
        "--evidence-unit",
        default="seconds",
        choices=["seconds", "normalized"],
        help="Unit for two-number spans or start/end mappings that omit unit.",
    )
    parser.add_argument(
        "--visual-id-field",
        default="visual_id",
        help="Dotted stable visual identity field; falls back to video_id/media_id/media_path.",
    )
    return parser.parse_args(argv)


def canonical_hash(payload: Any) -> str:
    return canonical_sha256(payload)


def view_content_hash(visual_id: str, spec: ViewSpec) -> str:
    """Content address shared by questions that request the same visual view."""

    return canonical_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "visual_id": str(visual_id),
            "view_spec": spec.to_dict(),
        }
    )


def media_content_identity(fingerprint: Mapping[str, Any]) -> dict[str, Any]:
    if fingerprint.get("sha256"):
        return {
            "sha256": str(fingerprint["sha256"]),
            "size_bytes": int(fingerprint["size_bytes"]),
        }
    return {
        "size_bytes": int(fingerprint["size_bytes"]),
        "mtime_ns": int(fingerprint["mtime_ns"]),
    }


def feature_content_hash(
    view_hash: str,
    *,
    encoder_config: Mapping[str, Any],
    media_identity: Mapping[str, Any],
    extraction_identity: Mapping[str, Any],
    decoded_frame_identity: Mapping[str, Any],
    sampling_identity: Mapping[str, Any],
) -> str:
    """Content address for the encoder-specific tensor artifact."""

    return canonical_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "view_content_hash": view_hash,
            "encoder_config": dict(encoder_config),
            "media_content_identity": dict(media_identity),
            "extraction_identity": dict(extraction_identity),
            "decoded_frame_identity": dict(decoded_frame_identity),
            "sampling_identity": dict(sampling_identity),
        }
    )


def _distribution_version(candidates: Sequence[str]) -> str | None:
    for name in candidates:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return None


def _decoder_implementation_version(backend: str) -> str:
    normalized = str(backend).strip().lower()
    package_version = (
        _distribution_version(("decord",))
        if normalized == "decord"
        else _distribution_version(
            ("opencv-python", "opencv-python-headless", "opencv-contrib-python")
        )
    )
    if package_version is not None:
        return package_version
    module_name = "decord" if normalized == "decord" else "cv2"
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def decoder_identity(
    *,
    requested_backend: str,
    sampling: Mapping[str, Any],
) -> dict[str, Any]:
    video = sampling.get("video")
    if not isinstance(video, Mapping):
        raise ValueError("sampling provenance has no video metadata")
    actual_backend = str(video.get("backend", "")).strip().lower()
    if actual_backend not in {"decord", "opencv"}:
        raise ValueError(
            f"sampling provenance has unsupported decoder backend: {actual_backend!r}"
        )
    package_version = _decoder_implementation_version(actual_backend)
    payload = {
        "requested_backend": str(requested_backend),
        "actual_backend": actual_backend,
        "implementation_version": package_version,
        "timestamp_source": video.get("timestamp_source"),
        "frame_timestamp_sha256": video.get("frame_timestamp_sha256"),
        "time_alignment_guaranteed": video.get("time_alignment_guaranteed"),
    }
    return {**payload, "identity_sha256": canonical_hash(payload)}


def extraction_pipeline_identity(
    *,
    encoder: Any,
    encoder_config: Mapping[str, Any],
    requested_dtype: str,
    requested_backend: str,
) -> dict[str, Any]:
    pretrained = resolved_pretrained_identity(
        requested_id=str(encoder_config["model_id"]),
        resolved_source=encoder.pretrained_source,
        model=encoder.model,
        auxiliaries={"processor": encoder.processor},
    )
    requested_revision = encoder_config.get("revision")
    if requested_revision not in (None, ""):
        try:
            validate_locked_pretrained_revision(
                pretrained,
                str(requested_revision),
                component_name="encoder/processor",
            )
        except ValueError as exc:
            raise ValueError(
                "loaded encoder/processor content identity does not match --model-revision"
            ) from exc
    payload = {
        "encoder_config": dict(encoder_config),
        "encoder_pretrained_identity": pretrained,
        "compute_dtype_requested": str(requested_dtype),
        "compute_dtype_resolved": str(encoder.dtype),
        "stored_feature_dtype": "torch.float32",
        "decoder_backend_requested": str(requested_backend),
        "decoder_runtime_versions": {
            backend: _decoder_implementation_version(backend)
            for backend in (
                ("decord", "opencv")
                if requested_backend == "auto"
                else (str(requested_backend),)
            )
        },
    }
    return {**payload, "identity_sha256": canonical_hash(payload)}


def feature_artifact_identity(
    *,
    visual_id: str,
    view_content_hash_value: str,
    feature_content_hash_value: str,
    encoder_config: Mapping[str, Any],
    extraction_identity: Mapping[str, Any],
    media_identity: Mapping[str, Any],
    decoded_frame_identity: Mapping[str, Any],
    sampling: Mapping[str, Any],
    feature_tensor_identity: Mapping[str, Any],
) -> str:
    return canonical_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "visual_id": str(visual_id),
            "view_content_hash": str(view_content_hash_value),
            "feature_content_hash": str(feature_content_hash_value),
            "encoder_config": dict(encoder_config),
            "extraction_identity": dict(extraction_identity),
            "media_content_identity": dict(media_identity),
            "decoded_frame_identity": dict(decoded_frame_identity),
            "sampling": dict(sampling),
            "feature_tensor_identity": dict(feature_tensor_identity),
        }
    )


def dotted_get(record: Mapping[str, Any], path: str, default: Any = MISSING) -> Any:
    current: Any = record
    for component in path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            if default is MISSING:
                raise KeyError(path)
            return default
        current = current[component]
    return current


def _normalize_span_container(
    value: Any, default_unit: str
) -> tuple[Sequence[Any], str]:
    if isinstance(value, Mapping) and "spans" in value:
        spans = value["spans"]
        unit = str(value.get("unit", default_unit))
    elif isinstance(value, Mapping):
        spans = [value]
        unit = default_unit
    else:
        spans = value
        unit = default_unit
    if spans is None:
        return (), unit
    if isinstance(spans, (str, bytes)) or not isinstance(spans, Sequence):
        raise EvidenceSpanError(
            f"Evidence field must contain a span mapping or sequence, got {type(spans).__name__}."
        )
    # A bare [start, end] is one span; [[start, end], ...] is a span list.
    if len(spans) == 2 and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in spans
    ):
        return [spans], unit
    return spans, unit


def evidence_for_record(
    record: Mapping[str, Any],
    *,
    evidence_field: str | None,
    default_unit: str,
) -> tuple[Sequence[Any], str, str | None]:
    fields = (evidence_field,) if evidence_field else DEFAULT_EVIDENCE_FIELDS
    for field in fields:
        if not field:
            continue
        value = dotted_get(record, field, default=MISSING)
        if value is MISSING:
            continue
        spans, unit = _normalize_span_container(value, default_unit)
        return spans, unit, field
    return (), default_unit, None


def visual_id_for_record(record: Mapping[str, Any], field: str) -> tuple[str, str]:
    candidates = (field, "visual_id", "video_id", "media_id")
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        value = dotted_get(record, candidate, default=MISSING)
        if value is not MISSING and value not in (None, ""):
            return str(value), candidate
    media_path = record.get("media_path")
    if media_path in (None, ""):
        raise ValueError(
            f"Record {record.get('id')} has neither a visual identity nor media_path."
        )
    return str(media_path), "media_path"


def media_fingerprint(path: Path, *, include_sha256: bool = False) -> dict[str, Any]:
    stat = path.stat()
    value: dict[str, Any] = {
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if include_sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        value["sha256"] = digest.hexdigest()
    return value


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_cached_feature(
    path: Path,
    *,
    expected_visual_id: str,
    expected_feature_hash: str,
    expected_view_hash: str,
    expected_encoder_config: Mapping[str, Any],
    expected_media_fingerprint: Mapping[str, Any],
    expected_extraction_identity: Mapping[str, Any],
    expected_decoded_frame_identity: Mapping[str, Any],
    expected_sampling: Mapping[str, Any],
) -> tuple[list[int], dict[str, Any], dict[str, Any], str] | None:
    if not path.is_file():
        return None
    try:
        loaded = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load cached feature artifact {path}: {exc}"
        ) from exc
    required = {
        "schema_version",
        "feature_content_hash",
        "view_content_hash",
        "encoder_config",
        "media_fingerprint",
        "media_content_identity",
        "extraction_identity",
        "decoded_frame_identity",
        "feature_tensor_identity",
        "feature_artifact_identity_sha256",
        "features",
        "sampling",
    }
    missing = (
        sorted(required - set(loaded))
        if isinstance(loaded, Mapping)
        else sorted(required)
    )
    if missing:
        raise RuntimeError(
            f"Cached feature artifact {path} is missing fields: {missing}"
        )
    checks = {
        "schema_version": (loaded["schema_version"], SCHEMA_VERSION),
        "visual_id": (str(loaded.get("visual_id", "")), str(expected_visual_id)),
        "feature_content_hash": (loaded["feature_content_hash"], expected_feature_hash),
        "view_content_hash": (loaded["view_content_hash"], expected_view_hash),
        "encoder_config": (loaded["encoder_config"], dict(expected_encoder_config)),
        "media_content_identity": (
            loaded["media_content_identity"],
            media_content_identity(expected_media_fingerprint),
        ),
        "extraction_identity": (
            loaded["extraction_identity"],
            dict(expected_extraction_identity),
        ),
        "decoded_frame_identity": (
            loaded["decoded_frame_identity"],
            dict(expected_decoded_frame_identity),
        ),
        "sampling": (loaded["sampling"], dict(expected_sampling)),
    }
    mismatches = [
        name for name, (actual, expected) in checks.items() if actual != expected
    ]
    if mismatches:
        raise RuntimeError(
            f"Cached feature artifact {path} does not match {', '.join(mismatches)}. "
            "Use a new visual_id/configuration, remove the stale artifact, or rerun with --force."
        )
    features = loaded["features"]
    if not torch.is_tensor(features) or features.ndim != 2:
        raise RuntimeError(
            f"Cached artifact {path} has invalid features; expected [tokens, dim] tensor."
        )
    computed_tensor_identity = tensor_identity(features)
    if loaded["feature_tensor_identity"] != computed_tensor_identity:
        raise RuntimeError(
            f"Cached artifact {path} feature tensor digest/metadata does not match its tensor."
        )
    computed_artifact_identity = feature_artifact_identity(
        visual_id=expected_visual_id,
        view_content_hash_value=expected_view_hash,
        feature_content_hash_value=expected_feature_hash,
        encoder_config=expected_encoder_config,
        extraction_identity=expected_extraction_identity,
        media_identity=media_content_identity(expected_media_fingerprint),
        decoded_frame_identity=expected_decoded_frame_identity,
        sampling=expected_sampling,
        feature_tensor_identity=computed_tensor_identity,
    )
    if loaded["feature_artifact_identity_sha256"] != computed_artifact_identity:
        raise RuntimeError(
            f"Cached artifact {path} has an invalid feature_artifact_identity_sha256."
        )
    return (
        list(features.shape),
        dict(loaded["sampling"]),
        computed_tensor_identity,
        computed_artifact_identity,
    )


def _unique_views(values: Sequence[str] | None) -> list[str]:
    requested = list(values or ["full"])
    output: list[str] = []
    for view in requested:
        if view not in output:
            output.append(view)
    return output


def _nested_label(record: Mapping[str, Any], name: str) -> Any:
    value = record.get(name, MISSING)
    if value is not MISSING:
        return value
    diagnostic = record.get("diagnostic")
    if isinstance(diagnostic, Mapping):
        return diagnostic.get(name)
    return None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    index_path = out_dir / "index.jsonl"
    metadata_path = out_dir / "metadata.json"
    resolved_inputs = {
        Path(value).expanduser().resolve()
        for value in (
            args.manifest,
            args.encoder_config,
            args.protocol_config,
            args.conditions_config,
            args.data_lock,
        )
        if value not in (None, "")
    }
    for label, path in (("index", index_path), ("metadata", metadata_path)):
        if path.expanduser().resolve() in resolved_inputs:
            raise ValueError(f"extract {label} output aliases an input: {path}")
        if (path.exists() or path.is_symlink()) and not args.overwrite:
            raise FileExistsError(
                f"extract {label} output exists; pass --overwrite: {path}"
            )
    protocol, protocol_metadata = load_protocol(args.protocol_config)
    require_confirmatory = not args.development
    if require_confirmatory and not args.data_lock:
        raise ValueError("confirmatory extraction requires --data-lock")
    data_protocol = validate_data_protocol(protocol) if require_confirmatory else None
    if require_confirmatory and args.limit is not None:
        raise ValueError("--limit is development-only; add --development explicitly")
    if require_confirmatory and args.allow_missing_media:
        raise ValueError(
            "--allow-missing-media is development-only; add --development explicitly"
        )
    if require_confirmatory and args.view:
        raise ValueError(
            "--view filtering is development-only for trial manifests; confirmatory extraction "
            "must index every visual_id"
        )

    manifest_attestation_sha256: str | None = None
    manifest_data_release_sha256: str | None = None
    trial_matrix_closure: dict[str, Any] | None = None
    data_lock_metadata: dict[str, Any] | None = None
    if require_confirmatory:
        trial_matrix_closure = validate_trial_matrix_closure(
            args.manifest,
            data_lock_path=args.data_lock,
            conditions_config_path=args.conditions_config,
            protocol=protocol,
        )
        manifest_attestation_sha256 = str(
            trial_matrix_closure["trial_build_attestation_sha256"]
        )
        manifest_data_release_sha256 = str(trial_matrix_closure["data_release_sha256"])
        if manifest_data_release_sha256 != data_protocol["data_release_sha256"]:
            raise ValueError(
                "trial-matrix closure data release differs from the locked protocol"
            )
        data_lock_metadata = {
            "path": str(Path(args.data_lock).expanduser().resolve()),
            "file_sha256": sha256_file(args.data_lock),
            "data_release_sha256": manifest_data_release_sha256,
            "manifest_semantic_record_set_sha256": trial_matrix_closure[
                "base_semantic_record_set_sha256"
            ],
            "records": trial_matrix_closure["base_records"],
            "verified_by_trial_matrix_closure": True,
        }
    else:
        attestation_values: set[str] = set()
        release_values: set[str] = set()
        preflight_rows = 0
        for preflight_row in iter_jsonl(args.manifest):
            preflight_rows += 1
            if "visual_spec" in preflight_row:
                attestation = validate_trial_build_attestation(
                    preflight_row,
                    protocol=protocol,
                    require_confirmatory=False,
                )
                attestation_values.add(str(attestation["attestation_sha256"]))
                if attestation.get("data_release_sha256") not in (None, ""):
                    release_values.add(str(attestation["data_release_sha256"]))
        if preflight_rows == 0:
            raise ValueError(f"Manifest is empty: {args.manifest}")
        if len(attestation_values) > 1:
            raise ValueError("trial manifest has mixed build attestations")
        if len(release_values) > 1:
            raise ValueError("trial manifest has mixed data release identities")
        if attestation_values:
            manifest_attestation_sha256 = next(iter(attestation_values))
        if release_values:
            manifest_data_release_sha256 = next(iter(release_values))
    sampling_protocol = protocol_section(protocol, "sampling")
    if "seed" not in sampling_protocol:
        raise ValueError("locked protocol sampling.seed is required")
    require_media_sha256 = sampling_protocol.get("require_media_sha256", False)
    if not isinstance(require_media_sha256, bool):
        raise ValueError(
            "locked protocol sampling.require_media_sha256 must be boolean"
        )
    if require_confirmatory and require_media_sha256 and not args.media_sha256:
        raise ValueError(
            "locked protocol requires content-safe media provenance; rerun extraction with "
            "--media-sha256 (or change the protocol before it is frozen for development only)"
        )
    require_strong_encoder_identity = sampling_protocol.get(
        "require_strong_encoder_identity", False
    )
    if not isinstance(require_strong_encoder_identity, bool):
        raise ValueError(
            "locked protocol sampling.require_strong_encoder_identity must be boolean"
        )
    if args.seed is not None and int(args.seed) != int(sampling_protocol["seed"]):
        raise ValueError(
            "--seed conflicts with locked protocol sampling.seed; update the protocol first"
        )
    resolved_seed = int(
        args.seed if args.seed is not None else sampling_protocol["seed"]
    )
    set_seed(resolved_seed)
    requested_views = _unique_views(args.view) if args.view else None
    base_views = requested_views or ["full"]
    cfg = resolve_encoder(
        args.encoder,
        args.encoder_config,
        overrides={
            "model_id": args.model_id,
            "revision": args.model_revision,
            "num_frames": args.num_frames,
            "max_tokens": args.max_tokens,
        },
    )
    records = iter_jsonl(args.manifest)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive when provided.")
        records = itertools.islice(records, args.limit)
    try:
        first_record = next(records)
    except StopIteration:
        raise ValueError(f"Manifest is empty: {args.manifest}")
    records = itertools.chain([first_record], records)

    feature_dir = out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Keep heavyweight Transformers imports out of module import / --help.  Loading
    # happens once, after cheap argument and manifest validation.
    from vlmevalbench.encoders import FrozenEncoder

    runtime_cfg, encoder_config = prepare_encoder_runtime(cfg)
    encoder = FrozenEncoder(
        runtime_cfg,
        device=args.device,
        dtype=get_dtype(args.dtype),
    )
    pipeline_identity = extraction_pipeline_identity(
        encoder=encoder,
        encoder_config=encoder_config,
        requested_dtype=args.dtype,
        requested_backend=args.backend,
    )
    if (
        require_confirmatory
        and require_strong_encoder_identity
        and pipeline_identity["encoder_pretrained_identity"]["identity_strength"]
        == "weak_mutable_identifier"
    ):
        raise ValueError(
            "locked protocol requires a resolved encoder/processor weight identity; pin an "
            "immutable hub revision in the encoder config or use a content-addressed local snapshot"
        )
    index_rows: list[dict[str, Any]] = []
    visual_sources: dict[str, str] = {}
    indexed_visuals: dict[str, tuple[str, str]] = {}
    failed_visuals: set[str] = set()
    computed = 0
    cache_hits = 0
    skipped_missing = 0
    skipped_decode = 0
    skipped_nonvisual_trials = 0
    skipped_view_filter = 0
    evidence_fields_used: set[str] = set()
    media_fingerprints: dict[str, dict[str, Any]] = {}
    records_considered = 0

    for record in tqdm(
        records,
        total=args.limit,
        desc=f"diagnostic_extract:{args.encoder}",
    ):
        records_considered += 1
        record_id = str(record.get("id") or "").strip()
        if not record_id:
            raise ValueError("Every manifest row must have a non-empty id.")
        is_trial_manifest = "visual_spec" in record
        raw_visual_spec = record.get("visual_spec") if is_trial_manifest else None
        if is_trial_manifest:
            declared_content_hash = str(record.get("trial_content_sha256", ""))
            computed_content_hash = trial_content_sha256(record)
            if (
                declared_content_hash != computed_content_hash
                or str(record.get("trial_id", record.get("id", "")))
                != f"trial::{computed_content_hash}"
            ):
                raise ValueError(
                    f"Trial id={record_id} is not bound to its scoring-relevant content"
                )
        if is_trial_manifest and raw_visual_spec is None:
            skipped_nonvisual_trials += 1
            continue
        if is_trial_manifest and not isinstance(raw_visual_spec, Mapping):
            raise ValueError(f"Trial id={record_id} has a non-object visual_spec.")

        media_type = (
            raw_visual_spec.get("media_type", record.get("media_type", "video"))
            if isinstance(raw_visual_spec, Mapping)
            else record.get("media_type", "video")
        )
        if str(media_type).lower() != "video":
            raise ValueError(
                f"Diagnostic extractor only accepts video rows; id={record_id} has "
                f"media_type={media_type!r}."
            )
        media_value = (
            raw_visual_spec.get("media_path", record.get("media_path"))
            if isinstance(raw_visual_spec, Mapping)
            else record.get("media_path")
        )
        media_path = Path(str(media_value)) if media_value not in (None, "") else None
        if media_path is None or not media_path.is_file():
            if args.allow_missing_media:
                skipped_missing += 1
                continue
            raise FileNotFoundError(f"Missing video for id={record_id}: {media_value}")

        media_key = str(media_path.resolve())
        if media_key not in media_fingerprints:
            media_fingerprints[media_key] = media_fingerprint(
                media_path, include_sha256=args.media_sha256
            )
        fingerprint = media_fingerprints[media_key]
        requests: list[tuple[str, str, str, ViewSpec, str, str | None]] = []
        if isinstance(raw_visual_spec, Mapping):
            # A trial manifest is authoritative: conditions.py already assigned
            # the view-specific visual_id and complete view spec.  Only inject the
            # encoder's fixed frame count.
            view = str(raw_visual_spec.get("view", "")).strip().lower()
            if requested_views is not None and view not in requested_views:
                skipped_view_filter += 1
                continue
            trial_visual_id = record.get("visual_id")
            if trial_visual_id in (None, ""):
                raise ValueError(
                    f"Visual trial id={record_id} must contain a non-empty visual_id."
                )
            raw_num_frames = raw_visual_spec.get("num_frames")
            if raw_num_frames is not None and int(raw_num_frames) != cfg.num_frames:
                raise ValueError(
                    f"Trial id={record_id} visual_spec.num_frames={raw_num_frames} conflicts with "
                    f"encoder num_frames={cfg.num_frames}."
                )
            visual_spans = raw_visual_spec.get("evidence_spans") or ()
            spec = ViewSpec.create(
                view=view,
                num_frames=cfg.num_frames,
                seed=int(raw_visual_spec.get("seed", resolved_seed)),
                evidence_spans=visual_spans,
                default_evidence_unit=args.evidence_unit,
                clip=raw_visual_spec.get("clip"),
            )
            visual_id = str(trial_visual_id)
            requests.append(
                (
                    visual_id,
                    visual_id,
                    "trial.visual_id",
                    spec,
                    view_content_hash(visual_id, spec),
                    "visual_spec.evidence_spans" if visual_spans else None,
                )
            )
        else:
            source_visual_id, visual_id_source = visual_id_for_record(
                record, args.visual_id_field
            )
            spans, evidence_unit, evidence_field = evidence_for_record(
                record,
                evidence_field=args.evidence_field,
                default_unit=args.evidence_unit,
            )
            for view in base_views:
                view_spans = spans if view in EVIDENCE_REQUIRED_VIEWS else ()
                spec = ViewSpec.create(
                    view=view,
                    num_frames=cfg.num_frames,
                    seed=resolved_seed,
                    evidence_spans=view_spans,
                    default_evidence_unit=evidence_unit,
                    clip=(
                        (record.get("diagnostic") or {}).get("media_clip")
                        if isinstance(record.get("diagnostic"), Mapping)
                        else None
                    ),
                )
                view_hash = view_content_hash(source_visual_id, spec)
                # A base visual can yield multiple experimental views.  Expose a
                # view-specific key so FeatureStore still has a one-to-one index.
                visual_id = f"visual::{view_hash[:20]}"
                requests.append(
                    (
                        visual_id,
                        source_visual_id,
                        visual_id_source,
                        spec,
                        view_hash,
                        evidence_field,
                    )
                )

        for (
            visual_id,
            source_visual_id,
            visual_id_source,
            spec,
            view_hash,
            evidence_field,
        ) in requests:
            view = spec.view
            if visual_id in failed_visuals:
                continue
            if evidence_field:
                evidence_fields_used.add(evidence_field)
            prior_path = visual_sources.setdefault(visual_id, str(media_path))
            if prior_path != str(media_path):
                raise ValueError(
                    f"visual_id={visual_id!r} maps to multiple media paths: {prior_path!r} and "
                    f"{str(media_path)!r}. Use distinct visual_id values."
                )
            prior_signature = indexed_visuals.get(visual_id)
            if prior_signature is not None:
                if prior_signature[0] != view_hash:
                    raise ValueError(
                        f"visual_id={visual_id!r} maps to conflicting view specs."
                    )
                continue

            # Decode before looking up the encoder cache.  This makes the cache
            # address depend on the actual backend, timestamp table, selected
            # source frames, and post-decoder RGB bytes instead of merely on a
            # mutable `backend=auto` request.
            try:
                decoded = load_video_view(
                    media_path,
                    spec,
                    visual_id=visual_id,
                    backend=args.backend,
                )
            except (EvidenceSpanError, ViewSamplingError) as exc:
                raise type(exc)(
                    f"id={record_id}, visual_id={visual_id}, view={view}: {exc}"
                ) from exc
            except (VideoDecodeError, OSError) as exc:
                if args.allow_missing_media:
                    skipped_decode += 1
                    failed_visuals.add(visual_id)
                    print(
                        f"Warning: skipped id={record_id} view={view} due to "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                raise

            sampling = decoded.selection.to_dict()
            frame_identity = decoded_frames_identity(decoded.frames)
            resolved_decoder_identity = decoder_identity(
                requested_backend=args.backend,
                sampling=sampling,
            )
            extraction_identity = {
                "pipeline": pipeline_identity,
                "decoder": resolved_decoder_identity,
            }
            extraction_identity["identity_sha256"] = canonical_hash(extraction_identity)
            content_identity = media_content_identity(fingerprint)
            feature_hash = feature_content_hash(
                view_hash,
                encoder_config=encoder_config,
                media_identity=content_identity,
                extraction_identity=extraction_identity,
                decoded_frame_identity=frame_identity,
                sampling_identity=sampling,
            )
            current_signature = (view_hash, feature_hash)
            feature_path = feature_dir / f"{feature_hash}.pt"
            cached = None
            if not args.force:
                cached = _load_cached_feature(
                    feature_path,
                    expected_visual_id=visual_id,
                    expected_feature_hash=feature_hash,
                    expected_view_hash=view_hash,
                    expected_encoder_config=encoder_config,
                    expected_media_fingerprint=fingerprint,
                    expected_extraction_identity=extraction_identity,
                    expected_decoded_frame_identity=frame_identity,
                    expected_sampling=sampling,
                )

            if cached is not None:
                shape, sampling, feature_tensor_identity, artifact_identity = cached
                cache_hits += 1
            else:
                features = encoder.encode_frames(decoded.frames)
                if not torch.is_tensor(features) or features.ndim != 2:
                    raise RuntimeError(
                        f"Encoder {args.encoder} returned {type(features).__name__} with shape "
                        f"{getattr(features, 'shape', None)}; expected [tokens, dim]."
                    )
                features = features.detach().float().cpu().contiguous()
                feature_tensor_identity = tensor_identity(features)
                artifact_identity = feature_artifact_identity(
                    visual_id=visual_id,
                    view_content_hash_value=view_hash,
                    feature_content_hash_value=feature_hash,
                    encoder_config=encoder_config,
                    extraction_identity=extraction_identity,
                    media_identity=content_identity,
                    decoded_frame_identity=frame_identity,
                    sampling=sampling,
                    feature_tensor_identity=feature_tensor_identity,
                )
                artifact = {
                    "schema_version": SCHEMA_VERSION,
                    "visual_id": visual_id,
                    "source_visual_id": source_visual_id,
                    "view": view,
                    "view_spec": spec.to_dict(),
                    "view_content_hash": view_hash,
                    "feature_content_hash": feature_hash,
                    "encoder": args.encoder,
                    "encoder_config": encoder_config,
                    "extraction_identity": extraction_identity,
                    "media_fingerprint": fingerprint,
                    "media_content_identity": content_identity,
                    "decoded_frame_identity": frame_identity,
                    "sampling": sampling,
                    "feature_tensor_identity": feature_tensor_identity,
                    "feature_artifact_identity_sha256": artifact_identity,
                    "features": features,
                }
                _atomic_torch_save(feature_path, artifact)
                shape = list(features.shape)
                computed += 1
            feature_file_sha256 = sha256_file(feature_path)

            base_id = str(record.get("base_id") or record_id)
            index_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "id": visual_id,
                    "first_record_id": record_id,
                    "base_id": base_id,
                    "visual_id": visual_id,
                    "source_visual_id": source_visual_id,
                    "visual_id_source": visual_id_source,
                    "pair_id": _nested_label(record, "pair_id"),
                    "pair_role": _nested_label(record, "pair_role"),
                    "information_family": _nested_label(record, "information_family"),
                    "question_family": _nested_label(record, "question_family"),
                    "source": record.get("source"),
                    "benchmark": record.get("benchmark"),
                    "view": view,
                    "view_spec": spec.to_dict(),
                    "view_content_hash": view_hash,
                    "feature_content_hash": feature_hash,
                    "extraction_identity": extraction_identity,
                    "media_content_identity": content_identity,
                    "media_fingerprint": fingerprint,
                    "decoded_frame_identity": frame_identity,
                    "feature_path": str(feature_path.resolve()),
                    "feature_file_sha256": feature_file_sha256,
                    "feature_tensor_identity": feature_tensor_identity,
                    "feature_artifact_identity_sha256": artifact_identity,
                    "shape": shape,
                    "sampling": sampling,
                }
            )
            indexed_visuals[visual_id] = current_signature

    if not index_rows:
        raise RuntimeError(
            "No diagnostic features were indexed. Check media paths, evidence spans, and decoder logs."
        )
    write_jsonl(index_path, index_rows)
    index_sha256 = sha256_file(index_path)
    artifact_root_sha256 = feature_artifact_root(index_rows)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "execution_mode": "confirmatory" if require_confirmatory else "development",
        "data_release_sha256": manifest_data_release_sha256,
        "trial_build_attestation_sha256": manifest_attestation_sha256,
        "trial_matrix_closure": trial_matrix_closure,
        "trial_matrix_closure_sha256": (
            trial_matrix_closure.get("closure_sha256")
            if trial_matrix_closure is not None
            else None
        ),
        "trial_set_identity": (
            {
                "schema_version": "information_upper_bound.trial_set.v1",
                "trial_count": trial_matrix_closure["trial_count"],
                "root_sha256": trial_matrix_closure["trial_set_root_sha256"],
            }
            if trial_matrix_closure is not None
            else None
        ),
        "data_lock": data_lock_metadata,
        "encoder": args.encoder,
        "encoder_config": encoder_config,
        "extraction_pipeline_identity": pipeline_identity,
        "extraction_pipeline_identity_sha256": pipeline_identity["identity_sha256"],
        "compute_dtype": args.dtype,
        "views_requested": requested_views,
        "views_indexed": sorted({row["view"] for row in index_rows}),
        "seed": resolved_seed,
        "protocol_config": protocol_metadata,
        "conditions_config": {
            "path": str(Path(args.conditions_config).expanduser().resolve()),
            "sha256": sha256_file(args.conditions_config),
        },
        "backend": args.backend,
        "resolved_decoder_backends": sorted(
            {
                row["extraction_identity"]["decoder"]["actual_backend"]
                for row in index_rows
            }
        ),
        "media_sha256_enabled": args.media_sha256,
        "media_sha256_required_by_protocol": require_media_sha256,
        "media_sha256_enforced": bool(require_confirmatory and require_media_sha256),
        "strong_encoder_identity_required_by_protocol": require_strong_encoder_identity,
        "strong_encoder_identity_enforced": bool(
            require_confirmatory and require_strong_encoder_identity
        ),
        "unique_media_fingerprinted": len(media_fingerprints),
        "evidence_field_requested": args.evidence_field,
        "evidence_fields_used": sorted(evidence_fields_used),
        "default_evidence_unit": args.evidence_unit,
        "visual_id_field": args.visual_id_field,
        "manifest_records_considered": records_considered,
        "indexed_trials": len(index_rows),
        "unique_feature_artifacts": len(
            {row["feature_content_hash"] for row in index_rows}
        ),
        "features_computed": computed,
        "cache_hits": cache_hits,
        "skipped_missing_media_records": skipped_missing,
        "skipped_decode_trials": skipped_decode,
        "skipped_decode_visual_ids": sorted(failed_visuals),
        "skipped_nonvisual_trials": skipped_nonvisual_trials,
        "skipped_by_view_filter": skipped_view_filter,
        "index_path": str(index_path.resolve()),
        "index_sha256": index_sha256,
        "feature_artifact_root_sha256": artifact_root_sha256,
    }
    write_json(metadata_path, metadata)
    print(
        f"Wrote {len(index_rows)} diagnostic feature index rows to {out_dir / 'index.jsonl'}"
    )
    print(
        f"Unique artifacts={metadata['unique_feature_artifacts']}, "
        f"computed={computed}, cache_hits={cache_hits}"
    )


if __name__ == "__main__":
    main()
