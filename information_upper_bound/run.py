from __future__ import annotations

import argparse
from collections import OrderedDict
import itertools
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

import torch
from tqdm import tqdm

from .attestation import validate_trial_build_attestation
from .conditions import trial_content_sha256
from .integrity import (
    RESULT_DESIGN_FIELDS,
    RESULT_INTEGRITY_SCHEMA_VERSION,
    canonical_sha256,
    feature_artifact_root,
    scored_result_sha256,
    tensor_identity,
    trial_set_identity,
)
from .io import iter_jsonl, sha256_file, write_json
from .protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
    protocol_section,
    validate_data_protocol,
    validate_frozen_model_protocol,
    validate_locked_projector_protocol,
)
from .schema import diagnostic_metadata
from .score_partition import (
    SCORING_PARTITION_ALGORITHM,
    SCORING_PARTITION_SCHEMA_VERSION,
    score_worker_index as _score_worker_index,
    validate_score_worker as _validate_score_worker,
)
from .scoring import SCORING_PROTOCOL_VERSION, FrozenMultipleChoiceScorer


class FeatureStore:
    _ARTIFACT_MATCH_FIELDS = (
        "schema_version",
        "visual_id",
        "view_content_hash",
        "feature_content_hash",
        "encoder_config",
        "extraction_identity",
        "media_content_identity",
        "decoded_frame_identity",
        "sampling",
        "feature_tensor_identity",
        "feature_artifact_identity_sha256",
    )

    def __init__(
        self,
        index_path: str | Path,
        *,
        cache_size: int = 16,
        metadata: Mapping[str, Any] | None = None,
        verify_all_files: bool = True,
    ) -> None:
        self.index_path = Path(index_path).resolve()
        self.cache_size = max(int(cache_size), 0)
        self.paths: dict[str, Path] = {}
        self.rows: dict[str, dict[str, Any]] = {}
        index_rows = list(iter_jsonl(self.index_path))
        for row in index_rows:
            key = str(row.get("visual_id", row.get("id", "")))
            if not key:
                raise ValueError(f"feature index row has no visual_id/id: {row}")
            if key in self.paths:
                raise ValueError(f"duplicate feature key in index: {key}")
            missing = sorted(
                (
                    set(self._ARTIFACT_MATCH_FIELDS)
                    | {"feature_path", "feature_file_sha256", "shape"}
                )
                - set(row)
            )
            if missing:
                raise ValueError(
                    f"feature index row visual_id={key!r} is missing integrity fields: {missing}"
                )
            path = Path(str(row["feature_path"]))
            if not path.is_absolute():
                path = (self.index_path.parent / path).resolve()
            self.paths[key] = path
            self.rows[key] = dict(row)
        if not self.paths:
            raise ValueError(f"empty feature index: {self.index_path}")
        self.artifact_root_sha256 = feature_artifact_root(index_rows)
        self.index_sha256 = sha256_file(self.index_path)
        if metadata is not None:
            expected_index = str(metadata.get("index_sha256", ""))
            expected_root = str(metadata.get("feature_artifact_root_sha256", ""))
            if not expected_index or not expected_root:
                raise ValueError(
                    "feature metadata must contain index_sha256 and "
                    "feature_artifact_root_sha256"
                )
            if expected_index != self.index_sha256:
                raise ValueError(
                    "feature metadata index_sha256 does not match the feature index"
                )
            if expected_root != self.artifact_root_sha256:
                raise ValueError(
                    "feature metadata feature_artifact_root_sha256 does not match the index"
                )
            expected_pipeline = metadata.get("extraction_pipeline_identity")
            expected_pipeline_sha = str(
                metadata.get("extraction_pipeline_identity_sha256", "")
            )
            if not isinstance(expected_pipeline, Mapping) or not expected_pipeline_sha:
                raise ValueError(
                    "feature metadata must contain extraction_pipeline_identity and its SHA256"
                )
            if (
                str(expected_pipeline.get("identity_sha256", ""))
                != expected_pipeline_sha
            ):
                raise ValueError(
                    "feature metadata extraction pipeline identity is inconsistent"
                )
            for key, row in self.rows.items():
                extraction = row.get("extraction_identity")
                if (
                    not isinstance(extraction, Mapping)
                    or extraction.get("pipeline") != expected_pipeline
                ):
                    raise ValueError(
                        f"feature index pipeline identity does not match metadata for {key}"
                    )
                if metadata.get("media_sha256_enabled") is True:
                    media_identity = row.get("media_content_identity")
                    digest = (
                        str(media_identity.get("sha256", ""))
                        if isinstance(media_identity, Mapping)
                        else ""
                    )
                    if len(digest) != 64 or any(
                        character not in "0123456789abcdef"
                        for character in digest.lower()
                    ):
                        raise ValueError(
                            f"feature index media identity is not SHA256-authenticated for {key}"
                        )
        if verify_all_files:
            for key in sorted(self.paths):
                self._verify_file_digest(key)
        self._cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    def _verify_file_digest(self, visual_id: str) -> None:
        path = self.paths[visual_id]
        if not path.is_file():
            raise FileNotFoundError(
                f"feature file does not exist for {visual_id}: {path}"
            )
        expected = str(self.rows[visual_id]["feature_file_sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"feature file digest mismatch for {visual_id}: expected={expected}, actual={actual}"
            )

    @staticmethod
    def _artifact_identity(payload: Mapping[str, Any]) -> str:
        return canonical_sha256(
            {
                "schema_version": payload["schema_version"],
                "visual_id": str(payload["visual_id"]),
                "view_content_hash": str(payload["view_content_hash"]),
                "feature_content_hash": str(payload["feature_content_hash"]),
                "encoder_config": dict(payload["encoder_config"]),
                "extraction_identity": dict(payload["extraction_identity"]),
                "media_content_identity": dict(payload["media_content_identity"]),
                "decoded_frame_identity": dict(payload["decoded_frame_identity"]),
                "sampling": dict(payload["sampling"]),
                "feature_tensor_identity": dict(payload["feature_tensor_identity"]),
            }
        )

    def load(self, visual_id: str) -> torch.Tensor:
        if visual_id in self._cache:
            value = self._cache.pop(visual_id)
            self._cache[visual_id] = value
            return value
        path = self.paths.get(visual_id)
        if path is None:
            raise KeyError(f"visual_id {visual_id!r} not found in feature index")
        self._verify_file_digest(visual_id)
        payload = torch.load(path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError(f"feature file {path} must contain a metadata mapping")
        expected = self.rows[visual_id]
        missing = sorted(
            (set(self._ARTIFACT_MATCH_FIELDS) | {"features"}) - set(payload)
        )
        if missing:
            raise ValueError(
                f"feature file {path} is missing integrity fields: {missing}"
            )
        mismatches = [
            field
            for field in self._ARTIFACT_MATCH_FIELDS
            if payload[field] != expected[field]
        ]
        if mismatches:
            raise ValueError(
                f"feature index/artifact metadata mismatch for {visual_id}: {mismatches}"
            )
        features = payload["features"]
        if not torch.is_tensor(features) or features.ndim != 2:
            raise ValueError(f"feature file {path} must contain tensor [N,D]")
        actual_tensor_identity = tensor_identity(features)
        if actual_tensor_identity != expected["feature_tensor_identity"]:
            raise ValueError(f"feature tensor digest mismatch for {visual_id}")
        if list(features.shape) != list(expected["shape"]):
            raise ValueError(
                f"feature tensor shape does not match index for {visual_id}"
            )
        actual_artifact_identity = self._artifact_identity(payload)
        if actual_artifact_identity != str(
            expected["feature_artifact_identity_sha256"]
        ):
            raise ValueError(f"feature artifact identity is invalid for {visual_id}")
        features = features.detach().float().contiguous()
        if self.cache_size:
            self._cache[visual_id] = features
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return features


def _load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _validate_score_output_paths(
    *,
    output_path: str | Path,
    metadata_path: str | Path,
    trials_path: str | Path,
    projector_checkpoint: str | Path,
    projector_metadata_path: str | Path,
    feature_index_path: str | Path | None,
    feature_metadata_path: str | Path | None,
    protocol_config_path: str | Path | None,
) -> None:
    """Refuse output aliases before overwrite can remove an input artifact."""

    outputs = {
        "score output": Path(output_path).resolve(),
        "score metadata sidecar": Path(metadata_path).resolve(),
    }
    if outputs["score output"] == outputs["score metadata sidecar"]:
        raise ValueError("score output and metadata sidecar resolve to the same path")
    raw_inputs = {
        "trial manifest": trials_path,
        "projector checkpoint": projector_checkpoint,
        "projector metadata": projector_metadata_path,
        "feature index": feature_index_path,
        "feature metadata": feature_metadata_path,
        "protocol config": protocol_config_path,
    }
    inputs = {
        name: Path(value).resolve()
        for name, value in raw_inputs.items()
        if value is not None
    }
    collisions = [
        (output_name, input_name, str(output))
        for output_name, output in outputs.items()
        for input_name, input_path in inputs.items()
        if output == input_path
    ]
    if collisions:
        detail = "; ".join(
            f"{output_name} aliases {input_name}: {path}"
            for output_name, input_name, path in collisions
        )
        raise ValueError(f"unsafe score output path collision: {detail}")


def _result_design_from_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    """Construct the exact flattened design fields copied into a result row."""

    trial_id = str(trial.get("trial_id", trial.get("id", "")))
    condition = trial.get("condition") or {}
    if not isinstance(condition, Mapping):
        raise ValueError(f"trial {trial_id!r} condition must be a mapping")
    row = {
        "trial_id": trial_id,
        "base_id": str(trial.get("base_id", "")),
        "trial_content_sha256": trial.get("trial_content_sha256"),
        "data_release_sha256": trial.get("data_release_sha256"),
        "trial_build_attestation_sha256": (
            (trial.get("trial_build_attestation") or {}).get("attestation_sha256")
            if isinstance(trial.get("trial_build_attestation"), Mapping)
            else None
        ),
        "visual_id": trial.get("visual_id"),
        **diagnostic_metadata(trial),
        "condition": str(condition.get("name", "")),
        "input_channel": condition.get("input_channel"),
        "visual_view": condition.get("visual_view"),
        "requested_dose": condition.get("requested_dose"),
        "effective_dose": condition.get("effective_dose"),
        "permutation_index": condition.get("permutation_index", 0),
        "seed": condition.get("seed"),
        "choices": trial.get("choices"),
        "answer": trial.get("answer"),
        "answer_text": trial.get("answer_text"),
    }
    return {name: row.get(name) for name in RESULT_DESIGN_FIELDS}


def _validate_existing_output(
    path: Path,
    *,
    expected_designs: Mapping[str, Mapping[str, Any]],
    run_signature_sha256: str,
    global_signature_sha256: str,
) -> set[str]:
    """Authenticate every durable row before resume is allowed to skip it."""

    completed: set[str] = set()
    for row_index, row in enumerate(iter_jsonl(path)):
        trial_id = str(row.get("trial_id", row.get("id", "")))
        if not trial_id or trial_id in completed:
            raise ValueError(
                f"cannot resume: existing output has empty/duplicate trial ID at row "
                f"{row_index}: {trial_id!r}"
            )
        expected = expected_designs.get(trial_id)
        if expected is None:
            raise ValueError(
                f"cannot resume: existing row {trial_id!r} is outside the current manifest"
            )
        actual_design = {name: row.get(name) for name in RESULT_DESIGN_FIELDS}
        if canonical_sha256(actual_design) != canonical_sha256(dict(expected)):
            mismatches = [
                name
                for name in RESULT_DESIGN_FIELDS
                if canonical_sha256(actual_design.get(name))
                != canonical_sha256(expected.get(name))
            ]
            raise ValueError(
                f"cannot resume: existing row {trial_id!r} differs from the current "
                f"trial manifest in design fields {mismatches}"
            )
        actual_run = str(row.get("scoring_run_signature_sha256", ""))
        if actual_run != run_signature_sha256:
            raise ValueError(
                f"cannot resume: existing row {trial_id!r} has scoring run signature "
                f"{actual_run!r}, expected {run_signature_sha256!r}"
            )
        actual_global = str(row.get("scoring_global_signature_sha256", ""))
        if actual_global != global_signature_sha256:
            raise ValueError(
                f"cannot resume: existing row {trial_id!r} has global experiment signature "
                f"{actual_global!r}, expected {global_signature_sha256!r}"
            )
        declared_digest = str(row.get("result_content_sha256", ""))
        recomputed_digest = scored_result_sha256(row)
        if declared_digest != recomputed_digest:
            raise ValueError(
                f"cannot resume: existing row {trial_id!r} has invalid result digest "
                f"{declared_digest!r}; recomputed {recomputed_digest!r}"
            )
        completed.add(trial_id)
    return completed


def _validate_existing_output_signature(path: Path, expected: str) -> None:
    """Compatibility helper retained for callers that only audit one field."""

    for row in iter_jsonl(path):
        actual = str(row.get("scoring_run_signature_sha256", ""))
        if actual != expected:
            trial_id = str(row.get("trial_id", row.get("id", "")))
            raise ValueError(
                f"cannot resume: existing row {trial_id!r} has scoring signature "
                f"{actual!r}, expected {expected!r}"
            )


def _trial_rows(path: Path, limit: int | None) -> Iterator[dict[str, Any]]:
    rows = iter_jsonl(path)
    return itertools.islice(rows, limit) if limit is not None else rows


def _selected_trial_rows(
    path: Path,
    limit: int | None,
    *,
    selected_trial_ids: set[str],
) -> Iterator[dict[str, Any]]:
    """Read the full manifest representation and yield one authenticated subset."""

    for row in _trial_rows(path, limit):
        trial_id = str(row.get("trial_id", row.get("id", "")))
        if trial_id in selected_trial_ids:
            yield row


def _validate_provenance(
    *,
    projector_metadata: Mapping[str, Any],
    feature_metadata: Mapping[str, Any] | None,
    llm_id: str | None,
) -> None:
    if (
        llm_id
        and projector_metadata.get("llm_id")
        and llm_id != projector_metadata["llm_id"]
    ):
        raise ValueError(
            f"--llm-id {llm_id!r} differs from projector llm_id={projector_metadata['llm_id']!r}"
        )
    if not feature_metadata:
        return
    projector_encoder = projector_metadata.get("encoder_name")
    feature_encoder = feature_metadata.get("encoder")
    if (
        projector_encoder
        and feature_encoder
        and str(projector_encoder) != str(feature_encoder)
    ):
        raise ValueError(
            f"projector encoder={projector_encoder!r} does not match feature encoder={feature_encoder!r}"
        )
    projector_pipeline = str(
        projector_metadata.get("encoder_extraction_pipeline_identity_sha256", "")
    )
    feature_pipeline = str(
        feature_metadata.get("extraction_pipeline_identity_sha256", "")
    )
    if not projector_pipeline or not feature_pipeline:
        raise ValueError(
            "confirmatory visual scoring requires the projector and feature metadata to bind "
            "encoder_extraction_pipeline_identity_sha256"
        )
    if projector_pipeline != feature_pipeline:
        raise ValueError(
            "projector training encoder/preprocessor identity does not match evaluation features"
        )


def _validate_evaluation_feature_lock(
    *,
    locked_projector: Mapping[str, Any],
    feature_store: FeatureStore,
    feature_metadata: Mapping[str, Any],
    feature_metadata_path: str | Path,
    trials_manifest_sha256: str,
) -> None:
    """Bind evaluation tensors and their extraction manifest to the final protocol."""

    actual = {
        "evaluation_feature_index_sha256": feature_store.index_sha256,
        "evaluation_feature_metadata_sha256": sha256_file(feature_metadata_path),
        "evaluation_feature_artifact_root_sha256": (feature_store.artifact_root_sha256),
    }
    mismatches = [
        name
        for name, digest in actual.items()
        if str(locked_projector.get(name, "")) != digest
    ]
    if mismatches:
        raise ValueError(
            "evaluation features do not match the projector/protocol lock: "
            + ", ".join(mismatches)
        )
    # Raw JSONL bytes are representation audit only: row order and remounted
    # media paths may change without changing any trial ID or the locked matrix
    # closure checked below.
    closure = feature_metadata.get("trial_matrix_closure")
    if not isinstance(closure, Mapping):
        raise ValueError(
            "evaluation feature metadata has no exact trial-matrix closure"
        )
    closure_payload = dict(closure)
    declared_closure_sha256 = str(closure_payload.pop("closure_sha256", "")).lower()
    expected_closure_sha256 = str(
        locked_projector.get("evaluation_trial_matrix_closure_sha256", "")
    ).lower()
    expected_trial_root = str(
        locked_projector.get("evaluation_trial_set_root_sha256", "")
    ).lower()
    expected_trial_count = locked_projector.get("evaluation_trial_count")
    if (
        closure_payload.get("status") != "exact"
        or canonical_sha256(closure_payload) != declared_closure_sha256
        or declared_closure_sha256 != expected_closure_sha256
        or str(feature_metadata.get("trial_matrix_closure_sha256", "")).lower()
        != expected_closure_sha256
        or str(closure_payload.get("trial_set_root_sha256", "")).lower()
        != expected_trial_root
        or closure_payload.get("trial_count") != expected_trial_count
    ):
        raise ValueError(
            "evaluation feature metadata trial-matrix closure differs from the "
            "projector/protocol lock"
        )


def run_trials(
    *,
    trials_path: str | Path,
    output_path: str | Path,
    projector_checkpoint: str | Path,
    projector_metadata_path: str | Path,
    feature_index_path: str | Path | None,
    feature_metadata_path: str | Path | None = None,
    llm_id: str | None = None,
    llm_revision: str | None = None,
    device: str = "cuda",
    dtype: str = "bf16",
    max_length: int = 4096,
    overflow_policy: str = "error",
    resume: bool = False,
    overwrite: bool = False,
    continue_on_error: bool = False,
    limit: int | None = None,
    feature_cache_size: int = 16,
    protocol_config_path: str | Path | None = None,
    worker_count: int = 1,
    worker_index: int = 0,
) -> dict[str, Any]:
    trials_file = Path(trials_path).resolve()
    output_file = Path(output_path).resolve()
    _validate_score_worker(worker_count=worker_count, worker_index=worker_index)
    if limit is not None and worker_count != 1:
        raise ValueError(
            "--limit cannot be combined with --worker-count greater than 1"
        )
    if output_file.suffix.casefold() == ".gz":
        raise ValueError(
            "score output must be an uncompressed JSONL for durable per-row fsync/resume; "
            "compress the completed file afterward if needed"
        )
    metadata_file = output_file.with_suffix(output_file.suffix + ".metadata.json")
    _validate_score_output_paths(
        output_path=output_file,
        metadata_path=metadata_file,
        trials_path=trials_file,
        projector_checkpoint=projector_checkpoint,
        projector_metadata_path=projector_metadata_path,
        feature_index_path=feature_index_path,
        feature_metadata_path=feature_metadata_path,
        protocol_config_path=protocol_config_path,
    )
    existing_artifacts = [
        path for path in (output_file, metadata_file) if path.exists()
    ]
    if existing_artifacts and not resume and not overwrite:
        raise FileExistsError(
            "run artifacts exist; pass --resume or --overwrite: "
            + ", ".join(str(path) for path in existing_artifacts)
        )
    if overwrite and output_file.exists():
        output_file.unlink()
    if overwrite and metadata_file.exists():
        metadata_file.unlink()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if limit is not None and limit <= 0:
        raise ValueError("--limit must be positive when provided")
    locked_protocol: Mapping[str, Any] | None = None
    locked_projector: Mapping[str, Any] | None = None
    locked_data_release_sha256: str | None = None
    manifest_representation_matches_projector: bool | None = None
    protocol_config_sha256 = (
        sha256_file(protocol_config_path) if protocol_config_path else None
    )
    if protocol_config_path is not None:
        locked_protocol, protocol_metadata = load_protocol(protocol_config_path)
        protocol_config_sha256 = str(protocol_metadata["sha256"])
        locked_data_release_sha256 = str(
            validate_data_protocol(locked_protocol)["data_release_sha256"]
        )
        if limit is not None:
            raise ValueError(
                "--limit is a development convenience and is forbidden for a locked "
                "confirmatory score run"
            )
        locked_sampling = protocol_section(locked_protocol, "sampling")
        if int(locked_sampling.get("trial_shards", 1)) != 1:
            raise ValueError(
                "locked confirmatory scoring currently requires sampling.trial_shards: 1: "
                "the projector lock authenticates one complete trial-matrix "
                "closure/root/count. Score that full matrix rather than independently "
                "built shards."
            )
    trials_manifest_sha256 = sha256_file(trials_file)
    trial_ids: set[str] = set()
    expected_designs: dict[str, dict[str, Any]] = {}
    trial_identity_rows: list[dict[str, str]] = []
    selected_trial_ids: set[str] = set()
    selected_expected_designs: dict[str, dict[str, Any]] = {}
    selected_trial_identity_rows: list[dict[str, str]] = []
    visual_ids: set[str] = set()
    num_trials = 0
    num_visual_trials = 0
    selected_num_visual_trials = 0
    trial_build_attestation_sha256_values: set[str] = set()
    for row in _trial_rows(trials_file, limit):
        trial_id = str(row.get("trial_id", row.get("id", "")))
        if not trial_id:
            raise ValueError("trial manifest contains an empty trial_id")
        if trial_id in trial_ids:
            raise ValueError(f"trial manifest contains duplicate trial_id: {trial_id}")
        declared_content_hash = str(row.get("trial_content_sha256", ""))
        computed_content_hash = trial_content_sha256(row)
        if declared_content_hash != computed_content_hash:
            raise ValueError(
                f"trial {trial_id} has stale/invalid trial_content_sha256: "
                f"declared={declared_content_hash!r}, computed={computed_content_hash!r}"
            )
        if trial_id != f"trial::{computed_content_hash}":
            raise ValueError(
                f"trial_id is not bound to trial_content_sha256 for {trial_id}"
            )
        if locked_protocol is not None:
            assert protocol_config_sha256 is not None
            try:
                attestation = validate_trial_build_attestation(
                    row,
                    protocol=locked_protocol,
                    require_confirmatory=True,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"trial {trial_id!r} has an invalid confirmatory build attestation: {exc}"
                ) from exc
            trial_build_attestation_sha256_values.add(
                str(attestation["attestation_sha256"])
            )
        design = _result_design_from_trial(row)
        identity_row = {
            "trial_id": trial_id,
            "trial_content_sha256": computed_content_hash,
        }
        trial_ids.add(trial_id)
        expected_designs[trial_id] = design
        trial_identity_rows.append(identity_row)
        num_trials += 1
        visual_id = row.get("visual_id")
        if visual_id not in (None, ""):
            visual_ids.add(str(visual_id))
            num_visual_trials += 1
        if (
            _score_worker_index(computed_content_hash, worker_count=worker_count)
            == worker_index
        ):
            selected_trial_ids.add(trial_id)
            selected_expected_designs[trial_id] = design
            selected_trial_identity_rows.append(identity_row)
            if visual_id not in (None, ""):
                selected_num_visual_trials += 1
    if not num_trials:
        raise ValueError(f"trial manifest is empty: {trials_file}")
    if locked_protocol is not None and len(trial_build_attestation_sha256_values) != 1:
        raise ValueError(
            "locked score manifest must carry one shared authenticated trial-build "
            f"attestation, found {len(trial_build_attestation_sha256_values)}"
        )
    trial_build_attestation_sha256 = (
        next(iter(trial_build_attestation_sha256_values))
        if trial_build_attestation_sha256_values
        else None
    )
    manifest_trial_set = trial_set_identity(trial_identity_rows)

    projector_metadata = _load_json(projector_metadata_path)
    feature_metadata = (
        _load_json(feature_metadata_path) if feature_metadata_path else None
    )
    projector_checkpoint_sha256 = sha256_file(projector_checkpoint)
    projector_metadata_sha256 = sha256_file(projector_metadata_path)

    if protocol_config_path is not None:
        assert locked_protocol is not None
        mismatched_data_locks = sorted(
            trial_id
            for trial_id, design in expected_designs.items()
            if str(design.get("data_release_sha256", "")) != locked_data_release_sha256
        )
        if mismatched_data_locks:
            raise ValueError(
                "trial manifest data_release_sha256 differs from locked protocol "
                f"data.data_release_sha256 for {len(mismatched_data_locks)} trials; "
                f"first IDs: {mismatched_data_locks[:5]}"
            )
        locked_model = validate_frozen_model_protocol(locked_protocol)
        expected_revision = locked_model.get("llm_revision")
        if str(llm_id or locked_model["llm_id"]) != str(locked_model["llm_id"]):
            raise ValueError("llm_id differs from the locked protocol")
        if (llm_revision or expected_revision) != expected_revision:
            raise ValueError("llm_revision differs from the locked protocol")
        if int(max_length) != int(locked_model["max_length"]):
            raise ValueError("max_length differs from the locked protocol")
        if str(dtype) != str(locked_model["dtype"]):
            raise ValueError("dtype differs from the locked protocol")
        if str(overflow_policy) != str(locked_model["overflow_policy"]):
            raise ValueError("overflow_policy differs from the locked protocol")
        locked_projector = validate_locked_projector_protocol(
            locked_protocol,
            checkpoint_sha256=projector_checkpoint_sha256,
            metadata_sha256=projector_metadata_sha256,
            projector_metadata=projector_metadata,
        )
        declared_evaluation_manifest = str(
            projector_metadata.get("evaluation_manifest_sha256", "")
        ).lower()
        manifest_representation_matches_projector = (
            declared_evaluation_manifest == trials_manifest_sha256
        )
        if (
            manifest_trial_set["root_sha256"]
            != locked_projector["evaluation_trial_set_root_sha256"]
            or manifest_trial_set["trial_count"]
            != locked_projector["evaluation_trial_count"]
        ):
            raise ValueError(
                "trial manifest is not the complete evaluation matrix locked by the "
                "projector/protocol"
            )

    _validate_provenance(
        projector_metadata=projector_metadata,
        feature_metadata=feature_metadata,
        llm_id=llm_id,
    )

    feature_store: FeatureStore | None = None
    if visual_ids:
        if feature_index_path is None:
            raise ValueError("visual trials exist, so --feature-index is required")
        if feature_metadata_path is None:
            raise ValueError(
                "visual trials exist, so --feature-metadata is required for encoder provenance"
            )
        assert feature_metadata is not None
        if locked_protocol is not None:
            sampling_protocol = protocol_section(locked_protocol, "sampling")
            if sampling_protocol.get("require_media_sha256") is True and not bool(
                feature_metadata.get("media_sha256_enabled")
                if feature_metadata
                else False
            ):
                raise ValueError(
                    "locked protocol requires SHA256 media provenance, but feature metadata "
                    "does not authenticate source media bytes"
                )
            if str(feature_metadata.get("data_release_sha256", "")) != str(
                locked_data_release_sha256
            ):
                raise ValueError(
                    "feature metadata data_release_sha256 differs from the authenticated "
                    "trial manifest and locked protocol"
                )
            if str(feature_metadata.get("trial_build_attestation_sha256", "")) != str(
                trial_build_attestation_sha256
            ):
                raise ValueError(
                    "feature metadata trial_build_attestation_sha256 differs from the "
                    "authenticated trial manifest"
                )
        feature_store = FeatureStore(
            feature_index_path,
            cache_size=feature_cache_size,
            metadata=feature_metadata,
            verify_all_files=True,
        )
        if locked_projector is not None:
            _validate_evaluation_feature_lock(
                locked_projector=locked_projector,
                feature_store=feature_store,
                feature_metadata=feature_metadata,
                feature_metadata_path=feature_metadata_path,
                trials_manifest_sha256=trials_manifest_sha256,
            )
        missing = sorted(visual_ids - set(feature_store.paths))
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(
                f"feature index is missing {len(missing)} visual inputs: {preview}"
            )

    if not selected_trial_identity_rows:
        raise ValueError(
            f"score worker {worker_index} of {worker_count} has no assigned trials"
        )
    selected_trial_set = trial_set_identity(selected_trial_identity_rows)
    selected_num_trials = len(selected_trial_identity_rows)

    scorer = FrozenMultipleChoiceScorer(
        projector_checkpoint=projector_checkpoint,
        projector_metadata=projector_metadata,
        llm_id=llm_id,
        llm_revision=llm_revision,
        device=device,
        dtype=dtype,
        max_length=max_length,
        overflow_policy=overflow_policy,
    )
    trained_llm_identity = str(
        projector_metadata.get("llm_pretrained_identity_sha256", "")
    )
    if trained_llm_identity and trained_llm_identity != str(
        scorer.pretrained_identity["identity_sha256"]
    ):
        training_identity = projector_metadata.get("llm_pretrained_identity") or {}
        training_source = (
            training_identity.get("source", {})
            if isinstance(training_identity, Mapping)
            else {}
        )
        scoring_source = scorer.pretrained_identity.get("source", {})
        raise ValueError(
            "projector training LLM/tokenizer identity does not match the scoring LLM; "
            f"expected_sha256={trained_llm_identity}, "
            f"actual_sha256={scorer.pretrained_identity['identity_sha256']}, "
            f"training_source_kind={training_source.get('kind')!r}, "
            f"scoring_source_kind={scoring_source.get('kind')!r}. "
            "A different VLMEB_LOCAL_FILES_ONLY mode can represent the same pinned Hub "
            "commit through a different provenance route. Reproduce the training mode; "
            "do not edit the projector metadata or protocol lock."
        )

    resolved_llm_id = str(llm_id or projector_metadata["llm_id"])
    global_signature = {
        "schema_version": "information_upper_bound.scoring_global_signature.v2",
        "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
        "protocol_config_sha256": protocol_config_sha256,
        "data_release_sha256": locked_data_release_sha256,
        "trial_build_attestation_sha256": trial_build_attestation_sha256,
        "trial_matrix_closure_sha256": (
            locked_projector.get("evaluation_trial_matrix_closure_sha256")
            if locked_projector is not None
            else None
        ),
        "full_trial_set_root_sha256": (
            locked_projector.get("evaluation_trial_set_root_sha256")
            if locked_projector is not None
            else manifest_trial_set["root_sha256"]
        ),
        "full_trial_count": (
            locked_projector.get("evaluation_trial_count")
            if locked_projector is not None
            else manifest_trial_set["trial_count"]
        ),
        "projector_checkpoint_sha256": projector_checkpoint_sha256,
        "projector_metadata_sha256": projector_metadata_sha256,
        "encoder_extraction_pipeline_identity_sha256": (
            feature_metadata.get("extraction_pipeline_identity_sha256")
            if feature_metadata
            else projector_metadata.get("encoder_extraction_pipeline_identity_sha256")
        ),
        "media_sha256_required": bool(
            protocol_section(locked_protocol, "sampling").get(
                "require_media_sha256", False
            )
            if locked_protocol is not None
            else feature_metadata.get("media_sha256_enabled")
            if feature_metadata
            else False
        ),
        "llm_id": resolved_llm_id,
        "llm_revision_requested": llm_revision,
        "llm_pretrained_identity": scorer.pretrained_identity,
        "dtype": dtype,
        "max_length": int(max_length),
        "overflow_policy": overflow_policy,
    }
    global_signature_sha256 = canonical_sha256(global_signature)
    run_signature: dict[str, Any] = {
        "schema_version": "information_upper_bound.scoring_run_signature.v2",
        "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
        "scoring_global_signature_sha256": global_signature_sha256,
        "protocol_config_sha256": protocol_config_sha256,
        "trials_manifest_sha256": trials_manifest_sha256,
        "manifest_representation_matches_projector": (
            manifest_representation_matches_projector
        ),
        "trial_set_identity": selected_trial_set,
        "data_release_sha256": locked_data_release_sha256,
        "trial_build_attestation_sha256": trial_build_attestation_sha256,
        "trial_matrix_closure_sha256": global_signature["trial_matrix_closure_sha256"],
        "full_trial_set_root_sha256": global_signature["full_trial_set_root_sha256"],
        "full_trial_count": global_signature["full_trial_count"],
        "projector_checkpoint_sha256": projector_checkpoint_sha256,
        "projector_metadata_sha256": projector_metadata_sha256,
        "feature_index_sha256": feature_store.index_sha256 if feature_store else None,
        "feature_metadata_sha256": (
            sha256_file(feature_metadata_path) if feature_metadata_path else None
        ),
        "feature_artifact_root_sha256": (
            feature_store.artifact_root_sha256 if feature_store else None
        ),
        "llm_id": resolved_llm_id,
        "llm_revision_requested": llm_revision,
        "llm_pretrained_identity": scorer.pretrained_identity,
        "dtype": dtype,
        "max_length": int(max_length),
        "overflow_policy": overflow_policy,
        "limit": limit,
    }
    score_partition = {
        "schema_version": SCORING_PARTITION_SCHEMA_VERSION,
        "algorithm": SCORING_PARTITION_ALGORITHM,
        "worker_count": worker_count,
        "worker_index": worker_index,
    }
    # Preserve the exact legacy single-worker run-signature payload so durable
    # pre-partition outputs remain resumable after this option is introduced.
    if worker_count != 1:
        run_signature["score_partition"] = score_partition
    run_signature_sha256 = canonical_sha256(run_signature)
    if resume and output_file.exists() and not metadata_file.exists():
        raise ValueError(
            f"cannot safely resume {output_file}: run metadata sidecar is missing"
        )
    if resume and metadata_file.exists():
        previous_metadata = _load_json(metadata_file)
        if previous_metadata.get("global_signature") != global_signature:
            raise ValueError(
                "resume global experiment signature differs from the existing run"
            )
        if (
            previous_metadata.get("global_signature_sha256") != global_signature_sha256
            or canonical_sha256(previous_metadata.get("global_signature"))
            != global_signature_sha256
        ):
            raise ValueError(
                "existing run metadata has an invalid global signature digest"
            )
        if previous_metadata.get("run_signature") != run_signature:
            raise ValueError(
                "resume signature differs from the existing run; use a new output path or --overwrite"
            )
        if (
            previous_metadata.get("run_signature_sha256") != run_signature_sha256
            or canonical_sha256(previous_metadata.get("run_signature"))
            != run_signature_sha256
        ):
            raise ValueError(
                "existing run metadata has an invalid run signature digest"
            )
        if output_file.exists():
            completed = _validate_existing_output(
                output_file,
                expected_designs=selected_expected_designs,
                run_signature_sha256=run_signature_sha256,
                global_signature_sha256=global_signature_sha256,
            )
        else:
            completed = set()
    else:
        completed = set()
    initial_report: dict[str, Any] = {
        "status": "running",
        "result_integrity_schema_version": RESULT_INTEGRITY_SCHEMA_VERSION,
        "global_signature": global_signature,
        "global_signature_sha256": global_signature_sha256,
        "run_signature": run_signature,
        "run_signature_sha256": run_signature_sha256,
        "trial_set_identity": selected_trial_set,
        "data_release_sha256": locked_data_release_sha256,
        "trial_build_attestation_sha256": trial_build_attestation_sha256,
        "trial_matrix_closure_sha256": global_signature["trial_matrix_closure_sha256"],
        "full_trial_set_root_sha256": global_signature["full_trial_set_root_sha256"],
        "full_trial_count": global_signature["full_trial_count"],
        "trials_manifest": str(trials_file),
        "manifest_representation_matches_projector": (
            manifest_representation_matches_projector
        ),
        "output": str(output_file),
        "num_trials_requested": selected_num_trials,
        "num_completed_before_run": len(completed),
    }
    if worker_count != 1:
        initial_report.update(
            {
                "score_partition": score_partition,
                "num_trials_in_full_manifest": num_trials,
                "num_visual_trials_in_full_manifest": num_visual_trials,
            }
        )
    write_json(metadata_file, initial_report)

    started = time.time()
    wrote = 0
    failures: list[dict[str, str]] = []
    skipped_completed = 0
    score_rows = (
        _trial_rows(trials_file, limit)
        if worker_count == 1
        else _selected_trial_rows(
            trials_file,
            limit,
            selected_trial_ids=selected_trial_ids,
        )
    )
    with output_file.open("a", encoding="utf-8") as handle:
        for trial in tqdm(
            score_rows,
            total=selected_num_trials,
            desc="information-upper-bound:score",
        ):
            trial_id = str(trial.get("trial_id", trial.get("id")))
            if trial_id in completed:
                skipped_completed += 1
                continue
            visual_id = trial.get("visual_id")
            try:
                features = (
                    feature_store.load(str(visual_id))
                    if visual_id and feature_store
                    else None
                )
                score = scorer.score(trial, features)
                condition = trial.get("condition") or {}
                row = {
                    "trial_id": trial_id,
                    "id": trial_id,
                    "base_id": str(trial.get("base_id", "")),
                    "trial_content_sha256": trial.get("trial_content_sha256"),
                    "data_release_sha256": trial.get("data_release_sha256"),
                    "trial_build_attestation_sha256": (
                        (trial.get("trial_build_attestation") or {}).get(
                            "attestation_sha256"
                        )
                        if isinstance(trial.get("trial_build_attestation"), Mapping)
                        else None
                    ),
                    "visual_id": visual_id,
                    **diagnostic_metadata(trial),
                    "condition": str(condition.get("name", "")),
                    "input_channel": condition.get("input_channel"),
                    "visual_view": condition.get("visual_view"),
                    "requested_dose": condition.get("requested_dose"),
                    "effective_dose": condition.get("effective_dose"),
                    "permutation_index": condition.get("permutation_index", 0),
                    "seed": condition.get("seed"),
                    "choices": trial.get("choices"),
                    "prediction": score.prediction,
                    "prediction_text": score.prediction_text,
                    "answer": trial.get("answer"),
                    "answer_text": trial.get("answer_text"),
                    "correct": score.correct,
                    "choice_nll": score.choice_nll,
                    "choice_probability": score.choice_probability,
                    "gold_nll": score.gold_nll,
                    "best_distractor_nll": score.best_distractor_nll,
                    "gold_margin": score.gold_margin,
                    "prompt_tokens": score.prompt_tokens,
                    "original_visual_tokens": score.original_visual_tokens,
                    "effective_visual_tokens": score.effective_visual_tokens,
                    "token_source": score.token_source,
                    "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
                    "scoring_global_signature_sha256": global_signature_sha256,
                    "scoring_run_signature_sha256": run_signature_sha256,
                }
                row["result_content_sha256"] = scored_result_sha256(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                completed.add(trial_id)
                wrote += 1
            except Exception as exc:
                failure = {
                    "trial_id": trial_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                failures.append(failure)
                if not continue_on_error:
                    raise

    report: dict[str, Any] = {
        "status": "complete" if not failures else "complete_with_failures",
        "result_integrity_schema_version": RESULT_INTEGRITY_SCHEMA_VERSION,
        "global_signature": global_signature,
        "global_signature_sha256": global_signature_sha256,
        "run_signature": run_signature,
        "run_signature_sha256": run_signature_sha256,
        "trial_set_identity": selected_trial_set,
        "data_release_sha256": locked_data_release_sha256,
        "trial_build_attestation_sha256": trial_build_attestation_sha256,
        "trial_matrix_closure_sha256": global_signature["trial_matrix_closure_sha256"],
        "full_trial_set_root_sha256": global_signature["full_trial_set_root_sha256"],
        "full_trial_count": global_signature["full_trial_count"],
        "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
        "trials_manifest": str(trials_file),
        "trials_manifest_sha256": trials_manifest_sha256,
        "manifest_representation_matches_projector": (
            manifest_representation_matches_projector
        ),
        "projector_checkpoint": str(Path(projector_checkpoint).resolve()),
        "projector_checkpoint_sha256": projector_checkpoint_sha256,
        "projector_metadata": str(Path(projector_metadata_path).resolve()),
        "feature_index": str(Path(feature_index_path).resolve())
        if feature_index_path
        else None,
        "feature_metadata": str(Path(feature_metadata_path).resolve())
        if feature_metadata_path
        else None,
        "feature_artifact_root_sha256": (
            feature_store.artifact_root_sha256 if feature_store else None
        ),
        "llm_id": scorer.llm_id,
        "llm_revision_requested": scorer.llm_revision,
        "llm_pretrained_identity": scorer.pretrained_identity,
        "dtype": dtype,
        "device": str(scorer.device),
        "max_length": max_length,
        "overflow_policy": overflow_policy,
        "num_trials_requested": selected_num_trials,
        "num_visual_trials": selected_num_visual_trials,
        "num_feature_entries": len(feature_store) if feature_store else 0,
        "num_written_this_run": wrote,
        "num_skipped_completed": skipped_completed,
        "num_failures": len(failures),
        "failures": failures,
        "elapsed_seconds": time.time() - started,
    }
    if worker_count != 1:
        report.update(
            {
                "score_partition": score_partition,
                "num_trials_in_full_manifest": num_trials,
                "num_visual_trials_in_full_manifest": num_visual_trials,
            }
        )
    write_json(metadata_file, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score all information-upper-bound trials with one frozen VideoLLM."
    )
    parser.add_argument("--trials", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--projector-ckpt", required=True)
    parser.add_argument("--projector-metadata", required=True)
    parser.add_argument(
        "--protocol-config",
        default=str(DEFAULT_PROTOCOL_PATH),
        help="locked model and evaluation protocol",
    )
    parser.add_argument("--feature-index")
    parser.add_argument("--feature-metadata")
    parser.add_argument("--llm-id")
    parser.add_argument("--llm-revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument(
        "--overflow-policy", default=None, choices=["error", "truncate_visual"]
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--feature-cache-size", type=int, default=16)
    parser.add_argument(
        "--worker-count",
        type=int,
        default=1,
        help=(
            "number of score workers reading the same full locked manifest; each worker "
            "executes one deterministic content-hash partition"
        ),
    )
    parser.add_argument(
        "--worker-index",
        type=int,
        default=0,
        help="zero-based score worker index in [0, --worker-count)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    protocol, _protocol_metadata = load_protocol(args.protocol_config)
    model_protocol = validate_frozen_model_protocol(protocol)
    resolved_llm_id = str(args.llm_id or model_protocol["llm_id"])
    resolved_llm_revision = args.llm_revision or model_protocol.get("llm_revision")
    resolved_dtype = str(args.dtype or model_protocol["dtype"])
    resolved_max_length = int(
        args.max_length if args.max_length is not None else model_protocol["max_length"]
    )
    resolved_overflow_policy = str(
        args.overflow_policy
        if args.overflow_policy is not None
        else model_protocol["overflow_policy"]
    )
    conflicts = []
    if args.llm_id is not None and resolved_llm_id != model_protocol["llm_id"]:
        conflicts.append("--llm-id")
    if args.llm_revision is not None and resolved_llm_revision != model_protocol.get(
        "llm_revision"
    ):
        conflicts.append("--llm-revision")
    if (
        args.max_length is not None
        and resolved_max_length != model_protocol["max_length"]
    ):
        conflicts.append("--max-length")
    if args.dtype is not None and resolved_dtype != model_protocol["dtype"]:
        conflicts.append("--dtype")
    if (
        args.overflow_policy is not None
        and resolved_overflow_policy != model_protocol["overflow_policy"]
    ):
        conflicts.append("--overflow-policy")
    if conflicts:
        raise ValueError(
            f"{', '.join(conflicts)} conflict with the locked protocol; update the protocol "
            "before starting the confirmatory run"
        )
    report = run_trials(
        trials_path=args.trials,
        output_path=args.out,
        projector_checkpoint=args.projector_ckpt,
        projector_metadata_path=args.projector_metadata,
        feature_index_path=args.feature_index,
        feature_metadata_path=args.feature_metadata,
        llm_id=resolved_llm_id,
        llm_revision=resolved_llm_revision,
        device=args.device,
        dtype=resolved_dtype,
        max_length=resolved_max_length,
        overflow_policy=resolved_overflow_policy,
        resume=args.resume,
        overwrite=args.overwrite,
        continue_on_error=args.continue_on_error,
        limit=args.limit,
        feature_cache_size=args.feature_cache_size,
        protocol_config_path=args.protocol_config,
        worker_count=args.worker_count,
        worker_index=args.worker_index,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
