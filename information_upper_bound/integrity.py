"""Content identities used by extraction and frozen-model scoring.

The functions in this module deliberately avoid importing Transformers.  They
operate on already-loaded objects, so provenance collection never causes a
second model load.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    import torch


_SNAPSHOT_REVISION = re.compile(r"^[0-9a-fA-F]{7,64}$")


RESULT_INTEGRITY_SCHEMA_VERSION = "information_upper_bound.scored_result.v2"
TRIAL_SET_SCHEMA_VERSION = "information_upper_bound.trial_set.v1"

# Keep this allowlist explicit.  Hashing the whole row would make harmless
# logging additions invalidate old results, while omitting a score/design field
# would permit a scientifically meaningful edit to go undetected.
RESULT_DESIGN_FIELDS = (
    "trial_id",
    "base_id",
    "trial_content_sha256",
    "data_release_sha256",
    "trial_build_attestation_sha256",
    "visual_id",
    "dataset",
    "split",
    "information_family",
    "question_family",
    "reasoning_depth",
    "resampling_unit_id",
    "pair_id",
    "pair_role",
    "independent_unit_id",
    "official_candidate_id",
    "official_candidate_count",
    "condition",
    "input_channel",
    "visual_view",
    "requested_dose",
    "effective_dose",
    "permutation_index",
    "seed",
    "choices",
    "answer",
    "answer_text",
)
RESULT_SCORE_FIELDS = (
    "prediction",
    "prediction_text",
    "correct",
    "choice_nll",
    "choice_probability",
    "gold_nll",
    "best_distractor_nll",
    "gold_margin",
    "prompt_tokens",
    "original_visual_tokens",
    "effective_visual_tokens",
    "token_source",
    "scoring_protocol_version",
    "scoring_global_signature_sha256",
    "scoring_run_signature_sha256",
)


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def scored_result_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return the stable, scientifically relevant payload of one score row."""

    return {
        "schema_version": RESULT_INTEGRITY_SCHEMA_VERSION,
        "design": {name: row.get(name) for name in RESULT_DESIGN_FIELDS},
        "score": {name: row.get(name) for name in RESULT_SCORE_FIELDS},
    }


def scored_result_sha256(row: Mapping[str, Any]) -> str:
    """Authenticate score outputs and their manifest/design bindings."""

    return canonical_sha256(scored_result_payload(row))


def trial_set_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a path/order-independent identity for one scored trial shard.

    The identity intentionally contains only trial IDs and their already
    validated content digests.  Analysis can therefore prove that the exact
    manifest subset represented by a score sidecar is complete even after
    multiple shard outputs are concatenated.
    """

    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        trial_id = str(row.get("trial_id", row.get("id", ""))).strip()
        content_sha256 = str(row.get("trial_content_sha256", "")).strip().lower()
        if not trial_id or trial_id in seen:
            raise ValueError(
                f"trial-set identity contains an empty/duplicate trial_id: {trial_id!r}"
            )
        if re.fullmatch(r"[0-9a-f]{64}", content_sha256) is None:
            raise ValueError(
                f"trial-set identity has invalid trial_content_sha256 for {trial_id!r}"
            )
        seen.add(trial_id)
        entries.append({"trial_id": trial_id, "trial_content_sha256": content_sha256})
    if not entries:
        raise ValueError("cannot compute a trial-set identity for an empty manifest")
    entries.sort(key=lambda value: value["trial_id"])
    root_sha256 = canonical_sha256(
        {"schema_version": TRIAL_SET_SCHEMA_VERSION, "entries": entries}
    )
    return {
        "schema_version": TRIAL_SET_SCHEMA_VERSION,
        "trial_count": len(entries),
        "root_sha256": root_sha256,
    }


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _local_source_identity(path: Path) -> dict[str, Any]:
    """Hash an arbitrary local checkpoint, or identify an immutable HF snapshot.

    Hugging Face snapshot directory names are content revisions.  For an
    arbitrary local directory there is no such contract, so all regular files
    are streamed into a deterministic Merkle-like manifest hash.
    """

    resolved = path.resolve()
    if resolved.is_file():
        return {
            "kind": "local_file",
            "sha256": _sha256_file(resolved),
            "size_bytes": int(resolved.stat().st_size),
        }
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"resolved pretrained source does not exist: {resolved}"
        )

    if resolved.parent.name == "snapshots" and _SNAPSHOT_REVISION.fullmatch(
        resolved.name
    ):
        return {
            "kind": "huggingface_snapshot",
            "revision": resolved.name.lower(),
        }

    files = sorted(path for path in resolved.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"local pretrained source contains no files: {resolved}")
    manifest_digest = hashlib.sha256()
    total_bytes = 0
    for file_path in files:
        relative = file_path.relative_to(resolved).as_posix()
        size = int(file_path.stat().st_size)
        file_digest = _sha256_file(file_path)
        total_bytes += size
        manifest_digest.update(
            json.dumps(
                {"path": relative, "sha256": file_digest, "size_bytes": size},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        manifest_digest.update(b"\n")
    return {
        "kind": "local_directory",
        "tree_sha256": manifest_digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
    }


def _commit_from_object(value: Any) -> str | None:
    candidates: list[Any] = []
    if value is not None:
        candidates.extend(
            [
                getattr(value, "_commit_hash", None),
                getattr(getattr(value, "config", None), "_commit_hash", None),
            ]
        )
        init_kwargs = getattr(value, "init_kwargs", None)
        if isinstance(init_kwargs, Mapping):
            candidates.extend(
                [
                    init_kwargs.get("_commit_hash"),
                    init_kwargs.get("commit_hash"),
                    init_kwargs.get("revision"),
                ]
            )
    for candidate in candidates:
        if candidate not in (None, ""):
            return str(candidate)
    return None


def resolved_pretrained_identity(
    *,
    requested_id: str,
    resolved_source: str | Path,
    model: Any,
    auxiliaries: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe the exact loaded pretrained source as strongly as available.

    For hub models, Transformers attaches the resolved repository commit to the
    loaded config.  For local sources, this uses the snapshot revision or a
    streamed content hash.  The returned identity excludes absolute paths, so
    identical artifacts have the same identity on different machines.
    """

    auxiliary_values = dict(auxiliaries or {})
    component_revisions: dict[str, str] = {}
    model_revision = _commit_from_object(model)
    if model_revision:
        component_revisions["model"] = model_revision
    for name, value in sorted(auxiliary_values.items()):
        revision = _commit_from_object(value)
        if revision:
            component_revisions[str(name)] = revision

    source_path = Path(str(resolved_source))
    local_identity: dict[str, Any] | None = None
    if source_path.exists():
        local_identity = _local_source_identity(source_path)

    revisions = sorted(set(component_revisions.values()))
    if len(revisions) > 1:
        raise ValueError(
            "loaded pretrained components resolve to different revisions: "
            + ", ".join(
                f"{name}={value}" for name, value in component_revisions.items()
            )
        )

    stable_source: dict[str, Any]
    if local_identity is not None:
        stable_source = local_identity
        strength = "strong_local_content_or_snapshot"
    elif revisions:
        stable_source = {
            "kind": "huggingface_hub_revision",
            "repo_id": str(requested_id),
            "revision": revisions[0],
        }
        strength = "strong_hub_revision"
    else:
        stable_source = {"kind": "unresolved_hub_id", "repo_id": str(requested_id)}
        strength = "weak_mutable_identifier"

    payload = {
        "requested_id": (
            str(requested_id) if local_identity is None else "<local_pretrained_source>"
        ),
        "source": stable_source,
        "component_revisions": component_revisions,
        "model_class": type(model).__qualname__,
        "auxiliary_classes": {
            str(name): type(value).__qualname__ if value is not None else None
            for name, value in sorted(auxiliary_values.items())
        },
        "identity_strength": strength,
    }
    return {**payload, "identity_sha256": canonical_sha256(payload)}


def validate_locked_pretrained_revision(
    identity: Mapping[str, Any],
    revision: str,
    *,
    component_name: str,
) -> None:
    """Require a requested revision to equal the identity actually loaded."""

    source_identity = identity.get("source")
    if not isinstance(source_identity, Mapping):
        raise ValueError(f"{component_name} pretrained identity has no source mapping")
    component_revisions = identity.get("component_revisions")
    revision_values = (
        component_revisions.values() if isinstance(component_revisions, Mapping) else ()
    )
    resolved_revisions = {
        str(value).lower()
        for value in (
            source_identity.get("revision"),
            source_identity.get("tree_sha256"),
            source_identity.get("sha256"),
            *revision_values,
        )
        if value not in (None, "")
    }
    if str(revision).lower() not in resolved_revisions:
        raise ValueError(
            f"loaded {component_name} content identity does not match the locked revision"
        )


def tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    # Keep torch out of lightweight CLI startup.  On some Linux Conda hosts,
    # importing torch before the stdlib sqlite3 extension causes the dynamic
    # loader to pin the host's older libstdc++.so.6.  sqlite3's Conda ICU then
    # cannot resolve newer CXXABI symbols.  The tensor-only path can safely pay
    # the import cost when it is actually used.
    import torch

    if not torch.is_tensor(tensor):
        raise TypeError("tensor_identity expects a torch.Tensor")
    value = tensor.detach().cpu().contiguous()
    header = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "numel": int(value.numel()),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\n")
    digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return {**header, "sha256": digest.hexdigest()}


def decoded_frames_identity(frames: Sequence[Any]) -> dict[str, Any]:
    """Hash ordered, post-decoder RGB pixels before processor transforms."""

    if not frames:
        raise ValueError("decoded frame identity requires at least one frame")
    digest = hashlib.sha256()
    shapes: list[list[int]] = []
    for index, frame in enumerate(frames):
        rgb = frame.convert("RGB")
        width, height = rgb.size
        header = {"index": index, "mode": "RGB", "width": width, "height": height}
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
        digest.update(rgb.tobytes())
        shapes.append([height, width, 3])
    return {
        "sha256": digest.hexdigest(),
        "num_frames": len(frames),
        "shapes": shapes,
        "pixel_format": "RGB_uint8",
    }


def feature_artifact_root(rows: Sequence[Mapping[str, Any]]) -> str:
    """Return a path- and row-order-independent root for a feature index."""

    required = (
        "visual_id",
        "feature_content_hash",
        "feature_file_sha256",
        "feature_tensor_identity",
        "feature_artifact_identity_sha256",
    )
    leaves: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(
                f"feature index row is missing integrity fields: {missing}"
            )
        visual_id = str(row["visual_id"])
        if not visual_id or visual_id in seen:
            raise ValueError(
                f"feature artifact root has empty/duplicate visual_id: {visual_id!r}"
            )
        seen.add(visual_id)
        leaves.append(
            {
                "visual_id": visual_id,
                "feature_content_hash": str(row["feature_content_hash"]),
                "feature_file_sha256": str(row["feature_file_sha256"]),
                "feature_tensor_identity": dict(row["feature_tensor_identity"]),
                "feature_artifact_identity_sha256": str(
                    row["feature_artifact_identity_sha256"]
                ),
            }
        )
    if not leaves:
        raise ValueError("cannot compute a feature artifact root for an empty index")
    leaves.sort(key=lambda value: value["visual_id"])
    return canonical_sha256(
        {
            "schema_version": "information_upper_bound.feature_artifact_root.v1",
            "entries": leaves,
        }
    )
