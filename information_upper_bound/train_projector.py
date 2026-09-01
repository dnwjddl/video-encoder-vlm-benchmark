#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from vlmevalbench.projector import MLPProjector
from information_upper_bound.projector_training import (
    FeatureTextDataset,
    build_inputs_embeds_and_labels,
    collate_feature_text,
)
from information_upper_bound.clevrer_pilot_contract import (
    validate_clevrer_selection_options,
)
from vlmevalbench.utils import get_dtype, save_json, set_seed
from information_upper_bound.integrity import resolved_pretrained_identity
from information_upper_bound.conditions import DEFAULT_CONDITION_PATH
from information_upper_bound.io import iter_jsonl, sha256_file
from information_upper_bound.protocol import DEFAULT_PROTOCOL_PATH, load_protocol
from information_upper_bound.split_integrity import audit_projector_split_disjointness
from information_upper_bound.trial_matrix import (
    DEVELOPMENT_TRIAL_MATRIX_CLOSURE_SCHEMA_VERSION,
    validate_development_trial_matrix_closure,
    validate_trial_base_release,
    validate_trial_matrix_closure,
)


STRICT_PROVENANCE_ARGUMENTS = (
    "feature_metadata",
    "eval_manifest",
    "eval_feature_index",
    "eval_feature_metadata",
    "eval_data_lock",
)
STRICT_PROJECTOR_LOCK_SCHEMA_VERSION = "information_upper_bound.projector_lock.v3"
CLEVRER_PROJECTOR_TRAIN_CONDITIONS_SHA256 = (
    "da7f1a51ee002be563b4ba2d67d5ffa14c45b768e1b7dceac70b98ca1caff697"
)


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_hf_source(
    model_id: str,
    *,
    revision: str | None,
    local_files_only: bool,
) -> str:
    if not local_files_only or Path(model_id).exists():
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id,
        repo_type="model",
        revision=revision,
        local_files_only=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train projector only with frozen visual features and frozen LLM."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--eval-manifest",
        help=(
            "Strict information-upper-bound mode: predeclared held-out trial manifest; "
            "must be family/media-disjoint from training."
        ),
    )
    parser.add_argument("--feature-index", required=True)
    parser.add_argument(
        "--feature-metadata",
        help=(
            "Strict information-upper-bound mode: training extraction metadata "
            "binding the feature index and encoder/preprocessor identity."
        ),
    )
    parser.add_argument(
        "--eval-feature-index",
        help=(
            "Strict information-upper-bound mode: feature index for the exact "
            "held-out trial manifest."
        ),
    )
    parser.add_argument(
        "--eval-feature-metadata",
        help=(
            "Strict information-upper-bound mode: extraction metadata for "
            "--eval-feature-index."
        ),
    )
    parser.add_argument(
        "--eval-data-lock",
        help=(
            "Strict mode: official release lock used to regenerate and authenticate "
            "the complete evaluation trial matrix."
        ),
    )
    parser.add_argument(
        "--train-data-lock",
        help=(
            "Strict mode: optional authenticated training-release lock. It is "
            "required by the registered CLEVRER pilot protocol."
        ),
    )
    parser.add_argument(
        "--train-conditions-config",
        help=(
            "Strict mode: exact development condition matrix used to build the "
            "projector-training manifest."
        ),
    )
    parser.add_argument(
        "--train-protocol-config",
        help=(
            "Strict mode: development protocol used to build the projector-training "
            "manifest."
        ),
    )
    parser.add_argument(
        "--conditions-config",
        default=str(DEFAULT_CONDITION_PATH),
        help="Strict mode: exact condition matrix used to build --eval-manifest.",
    )
    parser.add_argument(
        "--protocol-config",
        default=str(DEFAULT_PROTOCOL_PATH),
        help="Strict mode: preregistered protocol used to build --eval-manifest.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Strict information-upper-bound mode only: allow writing into an "
            "existing non-empty output directory. Use a dedicated directory per run."
        ),
    )
    parser.add_argument("--encoder-name", required=True)
    parser.add_argument("--llm-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm-revision", default=None)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--projector-depth", type=int, default=2)
    parser.add_argument("--projector-hidden-dim", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every-steps", type=int, default=1000)
    return parser.parse_args(argv)


def strict_information_upper_bound_mode(args: argparse.Namespace) -> bool:
    """Return whether the complete strict provenance interface was requested.

    Supplying none of the strict-only arguments preserves the repository's
    original generic projector-training path. A partial strict invocation is
    rejected rather than silently falling back to unauthenticated training.
    """

    supplied = {
        name: getattr(args, name, None) not in (None, "")
        for name in STRICT_PROVENANCE_ARGUMENTS
    }
    if any(supplied.values()) and not all(supplied.values()):
        missing = [
            "--" + name.replace("_", "-")
            for name, is_supplied in supplied.items()
            if not is_supplied
        ]
        present = [
            "--" + name.replace("_", "-")
            for name, is_supplied in supplied.items()
            if is_supplied
        ]
        raise ValueError(
            "strict information-upper-bound projector training requires the full "
            "provenance argument set; "
            f"present={present}, missing={missing}. Supply all strict provenance "
            "arguments or none."
        )
    return all(supplied.values())


def _validate_strict_output_directory(args: argparse.Namespace) -> None:
    """Fail before model loading when strict outputs could replace or alias inputs."""

    out_dir = Path(args.out_dir).expanduser()
    root_metadata = (out_dir / "metadata.json").resolve()
    resolved_inputs = {
        Path(value).expanduser().resolve()
        for value in (
            args.manifest,
            args.feature_index,
            args.feature_metadata,
            args.eval_manifest,
            args.eval_feature_index,
            args.eval_feature_metadata,
            args.eval_data_lock,
            args.train_data_lock,
            args.train_conditions_config,
            args.train_protocol_config,
            args.conditions_config,
            args.protocol_config,
        )
        if value not in (None, "")
    }
    if root_metadata in resolved_inputs:
        raise ValueError(
            "strict projector metadata output aliases an authenticated input: "
            f"{root_metadata}"
        )
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            "strict projector output directory is not empty; use a dedicated run "
            f"directory or pass --overwrite explicitly: {out_dir}"
        )


def _load_json_object(path: str | Path, *, option: str) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {option} JSON object at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{option} must contain a JSON object")
    return value


def _authenticate_feature_bundle(
    *,
    role: str,
    manifest_path: str | Path,
    feature_index_path: str | Path,
    feature_metadata_path: str | Path,
    encoder_name: str,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Authenticate one strict extraction bundle before loading the LLM."""

    # Imported lazily so the legacy CLI and --help do not pay the strict scorer
    # module's import cost or acquire a dependency on its artifact schema.
    from information_upper_bound.run import FeatureStore

    metadata = _load_json_object(
        feature_metadata_path,
        option=f"--{role}-feature-metadata"
        if role != "training"
        else "--feature-metadata",
    )
    manifest_sha256 = sha256_file(manifest_path)
    declared_manifest_sha256 = str(metadata.get("manifest_sha256", ""))
    manifest_representation_matches_metadata = (
        declared_manifest_sha256 == manifest_sha256
    )
    if role != "evaluation" and not manifest_representation_matches_metadata:
        raise ValueError(
            f"{role} feature metadata manifest_sha256 is missing or does not match "
            f"the exact {role} manifest"
        )
    feature_index_sha256 = sha256_file(feature_index_path)
    if str(metadata.get("index_sha256", "")) != feature_index_sha256:
        raise ValueError(
            f"{role} feature metadata index_sha256 is missing or does not match "
            f"the {role} feature index"
        )
    pipeline_sha256 = str(metadata.get("extraction_pipeline_identity_sha256", ""))
    if not pipeline_sha256:
        raise ValueError(
            f"{role} feature metadata must bind extraction_pipeline_identity_sha256"
        )
    if str(metadata.get("encoder", "")) != str(encoder_name):
        raise ValueError(
            f"--encoder-name {encoder_name!r} does not match {role} feature "
            f"metadata encoder={metadata.get('encoder')!r}"
        )
    store = FeatureStore(
        feature_index_path,
        cache_size=0,
        metadata=metadata,
        verify_all_files=True,
    )
    audit = {
        "manifest_sha256": manifest_sha256,
        "declared_manifest_sha256": declared_manifest_sha256,
        "manifest_representation_matches_metadata": (
            manifest_representation_matches_metadata
        ),
        "feature_index_sha256": feature_index_sha256,
        "feature_metadata_sha256": sha256_file(feature_metadata_path),
        "feature_artifact_root_sha256": store.artifact_root_sha256,
        "encoder_extraction_pipeline_identity_sha256": pipeline_sha256,
    }
    return metadata, store, audit


def _validate_visual_coverage(
    manifest: str | Path,
    feature_keys: set[str],
    *,
    role: str,
) -> dict[str, Any]:
    """Require a strict index to cover exactly every declared visual_id."""

    expected: set[str] = set()
    rows = 0
    visual_rows = 0
    for row in iter_jsonl(manifest):
        rows += 1
        visual_id = row.get("visual_id")
        if visual_id in (None, ""):
            continue
        visual_rows += 1
        expected.add(str(visual_id))
    if not rows:
        raise ValueError(f"strict {role} manifest is empty")
    if not expected:
        raise ValueError(f"strict {role} manifest contains no visual_id values")
    missing = sorted(expected - feature_keys)
    extra = sorted(feature_keys - expected)
    if missing or extra:
        raise ValueError(
            f"{role} feature index visual_id coverage does not exactly match the "
            f"{role} manifest; "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )
    return {
        f"{role}_rows": rows,
        f"{role}_visual_rows": visual_rows,
        f"{role}_unique_visual_ids": len(expected),
        "missing_visual_ids": 0,
        "extra_visual_ids": 0,
    }


def _validate_evaluation_visual_coverage(
    evaluation_manifest: str | Path,
    evaluation_feature_keys: set[str],
) -> dict[str, Any]:
    """Backward-compatible named wrapper for focused tests and callers."""

    return _validate_visual_coverage(
        evaluation_manifest,
        evaluation_feature_keys,
        role="evaluation",
    )


def infer_feature_dim(feature_index: str) -> int:
    import json

    with Path(feature_index).open("r", encoding="utf-8") as f:
        first = json.loads(f.readline())
    shape = first["shape"]
    return int(shape[-1])


def save_checkpoint(
    out_dir: Path, projector: torch.nn.Module, step: int, metadata: dict
) -> None:
    ckpt_dir = out_dir / f"step_{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    state = (
        projector.module.state_dict()
        if hasattr(projector, "module")
        else projector.state_dict()
    )
    checkpoint_path = ckpt_dir / "projector.pt"
    metadata_path = ckpt_dir / "metadata.json"
    torch.save(state, checkpoint_path)
    checkpoint_metadata = metadata | {"step": step}
    save_json(metadata_path, checkpoint_metadata)
    if metadata.get("projector_training_mode") == "information_upper_bound_strict":
        projector_lock = {
            "schema_version": STRICT_PROJECTOR_LOCK_SCHEMA_VERSION,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "metadata_sha256": sha256_file(metadata_path),
            "training_manifest_sha256": checkpoint_metadata["training_manifest_sha256"],
            "evaluation_manifest_sha256": checkpoint_metadata[
                "evaluation_manifest_sha256"
            ],
            "training_feature_index_sha256": checkpoint_metadata[
                "training_feature_index_sha256"
            ],
            "training_feature_artifact_root_sha256": checkpoint_metadata[
                "training_feature_artifact_root_sha256"
            ],
            "training_feature_metadata_sha256": checkpoint_metadata[
                "training_feature_metadata_sha256"
            ],
            "evaluation_feature_index_sha256": checkpoint_metadata[
                "evaluation_feature_index_sha256"
            ],
            "evaluation_feature_artifact_root_sha256": checkpoint_metadata[
                "evaluation_feature_artifact_root_sha256"
            ],
            "evaluation_feature_metadata_sha256": checkpoint_metadata[
                "evaluation_feature_metadata_sha256"
            ],
            "evaluation_trial_matrix_closure_sha256": checkpoint_metadata[
                "evaluation_trial_matrix_closure_sha256"
            ],
            "evaluation_trial_set_root_sha256": checkpoint_metadata[
                "evaluation_trial_set_root_sha256"
            ],
            "evaluation_trial_count": checkpoint_metadata["evaluation_trial_count"],
            "encoder_extraction_pipeline_identity_sha256": checkpoint_metadata[
                "encoder_extraction_pipeline_identity_sha256"
            ],
            "llm_pretrained_identity_sha256": checkpoint_metadata[
                "llm_pretrained_identity_sha256"
            ],
            "training_dtype": checkpoint_metadata["dtype"],
            "training_max_length": checkpoint_metadata["max_length"],
            "training_seed": checkpoint_metadata["seed"],
        }
        training_data_lock = checkpoint_metadata.get("training_data_lock")
        if isinstance(training_data_lock, Mapping):
            projector_lock["training_data_release_sha256"] = training_data_lock[
                "data_release_sha256"
            ]
        save_json(ckpt_dir / "protocol_projector_lock.json", projector_lock)


def _validate_registered_pilot_training_lock(
    protocol: Mapping[str, Any],
    training_data_lock: Mapping[str, Any] | None,
    training_trial_matrix_closure: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(protocol.get("pilot"), Mapping):
        return
    if training_data_lock is None:
        raise ValueError(
            "registered pilot projector training requires --train-data-lock"
        )
    if training_data_lock.get("datasets") != {
        "clevrer": training_data_lock.get("records")
    }:
        raise ValueError(
            "registered pilot training lock must contain only CLEVRER records"
        )
    training_runs = training_data_lock.get("adapter_runs")
    if not isinstance(training_runs, list) or len(training_runs) != 1:
        raise ValueError("registered pilot training lock must contain one adapter run")
    training_run = training_runs[0]
    if (
        not isinstance(training_run, Mapping)
        or training_run.get("dataset") != "clevrer"
        or training_run.get("canonical_split") != "train"
    ):
        raise ValueError(
            "registered pilot training lock must bind the CLEVRER train split"
        )
    training_options = training_run.get("adapter_options")
    selection_options = (
        training_options.get("resampling_unit_selection")
        if isinstance(training_options, Mapping)
        else None
    )
    if not isinstance(selection_options, Mapping):
        raise ValueError("registered pilot training lock has no unit-sampling contract")
    validate_clevrer_selection_options(
        selection_options,
        role="train",
        locked_record_count=int(training_data_lock["records"]),
    )
    expected_closure = {
        "schema_version": DEVELOPMENT_TRIAL_MATRIX_CLOSURE_SCHEMA_VERSION,
        "status": "exact",
        "mode": "development",
        "data_release_sha256": training_data_lock["data_release_sha256"],
        "conditions_sha256": CLEVRER_PROJECTOR_TRAIN_CONDITIONS_SHA256,
        "base_records": training_data_lock["records"],
        "conditions": ["full_video"],
        "trial_count": training_data_lock["records"],
        "sampling": {
            "seed": 42,
            "option_permutations": 1,
            "trial_shards": 1,
        },
    }
    if not isinstance(training_trial_matrix_closure, Mapping):
        raise ValueError(
            "registered pilot projector training requires exact training "
            "trial-matrix closure"
        )
    mismatches = {
        key: {"expected": value, "actual": training_trial_matrix_closure.get(key)}
        for key, value in expected_closure.items()
        if training_trial_matrix_closure.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"registered pilot training trial-matrix contract mismatch: {mismatches}"
        )


def main() -> None:
    args = parse_args()
    strict_mode = strict_information_upper_bound_mode(args)
    if (
        args.train_data_lock
        or args.train_conditions_config
        or args.train_protocol_config
    ) and not strict_mode:
        raise ValueError(
            "training release/closure arguments are available only with the "
            "strict interface"
        )
    if strict_mode:
        training_closure_args = (
            args.train_conditions_config,
            args.train_protocol_config,
        )
        if any(training_closure_args) and not all(training_closure_args):
            raise ValueError(
                "--train-conditions-config and --train-protocol-config must be "
                "provided together"
            )
        if any(training_closure_args) and not args.train_data_lock:
            raise ValueError(
                "training trial-matrix closure also requires --train-data-lock"
            )
        _validate_strict_output_directory(args)
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    strict_metadata: dict[str, Any] = {}
    if strict_mode:
        assert args.feature_metadata is not None
        assert args.eval_manifest is not None
        assert args.eval_feature_index is not None
        assert args.eval_feature_metadata is not None
        assert args.eval_data_lock is not None
        (
            training_feature_metadata,
            training_feature_store,
            training_feature_audit,
        ) = _authenticate_feature_bundle(
            role="training",
            manifest_path=args.manifest,
            feature_index_path=args.feature_index,
            feature_metadata_path=args.feature_metadata,
            encoder_name=args.encoder_name,
        )
        (
            evaluation_feature_metadata,
            evaluation_feature_store,
            evaluation_feature_audit,
        ) = _authenticate_feature_bundle(
            role="evaluation",
            manifest_path=args.eval_manifest,
            feature_index_path=args.eval_feature_index,
            feature_metadata_path=args.eval_feature_metadata,
            encoder_name=args.encoder_name,
        )
        trial_protocol, _trial_protocol_metadata = load_protocol(args.protocol_config)
        training_data_lock: dict[str, Any] | None = None
        training_trial_matrix_closure: dict[str, Any] | None = None
        if args.train_data_lock:
            if args.train_conditions_config and args.train_protocol_config:
                training_protocol, _training_protocol_metadata = load_protocol(
                    args.train_protocol_config
                )
                training_authentication = validate_development_trial_matrix_closure(
                    args.manifest,
                    data_lock_path=args.train_data_lock,
                    conditions_config_path=args.train_conditions_config,
                    protocol=training_protocol,
                )
                training_data_lock = training_authentication["data_lock"]
                training_trial_matrix_closure = training_authentication["closure"]
            else:
                training_data_lock = validate_trial_base_release(
                    args.manifest,
                    data_lock_path=args.train_data_lock,
                )
        _validate_registered_pilot_training_lock(
            trial_protocol,
            training_data_lock,
            training_trial_matrix_closure,
        )
        evaluation_trial_matrix_closure = validate_trial_matrix_closure(
            args.eval_manifest,
            data_lock_path=args.eval_data_lock,
            conditions_config_path=args.conditions_config,
            protocol=trial_protocol,
        )
        if evaluation_feature_metadata.get(
            "trial_matrix_closure"
        ) != evaluation_trial_matrix_closure or str(
            evaluation_feature_metadata.get("trial_matrix_closure_sha256", "")
        ) != str(evaluation_trial_matrix_closure["closure_sha256"]):
            raise ValueError(
                "evaluation feature metadata does not carry the exact regenerated "
                "trial-matrix closure"
            )
        expected_trial_set_identity = {
            "schema_version": "information_upper_bound.trial_set.v1",
            "trial_count": evaluation_trial_matrix_closure["trial_count"],
            "root_sha256": evaluation_trial_matrix_closure["trial_set_root_sha256"],
        }
        if (
            evaluation_feature_metadata.get("trial_set_identity")
            != expected_trial_set_identity
        ):
            raise ValueError(
                "evaluation feature metadata trial_set_identity differs from the "
                "regenerated complete matrix"
            )
        training_pipeline_sha256 = str(
            training_feature_audit["encoder_extraction_pipeline_identity_sha256"]
        )
        evaluation_pipeline_sha256 = str(
            evaluation_feature_audit["encoder_extraction_pipeline_identity_sha256"]
        )
        if training_pipeline_sha256 != evaluation_pipeline_sha256:
            raise ValueError(
                "training and evaluation features use different frozen "
                "encoder/preprocessor pipeline identities"
            )
        training_visual_coverage = _validate_visual_coverage(
            args.manifest,
            set(training_feature_store.paths),
            role="training",
        )
        evaluation_visual_coverage = _validate_evaluation_visual_coverage(
            args.eval_manifest,
            set(evaluation_feature_store.paths),
        )
        split_audit = audit_projector_split_disjointness(
            args.manifest,
            args.eval_manifest,
        )
        strict_metadata = {
            "projector_training_mode": "information_upper_bound_strict",
            "evaluation_manifest": args.eval_manifest,
            "feature_metadata": args.feature_metadata,
            "evaluation_feature_index": args.eval_feature_index,
            "evaluation_feature_metadata": args.eval_feature_metadata,
            "training_manifest_sha256": training_feature_audit["manifest_sha256"],
            "evaluation_manifest_sha256": evaluation_feature_audit["manifest_sha256"],
            "training_evaluation_split_audit": split_audit,
            "training_feature_index_sha256": training_feature_audit[
                "feature_index_sha256"
            ],
            "training_feature_artifact_root_sha256": training_feature_audit[
                "feature_artifact_root_sha256"
            ],
            "training_feature_metadata_sha256": training_feature_audit[
                "feature_metadata_sha256"
            ],
            "evaluation_feature_index_sha256": evaluation_feature_audit[
                "feature_index_sha256"
            ],
            "evaluation_feature_artifact_root_sha256": evaluation_feature_audit[
                "feature_artifact_root_sha256"
            ],
            "evaluation_feature_metadata_sha256": evaluation_feature_audit[
                "feature_metadata_sha256"
            ],
            "evaluation_trial_matrix_closure": evaluation_trial_matrix_closure,
            "evaluation_trial_matrix_closure_sha256": (
                evaluation_trial_matrix_closure["closure_sha256"]
            ),
            "evaluation_trial_set_root_sha256": (
                evaluation_trial_matrix_closure["trial_set_root_sha256"]
            ),
            "evaluation_trial_count": evaluation_trial_matrix_closure["trial_count"],
            "encoder_extraction_pipeline_identity_sha256": (training_pipeline_sha256),
            "evaluation_visual_id_coverage": evaluation_visual_coverage,
            "training_visual_id_coverage": training_visual_coverage,
            "training_feature_extraction_metadata": {
                "schema_version": training_feature_metadata.get("schema_version"),
                "execution_mode": training_feature_metadata.get("execution_mode"),
                "data_release_sha256": training_feature_metadata.get(
                    "data_release_sha256"
                ),
                "trial_build_attestation_sha256": training_feature_metadata.get(
                    "trial_build_attestation_sha256"
                ),
            },
            "training_data_lock": (
                {
                    "data_release_sha256": training_data_lock["data_release_sha256"],
                    "file_sha256": training_data_lock["file_sha256"],
                    "records": training_data_lock["records"],
                    "datasets": training_data_lock["datasets"],
                    "adapter_run_ids": [
                        run["adapter_run_id"]
                        for run in training_data_lock["adapter_runs"]
                    ],
                }
                if training_data_lock is not None
                else None
            ),
            "training_trial_matrix_closure": training_trial_matrix_closure,
            "evaluation_feature_extraction_metadata": {
                "schema_version": evaluation_feature_metadata.get("schema_version"),
                "execution_mode": evaluation_feature_metadata.get("execution_mode"),
                "data_release_sha256": evaluation_feature_metadata.get(
                    "data_release_sha256"
                ),
                "trial_build_attestation_sha256": evaluation_feature_metadata.get(
                    "trial_build_attestation_sha256"
                ),
            },
        }

    # Keep heavyweight/runtime-only dependencies out of parser and provenance
    # helper imports. They are required only after all cheap strict checks pass.
    from accelerate import Accelerator
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    dtype = get_dtype(args.dtype)

    local_files_only = env_flag("VLMEB_LOCAL_FILES_ONLY")
    llm_source = resolve_hf_source(
        args.llm_id,
        revision=args.llm_revision,
        local_files_only=local_files_only,
    )
    revision_kwargs = (
        {"revision": args.llm_revision}
        if args.llm_revision is not None and not Path(llm_source).exists()
        else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(
        llm_source,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_files_only,
        **revision_kwargs,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        llm_source,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
        **revision_kwargs,
    )
    llm.eval()
    llm.requires_grad_(False)
    llm_identity: Mapping[str, Any] | None = None
    if strict_mode:
        llm_identity = resolved_pretrained_identity(
            requested_id=args.llm_id,
            resolved_source=llm_source,
            model=llm,
            auxiliaries={"tokenizer": tokenizer},
        )
        if llm_identity["identity_strength"] == "weak_mutable_identifier":
            raise ValueError(
                "strict projector training requires a resolved LLM/tokenizer "
                "revision or a content-addressed local snapshot"
            )

    input_dim = infer_feature_dim(args.feature_index)
    output_dim = int(llm.config.hidden_size)
    projector = MLPProjector(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.projector_hidden_dim,
        depth=args.projector_depth,
    )

    dataset = FeatureTextDataset(
        args.manifest,
        args.feature_index,
        require_integrity=strict_mode,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_feature_text,
        pin_memory=True,
    )
    optimizer = torch.optim.AdamW(
        projector.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    projector, llm, optimizer, dataloader = accelerator.prepare(
        projector, llm, optimizer, dataloader
    )

    metadata = {
        "encoder_name": args.encoder_name,
        "llm_id": args.llm_id,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "seed": args.seed,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "projector_depth": args.projector_depth,
        "projector_hidden_dim": args.projector_hidden_dim,
        "manifest": args.manifest,
        "feature_index": args.feature_index,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
    }
    if strict_mode:
        assert llm_identity is not None
        metadata.update(
            {
                **strict_metadata,
                "llm_revision": args.llm_revision,
                "llm_pretrained_identity": dict(llm_identity),
                "llm_pretrained_identity_sha256": llm_identity["identity_sha256"],
            }
        )
    if accelerator.is_main_process:
        save_json(out_dir / "metadata.json", metadata)
    accelerator.wait_for_everyone()

    global_step = 0
    projector.train()
    for epoch in range(args.epochs):
        progress = tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"epoch {epoch + 1}",
        )
        for batch in progress:
            with accelerator.accumulate(projector):
                features = batch["features"].to(accelerator.device, dtype=torch.float32)
                feature_mask = batch["feature_mask"].to(accelerator.device)
                visual_embeds = projector(features)
                inputs_embeds, attention_mask, labels = build_inputs_embeds_and_labels(
                    tokenizer=tokenizer,
                    llm=llm,
                    visual_embeds=visual_embeds,
                    feature_mask=feature_mask,
                    prefixes=batch["prefixes"],
                    suffixes=batch["suffixes"],
                    answers=batch["answers"],
                    max_length=args.max_length,
                )
                outputs = llm(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            progress.set_postfix(loss=f"{loss.detach().float().item():.4f}")
            if (
                accelerator.is_main_process
                and args.save_every_steps > 0
                and global_step % args.save_every_steps == 0
            ):
                save_checkpoint(
                    out_dir, accelerator.unwrap_model(projector), global_step, metadata
                )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(
            out_dir, accelerator.unwrap_model(projector), global_step, metadata
        )
        print(f"Saved final projector checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
