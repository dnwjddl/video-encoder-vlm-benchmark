"""Shared loading and validation for the locked experimental protocol."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import re
from typing import Any, Mapping

import yaml

from .io import sha256_file
from .schema import InformationFamily


DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("configs") / "protocol.yaml"
TRIAL_BUILD_PROTOCOL_SCHEMA_VERSION = "information_upper_bound.trial_build_protocol.v1"
TRIAL_BUILD_LATE_BOUND_SECTIONS = frozenset({"projector"})


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def trial_build_protocol_payload(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return the preregistered protocol content available before projector training.

    The projector lock is necessarily produced *after* the held-out trial
    manifest exists, because its metadata binds that manifest. Excluding only
    this explicitly late-bound section prevents a hash cycle while keeping the
    data release, condition matrix, model, sampling, analysis, comparisons, and
    dataset roles committed by every trial-build attestation.
    """

    if not isinstance(protocol, Mapping):
        raise ValueError("trial-build protocol must be a mapping")
    committed = {
        str(key): value
        for key, value in protocol.items()
        if str(key) not in TRIAL_BUILD_LATE_BOUND_SECTIONS
    }
    if not committed:
        raise ValueError("trial-build protocol has no preregistered content")
    return {
        "schema_version": TRIAL_BUILD_PROTOCOL_SCHEMA_VERSION,
        "excluded_late_bound_sections": sorted(TRIAL_BUILD_LATE_BOUND_SECTIONS),
        "protocol": committed,
    }


def trial_build_protocol_sha256(protocol: Mapping[str, Any]) -> str:
    """Hash preregistered protocol content without late-bound projector locks."""

    return _canonical_sha256(trial_build_protocol_payload(protocol))


def _locked_sha256(value: Any, *, field: str) -> str:
    converted = str(value or "").strip().lower()
    if len(converted) != 64 or any(
        character not in "0123456789abcdef" for character in converted
    ):
        raise ValueError(
            f"locked protocol {field} must be a 64-character SHA256; replace template values "
            "before the confirmatory run"
        )
    return converted


def validate_dataset_roles(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_roles = protocol.get("dataset_roles")
    if not isinstance(raw_roles, Mapping) or not raw_roles:
        raise ValueError("locked protocol dataset_roles must be a non-empty mapping")
    allowed_families = {value.value for value in InformationFamily}
    validated: dict[str, dict[str, Any]] = {}
    for raw_name, raw_role in raw_roles.items():
        name = str(raw_name).strip()
        if not name:
            raise ValueError(
                "locked protocol dataset_roles contains an empty dataset name"
            )
        if not isinstance(raw_role, Mapping):
            raise ValueError(f"locked protocol dataset_roles.{name} must be a mapping")
        families = raw_role.get("information_families")
        if (
            not isinstance(families, (list, tuple))
            or isinstance(families, (str, bytes))
            or not families
        ):
            raise ValueError(
                f"locked protocol dataset_roles.{name}.information_families must be "
                "a non-empty sequence"
            )
        normalized = [str(value).strip() for value in families]
        unknown = sorted(set(normalized) - allowed_families)
        if unknown:
            raise ValueError(
                f"locked protocol dataset_roles.{name} has unknown information families: "
                f"{unknown}"
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                f"locked protocol dataset_roles.{name}.information_families contains duplicates"
            )
        primary_use = str(raw_role.get("primary_use", "")).strip()
        if not primary_use:
            raise ValueError(
                f"locked protocol dataset_roles.{name}.primary_use must be non-empty"
            )
        validated[name] = {
            **dict(raw_role),
            "information_families": normalized,
            "primary_use": primary_use,
        }
    return validated


def load_protocol(
    path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path) if path is not None else DEFAULT_PROTOCOL_PATH
    if not source.is_file():
        raise FileNotFoundError(f"locked protocol config does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"locked protocol must be a YAML mapping: {source}")
    protocol = dict(value)
    schema_version = str(protocol.get("schema_version", "")).strip()
    if not schema_version:
        raise ValueError(f"locked protocol has no schema_version: {source}")
    validate_dataset_roles(protocol)
    return protocol, {
        "path": str(source.resolve()),
        "sha256": sha256_file(source),
        "schema_version": schema_version,
    }


def protocol_section(protocol: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = protocol.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"locked protocol section {name!r} must be a mapping")
    return dict(value)


def validate_data_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the preregistered official-data lock and dataset coverage."""

    value = protocol_section(protocol, "data")
    data_release_sha256 = _locked_sha256(
        value.get("data_release_sha256"), field="data.data_release_sha256"
    )
    conditions_sha256 = _locked_sha256(
        value.get("conditions_sha256"), field="data.conditions_sha256"
    )
    raw_required = value.get("required_datasets")
    if (
        not isinstance(raw_required, (list, tuple))
        or isinstance(raw_required, (str, bytes))
        or not raw_required
    ):
        raise ValueError(
            "locked protocol data.required_datasets must be a non-empty sequence"
        )
    required = [str(item).strip() for item in raw_required]
    if any(not item for item in required) or len(set(required)) != len(required):
        raise ValueError(
            "locked protocol data.required_datasets must contain unique non-empty names"
        )
    declared_roles = validate_dataset_roles(protocol)
    unknown = sorted(set(required) - set(declared_roles))
    if unknown:
        raise ValueError(
            "locked protocol data.required_datasets are missing from dataset_roles: "
            f"{unknown}"
        )
    raw_contract = value.get("coverage_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("locked protocol data.coverage_contract must be a mapping")
    if (
        raw_contract.get("schema_version")
        != "information_upper_bound.coverage_contract.v1"
    ):
        raise ValueError("locked protocol has an unsupported coverage_contract schema")
    raw_dataset_contracts = raw_contract.get("datasets")
    if not isinstance(raw_dataset_contracts, Mapping):
        raise ValueError("locked protocol coverage_contract.datasets must be a mapping")
    if set(map(str, raw_dataset_contracts)) != set(required):
        raise ValueError(
            "coverage_contract dataset keys must exactly match data.required_datasets"
        )
    dataset_contracts: dict[str, dict[str, Any]] = {}
    adapter_run_pattern = re.compile(r"^adapter-run::[0-9a-f]{64}$")
    for dataset in required:
        raw_dataset = raw_dataset_contracts.get(dataset)
        if not isinstance(raw_dataset, Mapping):
            raise ValueError(f"coverage contract for {dataset} must be a mapping")
        try:
            required_runs = int(raw_dataset.get("required_adapter_runs"))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"coverage contract {dataset}.required_adapter_runs must be positive"
            ) from exc
        if required_runs < 1:
            raise ValueError(
                f"coverage contract {dataset}.required_adapter_runs must be positive"
            )
        raw_run_ids = raw_dataset.get("required_adapter_run_ids")
        if not isinstance(raw_run_ids, list) or len(raw_run_ids) != required_runs:
            raise ValueError(
                f"coverage contract {dataset}.required_adapter_run_ids must contain "
                f"exactly {required_runs} IDs"
            )
        run_ids = [str(value).strip() for value in raw_run_ids]
        if len(set(run_ids)) != len(run_ids) or any(
            adapter_run_pattern.fullmatch(value) is None for value in run_ids
        ):
            raise ValueError(
                f"coverage contract {dataset}.required_adapter_run_ids must be unique "
                "adapter-run::<64 hex> identities; replace all templates"
            )

        def string_list(name: str, *, allow_empty: bool = False) -> list[str]:
            raw_values = raw_dataset.get(name)
            if not isinstance(raw_values, list) or (not allow_empty and not raw_values):
                qualifier = "a sequence" if allow_empty else "a non-empty sequence"
                raise ValueError(
                    f"coverage contract {dataset}.{name} must be {qualifier}"
                )
            values = [str(item).strip() for item in raw_values]
            if any(not item or "REPLACE_" in item for item in values):
                raise ValueError(
                    f"coverage contract {dataset}.{name} contains an empty/template value"
                )
            if len(set(values)) != len(values):
                raise ValueError(
                    f"coverage contract {dataset}.{name} contains duplicates"
                )
            return values

        required_splits = string_list("required_splits")
        required_families = string_list("required_information_families")
        unknown_families = sorted(
            set(required_families)
            - set(declared_roles[dataset]["information_families"])
        )
        if unknown_families:
            raise ValueError(
                f"coverage contract {dataset} declares families outside dataset_roles: "
                f"{unknown_families}"
            )
        question_families = string_list("required_question_families", allow_empty=True)
        source_roles = string_list("required_source_roles")
        minimums: dict[str, int] = {}
        for name in (
            "minimum_records",
            "minimum_records_with_evidence",
            "minimum_records_with_safe_oracles",
        ):
            raw_minimum = raw_dataset.get(name, 0)
            if isinstance(raw_minimum, bool):
                raise ValueError(f"coverage contract {dataset}.{name} must be >= 0")
            try:
                minimum = int(raw_minimum)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"coverage contract {dataset}.{name} must be >= 0"
                ) from exc
            if minimum < 0:
                raise ValueError(f"coverage contract {dataset}.{name} must be >= 0")
            minimums[name] = minimum
        dataset_contracts[dataset] = {
            **dict(raw_dataset),
            "required_adapter_runs": required_runs,
            "required_adapter_run_ids": run_ids,
            "required_splits": required_splits,
            "required_information_families": required_families,
            "required_question_families": question_families,
            "required_source_roles": source_roles,
            **minimums,
        }
    return {
        **value,
        "data_release_sha256": data_release_sha256,
        "conditions_sha256": conditions_sha256,
        "required_datasets": required,
        "coverage_contract": {
            **dict(raw_contract),
            "datasets": dataset_contracts,
        },
    }


def validate_release_coverage(
    data_protocol: Mapping[str, Any],
    lock_metadata: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Enforce the preregistered release scope against rows and adapter runs."""

    contract = data_protocol.get("coverage_contract")
    if not isinstance(contract, Mapping) or not isinstance(
        contract.get("datasets"), Mapping
    ):
        raise ValueError("validated data protocol has no coverage contract")
    adapter_runs = lock_metadata.get("adapter_runs")
    if not isinstance(adapter_runs, list):
        raise ValueError("validated data lock does not expose canonical adapter_runs")

    rows_by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for row in records:
        diagnostic = row.get("diagnostic")
        if not isinstance(diagnostic, Mapping):
            raise ValueError(f"record {row.get('id')!r} has no diagnostic metadata")
        dataset = str(diagnostic.get("dataset", "")).strip()
        rows_by_dataset.setdefault(dataset, []).append(row)
    runs_by_dataset: dict[str, list[Mapping[str, Any]]] = {}
    for run in adapter_runs:
        if not isinstance(run, Mapping):
            raise ValueError("validated data lock contains a malformed adapter run")
        runs_by_dataset.setdefault(str(run.get("dataset", "")), []).append(run)

    required_datasets = set(map(str, data_protocol.get("required_datasets") or []))
    if (
        set(rows_by_dataset) != required_datasets
        or set(runs_by_dataset) != required_datasets
    ):
        raise ValueError(
            "coverage contract requires an exact dataset set; "
            f"rows={sorted(rows_by_dataset)}, runs={sorted(runs_by_dataset)}, "
            f"required={sorted(required_datasets)}"
        )

    observed: dict[str, Any] = {}
    for dataset in sorted(required_datasets):
        dataset_contract = contract["datasets"].get(dataset)
        if not isinstance(dataset_contract, Mapping):
            raise ValueError(f"coverage contract is missing dataset {dataset}")
        dataset_rows = rows_by_dataset[dataset]
        dataset_runs = runs_by_dataset[dataset]
        actual_run_ids = {str(run.get("adapter_run_id", "")) for run in dataset_runs}
        expected_run_ids = set(dataset_contract["required_adapter_run_ids"])
        if actual_run_ids != expected_run_ids:
            raise ValueError(
                f"coverage contract adapter runs mismatch for {dataset}: "
                f"actual={sorted(actual_run_ids)}, required={sorted(expected_run_ids)}"
            )
        diagnostic_rows = [dict(row["diagnostic"]) for row in dataset_rows]
        splits = {str(value.get("split", "")) for value in diagnostic_rows}
        required_splits = set(dataset_contract["required_splits"])
        run_splits = {str(run.get("canonical_split", "")) for run in dataset_runs}
        if splits != required_splits or run_splits != required_splits:
            raise ValueError(
                f"coverage contract split mismatch for {dataset}: "
                f"rows={sorted(splits)}, runs={sorted(run_splits)}, "
                f"required={sorted(required_splits)}"
            )
        information_families = {
            str(value.get("information_family", "")) for value in diagnostic_rows
        }
        required_families = set(dataset_contract["required_information_families"])
        if information_families != required_families:
            raise ValueError(
                f"coverage contract information-family mismatch for {dataset}: "
                f"actual={sorted(information_families)}, required={sorted(required_families)}"
            )
        question_families = {
            str(value.get("question_family", "")) for value in diagnostic_rows
        }
        required_questions = set(dataset_contract["required_question_families"])
        exact_questions = dataset_contract.get("exact_question_family_set") is True
        question_ok = (
            question_families == required_questions
            if exact_questions
            else required_questions.issubset(question_families)
        )
        if not question_ok:
            raise ValueError(
                f"coverage contract question-family mismatch for {dataset}: "
                f"actual={sorted(question_families)}, required={sorted(required_questions)}, "
                f"exact={exact_questions}"
            )
        source_roles = {
            str(artifact.get("role", ""))
            for run in dataset_runs
            for artifact in (run.get("source_artifacts") or [])
            if isinstance(artifact, Mapping)
        }
        required_source_roles = set(dataset_contract["required_source_roles"])
        if not required_source_roles.issubset(source_roles):
            raise ValueError(
                f"coverage contract source roles missing for {dataset}: "
                f"missing={sorted(required_source_roles - source_roles)}"
            )
        evidence_records = sum(
            bool(value.get("evidence_spans")) for value in diagnostic_rows
        )
        safe_oracle_records = 0
        for diagnostic in diagnostic_rows:
            oracles = diagnostic.get("oracles")
            facts = (
                [
                    fact
                    for values in oracles.values()
                    if isinstance(values, list)
                    for fact in values
                    if isinstance(fact, Mapping)
                ]
                if isinstance(oracles, Mapping)
                else []
            )
            safe_oracle_records += any(
                fact.get("access") == "safe_visual_gt" for fact in facts
            )
        minimum_checks = {
            "minimum_records": len(dataset_rows),
            "minimum_records_with_evidence": evidence_records,
            "minimum_records_with_safe_oracles": safe_oracle_records,
        }
        for name, actual in minimum_checks.items():
            required_minimum = int(dataset_contract.get(name, 0))
            if actual < required_minimum:
                raise ValueError(
                    f"coverage contract {dataset}.{name} requires >= {required_minimum}, "
                    f"observed {actual}"
                )
        observed[dataset] = {
            "records": len(dataset_rows),
            "adapter_run_ids": sorted(actual_run_ids),
            "splits": sorted(splits),
            "information_families": sorted(information_families),
            "question_families": sorted(question_families),
            "source_roles": sorted(source_roles),
            "records_with_evidence": evidence_records,
            "records_with_safe_oracles": safe_oracle_records,
        }
    return {
        "schema_version": "information_upper_bound.coverage_validation.v1",
        "valid": True,
        "coverage_contract_sha256": _canonical_sha256(contract),
        "datasets": observed,
    }


def validate_frozen_model_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    model = protocol_section(protocol, "model")
    for key in (
        "llm_frozen",
        "visual_encoder_frozen",
        "projector_frozen_during_evaluation",
    ):
        if model.get(key) is not True:
            raise ValueError(f"locked protocol requires model.{key}: true")
    llm_id = str(model.get("llm_id", "")).strip()
    if not llm_id:
        raise ValueError("locked protocol model.llm_id must be non-empty")
    llm_revision = str(model.get("llm_revision") or "").strip()
    if (
        len(llm_revision) < 40
        or len(llm_revision) > 64
        or any(
            character not in "0123456789abcdef" for character in llm_revision.lower()
        )
    ):
        raise ValueError(
            "locked protocol model.llm_revision must pin a 40-64 character hexadecimal "
            "immutable commit before confirmatory scoring"
        )
    try:
        max_length = int(model.get("max_length"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "locked protocol model.max_length must be a positive integer"
        ) from exc
    if max_length < 1:
        raise ValueError("locked protocol model.max_length must be a positive integer")
    overflow_policy = str(model.get("overflow_policy", "")).strip()
    if overflow_policy not in {"error", "truncate_visual"}:
        raise ValueError(
            "locked protocol model.overflow_policy must be 'error' or 'truncate_visual'"
        )
    dtype = str(model.get("dtype", "")).strip().lower()
    if dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(
            "locked protocol model.dtype must be 'bf16', 'fp16', or 'fp32'"
        )
    return {
        **model,
        "llm_id": llm_id,
        "llm_revision": llm_revision,
        "max_length": max_length,
        "dtype": dtype,
        "overflow_policy": overflow_policy,
    }


def validate_locked_projector_protocol(
    protocol: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    metadata_sha256: str,
    projector_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the selected projector and the feature pipeline it learned."""

    value = protocol_section(protocol, "projector")
    locked = {
        name: _locked_sha256(value.get(name), field=f"projector.{name}")
        for name in (
            "checkpoint_sha256",
            "metadata_sha256",
            "training_manifest_sha256",
            "evaluation_manifest_sha256",
            "training_feature_index_sha256",
            "training_feature_metadata_sha256",
            "training_feature_artifact_root_sha256",
            "evaluation_feature_index_sha256",
            "evaluation_feature_metadata_sha256",
            "evaluation_feature_artifact_root_sha256",
            "evaluation_trial_matrix_closure_sha256",
            "evaluation_trial_set_root_sha256",
            "encoder_extraction_pipeline_identity_sha256",
            "llm_pretrained_identity_sha256",
        )
    }
    actual = {
        "checkpoint_sha256": str(checkpoint_sha256).lower(),
        "metadata_sha256": str(metadata_sha256).lower(),
        "training_manifest_sha256": str(
            projector_metadata.get("training_manifest_sha256", "")
        ).lower(),
        "evaluation_manifest_sha256": str(
            projector_metadata.get("evaluation_manifest_sha256", "")
        ).lower(),
        "training_feature_index_sha256": str(
            projector_metadata.get("training_feature_index_sha256", "")
        ).lower(),
        "training_feature_metadata_sha256": str(
            projector_metadata.get("training_feature_metadata_sha256", "")
        ).lower(),
        "training_feature_artifact_root_sha256": str(
            projector_metadata.get("training_feature_artifact_root_sha256", "")
        ).lower(),
        "evaluation_feature_index_sha256": str(
            projector_metadata.get("evaluation_feature_index_sha256", "")
        ).lower(),
        "evaluation_feature_metadata_sha256": str(
            projector_metadata.get("evaluation_feature_metadata_sha256", "")
        ).lower(),
        "evaluation_feature_artifact_root_sha256": str(
            projector_metadata.get("evaluation_feature_artifact_root_sha256", "")
        ).lower(),
        "evaluation_trial_matrix_closure_sha256": str(
            projector_metadata.get("evaluation_trial_matrix_closure_sha256", "")
        ).lower(),
        "evaluation_trial_set_root_sha256": str(
            projector_metadata.get("evaluation_trial_set_root_sha256", "")
        ).lower(),
        "encoder_extraction_pipeline_identity_sha256": str(
            projector_metadata.get("encoder_extraction_pipeline_identity_sha256", "")
        ).lower(),
        "llm_pretrained_identity_sha256": str(
            projector_metadata.get("llm_pretrained_identity_sha256", "")
        ).lower(),
    }
    if value.get("training_data_release_sha256") not in (None, ""):
        locked["training_data_release_sha256"] = _locked_sha256(
            value.get("training_data_release_sha256"),
            field="projector.training_data_release_sha256",
        )
        training_data_lock = projector_metadata.get("training_data_lock")
        actual["training_data_release_sha256"] = str(
            training_data_lock.get("data_release_sha256", "")
            if isinstance(training_data_lock, Mapping)
            else ""
        ).lower()
    training_dtype = str(value.get("training_dtype", "")).strip().lower()
    if training_dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(
            "locked protocol projector.training_dtype must be 'bf16', 'fp16', or 'fp32'"
        )
    raw_evaluation_trial_count = value.get("evaluation_trial_count")
    if isinstance(raw_evaluation_trial_count, bool):
        raise ValueError(
            "locked protocol projector.evaluation_trial_count must be a positive integer"
        )
    try:
        training_max_length = int(value.get("training_max_length"))
        training_seed = int(value.get("training_seed"))
        evaluation_trial_count = int(raw_evaluation_trial_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "locked protocol projector training_max_length, training_seed, and "
            "evaluation_trial_count must be integers"
        ) from exc
    if training_max_length < 1:
        raise ValueError(
            "locked protocol projector.training_max_length must be positive"
        )
    if evaluation_trial_count < 1:
        raise ValueError(
            "locked protocol projector.evaluation_trial_count must be positive"
        )
    locked.update(
        {
            "training_dtype": training_dtype,
            "training_max_length": training_max_length,
            "training_seed": training_seed,
            "evaluation_trial_count": evaluation_trial_count,
        }
    )
    actual_evaluation_trial_count = projector_metadata.get("evaluation_trial_count")
    if isinstance(actual_evaluation_trial_count, bool):
        raise ValueError(
            "projector metadata evaluation_trial_count must be a positive integer"
        )
    actual.update(
        {
            "training_dtype": str(projector_metadata.get("dtype", "")).lower(),
            "training_max_length": projector_metadata.get("max_length"),
            "training_seed": projector_metadata.get("seed"),
            "evaluation_trial_count": actual_evaluation_trial_count,
        }
    )
    raw_closure = projector_metadata.get("evaluation_trial_matrix_closure")
    if not isinstance(raw_closure, Mapping):
        raise ValueError(
            "projector metadata has no evaluation_trial_matrix_closure object"
        )
    closure_payload = dict(raw_closure)
    declared_closure_sha256 = str(closure_payload.pop("closure_sha256", "")).lower()
    if (
        closure_payload.get("status") != "exact"
        or _canonical_sha256(closure_payload) != declared_closure_sha256
        or declared_closure_sha256 != actual["evaluation_trial_matrix_closure_sha256"]
        or str(closure_payload.get("trial_set_root_sha256", "")).lower()
        != actual["evaluation_trial_set_root_sha256"]
        or closure_payload.get("trial_count") != actual["evaluation_trial_count"]
    ):
        raise ValueError(
            "projector evaluation trial-matrix closure is missing or internally inconsistent"
        )
    mismatches = [name for name in locked if actual[name] != locked[name]]
    if mismatches:
        details = ", ".join(
            f"{name}: locked={locked[name]!r}, actual={actual[name]!r}"
            for name in mismatches
        )
        raise ValueError(
            f"projector provenance does not match the locked protocol ({details})"
        )
    return locked
