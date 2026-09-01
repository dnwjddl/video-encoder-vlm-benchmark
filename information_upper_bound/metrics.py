"""Deterministic statistics for the information-upper-bound protocol.

The analysis deliberately keeps coverage separate from performance.  A missing
counterfactual, option permutation, or reference trial is never interpreted as
a failure *or* a success: it is excluded from the corresponding estimand and
reported in the coverage fields and in ``report.json``.

The numerical analysis itself uses only NumPy and the Python standard library;
shared package modules provide protocol and provenance authentication.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import re
import tempfile
import os
from typing import Any, Callable, Iterable, Mapping, Sequence
import unicodedata

try:
    import numpy as np
except ImportError:  # pragma: no cover - requirements.txt installs NumPy.
    np = None  # type: ignore[assignment]

from .conditions import trial_content_sha256
from .attestation import validate_trial_build_attestation
from .integrity import (
    RESULT_INTEGRITY_SCHEMA_VERSION,
    canonical_sha256,
    scored_result_sha256,
    trial_set_identity,
)
from .io import sha256_file
from .protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
    protocol_section,
    validate_data_protocol,
    validate_frozen_model_protocol,
)
from .score_partition import score_worker_index, validate_score_partition
from .scoring import SCORING_PROTOCOL_VERSION

DEFAULT_BOOTSTRAP_REPLICATES = 2000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SEED = 42
DEFAULT_REFERENCE_CONDITION = "full_video"
DEFAULT_ECE_BINS = 10
DEFAULT_MINIMUM_CONFIRMATORY_RESAMPLING_UNITS = 1
BOOTSTRAP_MAX_BATCH_REPLICATES = 256
BOOTSTRAP_MAX_COUNT_CELLS = 1_000_000
ANALYSIS_OUTPUT_FILENAMES = (
    "summary.csv",
    "comparisons.csv",
    "pair_metrics.csv",
    "dose_curves.csv",
    "report.json",
)
CLEVRER_SEMANTIC_AGGREGATION_RULE = (
    "normalize option text with NFKC, case-folding, and whitespace collapse; "
    "when the build attestation declares option_permutations='all', require each "
    "candidate's permutation indices to be exactly 0..number_of_options-1; "
    "normalize each permutation's option probabilities; mean the probability "
    "assigned to each semantic option across every authenticated permutation; "
    "predict only the unique maximum; score an official question correct only "
    "when every official candidate is correct (v1)"
)
CLEVRER_QUESTION_METRIC_NAMES = (
    "official_question_exact_set_accuracy",
    "official_question_permutation_robustness_accuracy",
)
CLEVRER_PAIRED_METRIC_NAMES = (
    "condition_official_question_exact_set_accuracy",
    "reference_official_question_exact_set_accuracy",
    "official_question_exact_set_accuracy_gain",
)
SUMMARY_METRIC_NAMES = (
    "accuracy",
    "row_micro_accuracy",
    "cluster_macro_accuracy",
    "cluster_all_rows_correct",
    "gold_margin",
    "gold_probability",
    "brier",
    "confidence_brier",
    "ece",
)
PAIRED_METRIC_NAMES = (
    "condition_accuracy",
    "reference_accuracy",
    "accuracy_gain",
    "condition_gold_margin",
    "reference_gold_margin",
    "gold_margin_gain",
)

STRATUM_FIELDS = (
    "dataset",
    "information_family",
    "question_family",
    "reasoning_depth",
)
GROUP_FIELDS = STRATUM_FIELDS + (
    "condition",
    "requested_dose",
    "effective_dose",
)
PREDICTION_FIELDS = (
    "trial_id",
    "base_id",
    "visual_id",
    "dataset",
    "information_family",
    "question_family",
    "reasoning_depth",
    "resampling_unit_id",
    "pair_id",
    "pair_role",
    "independent_unit_id",
    "condition",
    "input_channel",
    "requested_dose",
    "effective_dose",
    "permutation_index",
    "prediction",
    "prediction_text",
    "scoring_global_signature_sha256",
    "scoring_run_signature_sha256",
    "result_content_sha256",
    "trial_content_sha256",
    "data_release_sha256",
    "trial_build_attestation_sha256",
    "choices",
    "answer",
    "answer_text",
    "correct",
    "choice_probability",
    "choice_nll",
    "gold_nll",
    "best_distractor_nll",
    "gold_margin",
    "original_visual_tokens",
    "effective_visual_tokens",
    "token_source",
    "official_candidate_id",
    "official_candidate_count",
)
DESIGN_FIELDS = (
    "base_id",
    "visual_id",
    "dataset",
    "information_family",
    "question_family",
    "reasoning_depth",
    "resampling_unit_id",
    "pair_id",
    "pair_role",
    "independent_unit_id",
    "condition",
    "input_channel",
    "requested_dose",
    "effective_dose",
    "permutation_index",
    "trial_content_sha256",
    "data_release_sha256",
    "trial_build_attestation_sha256",
    "choices",
    "answer",
    "answer_text",
    "official_candidate_id",
    "official_candidate_count",
)


def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _positive_int(value: Any) -> int | None:
    """Return a strict JSON/Python integer token count, excluding bool/float."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _bool_value(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1) and not isinstance(value, str):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _semantic(value: Any) -> str | None:
    if _missing(value):
        return None
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def _dose_value(value: Any) -> Any:
    if _missing(value):
        return "<missing>"
    if isinstance(value, bool):
        return str(value).casefold()
    number = _finite_float(value)
    if number is not None and str(value).strip().casefold() != "all":
        return int(number) if number.is_integer() else number
    return str(value).strip()


def _group_value(value: Any) -> Any:
    if _missing(value):
        return "<missing>"
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _sort_key(values: Sequence[Any]) -> tuple[tuple[int, Any], ...]:
    result: list[tuple[int, Any]] = []
    for value in values:
        number = _finite_float(value)
        if number is not None and str(value).casefold() != "all":
            result.append((0, number))
        else:
            result.append((1, str(value)))
    return tuple(result)


def _stable_seed(seed: int, *context: Any) -> int:
    payload = json.dumps(
        [int(seed), *context], ensure_ascii=False, sort_keys=True, default=str
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _mean(values: Sequence[float | int | bool]) -> float:
    if not values:
        raise ValueError("mean requires at least one value")
    return math.fsum(float(value) for value in values) / len(values)


def _linear_quantile(sorted_values: Sequence[float], quantile: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = min(max(float(quantile), 0.0), 1.0) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
    )


def _percentile_interval(
    values: Sequence[float], confidence_level: float
) -> tuple[float | None, float | None]:
    finite = sorted(float(value) for value in values if math.isfinite(value))
    if not finite:
        return None, None
    alpha = (1.0 - confidence_level) / 2.0
    return _linear_quantile(finite, alpha), _linear_quantile(finite, 1.0 - alpha)


def flatten_trial_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten either a trial-manifest row or a scored prediction row."""

    result = dict(row)
    attestation = row.get("trial_build_attestation")
    if _missing(result.get("trial_build_attestation_sha256")) and isinstance(
        attestation, Mapping
    ):
        result["trial_build_attestation_sha256"] = attestation.get("attestation_sha256")
    diagnostic = row.get("diagnostic")
    if isinstance(diagnostic, Mapping):
        for key in (
            "dataset",
            "information_family",
            "question_family",
            "reasoning_depth",
            "resampling_unit_id",
            "pair_id",
            "pair_role",
            "independent_unit_id",
            "official_candidate_id",
            "official_candidate_count",
        ):
            if _missing(result.get(key)) and key in diagnostic:
                result[key] = diagnostic.get(key)

    condition = row.get("condition")
    if isinstance(condition, Mapping):
        result["condition"] = condition.get("name")
        for key in (
            "input_channel",
            "requested_dose",
            "effective_dose",
            "permutation_index",
        ):
            if _missing(result.get(key)) and key in condition:
                result[key] = condition.get(key)
    if _missing(result.get("trial_id")) and not _missing(result.get("id")):
        result["trial_id"] = result.get("id")
    if _missing(result.get("base_id")) and not _missing(result.get("trial_id")):
        result["base_id"] = result.get("trial_id")
    if _missing(result.get("permutation_index")):
        result["permutation_index"] = 0
    return result


class _Issues:
    def __init__(self, *, example_limit: int = 100) -> None:
        self.counts: Counter[str] = Counter()
        self.examples: list[dict[str, Any]] = []
        self.example_limit = example_limit

    def add(self, kind: str, **detail: Any) -> None:
        self.counts[kind] += 1
        if len(self.examples) < self.example_limit:
            self.examples.append({"kind": kind, **detail})

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "examples": self.examples,
            "examples_truncated": sum(self.counts.values()) > len(self.examples),
        }


def _unique_index(
    rows: Iterable[Mapping[str, Any]], *, source: str, issues: _Issues
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    id_counts: Counter[str] = Counter()
    missing_ids = 0
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            issues.add(f"{source}_non_object_row", row_index=index)
            continue
        if source == "manifest":
            declared_hash = raw.get("trial_content_sha256")
            if _missing(declared_hash):
                issues.add("manifest_missing_trial_content_sha256", row_index=index)
            else:
                try:
                    recomputed_hash = trial_content_sha256(raw)
                except (TypeError, ValueError) as exc:
                    issues.add(
                        "manifest_trial_content_hash_recompute_error",
                        row_index=index,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                else:
                    if str(declared_hash) != recomputed_hash:
                        issues.add(
                            "stale_manifest_trial_content_sha256",
                            row_index=index,
                            declared=str(declared_hash),
                            recomputed=recomputed_hash,
                        )
        row = flatten_trial_row(raw)
        trial_id = row.get("trial_id")
        if _missing(trial_id):
            missing_ids += 1
            issues.add(f"{source}_missing_trial_id", row_index=index)
            continue
        row["trial_id"] = str(trial_id)
        row["_row_index"] = index
        flattened.append(row)
        id_counts[row["trial_id"]] += 1

    duplicates = sorted(key for key, count in id_counts.items() if count > 1)
    for trial_id in duplicates:
        issues.add(
            f"duplicate_{source}_trial_id", trial_id=trial_id, count=id_counts[trial_id]
        )
    duplicate_set = set(duplicates)
    unique = {
        row["trial_id"]: row
        for row in flattened
        if row["trial_id"] not in duplicate_set
    }
    return unique, {
        "raw_rows": len(flattened) + missing_ids,
        "rows_with_trial_id": len(flattened),
        "missing_trial_id_rows": missing_ids,
        "duplicate_trial_ids": duplicates,
        "duplicate_rows": sum(id_counts[key] for key in duplicate_set),
        "usable_unique_trials": len(unique),
    }


def _audit_official_candidate_sets(
    rows: Sequence[Mapping[str, Any]], issues: _Issues
) -> None:
    """Authenticate every CLEVRER candidate set in the expected manifest."""

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("dataset", "")).strip().casefold() != "clevrer":
            continue
        if _missing(row.get("resampling_unit_id")):
            issues.add(
                "clevrer_missing_resampling_unit_id",
                trial_id=row.get("trial_id"),
            )
        independent_unit_id = row.get("independent_unit_id")
        if _missing(independent_unit_id):
            issues.add(
                "clevrer_missing_independent_unit_id",
                trial_id=row.get("trial_id"),
            )
            continue
        key = (
            str(independent_unit_id),
            _group_value(row.get("condition")),
            _dose_value(row.get("requested_dose")),
            _dose_value(row.get("effective_dose")),
            _group_value(row.get("permutation_index", 0)),
        )
        groups[key].append(row)
    for key, candidates in sorted(groups.items(), key=lambda item: _sort_key(item[0])):
        counts = {
            _positive_int(row.get("official_candidate_count")) for row in candidates
        }
        candidate_ids = [
            str(row.get("official_candidate_id", "")).strip() for row in candidates
        ]
        if None in counts or len(counts) != 1:
            issues.add(
                "clevrer_invalid_official_candidate_count",
                group=list(key),
                values=sorted(str(value) for value in counts),
            )
            continue
        expected_count = next(iter(counts))
        assert expected_count is not None
        if any(not value for value in candidate_ids):
            issues.add(
                "clevrer_missing_official_candidate_id",
                group=list(key),
            )
        if len(set(candidate_ids)) != len(candidate_ids):
            issues.add(
                "clevrer_duplicate_official_candidate_id",
                group=list(key),
                candidate_ids=candidate_ids,
            )
        if (
            len(candidates) != expected_count
            or len(set(candidate_ids)) != expected_count
        ):
            issues.add(
                "clevrer_incomplete_official_candidate_set",
                group=list(key),
                expected_candidate_count=expected_count,
                observed_rows=len(candidates),
                observed_unique_candidate_ids=len(set(candidate_ids)),
            )


def _prepare_rows(
    prediction_rows: Iterable[Mapping[str, Any]],
    expected_rows: Iterable[Mapping[str, Any]] | None,
    *,
    external_issues: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    issues = _Issues()
    for issue in external_issues:
        issues.add(
            str(issue.get("kind", "input_issue")),
            **{k: v for k, v in issue.items() if k != "kind"},
        )

    prediction_index, prediction_stats = _unique_index(
        prediction_rows, source="prediction", issues=issues
    )
    if expected_rows is None:
        expected_index: dict[str, dict[str, Any]] = {}
        expected_stats: dict[str, Any] | None = None
    else:
        expected_index, expected_stats = _unique_index(
            expected_rows, source="manifest", issues=issues
        )
        _audit_official_candidate_sets(list(expected_index.values()), issues)

    observed: list[dict[str, Any]] = []
    unexpected: list[str] = []
    metadata_mismatches = 0
    scoring_run_signatures: set[str] = set()
    scoring_global_signatures: set[str] = set()
    projected_token_counts: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for trial_id, prediction in prediction_index.items():
        row = dict(prediction)
        declared_result_digest = str(row.get("result_content_sha256", ""))
        if re.fullmatch(r"[0-9a-f]{64}", declared_result_digest) is None:
            issues.add(
                "missing_or_invalid_result_content_sha256",
                trial_id=trial_id,
                value=declared_result_digest,
            )
        else:
            recomputed_result_digest = scored_result_sha256(row)
            if declared_result_digest != recomputed_result_digest:
                issues.add(
                    "result_content_sha256_mismatch",
                    trial_id=trial_id,
                    declared=declared_result_digest,
                    recomputed=recomputed_result_digest,
                )
        global_signature = row.get("scoring_global_signature_sha256")
        if _missing(global_signature):
            issues.add("missing_scoring_global_signature_sha256", trial_id=trial_id)
        else:
            normalized_global = str(global_signature)
            scoring_global_signatures.add(normalized_global)
            if re.fullmatch(r"[0-9a-f]{64}", normalized_global) is None:
                issues.add(
                    "invalid_scoring_global_signature_sha256",
                    trial_id=trial_id,
                    value=normalized_global,
                )
        scoring_signature = row.get("scoring_run_signature_sha256")
        if _missing(scoring_signature):
            issues.add("missing_scoring_run_signature_sha256", trial_id=trial_id)
        else:
            normalized_signature = str(scoring_signature)
            scoring_run_signatures.add(normalized_signature)
            if re.fullmatch(r"[0-9a-f]{64}", normalized_signature) is None:
                issues.add(
                    "invalid_scoring_run_signature_sha256",
                    trial_id=trial_id,
                    value=normalized_signature,
                )
        if expected_rows is not None and trial_id not in expected_index:
            unexpected.append(trial_id)
            issues.add("unexpected_prediction_trial", trial_id=trial_id)
            continue
        expected = expected_index.get(trial_id)
        if expected is not None:
            for field in DESIGN_FIELDS:
                expected_value = expected.get(field)
                prediction_value = row.get(field)
                if not _missing(expected_value):
                    if field in {"trial_content_sha256", "choices"} and _missing(
                        prediction_value
                    ):
                        issues.add(
                            "prediction_manifest_missing_binding_field",
                            trial_id=trial_id,
                            field=field,
                        )
                    if not _missing(prediction_value) and _group_value(
                        prediction_value
                    ) != _group_value(expected_value):
                        metadata_mismatches += 1
                        issues.add(
                            "prediction_manifest_mismatch",
                            trial_id=trial_id,
                            field=field,
                            prediction=prediction_value,
                            expected=expected_value,
                        )
                    row[field] = expected_value
        for field in GROUP_FIELDS:
            if _missing(row.get(field)):
                issues.add("missing_group_field", trial_id=trial_id, field=field)
        if _missing(row.get("input_channel")):
            issues.add("missing_input_channel", trial_id=trial_id)
        correct = _bool_value(row.get("correct"))
        if correct is None:
            issues.add("invalid_correct", trial_id=trial_id, value=row.get("correct"))
        if _finite_float(row.get("gold_margin")) is None:
            issues.add(
                "invalid_gold_margin", trial_id=trial_id, value=row.get("gold_margin")
            )

        if row.get("token_source") == "projected_visual_features":
            original_visual_tokens = _positive_int(row.get("original_visual_tokens"))
            effective_visual_tokens = _positive_int(row.get("effective_visual_tokens"))
            if original_visual_tokens is None:
                issues.add(
                    "invalid_projected_original_visual_tokens",
                    trial_id=trial_id,
                    value=row.get("original_visual_tokens"),
                )
            if effective_visual_tokens is None:
                issues.add(
                    "invalid_projected_effective_visual_tokens",
                    trial_id=trial_id,
                    value=row.get("effective_visual_tokens"),
                )
            if (
                original_visual_tokens is not None
                and effective_visual_tokens is not None
            ):
                if effective_visual_tokens != original_visual_tokens:
                    issues.add(
                        "projected_visual_token_truncation",
                        trial_id=trial_id,
                        original_visual_tokens=original_visual_tokens,
                        effective_visual_tokens=effective_visual_tokens,
                    )
                visual_id = row.get("visual_id")
                if _missing(visual_id):
                    issues.add(
                        "projected_visual_tokens_missing_visual_id", trial_id=trial_id
                    )
                else:
                    projected_token_counts[str(visual_id)].add(
                        (original_visual_tokens, effective_visual_tokens)
                    )

        labels = _choice_labels(row)
        if labels is None:
            # A manifest-backed run must carry the semantic options required to
            # interpret the scored option labels. Prediction-only legacy files
            # remain analyzable, but their probabilities receive the older
            # structural validation below.
            if expected is not None or "choices" in row:
                issues.add(
                    "invalid_choices", trial_id=trial_id, value=row.get("choices")
                )
        else:
            prediction_label = (
                str(row.get("prediction")).strip()
                if not _missing(row.get("prediction"))
                else None
            )
            answer_label = (
                str(row.get("answer")).strip()
                if not _missing(row.get("answer"))
                else None
            )
            if prediction_label not in labels:
                issues.add(
                    "invalid_prediction_label",
                    trial_id=trial_id,
                    prediction=row.get("prediction"),
                    expected_labels=labels,
                )
            if (
                correct is not None
                and prediction_label is not None
                and answer_label is not None
            ):
                implied_correct = prediction_label == answer_label
                if correct != implied_correct:
                    issues.add(
                        "correct_prediction_inconsistency",
                        trial_id=trial_id,
                        correct=correct,
                        prediction=prediction_label,
                        answer=answer_label,
                    )
            if prediction_label in labels:
                if not _missing(row.get("prediction_text")):
                    selected_text = str(
                        row["choices"][ord(prediction_label) - ord("A")]
                    )
                    if _semantic(row.get("prediction_text")) != _semantic(
                        selected_text
                    ):
                        issues.add(
                            "prediction_text_choice_mismatch",
                            trial_id=trial_id,
                            prediction=prediction_label,
                            prediction_text=row.get("prediction_text"),
                            selected_choice=selected_text,
                        )

        probability_error = _choice_probability_error(row)
        if probability_error is not None:
            kind, detail = probability_error
            issues.add(kind, trial_id=trial_id, **detail)
        for kind, detail in _choice_nll_integrity_errors(row):
            issues.add(kind, trial_id=trial_id, **detail)
        canonical_scores = _canonical_choice_scores(row)
        if canonical_scores is not None:
            # All estimands consume one canonical score algebra. Reported
            # convenience fields were authenticated above, but tiny tolerated
            # serialization differences must never flip a near-tie estimand.
            row["choice_probability"] = canonical_scores["choice_probability"]
            row["prediction"] = canonical_scores["prediction"]
            row["prediction_text"] = canonical_scores["prediction_text"]
            if canonical_scores["correct"] is not None:
                correct = bool(canonical_scores["correct"])
                row["correct"] = correct
                row["gold_nll"] = canonical_scores["gold_nll"]
                row["best_distractor_nll"] = canonical_scores["best_distractor_nll"]
                row["gold_margin"] = canonical_scores["gold_margin"]
        row["_correct"] = correct
        observed.append(row)

    for visual_id, token_counts in sorted(projected_token_counts.items()):
        if len(token_counts) > 1:
            issues.add(
                "inconsistent_projected_visual_token_budget",
                visual_id=visual_id,
                token_counts=[list(value) for value in sorted(token_counts)],
            )

    if len(scoring_run_signatures) > 1:
        issues.add(
            "mixed_scoring_run_signatures",
            count=len(scoring_run_signatures),
            signatures=sorted(scoring_run_signatures),
        )
    if len(scoring_global_signatures) > 1:
        issues.add(
            "mixed_scoring_global_signatures",
            count=len(scoring_global_signatures),
            signatures=sorted(scoring_global_signatures),
        )

    expected = list(expected_index.values())
    observed.sort(
        key=lambda row: (str(row.get("trial_id")), int(row.get("_row_index", 0)))
    )
    expected.sort(key=lambda row: str(row.get("trial_id")))
    prediction_ids_seen = set(prediction_index) | set(
        prediction_stats["duplicate_trial_ids"]
    )
    missing_ids = (
        sorted(set(expected_index) - prediction_ids_seen)
        if expected_rows is not None
        else []
    )
    unusable_expected_ids = (
        sorted(set(expected_index) - set(prediction_index))
        if expected_rows is not None
        else []
    )
    for trial_id in missing_ids[:100]:
        issues.add("missing_prediction_trial", trial_id=trial_id)

    field_present: dict[str, int] = {}
    field_non_null: dict[str, int] = {}
    for field in PREDICTION_FIELDS:
        field_present[field] = sum(field in row for row in observed)
        field_non_null[field] = sum(not _missing(row.get(field)) for row in observed)

    expected_count = len(expected_index) if expected_rows is not None else None
    joined_count = len(observed)
    coverage = (
        None
        if expected_count is None
        else (joined_count / expected_count if expected_count else None)
    )
    report = {
        "prediction_input": prediction_stats,
        "manifest_input": expected_stats,
        "expected_trials": expected_count,
        "joined_prediction_trials": joined_count,
        "joined_coverage": coverage,
        "missing_prediction_count": len(missing_ids),
        "missing_prediction_trial_ids": missing_ids,
        "unusable_expected_count": len(unusable_expected_ids),
        "unusable_expected_trial_ids": unusable_expected_ids,
        "unexpected_prediction_count": len(unexpected),
        "unexpected_prediction_trial_ids": sorted(unexpected),
        "metadata_mismatch_count": metadata_mismatches,
        "scoring_run_signature_count": len(scoring_run_signatures),
        "scoring_run_signatures": sorted(scoring_run_signatures),
        "scoring_global_signature_count": len(scoring_global_signatures),
        "scoring_global_signatures": sorted(scoring_global_signatures),
        "valid_correct_rows": sum(row.get("_correct") is not None for row in observed),
        "valid_gold_margin_rows": sum(
            _finite_float(row.get("gold_margin")) is not None for row in observed
        ),
        "valid_probability_rows": sum(
            _probability_observation(row) is not None for row in observed
        ),
        "field_present_count": field_present,
        "field_non_null_count": field_non_null,
        "issues": issues.to_dict(),
    }
    return observed, expected, report


def _probability_observation(row: Mapping[str, Any]) -> dict[str, float] | None:
    if _choice_probability_error(row) is not None:
        return None
    raw = row.get("choice_probability")
    assert isinstance(raw, Mapping)
    probabilities: dict[str, float] = {}
    for label, value in raw.items():
        number = _finite_float(value)
        assert number is not None and number >= 0
        probabilities[str(label)] = number
    total = math.fsum(probabilities.values())
    probabilities = {key: value / total for key, value in probabilities.items()}
    answer = str(row.get("answer")) if not _missing(row.get("answer")) else None
    if answer is None or answer not in probabilities:
        return None
    correct = (
        row.get("_correct") if "_correct" in row else _bool_value(row.get("correct"))
    )
    if correct is None:
        return None
    prediction = (
        str(row.get("prediction")) if not _missing(row.get("prediction")) else None
    )
    confidence = probabilities.get(prediction, max(probabilities.values()))
    multiclass_brier = sum(
        (probability - (1.0 if label == answer else 0.0)) ** 2
        for label, probability in probabilities.items()
    )
    return {
        "gold_probability": probabilities[answer],
        "confidence": confidence,
        "brier": multiclass_brier,
        "confidence_brier": (confidence - float(correct)) ** 2,
        "correct": float(correct),
    }


def _choice_labels(row: Mapping[str, Any]) -> list[str] | None:
    choices = row.get("choices")
    if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
        return None
    if any(_missing(value) for value in choices):
        return None
    return [chr(ord("A") + index) for index in range(len(choices))]


def _canonical_choice_scores(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Derive probabilities, prediction, margin, and correctness from NLL only."""

    labels = _choice_labels(row)
    raw_nll = row.get("choice_nll")
    if labels is None or not isinstance(raw_nll, Mapping):
        return None
    if set(raw_nll) != set(labels):
        return None
    nll: dict[str, float] = {}
    for label in labels:
        value = _finite_float(raw_nll.get(label))
        if value is None or value < 0:
            return None
        nll[label] = value
    minimum = min(nll.values())
    weights = {label: math.exp(-(nll[label] - minimum)) for label in labels}
    denominator = math.fsum(weights.values())
    probabilities = {label: weights[label] / denominator for label in labels}
    predicted = min(labels, key=lambda label: nll[label])
    choices = row["choices"]
    assert isinstance(choices, list)
    result: dict[str, Any] = {
        "choice_probability": probabilities,
        "prediction": predicted,
        "prediction_text": str(choices[labels.index(predicted)]),
        "correct": None,
        "gold_nll": None,
        "best_distractor_nll": None,
        "gold_margin": None,
    }
    answer = str(row.get("answer", "")).strip()
    if answer in labels:
        gold_nll = nll[answer]
        best_distractor_nll = min(nll[label] for label in labels if label != answer)
        result.update(
            {
                "correct": predicted == answer,
                "gold_nll": gold_nll,
                "best_distractor_nll": best_distractor_nll,
                "gold_margin": best_distractor_nll - gold_nll,
            }
        )
    return result


def _choice_nll_integrity_errors(
    row: Mapping[str, Any],
    *,
    tolerance: float = 2e-6,
) -> list[tuple[str, dict[str, Any]]]:
    """Re-derive every reported MCQ score from the canonical choice NLLs."""

    labels = _choice_labels(row)
    raw_nll = row.get("choice_nll")
    if labels is None:
        return []
    if not isinstance(raw_nll, Mapping) or set(raw_nll) != set(labels):
        return [
            (
                "invalid_choice_nll_key_set",
                {
                    "keys": (
                        sorted(str(value) for value in raw_nll)
                        if isinstance(raw_nll, Mapping)
                        else None
                    ),
                    "expected_keys": labels,
                },
            )
        ]
    nll: dict[str, float] = {}
    for label in labels:
        value = _finite_float(raw_nll.get(label))
        if value is None or value < 0:
            return [
                (
                    "invalid_choice_nll_value",
                    {"label": label, "value": raw_nll.get(label)},
                )
            ]
        nll[label] = value

    errors: list[tuple[str, dict[str, Any]]] = []
    minimum = min(nll.values())
    weights = {label: math.exp(-(nll[label] - minimum)) for label in labels}
    denominator = math.fsum(weights.values())
    expected_probability = {label: weights[label] / denominator for label in labels}
    raw_probability = row.get("choice_probability")
    if isinstance(raw_probability, Mapping) and set(raw_probability) == set(labels):
        for label in labels:
            actual = _finite_float(raw_probability.get(label))
            expected = expected_probability[label]
            if actual is None or not math.isclose(
                actual, expected, rel_tol=tolerance, abs_tol=tolerance
            ):
                errors.append(
                    (
                        "choice_probability_nll_inconsistency",
                        {"label": label, "reported": actual, "recomputed": expected},
                    )
                )
                break

    # Python's min is stable in label order, matching torch.argmin's first
    # minimum convention used by FrozenMultipleChoiceScorer.
    predicted = min(labels, key=lambda label: nll[label])
    reported_prediction = str(row.get("prediction", "")).strip()
    if reported_prediction != predicted:
        errors.append(
            (
                "prediction_nll_inconsistency",
                {"reported": reported_prediction, "recomputed": predicted},
            )
        )
    answer = str(row.get("answer", "")).strip()
    if answer not in labels:
        return errors
    expected_gold_nll = nll[answer]
    expected_distractor_nll = min(nll[label] for label in labels if label != answer)
    expected_margin = expected_distractor_nll - expected_gold_nll
    for field, expected in (
        ("gold_nll", expected_gold_nll),
        ("best_distractor_nll", expected_distractor_nll),
        ("gold_margin", expected_margin),
    ):
        reported = _finite_float(row.get(field))
        if reported is None or not math.isclose(
            reported, expected, rel_tol=tolerance, abs_tol=tolerance
        ):
            issue_kind = (
                f"{field}_inconsistency"
                if field in {"gold_nll", "best_distractor_nll"}
                else "gold_margin_nll_inconsistency"
            )
            errors.append(
                (
                    issue_kind,
                    {"reported": reported, "recomputed": expected},
                )
            )
    expected_correct = predicted == answer
    reported_correct = _bool_value(row.get("correct"))
    if reported_correct is None or reported_correct != expected_correct:
        errors.append(
            (
                "correct_nll_inconsistency",
                {"reported": reported_correct, "recomputed": expected_correct},
            )
        )
    return errors


def _choice_probability_error(
    row: Mapping[str, Any],
) -> tuple[str, dict[str, Any]] | None:
    """Return the first structural probability error, if any.

    Scores need not already sum to one: finite, non-negative weights with a
    finite positive total are deliberately accepted and normalized by the
    analysis. When semantic choices are available, however, the option-label
    key set must match them exactly so an omitted or extra option cannot hide
    behind renormalization.
    """

    raw = row.get("choice_probability")
    if not isinstance(raw, Mapping) or not raw:
        return "invalid_choice_probability", {"value": raw}

    labels = _choice_labels(row)
    raw_labels = list(raw.keys())
    if any(not isinstance(label, str) for label in raw_labels):
        return "invalid_choice_probability_key_set", {
            "keys": [str(label) for label in raw_labels],
            "expected_keys": labels,
        }
    if labels is not None and set(raw_labels) != set(labels):
        return "invalid_choice_probability_key_set", {
            "keys": sorted(raw_labels),
            "expected_keys": labels,
        }

    values: list[float] = []
    for label, value in raw.items():
        number = _finite_float(value)
        if number is None or number < 0:
            return "invalid_choice_probability_value", {
                "label": label,
                "value": (
                    value
                    if not isinstance(value, float) or math.isfinite(value)
                    else repr(value)
                ),
            }
        values.append(number)
    try:
        total = math.fsum(values)
    except OverflowError:
        total = math.inf
    if not math.isfinite(total) or total <= 0:
        return "invalid_choice_probability_total", {
            "total": total if math.isfinite(total) else "non_finite"
        }
    return None


def calibration_metrics(
    rows: Sequence[Mapping[str, Any]], *, ece_bins: int = DEFAULT_ECE_BINS
) -> dict[str, float | int | None]:
    """Return multiclass Brier and confidence ECE for valid rows.

    ``choice_probability`` is the softmax over option sequence scores and is
    therefore a pseudo-probability rather than a calibrated generative
    probability.  ``brier`` is the standard unnormalised multiclass Brier
    score; ``confidence_brier`` and ECE use max-option confidence vs.
    correctness.
    """

    if ece_bins < 1:
        raise ValueError("ece_bins must be >= 1")
    observations = [
        value for row in rows if (value := _probability_observation(row)) is not None
    ]
    if not observations:
        return {
            "n": 0,
            "gold_probability": None,
            "brier": None,
            "confidence_brier": None,
            "ece": None,
        }
    bins = [[0.0, 0.0, 0.0] for _ in range(ece_bins)]  # count, correct, confidence
    for observation in observations:
        index = min(int(observation["confidence"] * ece_bins), ece_bins - 1)
        bins[index][0] += 1.0
        bins[index][1] += observation["correct"]
        bins[index][2] += observation["confidence"]
    ece = 0.0
    for count, correct_sum, confidence_sum in bins:
        if count:
            ece += (
                count
                / len(observations)
                * abs(correct_sum / count - confidence_sum / count)
            )
    return {
        "n": len(observations),
        "gold_probability": _mean(
            [value["gold_probability"] for value in observations]
        ),
        "brier": _mean([value["brier"] for value in observations]),
        "confidence_brier": _mean(
            [value["confidence_brier"] for value in observations]
        ),
        "ece": float(ece),
    }


def _aggregation_unit_id(row: Mapping[str, Any]) -> str:
    """Unit for point estimates and all-rows-correct aggregation."""

    pair_id = row.get("pair_id")
    pair_role = str(row.get("pair_role", "")).casefold()
    if not _missing(pair_id) and pair_role in {
        "original",
        "counterfactual",
        "nuisance",
    }:
        return f"pair::{pair_id}"
    independent_unit_id = row.get("independent_unit_id")
    if not _missing(independent_unit_id):
        return f"independent::{independent_unit_id}"
    base_id = row.get("base_id")
    if not _missing(base_id):
        return f"base::{base_id}"
    return f"trial::{row.get('trial_id', '<missing>')}"


def _cluster_id(row: Mapping[str, Any]) -> str:
    """Highest-level dependence unit used for bootstrap resampling."""

    resampling_unit_id = row.get("resampling_unit_id")
    if not _missing(resampling_unit_id):
        return f"resampling::{resampling_unit_id}"
    return _aggregation_unit_id(row)


def _rows_to_units(rows: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_aggregation_unit_id(row)].append(row)
    return [grouped[key] for key in sorted(grouped)]


def _unit_resampling_ids(
    units: Sequence[Sequence[Mapping[str, Any]]],
) -> list[str]:
    """Resolve one source-video/scene cluster for each aggregation unit."""

    result: list[str] = []
    for unit in units:
        identifiers = {_cluster_id(row) for row in unit}
        if len(identifiers) != 1:
            raise ValueError(
                "one aggregation unit maps to multiple resampling units: "
                f"{sorted(identifiers)}"
            )
        result.append(next(iter(identifiers)))
    return result


def _summary_metrics_from_units(
    units: Sequence[Sequence[Mapping[str, Any]]], *, ece_bins: int
) -> dict[str, float]:
    result: dict[str, float] = {}
    decomposable: dict[str, list[float]] = defaultdict(list)
    row_correct: list[float] = []
    calibration_units: list[list[dict[str, float]]] = []
    for unit in units:
        correct = [
            value
            for row in unit
            if (
                value := (
                    row.get("_correct")
                    if "_correct" in row
                    else _bool_value(row.get("correct"))
                )
            )
            is not None
        ]
        margin = [
            value
            for row in unit
            if (value := _finite_float(row.get("gold_margin"))) is not None
        ]
        probabilities = [
            value
            for row in unit
            if (value := _probability_observation(row)) is not None
        ]
        if correct:
            unit_accuracy = _mean(correct)
            decomposable["cluster_macro_accuracy"].append(unit_accuracy)
            row_correct.extend(float(value) for value in correct)
            # This is deliberately stricter than mean row accuracy.  For a
            # CLEVRER independent unit it requires every binary candidate and
            # option permutation in the official question to be correct.
            if len(correct) == len(unit):
                decomposable["cluster_all_rows_correct"].append(float(all(correct)))
        if margin:
            decomposable["gold_margin"].append(_mean(margin))
        if probabilities:
            for name in ("gold_probability", "brier", "confidence_brier"):
                decomposable[name].append(
                    _mean([value[name] for value in probabilities])
                )
            calibration_units.append(probabilities)
    result["row_micro_accuracy"] = _mean(row_correct) if row_correct else math.nan
    cluster_accuracies = decomposable.get("cluster_macro_accuracy", [])
    result["cluster_macro_accuracy"] = (
        _mean(cluster_accuracies) if cluster_accuracies else math.nan
    )
    # Backward-compatible alias: historical ``accuracy`` was already a macro
    # mean over bootstrap clusters, rather than a micro mean over rows.
    result["accuracy"] = result["cluster_macro_accuracy"]

    for name in (
        "cluster_all_rows_correct",
        "gold_margin",
        "gold_probability",
        "brier",
        "confidence_brier",
    ):
        values = decomposable.get(name, [])
        result[name] = _mean(values) if values else math.nan

    if calibration_units:
        bins = [[0.0, 0.0, 0.0] for _ in range(ece_bins)]
        for observations in calibration_units:
            row_weight = 1.0 / len(observations)
            for observation in observations:
                index = min(int(observation["confidence"] * ece_bins), ece_bins - 1)
                bins[index][0] += row_weight
                bins[index][1] += row_weight * observation["correct"]
                bins[index][2] += row_weight * observation["confidence"]
        ece = 0.0
        total_weight = float(len(calibration_units))
        for count, correct_sum, confidence_sum in bins:
            if count:
                ece += (
                    count
                    / total_weight
                    * abs(correct_sum / count - confidence_sum / count)
                )
        result["ece"] = ece
    else:
        result["ece"] = math.nan
    return result


def _resampling_memberships(
    num_units: int, resampling_ids: Sequence[str] | None
) -> list[list[int]]:
    if resampling_ids is None:
        return [[index] for index in range(num_units)]
    if len(resampling_ids) != num_units:
        raise ValueError("resampling_ids must have one entry per aggregation unit")
    grouped: dict[str, list[int]] = {}
    for index, identifier in enumerate(resampling_ids):
        grouped.setdefault(str(identifier), []).append(index)
    return list(grouped.values())


def _bootstrap_grouped_stat_map(
    units: Sequence[Any],
    memberships: Sequence[Sequence[int]],
    calculator: Callable[[Sequence[Any]], Mapping[str, float]],
    *,
    seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
) -> dict[str, float | None]:
    """Reference implementation: resample source clusters, retain child units."""

    point = dict(calculator(units)) if units else {}
    output: dict[str, float | None] = {
        key: (None if not math.isfinite(value) else float(value))
        for key, value in point.items()
    }
    names = list(point)
    if not units or not memberships or bootstrap_replicates <= 0:
        for name in names:
            output[f"{name}_ci_low"] = None
            output[f"{name}_ci_high"] = None
        return output

    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(bootstrap_replicates):
        sampled_units: list[Any] = []
        for _draw in memberships:
            sampled_group = memberships[rng.randrange(len(memberships))]
            sampled_units.extend(units[index] for index in sampled_group)
        values = calculator(sampled_units)
        for name in names:
            value = values.get(name, math.nan)
            if math.isfinite(value):
                distributions[name].append(float(value))
    for name in names:
        low, high = _percentile_interval(distributions[name], confidence_level)
        output[f"{name}_ci_low"] = low
        output[f"{name}_ci_high"] = high
    return output


def _bootstrap_stat_map(
    units: Sequence[Any],
    calculator: Callable[[Sequence[Any]], Mapping[str, float]],
    *,
    seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
) -> dict[str, float | None]:
    point = dict(calculator(units)) if units else {}
    output: dict[str, float | None] = {
        key: (None if not math.isfinite(value) else float(value))
        for key, value in point.items()
    }
    names = list(point)
    if not units or bootstrap_replicates <= 0:
        for name in names:
            output[f"{name}_ci_low"] = None
            output[f"{name}_ci_high"] = None
        return output

    rng = random.Random(seed)
    distributions: dict[str, list[float]] = {name: [] for name in names}
    for _ in range(bootstrap_replicates):
        sample = [units[rng.randrange(len(units))] for _ in range(len(units))]
        values = calculator(sample)
        for name in names:
            value = values.get(name, math.nan)
            if math.isfinite(value):
                distributions[name].append(float(value))
    for name in names:
        low, high = _percentile_interval(distributions[name], confidence_level)
        output[f"{name}_ci_low"] = low
        output[f"{name}_ci_high"] = high
    return output


def _bootstrap_count_batches(
    *,
    num_units: int,
    bootstrap_replicates: int,
    seed: int,
    max_count_cells: int = BOOTSTRAP_MAX_COUNT_CELLS,
) -> Iterable[np.ndarray]:
    """Yield bounded cluster multiplicity matrices using the legacy RNG stream.

    Each output row is one ordinary cluster-bootstrap resample represented by
    counts, not by a Python list containing repeated row-unit objects. Drawing
    remains replicate-major with ``random.Random.randrange`` so a fixed seed
    selects exactly the same clusters as the original implementation.
    """

    if num_units < 1:
        raise ValueError("num_units must be >= 1")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be >= 0")
    if max_count_cells < 1:
        raise ValueError("max_count_cells must be >= 1")
    if np is None:
        raise RuntimeError("NumPy is required for batched bootstrap counts")

    batch_size = min(
        BOOTSTRAP_MAX_BATCH_REPLICATES,
        max(1, max_count_cells // num_units),
    )
    rng = random.Random(seed)
    remaining = bootstrap_replicates
    while remaining:
        current = min(batch_size, remaining)
        draw_count = current * num_units
        draws = np.fromiter(
            (rng.randrange(num_units) for _ in range(draw_count)),
            dtype=np.int64,
            count=draw_count,
        ).reshape(current, num_units)
        counts = np.empty((current, num_units), dtype=np.float64)
        for replicate_index in range(current):
            counts[replicate_index] = np.bincount(
                draws[replicate_index], minlength=num_units
            )
        yield counts
        remaining -= current


def _bootstrap_unit_macro_stat_map(
    units: Sequence[Any],
    calculator: Callable[[Sequence[Any]], Mapping[str, float]],
    names: Sequence[str],
    *,
    resampling_ids: Sequence[str] | None = None,
    seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
) -> dict[str, float | None]:
    """Vectorized bootstrap for metrics that macro-average per-cluster values."""

    memberships = _resampling_memberships(len(units), resampling_ids)
    if np is None:
        return _bootstrap_grouped_stat_map(
            units,
            memberships,
            calculator,
            seed=seed,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )

    point = dict(calculator(units)) if units else {}
    output: dict[str, float | None] = {
        key: (None if not math.isfinite(value) else float(value))
        for key, value in point.items()
    }
    if not units or bootstrap_replicates <= 0:
        for name in point:
            output[f"{name}_ci_low"] = None
            output[f"{name}_ci_high"] = None
        return output

    per_unit = np.full((len(units), len(names)), np.nan, dtype=np.float64)
    for unit_index, unit in enumerate(units):
        values = calculator([unit])
        for metric_index, name in enumerate(names):
            number = _finite_float(values.get(name))
            if number is not None:
                per_unit[unit_index, metric_index] = number
    eligible = np.isfinite(per_unit).astype(np.float64)
    numerators = np.nan_to_num(per_unit, nan=0.0)
    grouped_numerators = np.zeros((len(memberships), len(names)), dtype=np.float64)
    grouped_eligible = np.zeros_like(grouped_numerators)
    for group_index, member_indices in enumerate(memberships):
        grouped_numerators[group_index] = numerators[member_indices].sum(axis=0)
        grouped_eligible[group_index] = eligible[member_indices].sum(axis=0)
    distributions: dict[str, list[float]] = {name: [] for name in names}
    for counts in _bootstrap_count_batches(
        num_units=len(memberships),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    ):
        batch_numerators = counts @ grouped_numerators
        batch_denominators = counts @ grouped_eligible
        batch_values = np.full_like(batch_numerators, np.nan)
        np.divide(
            batch_numerators,
            batch_denominators,
            out=batch_values,
            where=batch_denominators > 0,
        )
        for metric_index, name in enumerate(names):
            values = batch_values[:, metric_index]
            distributions[name].extend(values[np.isfinite(values)].tolist())
    for name in point:
        low, high = _percentile_interval(distributions.get(name, []), confidence_level)
        output[f"{name}_ci_low"] = low
        output[f"{name}_ci_high"] = high
    return output


def _bootstrap_summary_stat_map(
    units: Sequence[Sequence[Mapping[str, Any]]],
    *,
    resampling_ids: Sequence[str] | None = None,
    ece_bins: int,
    seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
) -> dict[str, float | None]:
    """Bootstrap summary metrics from per-cluster sufficient statistics."""

    memberships = _resampling_memberships(len(units), resampling_ids)
    if np is None:
        return _bootstrap_grouped_stat_map(
            units,
            memberships,
            lambda sample: _summary_metrics_from_units(sample, ece_bins=ece_bins),
            seed=seed,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )

    point = _summary_metrics_from_units(units, ece_bins=ece_bins) if units else {}
    output: dict[str, float | None] = {
        key: (None if not math.isfinite(value) else float(value))
        for key, value in point.items()
    }
    if not units or bootstrap_replicates <= 0:
        for name in point:
            output[f"{name}_ci_low"] = None
            output[f"{name}_ci_high"] = None
        return output

    mean_names = (
        "row_micro_accuracy",
        "cluster_macro_accuracy",
        "cluster_all_rows_correct",
        "gold_margin",
        "gold_probability",
        "brier",
        "confidence_brier",
    )
    numerators = np.zeros((len(units), len(mean_names)), dtype=np.float64)
    denominators = np.zeros_like(numerators)
    ece_deltas = np.zeros((len(units), ece_bins), dtype=np.float64)
    ece_eligible = np.zeros(len(units), dtype=np.float64)

    for unit_index, unit in enumerate(units):
        correct = [
            value
            for row in unit
            if (
                value := (
                    row.get("_correct")
                    if "_correct" in row
                    else _bool_value(row.get("correct"))
                )
            )
            is not None
        ]
        margin = [
            value
            for row in unit
            if (value := _finite_float(row.get("gold_margin"))) is not None
        ]
        probabilities = [
            value
            for row in unit
            if (value := _probability_observation(row)) is not None
        ]

        if correct:
            correct_sum = math.fsum(float(value) for value in correct)
            numerators[unit_index, 0] = correct_sum
            denominators[unit_index, 0] = len(correct)
            numerators[unit_index, 1] = correct_sum / len(correct)
            denominators[unit_index, 1] = 1.0
            if len(correct) == len(unit):
                numerators[unit_index, 2] = float(all(correct))
                denominators[unit_index, 2] = 1.0
        if margin:
            numerators[unit_index, 3] = _mean(margin)
            denominators[unit_index, 3] = 1.0
        if probabilities:
            for offset, name in enumerate(
                ("gold_probability", "brier", "confidence_brier"), start=4
            ):
                numerators[unit_index, offset] = _mean(
                    [value[name] for value in probabilities]
                )
                denominators[unit_index, offset] = 1.0
            row_weight = 1.0 / len(probabilities)
            for observation in probabilities:
                bin_index = min(int(observation["confidence"] * ece_bins), ece_bins - 1)
                ece_deltas[unit_index, bin_index] += row_weight * (
                    observation["correct"] - observation["confidence"]
                )
            ece_eligible[unit_index] = 1.0

    grouped_numerators = np.zeros((len(memberships), len(mean_names)), dtype=np.float64)
    grouped_denominators = np.zeros_like(grouped_numerators)
    grouped_ece_deltas = np.zeros((len(memberships), ece_bins), dtype=np.float64)
    grouped_ece_eligible = np.zeros(len(memberships), dtype=np.float64)
    for group_index, member_indices in enumerate(memberships):
        grouped_numerators[group_index] = numerators[member_indices].sum(axis=0)
        grouped_denominators[group_index] = denominators[member_indices].sum(axis=0)
        grouped_ece_deltas[group_index] = ece_deltas[member_indices].sum(axis=0)
        grouped_ece_eligible[group_index] = ece_eligible[member_indices].sum()

    distributions: dict[str, list[float]] = {
        name: [] for name in (*mean_names, "accuracy", "ece")
    }
    for counts in _bootstrap_count_batches(
        num_units=len(memberships),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    ):
        batch_numerators = counts @ grouped_numerators
        batch_denominators = counts @ grouped_denominators
        batch_values = np.full_like(batch_numerators, np.nan)
        np.divide(
            batch_numerators,
            batch_denominators,
            out=batch_values,
            where=batch_denominators > 0,
        )
        for metric_index, name in enumerate(mean_names):
            values = batch_values[:, metric_index]
            distributions[name].extend(values[np.isfinite(values)].tolist())
        macro_values = batch_values[:, mean_names.index("cluster_macro_accuracy")]
        distributions["accuracy"].extend(
            macro_values[np.isfinite(macro_values)].tolist()
        )

        batch_ece_denominator = counts @ grouped_ece_eligible
        batch_ece_numerator = np.abs(counts @ grouped_ece_deltas).sum(axis=1)
        batch_ece = np.full(counts.shape[0], np.nan, dtype=np.float64)
        np.divide(
            batch_ece_numerator,
            batch_ece_denominator,
            out=batch_ece,
            where=batch_ece_denominator > 0,
        )
        distributions["ece"].extend(batch_ece[np.isfinite(batch_ece)].tolist())

    for name in point:
        low, high = _percentile_interval(distributions.get(name, []), confidence_level)
        output[f"{name}_ci_low"] = low
        output[f"{name}_ci_high"] = high
    return output


def bootstrap_mean_ci(
    values: Sequence[float],
    cluster_ids: Sequence[str],
    *,
    seed: int = DEFAULT_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[float | None, float | None, float | None]:
    """Cluster-bootstrap a mean; exposed for formula-level tests and reuse."""

    if len(values) != len(cluster_ids):
        raise ValueError("values and cluster_ids must have the same length")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster_id in zip(values, cluster_ids):
        number = _finite_float(value)
        if number is not None:
            grouped[str(cluster_id)].append(number)
    units = [grouped[key] for key in sorted(grouped)]
    stats = _bootstrap_unit_macro_stat_map(
        units,
        lambda sample: {"mean": _mean([_mean(unit) for unit in sample])},
        ("mean",),
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )
    return stats.get("mean"), stats.get("mean_ci_low"), stats.get("mean_ci_high")


def _group_key(
    row: Mapping[str, Any], fields: Sequence[str] = GROUP_FIELDS
) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in fields:
        value = row.get(field)
        values.append(
            _dose_value(value)
            if field in {"requested_dose", "effective_dose"}
            else _group_value(value)
        )
    return tuple(values)


def _unique_count(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return len({str(row.get(field)) for row in rows if not _missing(row.get(field))})


def _condition_input_channel(rows: Sequence[Mapping[str, Any]]) -> str:
    channels = sorted(
        {
            str(row.get("input_channel"))
            for row in rows
            if not _missing(row.get("input_channel"))
        }
    )
    if not channels:
        return "<missing>"
    return channels[0] if len(channels) == 1 else "<mixed>"


def _dose_reference_condition(
    *, input_channel: Any, condition: Any, visual_reference_condition: str
) -> str:
    channel = str(input_channel or "").strip().casefold()
    condition_name = str(condition or "").strip().casefold()
    if channel in {"text_oracle", "embedding_oracle"}:
        return "question_only"
    if channel == "visual_plus_text":
        return visual_reference_condition
    # Old result files may predate the flattened input_channel.  This fallback
    # is explicit in the output and preserves the intended baseline rather
    # than silently treating a text oracle as an improvement over pixels.
    if not channel or channel == "<missing>":
        if "video_plus" in condition_name or "visual_plus" in condition_name:
            return visual_reference_condition
        if "oracle" in condition_name:
            return "question_only"
    return visual_reference_condition


def _semantic_option_probabilities(
    row: Mapping[str, Any],
) -> dict[str, float] | None:
    """Map one scored permutation from letter labels to semantic option text."""

    labels = _choice_labels(row)
    choices = row.get("choices")
    if labels is None or not isinstance(choices, list):
        return None
    if _choice_probability_error(row) is not None:
        return None
    raw_probability = row.get("choice_probability")
    assert isinstance(raw_probability, Mapping)
    total = math.fsum(float(raw_probability[label]) for label in labels)
    semantic_probabilities: dict[str, float] = {}
    for index, label in enumerate(labels):
        semantic_text = _semantic(choices[index])
        if semantic_text is None or semantic_text in semantic_probabilities:
            return None
        semantic_probabilities[semantic_text] = float(raw_probability[label]) / total
    return semantic_probabilities


def _semantic_gold_option(row: Mapping[str, Any]) -> str | None:
    labels = _choice_labels(row)
    choices = row.get("choices")
    answer = str(row.get("answer", "")).strip()
    if labels is None or not isinstance(choices, list) or answer not in labels:
        return None
    semantic_choices = [_semantic(choice) for choice in choices]
    if any(choice is None for choice in semantic_choices) or len(
        set(semantic_choices)
    ) != len(semantic_choices):
        return None
    gold = semantic_choices[labels.index(answer)]
    answer_text = _semantic(row.get("answer_text"))
    if answer_text is not None and answer_text != gold:
        return None
    return gold


def _clevrer_question_evaluation(
    rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Authenticate and score complete CLEVRER questions semantically.

    Each binary candidate is one semantic decision repeated under option-letter
    permutations.  Averaging by normalized option text prevents a single noisy
    letter-position decision from changing the scientific estimand.
    """

    expected_by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    observed_by_question: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in expected_rows:
        expected_by_question[str(row.get("independent_unit_id", "")).strip()].append(
            row
        )
    for row in rows:
        observed_by_question[str(row.get("independent_unit_id", "")).strip()].append(
            row
        )

    failures: Counter[str] = Counter()
    if not expected_by_question:
        failures["missing_expected_questions"] += 1
    if "" in expected_by_question:
        failures["missing_question_id"] += 1
    if set(observed_by_question) != set(expected_by_question):
        failures["question_set_mismatch"] += 1

    question_observations: list[dict[str, Any]] = []
    semantic_argmax_ties = 0
    candidate_total = 0
    for question_id, expected_question in sorted(expected_by_question.items()):
        if not question_id:
            continue
        observed_question = observed_by_question.get(question_id, [])
        expected_trial_ids = [str(row.get("trial_id", "")) for row in expected_question]
        observed_trial_ids = [str(row.get("trial_id", "")) for row in observed_question]
        question_valid = True
        if (
            any(not trial_id for trial_id in expected_trial_ids)
            or len(set(expected_trial_ids)) != len(expected_trial_ids)
            or len(set(observed_trial_ids)) != len(observed_trial_ids)
            or set(observed_trial_ids) != set(expected_trial_ids)
            or len(observed_question) != len(expected_question)
        ):
            failures["trial_set_mismatch"] += 1
            question_valid = False

        expected_by_permutation: dict[Any, list[Mapping[str, Any]]] = defaultdict(list)
        for row in expected_question:
            expected_by_permutation[
                _group_value(row.get("permutation_index", 0))
            ].append(row)
        candidate_sets: list[set[str]] = []
        declared_counts: set[int | None] = set()
        for permutation_rows in expected_by_permutation.values():
            declared_counts.update(
                _positive_int(row.get("official_candidate_count"))
                for row in permutation_rows
            )
            candidate_ids = [
                str(row.get("official_candidate_id", "")).strip()
                for row in permutation_rows
            ]
            candidate_sets.append(set(candidate_ids))
            declared = _positive_int(
                permutation_rows[0].get("official_candidate_count")
            )
            if (
                declared is None
                or "" in candidate_ids
                or len(set(candidate_ids)) != len(candidate_ids)
                or len(candidate_ids) != declared
            ):
                question_valid = False
        if (
            None in declared_counts
            or len(declared_counts) != 1
            or not candidate_sets
            or any(
                candidate_set != candidate_sets[0] for candidate_set in candidate_sets
            )
        ):
            failures["invalid_candidate_permutation_design"] += 1
            question_valid = False

        observed_by_trial = {
            str(row.get("trial_id", "")): row for row in observed_question
        }
        candidate_correctness: list[bool] = []
        candidate_permutation_robustness: list[bool] = []
        candidate_ids = sorted(candidate_sets[0]) if candidate_sets else []
        candidate_total += len(candidate_ids)
        for candidate_id in candidate_ids:
            expected_candidate = [
                row
                for row in expected_question
                if str(row.get("official_candidate_id", "")).strip() == candidate_id
            ]
            expected_permutations = [
                _group_value(row.get("permutation_index", 0))
                for row in expected_candidate
            ]
            if len(expected_candidate) != len(expected_by_permutation) or len(
                set(expected_permutations)
            ) != len(expected_permutations):
                failures["invalid_candidate_permutation_design"] += 1
                question_valid = False
                continue

            attested_all_permutations: list[bool] = []
            for row in expected_candidate:
                attestation = row.get("trial_build_attestation")
                sampling = (
                    attestation.get("sampling")
                    if isinstance(attestation, Mapping)
                    else None
                )
                permutation_mode = (
                    sampling.get("option_permutations")
                    if isinstance(sampling, Mapping)
                    else None
                )
                attested_all_permutations.append(
                    isinstance(permutation_mode, str)
                    and permutation_mode.strip().casefold() == "all"
                )
            if any(attested_all_permutations):
                choice_lengths = {
                    len(row["choices"])
                    for row in expected_candidate
                    if isinstance(row.get("choices"), list)
                }
                raw_permutation_indices = [
                    row.get("permutation_index", 0) for row in expected_candidate
                ]
                exact_index_set = (
                    all(attested_all_permutations)
                    and len(choice_lengths) == 1
                    and len(raw_permutation_indices) == next(iter(choice_lengths))
                    and all(
                        isinstance(index, int) and not isinstance(index, bool)
                        for index in raw_permutation_indices
                    )
                    and set(raw_permutation_indices)
                    == set(range(next(iter(choice_lengths))))
                )
                if not exact_index_set:
                    failures["attested_all_permutation_set_mismatch"] += 1
                    question_valid = False
                    continue

            gold_options = {_semantic_gold_option(row) for row in expected_candidate}
            semantic_choice_sets: set[tuple[str, ...]] = set()
            observed_candidate: list[Mapping[str, Any]] = []
            probability_vectors: list[dict[str, float]] = []
            row_correctness: list[bool] = []
            for expected_row in expected_candidate:
                choices = expected_row.get("choices")
                if isinstance(choices, list):
                    normalized_choices = [_semantic(choice) for choice in choices]
                    if all(choice is not None for choice in normalized_choices) and len(
                        set(normalized_choices)
                    ) == len(normalized_choices):
                        semantic_choice_sets.add(
                            tuple(sorted(str(choice) for choice in normalized_choices))
                        )
                observed_row = observed_by_trial.get(
                    str(expected_row.get("trial_id", ""))
                )
                if observed_row is None:
                    continue
                observed_candidate.append(observed_row)
                probabilities = _semantic_option_probabilities(observed_row)
                if probabilities is not None:
                    probability_vectors.append(probabilities)
                correct = (
                    observed_row.get("_correct")
                    if "_correct" in observed_row
                    else _bool_value(observed_row.get("correct"))
                )
                if correct is not None:
                    row_correctness.append(bool(correct))
            if (
                None in gold_options
                or len(gold_options) != 1
                or len(semantic_choice_sets) != 1
                or len(observed_candidate) != len(expected_candidate)
                or len(probability_vectors) != len(expected_candidate)
                or len(row_correctness) != len(expected_candidate)
                or any(
                    tuple(sorted(probabilities)) != next(iter(semantic_choice_sets))
                    for probabilities in probability_vectors
                )
            ):
                failures["invalid_semantic_candidate_scores"] += 1
                question_valid = False
                continue

            option_scores = {
                semantic_option: _mean(
                    [
                        probabilities[semantic_option]
                        for probabilities in probability_vectors
                    ]
                )
                for semantic_option in next(iter(semantic_choice_sets))
            }
            maximum = max(option_scores.values())
            winners = [
                semantic_option
                for semantic_option, probability in option_scores.items()
                if probability == maximum
            ]
            if len(winners) != 1:
                semantic_argmax_ties += 1
            gold_option = next(iter(gold_options))
            candidate_correctness.append(
                len(winners) == 1 and winners[0] == gold_option
            )
            candidate_permutation_robustness.append(all(row_correctness))

        resampling_ids = {
            str(row.get("resampling_unit_id", "")).strip()
            for row in [*expected_question, *observed_question]
        }
        if "" in resampling_ids or len(resampling_ids) != 1:
            failures["question_maps_to_multiple_resampling_units"] += 1
            question_valid = False
        if len(candidate_correctness) != len(candidate_ids):
            question_valid = False
        if question_valid:
            question_observations.append(
                {
                    "question_id": question_id,
                    "resampling_cluster_id": next(iter(resampling_ids)),
                    "official_question_exact_set_accuracy": float(
                        all(candidate_correctness)
                    ),
                    "official_question_permutation_robustness_accuracy": float(
                        all(candidate_permutation_robustness)
                    ),
                }
            )

    authenticated = (
        not failures
        and bool(expected_by_question)
        and len(question_observations) == len(expected_by_question)
    )
    return {
        "authenticated": authenticated,
        "expected_question_count": len(expected_by_question),
        "candidate_count": candidate_total,
        "semantic_argmax_ties": semantic_argmax_ties,
        "failures": dict(sorted(failures.items())),
        "questions": question_observations if authenticated else [],
    }


def _clevrer_question_stats(
    evaluation: Mapping[str, Any],
    *,
    seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
) -> dict[str, float | None]:
    questions = evaluation.get("questions")
    if not evaluation.get("authenticated") or not isinstance(questions, list):
        return {
            field: None
            for name in CLEVRER_QUESTION_METRIC_NAMES
            for field in (name, f"{name}_ci_low", f"{name}_ci_high")
        }
    units = [[question] for question in questions]

    def calculate(sample: Sequence[Sequence[Mapping[str, Any]]]) -> Mapping[str, float]:
        return {
            name: _mean([float(unit[0][name]) for unit in sample])
            for name in CLEVRER_QUESTION_METRIC_NAMES
        }

    return _bootstrap_unit_macro_stat_map(
        units,
        calculate,
        CLEVRER_QUESTION_METRIC_NAMES,
        resampling_ids=[
            str(question["resampling_cluster_id"]) for question in questions
        ],
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )


def _clevrer_exact_set_fields(
    rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    *,
    seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Expose semantic official-question accuracy only for authenticated sets."""

    dataset_values = {
        str(row.get("dataset", "")).strip().casefold()
        for row in [*rows, *expected_rows]
    }
    if dataset_values != {"clevrer"}:
        return {
            "primary_accuracy_metric": "cluster_macro_accuracy",
            "primary_accuracy": stats.get("cluster_macro_accuracy"),
            "primary_accuracy_ci_low": stats.get("cluster_macro_accuracy_ci_low"),
            "primary_accuracy_ci_high": stats.get("cluster_macro_accuracy_ci_high"),
            "official_question_exact_set_accuracy": None,
            "official_question_exact_set_accuracy_ci_low": None,
            "official_question_exact_set_accuracy_ci_high": None,
            "official_question_exact_set_questions": 0,
            "official_question_exact_set_authenticated": False,
            "candidate_level_accuracy_primary": None,
        }

    evaluation = _clevrer_question_evaluation(rows, expected_rows)
    exact_stats = _clevrer_question_stats(
        evaluation,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )
    exact_accuracy = exact_stats["official_question_exact_set_accuracy"]
    return {
        "primary_accuracy_metric": "official_question_exact_set_accuracy",
        "primary_accuracy": exact_accuracy,
        "primary_accuracy_ci_low": exact_stats[
            "official_question_exact_set_accuracy_ci_low"
        ],
        "primary_accuracy_ci_high": exact_stats[
            "official_question_exact_set_accuracy_ci_high"
        ],
        **exact_stats,
        "official_question_exact_set_questions": evaluation["expected_question_count"],
        "official_question_exact_set_authenticated": evaluation["authenticated"],
        "official_question_semantic_aggregation_rule": (
            CLEVRER_SEMANTIC_AGGREGATION_RULE
        ),
        "official_question_semantic_argmax_ties": evaluation["semantic_argmax_ties"],
        "official_question_semantic_aggregation_failures": evaluation["failures"],
        # Candidate rows and the older every-row exact result remain useful
        # diagnostics, but neither is the CLEVRER primary estimand.
        "candidate_level_accuracy_primary": False,
    }


def summarize_predictions(
    observed: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]] = (),
    *,
    seed: int = DEFAULT_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    ece_bins: int = DEFAULT_ECE_BINS,
) -> list[dict[str, Any]]:
    observed_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    expected_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observed:
        observed_groups[_group_key(row)].append(row)
    for row in expected:
        expected_groups[_group_key(row)].append(row)
    keys = sorted(set(observed_groups) | set(expected_groups), key=_sort_key)

    summaries: list[dict[str, Any]] = []
    for key in keys:
        rows = observed_groups.get(key, [])
        expected_rows = expected_groups.get(key, [])
        units = _rows_to_units(rows)
        resampling_ids = _unit_resampling_ids(units)
        stats = _bootstrap_summary_stat_map(
            units,
            resampling_ids=resampling_ids,
            ece_bins=ece_bins,
            seed=_stable_seed(seed, "summary", key),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        for name in SUMMARY_METRIC_NAMES:
            stats.setdefault(name, None)
            stats.setdefault(f"{name}_ci_low", None)
            stats.setdefault(f"{name}_ci_high", None)
        output = dict(zip(GROUP_FIELDS, key))
        expected_count = len(expected_rows) if expected else None
        observed_base_count = _unique_count(rows, "base_id")
        expected_base_count = (
            _unique_count(expected_rows, "base_id") if expected else None
        )
        observed_pair_count = _unique_count(rows, "pair_id")
        expected_pair_count = (
            _unique_count(expected_rows, "pair_id") if expected else None
        )
        expected_cluster_count = (
            len(_rows_to_units(expected_rows)) if expected else None
        )
        expected_resampling_count = (
            len(set(_unit_resampling_ids(_rows_to_units(expected_rows))))
            if expected
            else None
        )
        output.update(
            {
                "input_channel": _condition_input_channel([*rows, *expected_rows]),
                "observed_rows": len(rows),
                "expected_rows": expected_count,
                "row_coverage": (
                    len(rows) / expected_count
                    if expected_count
                    else (None if expected else 1.0)
                ),
                "accuracy_rows": sum(
                    (
                        row.get("_correct")
                        if "_correct" in row
                        else _bool_value(row.get("correct"))
                    )
                    is not None
                    for row in rows
                ),
                "gold_margin_rows": sum(
                    _finite_float(row.get("gold_margin")) is not None for row in rows
                ),
                "probability_rows": sum(
                    _probability_observation(row) is not None for row in rows
                ),
                "observed_unique_base_ids": observed_base_count,
                "expected_unique_base_ids": expected_base_count,
                "base_coverage": (
                    observed_base_count / expected_base_count
                    if expected_base_count
                    else (None if expected else 1.0)
                ),
                "observed_unique_pair_ids": observed_pair_count,
                "expected_unique_pair_ids": expected_pair_count,
                "pair_coverage": (
                    observed_pair_count / expected_pair_count
                    if expected_pair_count
                    else (None if expected else 1.0)
                ),
                "observed_unique_clusters": len(units),
                "expected_unique_clusters": expected_cluster_count,
                "cluster_coverage": (
                    len(units) / expected_cluster_count
                    if expected_cluster_count
                    else (None if expected else 1.0)
                ),
                "observed_unique_resampling_units": len(set(resampling_ids)),
                "expected_unique_resampling_units": expected_resampling_count,
                "resampling_unit_coverage": (
                    len(set(resampling_ids)) / expected_resampling_count
                    if expected_resampling_count
                    else (None if expected else 1.0)
                ),
                "confidence_level": confidence_level,
                "bootstrap_replicates": bootstrap_replicates,
                **stats,
                **_clevrer_exact_set_fields(
                    rows,
                    expected_rows,
                    stats,
                    seed=_stable_seed(seed, "clevrer_semantic_exact_set", key),
                    bootstrap_replicates=bootstrap_replicates,
                    confidence_level=confidence_level,
                ),
            }
        )
        summaries.append(output)
    return summaries


def _alignment_key(row: Mapping[str, Any]) -> tuple[str, int | str]:
    permutation = row.get("permutation_index", 0)
    number = _finite_float(permutation)
    normalized: int | str = (
        int(number) if number is not None and number.is_integer() else str(permutation)
    )
    return str(row.get("base_id", "<missing>")), normalized


def _paired_units(pairs: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["cluster_id"])].append(pair)
    return [grouped[key] for key in sorted(grouped)]


def _observation_unit_resampling_ids(
    units: Sequence[Sequence[Mapping[str, Any]]],
) -> list[str]:
    result: list[str] = []
    for unit in units:
        identifiers = {
            str(item.get("resampling_cluster_id", item.get("cluster_id")))
            for item in unit
        }
        if len(identifiers) != 1:
            raise ValueError(
                "one observation unit maps to multiple resampling units: "
                f"{sorted(identifiers)}"
            )
        result.append(next(iter(identifiers)))
    return result


def _paired_metrics_from_units(
    units: Sequence[Sequence[Mapping[str, Any]]],
) -> dict[str, float]:
    names = PAIRED_METRIC_NAMES
    unit_values: dict[str, list[float]] = defaultdict(list)
    for unit in units:
        left_correct: list[float] = []
        right_correct: list[float] = []
        accuracy_gain: list[float] = []
        left_margin: list[float] = []
        right_margin: list[float] = []
        margin_gain: list[float] = []
        for pair in unit:
            left = pair.get("left_correct")
            right = pair.get("right_correct")
            if left is not None and right is not None:
                left_correct.append(float(left))
                right_correct.append(float(right))
                accuracy_gain.append(float(left) - float(right))
            left_m = pair.get("left_margin")
            right_m = pair.get("right_margin")
            if left_m is not None and right_m is not None:
                left_margin.append(float(left_m))
                right_margin.append(float(right_m))
                margin_gain.append(float(left_m) - float(right_m))
        for name, values in (
            ("condition_accuracy", left_correct),
            ("reference_accuracy", right_correct),
            ("accuracy_gain", accuracy_gain),
            ("condition_gold_margin", left_margin),
            ("reference_gold_margin", right_margin),
            ("gold_margin_gain", margin_gain),
        ):
            if values:
                unit_values[name].append(_mean(values))
    return {
        name: (_mean(unit_values[name]) if unit_values.get(name) else math.nan)
        for name in names
    }


def _bootstrap_paired_stat_map(
    units: Sequence[Sequence[Mapping[str, Any]]],
    *,
    resampling_ids: Sequence[str] | None = None,
    seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
) -> dict[str, float | None]:
    return _bootstrap_unit_macro_stat_map(
        units,
        _paired_metrics_from_units,
        PAIRED_METRIC_NAMES,
        resampling_ids=resampling_ids,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )


def _clevrer_paired_exact_set_fields(
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    expected_left_rows: Sequence[Mapping[str, Any]],
    expected_right_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Paired CLEVRER contrast after semantic permutation aggregation."""

    dataset_values = {
        str(row.get("dataset", "")).strip().casefold()
        for row in [
            *left_rows,
            *right_rows,
            *expected_left_rows,
            *expected_right_rows,
        ]
    }
    if dataset_values != {"clevrer"}:
        return {}

    left_evaluation = _clevrer_question_evaluation(left_rows, expected_left_rows)
    right_evaluation = _clevrer_question_evaluation(right_rows, expected_right_rows)
    left_questions = {
        str(question["question_id"]): question
        for question in left_evaluation["questions"]
    }
    right_questions = {
        str(question["question_id"]): question
        for question in right_evaluation["questions"]
    }
    authenticated = bool(
        left_evaluation["authenticated"]
        and right_evaluation["authenticated"]
        and left_questions
        and set(left_questions) == set(right_questions)
    )
    if authenticated and any(
        left_questions[question_id]["resampling_cluster_id"]
        != right_questions[question_id]["resampling_cluster_id"]
        for question_id in left_questions
    ):
        authenticated = False

    stats: dict[str, float | None]
    if authenticated:
        observations = [
            {
                "question_id": question_id,
                "resampling_cluster_id": left_questions[question_id][
                    "resampling_cluster_id"
                ],
                "condition_official_question_exact_set_accuracy": left_questions[
                    question_id
                ]["official_question_exact_set_accuracy"],
                "reference_official_question_exact_set_accuracy": right_questions[
                    question_id
                ]["official_question_exact_set_accuracy"],
                "official_question_exact_set_accuracy_gain": (
                    left_questions[question_id]["official_question_exact_set_accuracy"]
                    - right_questions[question_id][
                        "official_question_exact_set_accuracy"
                    ]
                ),
            }
            for question_id in sorted(left_questions)
        ]
        units = [[observation] for observation in observations]

        def calculate(
            sample: Sequence[Sequence[Mapping[str, Any]]],
        ) -> Mapping[str, float]:
            return {
                name: _mean([float(unit[0][name]) for unit in sample])
                for name in CLEVRER_PAIRED_METRIC_NAMES
            }

        stats = _bootstrap_unit_macro_stat_map(
            units,
            calculate,
            CLEVRER_PAIRED_METRIC_NAMES,
            resampling_ids=[
                str(observation["resampling_cluster_id"])
                for observation in observations
            ],
            seed=seed,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
    else:
        stats = {
            field: None
            for name in CLEVRER_PAIRED_METRIC_NAMES
            for field in (name, f"{name}_ci_low", f"{name}_ci_high")
        }
    return {
        **stats,
        "official_question_exact_set_comparison_authenticated": authenticated,
        "official_question_exact_set_comparison_questions": (
            len(left_questions) if authenticated else 0
        ),
        "official_question_semantic_aggregation_rule": (
            CLEVRER_SEMANTIC_AGGREGATION_RULE
        ),
        "official_question_exact_set_comparison_failures": {
            "condition": left_evaluation["failures"],
            "reference": right_evaluation["failures"],
            "question_sets_match": set(left_questions) == set(right_questions),
        },
        "candidate_level_accuracy_metrics_primary": False,
    }


def _condition_index(
    rows: Sequence[Mapping[str, Any]], condition: str
) -> tuple[dict[tuple[Any, ...], dict[tuple[str, int | str], Mapping[str, Any]]], int]:
    temporary: dict[
        tuple[Any, ...], dict[tuple[str, int | str], list[Mapping[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row.get("condition")) != condition:
            continue
        stratum = _group_key(row, STRATUM_FIELDS)
        temporary[stratum][_alignment_key(row)].append(row)
    ambiguous = 0
    result: dict[tuple[Any, ...], dict[tuple[str, int | str], Mapping[str, Any]]] = {}
    for stratum, values in temporary.items():
        result[stratum] = {}
        for key, matches in values.items():
            if len(matches) == 1:
                result[stratum][key] = matches[0]
            else:
                ambiguous += 1
    return result, ambiguous


def _contrast_rows(
    observed: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    left_condition: str,
    right_condition: str,
    contrast: str,
    comparison_type: str,
    seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    right_observed, ambiguous_right = _condition_index(observed, right_condition)
    right_expected, ambiguous_expected_right = _condition_index(
        expected, right_condition
    )
    left_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    left_expected_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in observed:
        if str(row.get("condition")) == left_condition:
            left_groups[_group_key(row)].append(row)
    for row in expected:
        if str(row.get("condition")) == left_condition:
            left_expected_groups[_group_key(row)].append(row)
    keys = sorted(set(left_groups) | set(left_expected_groups), key=_sort_key)
    outputs: list[dict[str, Any]] = []
    unmatched = 0
    ambiguous_left = 0
    ambiguous_expected_left = 0
    for key in keys:
        raw_rows = left_groups.get(key, [])
        raw_expected_rows = left_expected_groups.get(key, [])
        left_by_alignment: dict[tuple[str, int | str], list[Mapping[str, Any]]] = (
            defaultdict(list)
        )
        expected_left_by_alignment: dict[
            tuple[str, int | str], list[Mapping[str, Any]]
        ] = defaultdict(list)
        for row in raw_rows:
            left_by_alignment[_alignment_key(row)].append(row)
        for row in raw_expected_rows:
            expected_left_by_alignment[_alignment_key(row)].append(row)
        ambiguous_left += sum(
            len(matches) != 1 for matches in left_by_alignment.values()
        )
        ambiguous_expected_left += sum(
            len(matches) != 1 for matches in expected_left_by_alignment.values()
        )
        rows = [
            matches[0] for matches in left_by_alignment.values() if len(matches) == 1
        ]
        expected_rows = [
            matches[0]
            for matches in expected_left_by_alignment.values()
            if len(matches) == 1
        ]
        stratum = key[: len(STRATUM_FIELDS)]
        reference = right_observed.get(stratum, {})
        expected_reference = right_expected.get(stratum, {})
        pairs: list[dict[str, Any]] = []
        for row in rows:
            other = reference.get(_alignment_key(row))
            if other is None:
                unmatched += 1
                continue
            pairs.append(
                {
                    "cluster_id": _aggregation_unit_id(row),
                    "resampling_cluster_id": _cluster_id(row),
                    "left_correct": (
                        row.get("_correct")
                        if "_correct" in row
                        else _bool_value(row.get("correct"))
                    ),
                    "right_correct": (
                        other.get("_correct")
                        if "_correct" in other
                        else _bool_value(other.get("correct"))
                    ),
                    "left_margin": _finite_float(row.get("gold_margin")),
                    "right_margin": _finite_float(other.get("gold_margin")),
                }
            )
        expected_aligned = (
            sum(_alignment_key(row) in expected_reference for row in expected_rows)
            if expected
            else None
        )
        units = _paired_units(pairs)
        stats = _bootstrap_paired_stat_map(
            units,
            resampling_ids=_observation_unit_resampling_ids(units),
            seed=_stable_seed(seed, "contrast", contrast, key),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        for name in PAIRED_METRIC_NAMES:
            stats.setdefault(name, None)
            stats.setdefault(f"{name}_ci_low", None)
            stats.setdefault(f"{name}_ci_high", None)
        output = dict(zip(GROUP_FIELDS, key))
        output.update(
            {
                "comparison_type": comparison_type,
                "contrast": contrast,
                "condition": left_condition,
                "input_channel": _condition_input_channel([*rows, *expected_rows]),
                "reference_condition": right_condition,
                "aligned_rows": len(pairs),
                "expected_aligned_rows": expected_aligned,
                "alignment_coverage": (
                    len(pairs) / expected_aligned
                    if expected_aligned
                    else (None if expected else (1.0 if pairs else None))
                ),
                "unique_aligned_bases": len(
                    {
                        _alignment_key(row)[0]
                        for row in rows
                        if _alignment_key(row) in reference
                    }
                ),
                "unique_aligned_clusters": len(units),
                "unique_aligned_resampling_units": len(
                    set(_observation_unit_resampling_ids(units))
                ),
                "confidence_level": confidence_level,
                "bootstrap_replicates": bootstrap_replicates,
                **stats,
            }
        )
        outputs.append(output)
    return outputs, {
        "unmatched_left_rows": unmatched,
        "ambiguous_left_alignment_keys": ambiguous_left,
        "ambiguous_expected_left_alignment_keys": ambiguous_expected_left,
        "ambiguous_right_alignment_keys": ambiguous_right,
        "ambiguous_expected_right_alignment_keys": ambiguous_expected_right,
    }


def _condition_dose_sets(
    rows: Sequence[Mapping[str, Any]], condition: str
) -> dict[tuple[Any, ...], set[Any]]:
    result: dict[tuple[Any, ...], set[Any]] = defaultdict(set)
    for row in rows:
        if str(row.get("condition")) == condition:
            result[_group_key(row, STRATUM_FIELDS)].add(
                _dose_value(row.get("requested_dose"))
            )
    return result


def _dose_aware_condition_index(
    rows: Sequence[Mapping[str, Any]],
    condition: str,
    design_doses: Mapping[tuple[Any, ...], set[Any]],
) -> tuple[
    dict[tuple[Any, ...], dict[tuple[Any, tuple[str, int | str]], Mapping[str, Any]]],
    int,
]:
    temporary: dict[
        tuple[Any, ...],
        dict[tuple[Any, tuple[str, int | str]], list[Mapping[str, Any]]],
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row.get("condition")) != condition:
            continue
        stratum = _group_key(row, STRATUM_FIELDS)
        doses = design_doses.get(stratum) or {_dose_value(row.get("requested_dose"))}
        dose_key = _dose_value(row.get("requested_dose")) if len(doses) > 1 else None
        temporary[stratum][(dose_key, _alignment_key(row))].append(row)
    ambiguous = 0
    output: dict[
        tuple[Any, ...], dict[tuple[Any, tuple[str, int | str]], Mapping[str, Any]]
    ] = {}
    for stratum, indexed in temporary.items():
        output[stratum] = {}
        for key, matches in indexed.items():
            if len(matches) == 1:
                output[stratum][key] = matches[0]
            else:
                ambiguous += 1
    return output, ambiguous


def _confirmatory_contrast_rows(
    observed: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]],
    *,
    left_condition: str,
    right_condition: str,
    seed: int,
    bootstrap_replicates: int,
    confidence_level: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute a locked left-minus-right contrast with dose-aware alignment."""

    observed_right_doses = _condition_dose_sets(observed, right_condition)
    expected_right_doses = _condition_dose_sets(expected, right_condition)
    design_doses: dict[tuple[Any, ...], set[Any]] = {}
    for stratum in set(observed_right_doses) | set(expected_right_doses):
        # The expected manifest is authoritative.  This prevents a missing
        # observed dose from turning a multi-dose condition into a broadcast.
        design_doses[stratum] = set(
            expected_right_doses.get(stratum)
            or observed_right_doses.get(stratum)
            or set()
        )

    right_observed, ambiguous_right = _dose_aware_condition_index(
        observed, right_condition, design_doses
    )
    right_expected, ambiguous_expected_right = _dose_aware_condition_index(
        expected, right_condition, design_doses
    )
    left_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    left_expected_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(
        list
    )
    for row in observed:
        if str(row.get("condition")) == left_condition:
            left_groups[_group_key(row)].append(row)
    for row in expected:
        if str(row.get("condition")) == left_condition:
            left_expected_groups[_group_key(row)].append(row)

    outputs: list[dict[str, Any]] = []
    unmatched = 0
    expected_unmatched = 0
    ambiguous_left = 0
    ambiguous_expected_left = 0
    expected_left_rows_total = 0
    expected_aligned_rows_total = 0
    aligned_rows_total = 0
    expected_aligned_clusters: set[str] = set()
    expected_aligned_resampling_units: set[str] = set()
    aligned_clusters: set[str] = set()
    aligned_resampling_units: set[str] = set()
    estimands: list[dict[str, Any]] = []
    keys = sorted(set(left_groups) | set(left_expected_groups), key=_sort_key)
    for key in keys:
        raw_rows = left_groups.get(key, [])
        raw_expected_rows = left_expected_groups.get(key, [])
        left_by_alignment: dict[tuple[str, int | str], list[Mapping[str, Any]]] = (
            defaultdict(list)
        )
        expected_left_by_alignment: dict[
            tuple[str, int | str], list[Mapping[str, Any]]
        ] = defaultdict(list)
        for row in raw_rows:
            left_by_alignment[_alignment_key(row)].append(row)
        for row in raw_expected_rows:
            expected_left_by_alignment[_alignment_key(row)].append(row)
        ambiguous_left += sum(
            len(matches) != 1 for matches in left_by_alignment.values()
        )
        ambiguous_expected_left += sum(
            len(matches) != 1 for matches in expected_left_by_alignment.values()
        )
        rows = [
            matches[0] for matches in left_by_alignment.values() if len(matches) == 1
        ]
        expected_rows = [
            matches[0]
            for matches in expected_left_by_alignment.values()
            if len(matches) == 1
        ]
        expected_left_rows_total += len(raw_expected_rows)

        stratum = key[: len(STRATUM_FIELDS)]
        requested_dose = key[GROUP_FIELDS.index("requested_dose")]
        right_doses = design_doses.get(stratum, set())
        right_is_multi_dose = len(right_doses) > 1
        right_dose_key = requested_dose if right_is_multi_dose else None
        reference_requested_dose = (
            requested_dose
            if right_is_multi_dose
            else (next(iter(right_doses)) if len(right_doses) == 1 else None)
        )
        reference = right_observed.get(stratum, {})
        expected_reference = right_expected.get(stratum, {})
        pairs: list[dict[str, Any]] = []
        matched_left_rows: list[Mapping[str, Any]] = []
        matched_right_rows: list[Mapping[str, Any]] = []
        for row in rows:
            other = reference.get((right_dose_key, _alignment_key(row)))
            if other is None:
                unmatched += 1
                continue
            matched_left_rows.append(row)
            matched_right_rows.append(other)
            pairs.append(
                {
                    "cluster_id": _aggregation_unit_id(row),
                    "resampling_cluster_id": _cluster_id(row),
                    "left_correct": (
                        row.get("_correct")
                        if "_correct" in row
                        else _bool_value(row.get("correct"))
                    ),
                    "right_correct": (
                        other.get("_correct")
                        if "_correct" in other
                        else _bool_value(other.get("correct"))
                    ),
                    "left_margin": _finite_float(row.get("gold_margin")),
                    "right_margin": _finite_float(other.get("gold_margin")),
                }
            )
        expected_aligned_rows = (
            [
                row
                for row in expected_rows
                if (right_dose_key, _alignment_key(row)) in expected_reference
            ]
            if expected
            else []
        )
        expected_aligned = len(expected_aligned_rows) if expected else None
        expected_reference_rows = [
            expected_reference[(right_dose_key, _alignment_key(row))]
            for row in expected_aligned_rows
        ]
        expected_unmatched_in_group = (
            len(expected_rows) - len(expected_aligned_rows) if expected else 0
        )
        expected_unmatched += expected_unmatched_in_group
        expected_aligned_rows_total += len(expected_aligned_rows)
        aligned_rows_total += len(pairs)
        expected_group_clusters = {
            _aggregation_unit_id(row) for row in expected_aligned_rows
        }
        expected_group_resampling_units = {
            _cluster_id(row) for row in expected_aligned_rows
        }
        group_clusters = {str(pair["cluster_id"]) for pair in pairs}
        group_resampling_units = {str(pair["resampling_cluster_id"]) for pair in pairs}
        expected_aligned_clusters.update(expected_group_clusters)
        expected_aligned_resampling_units.update(expected_group_resampling_units)
        aligned_clusters.update(group_clusters)
        aligned_resampling_units.update(group_resampling_units)
        units = _paired_units(pairs)
        contrast = f"{left_condition}_minus_{right_condition}"
        stats = _bootstrap_paired_stat_map(
            units,
            resampling_ids=_observation_unit_resampling_ids(units),
            seed=_stable_seed(seed, "confirmatory", contrast, key),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        for name in PAIRED_METRIC_NAMES:
            stats.setdefault(name, None)
            stats.setdefault(f"{name}_ci_low", None)
            stats.setdefault(f"{name}_ci_high", None)
        clevrer_exact_set_stats = _clevrer_paired_exact_set_fields(
            matched_left_rows,
            matched_right_rows,
            expected_aligned_rows,
            expected_reference_rows,
            seed=_stable_seed(
                seed, "confirmatory_clevrer_semantic_exact_set", contrast, key
            ),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        output = dict(zip(GROUP_FIELDS, key))
        output.update(
            {
                "comparison_type": "confirmatory",
                "contrast": contrast,
                "condition": left_condition,
                "input_channel": _condition_input_channel([*rows, *expected_rows]),
                "reference_condition": right_condition,
                "reference_requested_dose": reference_requested_dose,
                "dose_alignment": (
                    "matched_requested_dose"
                    if right_is_multi_dose
                    else "broadcast_single_dose"
                ),
                "aligned_rows": len(pairs),
                "expected_left_rows": len(raw_expected_rows) if expected else None,
                "expected_aligned_rows": expected_aligned,
                "expected_unmatched_left_rows": (
                    expected_unmatched_in_group if expected else None
                ),
                "expected_alignment_coverage": (
                    len(expected_aligned_rows) / len(raw_expected_rows)
                    if expected and raw_expected_rows
                    else (None if expected else None)
                ),
                "alignment_coverage": (
                    len(pairs) / expected_aligned
                    if expected_aligned
                    else (None if expected else (1.0 if pairs else None))
                ),
                "unique_aligned_bases": len(
                    {
                        _alignment_key(row)[0]
                        for row in rows
                        if (right_dose_key, _alignment_key(row)) in reference
                    }
                ),
                "unique_aligned_clusters": len(units),
                "expected_unique_aligned_clusters": (
                    len(expected_group_clusters) if expected else None
                ),
                "unique_aligned_resampling_units": len(
                    set(_observation_unit_resampling_ids(units))
                ),
                "expected_unique_aligned_resampling_units": (
                    len(expected_group_resampling_units) if expected else None
                ),
                "confidence_level": confidence_level,
                "bootstrap_replicates": bootstrap_replicates,
                **stats,
                **clevrer_exact_set_stats,
            }
        )
        outputs.append(output)
        estimands.append(
            {
                **{field: output.get(field) for field in GROUP_FIELDS},
                "expected_left_rows": output.get("expected_left_rows"),
                "expected_aligned_rows": output.get("expected_aligned_rows"),
                "expected_unmatched_left_rows": output.get(
                    "expected_unmatched_left_rows"
                ),
                "expected_alignment_coverage": output.get(
                    "expected_alignment_coverage"
                ),
                "aligned_rows": output.get("aligned_rows"),
                "alignment_coverage": output.get("alignment_coverage"),
                "expected_unique_aligned_clusters": output.get(
                    "expected_unique_aligned_clusters"
                ),
                "unique_aligned_clusters": output.get("unique_aligned_clusters"),
                "expected_unique_aligned_resampling_units": output.get(
                    "expected_unique_aligned_resampling_units"
                ),
                "unique_aligned_resampling_units": output.get(
                    "unique_aligned_resampling_units"
                ),
            }
        )
    return outputs, {
        "left_condition": left_condition,
        "right_condition": right_condition,
        "dose_rule": "match requested_dose when right has multiple requested doses; otherwise broadcast",
        "unmatched_left_rows": unmatched,
        "expected_unmatched_left_rows": expected_unmatched,
        "expected_left_rows": expected_left_rows_total,
        "expected_aligned_rows": expected_aligned_rows_total,
        "aligned_rows": aligned_rows_total,
        "expected_unique_aligned_clusters": len(expected_aligned_clusters),
        "unique_aligned_clusters": len(aligned_clusters),
        "expected_unique_aligned_resampling_units": len(
            expected_aligned_resampling_units
        ),
        "unique_aligned_resampling_units": len(aligned_resampling_units),
        "estimands": estimands,
        "ambiguous_left_alignment_keys": ambiguous_left,
        "ambiguous_expected_left_alignment_keys": ambiguous_expected_left,
        "ambiguous_right_alignment_keys": ambiguous_right,
        "ambiguous_expected_right_alignment_keys": ambiguous_expected_right,
    }


def paired_comparisons(
    observed: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]] = (),
    *,
    reference_condition: str = DEFAULT_REFERENCE_CONDITION,
    confirmatory_comparisons: Sequence[Sequence[str]] = (),
    seed: int = DEFAULT_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    conditions = sorted(
        {
            str(row.get("condition"))
            for row in [*observed, *expected]
            if not _missing(row.get("condition"))
        }
    )
    outputs: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}

    for left_condition, right_condition in confirmatory_comparisons:
        name = f"{left_condition}_minus_{right_condition}"
        if left_condition not in conditions or right_condition not in conditions:
            diagnostics[f"confirmatory::{name}"] = {
                "left_condition": left_condition,
                "right_condition": right_condition,
                "not_computed": 1,
            }
            continue
        rows, details = _confirmatory_contrast_rows(
            observed,
            expected,
            left_condition=left_condition,
            right_condition=right_condition,
            seed=seed,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        outputs.extend(rows)
        diagnostics[f"confirmatory::{name}"] = details

    for condition in conditions:
        if condition == reference_condition:
            continue
        rows, details = _contrast_rows(
            observed,
            expected,
            left_condition=condition,
            right_condition=reference_condition,
            contrast=f"{condition}_minus_{reference_condition}",
            comparison_type="condition_vs_reference",
            seed=seed,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        outputs.extend(rows)
        diagnostics[f"{condition}_vs_{reference_condition}"] = details

    rows_by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in [*observed, *expected]:
        if not _missing(row.get("condition")):
            rows_by_condition[str(row.get("condition"))].append(row)
    if "question_only" in conditions and reference_condition != "question_only":
        for condition in conditions:
            if condition in {reference_condition, "question_only"}:
                continue
            channel = _condition_input_channel(rows_by_condition.get(condition, []))
            if channel.casefold() not in {"text_oracle", "embedding_oracle"}:
                # Backward-compatible inference is recorded through the
                # emitted <missing> channel rather than changing row metadata.
                inferred_reference = _dose_reference_condition(
                    input_channel=channel,
                    condition=condition,
                    visual_reference_condition=reference_condition,
                )
                if inferred_reference != "question_only":
                    continue
            rows, details = _contrast_rows(
                observed,
                expected,
                left_condition=condition,
                right_condition="question_only",
                contrast=f"{condition}_minus_question_only",
                comparison_type="oracle_vs_question_only",
                seed=seed,
                bootstrap_replicates=bootstrap_replicates,
                confidence_level=confidence_level,
            )
            outputs.extend(rows)
            diagnostics[f"{condition}_vs_question_only"] = details

    evidence_contrasts = (
        ("evidence_only", reference_condition, "evidence_sufficiency"),
        ("evidence_present", "evidence_removed", "evidence_comprehensiveness"),
        ("evidence_present", "random_position_mask", "random_mask_placebo_cost"),
        ("random_position_mask", "evidence_removed", "evidence_mask_specificity"),
        ("random_matched", reference_condition, "random_control"),
        ("evidence_only", "random_matched", "evidence_specificity"),
    )
    for left, right, name in evidence_contrasts:
        if left not in conditions or right not in conditions:
            diagnostics[name] = {"not_computed": 1}
            continue
        rows, details = _contrast_rows(
            observed,
            expected,
            left_condition=left,
            right_condition=right,
            contrast=name,
            comparison_type="evidence_metric",
            seed=seed,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        outputs.extend(rows)
        diagnostics[name] = details

    placebo_contrasts = (
        (
            "ordered_oracle",
            "ordered_timestamp_sham",
            "timestamp_information_over_sham",
        ),
        (
            "ordered_timestamp_sham",
            "atomic_oracle",
            "timestamp_format_placebo",
        ),
        (
            "reasoning_oracle",
            "reasoning_operator_sham",
            "operator_information_over_sham",
        ),
        (
            "reasoning_operator_sham",
            "ordered_oracle",
            "operator_format_placebo",
        ),
    )
    for left, right, name in placebo_contrasts:
        if left not in conditions or right not in conditions:
            diagnostics[name] = {"not_computed": 1}
            continue
        rows, details = _contrast_rows(
            observed,
            expected,
            left_condition=left,
            right_condition=right,
            contrast=name,
            comparison_type="placebo_metric",
            seed=seed,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        outputs.extend(rows)
        diagnostics[name] = details
    outputs.sort(
        key=lambda row: _sort_key(
            [
                *(row.get(field) for field in STRATUM_FIELDS),
                row.get("comparison_type"),
                row.get("contrast"),
                row.get("requested_dose"),
                row.get("effective_dose"),
            ]
        )
    )
    return outputs, diagnostics


def _role_units(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[tuple[str, Any], dict[str, list[Mapping[str, Any]]]], int]:
    units: dict[tuple[str, Any], dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        pair_id = row.get("pair_id")
        role = str(row.get("pair_role", "")).casefold()
        if _missing(pair_id) or role not in {"original", "counterfactual", "nuisance"}:
            continue
        units[(str(pair_id), row.get("permutation_index", 0))][role].append(row)
    ambiguous = sum(
        1
        for role_map in units.values()
        for matches in role_map.values()
        if len(matches) != 1
    )
    return units, ambiguous


def _pair_stat_calculator(
    names: Sequence[str],
) -> Callable[[Sequence[Sequence[Mapping[str, Any]]]], Mapping[str, float]]:
    def calculate(units: Sequence[Sequence[Mapping[str, Any]]]) -> Mapping[str, float]:
        values: dict[str, list[float]] = defaultdict(list)
        for unit in units:
            for name in names:
                present = [
                    float(item[name]) for item in unit if item.get(name) is not None
                ]
                if present:
                    values[name].append(_mean(present))
        return {
            name: (_mean(values[name]) if values.get(name) else math.nan)
            for name in names
        }

    return calculate


def _cluster_pair_observations(
    observations: Sequence[Mapping[str, Any]], cluster_field: str = "cluster_id"
) -> list[list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for observation in observations:
        grouped[str(observation[cluster_field])].append(observation)
    return [grouped[key] for key in sorted(grouped)]


def _build_role_observations(
    rows: Sequence[Mapping[str, Any]], other_role: str
) -> tuple[list[dict[str, Any]], int, int]:
    role_units, ambiguous = _role_units(rows)
    observations: list[dict[str, Any]] = []
    possible = 0
    for (pair_id, _permutation), role_map in role_units.items():
        if "original" in role_map or other_role in role_map:
            possible += 1
        if (
            len(role_map.get("original", [])) != 1
            or len(role_map.get(other_role, [])) != 1
        ):
            continue
        original = role_map["original"][0]
        other = role_map[other_role][0]
        original_correct = (
            original.get("_correct")
            if "_correct" in original
            else _bool_value(original.get("correct"))
        )
        other_correct = (
            other.get("_correct")
            if "_correct" in other
            else _bool_value(other.get("correct"))
        )
        prediction_original = _semantic(original.get("prediction_text"))
        prediction_other = _semantic(other.get("prediction_text"))
        answer_original = _semantic(original.get("answer_text"))
        answer_other = _semantic(other.get("answer_text"))
        both_correct = (
            bool(original_correct and other_correct)
            if original_correct is not None and other_correct is not None
            else None
        )
        observation: dict[str, Any] = {
            "cluster_id": f"pair::{pair_id}",
            "resampling_cluster_id": _cluster_id(original),
            "both_correct": both_correct,
        }
        if prediction_original is not None and prediction_other is not None:
            if other_role == "counterfactual":
                predicted_flip = prediction_original != prediction_other
                gold_flip = (
                    answer_original != answer_other
                    if answer_original is not None and answer_other is not None
                    else None
                )
                observation["semantic_flip"] = predicted_flip
                observation["correct_semantic_flip"] = (
                    bool(both_correct and predicted_flip and gold_flip)
                    if both_correct is not None and gold_flip is not None
                    else None
                )
            else:
                invariant = prediction_original == prediction_other
                gold_invariant = (
                    answer_original == answer_other
                    if answer_original is not None and answer_other is not None
                    else None
                )
                observation["invariance"] = invariant
                observation["correct_invariance"] = (
                    bool(both_correct and invariant and gold_invariant)
                    if both_correct is not None and gold_invariant is not None
                    else None
                )
        observations.append(observation)
    return observations, possible, ambiguous


def _permutation_observations(
    rows: Sequence[Mapping[str, Any]], expected_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    observed_by_base: dict[str, dict[Any, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    expected_by_base: dict[str, set[Any]] = defaultdict(set)
    for row in rows:
        observed_by_base[str(row.get("base_id", "<missing>"))][
            row.get("permutation_index", 0)
        ].append(row)
    if expected_rows:
        for row in expected_rows:
            expected_by_base[str(row.get("base_id", "<missing>"))].add(
                row.get("permutation_index", 0)
            )
    else:
        inferred = {row.get("permutation_index", 0) for row in rows}
        for base_id in observed_by_base:
            expected_by_base[base_id] = set(inferred)

    observations: list[dict[str, Any]] = []
    ambiguous = 0
    incomplete = 0
    for base_id, expected_permutations in expected_by_base.items():
        observed_map = observed_by_base.get(base_id, {})
        if any(len(matches) != 1 for matches in observed_map.values()):
            ambiguous += 1
            continue
        if set(observed_map) != expected_permutations:
            incomplete += 1
            continue
        ordered = [
            observed_map[key][0]
            for key in sorted(
                expected_permutations, key=lambda value: _sort_key([value])
            )
        ]
        predictions = [_semantic(row.get("prediction_text")) for row in ordered]
        correctness = [
            (
                row.get("_correct")
                if "_correct" in row
                else _bool_value(row.get("correct"))
            )
            for row in ordered
        ]
        semantic_consistency = (
            None
            if any(value is None for value in predictions)
            else len(set(predictions)) == 1
        )
        stability = None
        if not any(value is None for value in predictions):
            counts = Counter(predictions)
            stability = max(counts.values()) / len(predictions)
        all_correct = (
            None if any(value is None for value in correctness) else all(correctness)
        )
        source = ordered[0]
        observations.append(
            {
                "cluster_id": _aggregation_unit_id(source),
                "resampling_cluster_id": _cluster_id(source),
                "semantic_consistency": semantic_consistency,
                "semantic_stability": stability,
                "all_permutations_correct": all_correct,
                "permutation_count": len(ordered),
            }
        )
    return observations, {
        "expected_units": len(expected_by_base),
        "complete_units": len(observations),
        "incomplete_units": incomplete,
        "ambiguous_units": ambiguous,
    }


def compute_pair_metrics(
    observed: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]] = (),
    *,
    seed: int = DEFAULT_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    expected_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observed:
        observed_groups[_group_key(row)].append(row)
    for row in expected:
        expected_groups[_group_key(row)].append(row)
    keys = sorted(set(observed_groups) | set(expected_groups), key=_sort_key)
    outputs: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for key in keys:
        rows = observed_groups.get(key, [])
        expected_rows = expected_groups.get(key, [])
        cf, cf_possible, cf_ambiguous = _build_role_observations(rows, "counterfactual")
        nuisance, nuisance_possible, nuisance_ambiguous = _build_role_observations(
            rows, "nuisance"
        )
        expected_cf, expected_cf_possible, _ = (
            _build_role_observations(expected_rows, "counterfactual")
            if expected
            else ([], cf_possible, 0)
        )
        expected_nuisance, expected_nuisance_possible, _ = (
            _build_role_observations(expected_rows, "nuisance")
            if expected
            else ([], nuisance_possible, 0)
        )
        permutations, permutation_coverage = _permutation_observations(
            rows, expected_rows
        )

        cf_names = ("both_correct", "semantic_flip", "correct_semantic_flip")
        nuisance_names = ("both_correct", "invariance", "correct_invariance")
        permutation_names = (
            "semantic_consistency",
            "semantic_stability",
            "all_permutations_correct",
        )
        cf_units = _cluster_pair_observations(cf)
        nuisance_units = _cluster_pair_observations(nuisance)
        permutation_units = _cluster_pair_observations(permutations)
        cf_stats = _bootstrap_unit_macro_stat_map(
            cf_units,
            _pair_stat_calculator(cf_names),
            cf_names,
            resampling_ids=_observation_unit_resampling_ids(cf_units),
            seed=_stable_seed(seed, "counterfactual", key),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        nuisance_stats = _bootstrap_unit_macro_stat_map(
            nuisance_units,
            _pair_stat_calculator(nuisance_names),
            nuisance_names,
            resampling_ids=_observation_unit_resampling_ids(nuisance_units),
            seed=_stable_seed(seed, "nuisance", key),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        permutation_stats = _bootstrap_unit_macro_stat_map(
            permutation_units,
            _pair_stat_calculator(permutation_names),
            permutation_names,
            resampling_ids=_observation_unit_resampling_ids(permutation_units),
            seed=_stable_seed(seed, "permutation", key),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        for stats, names in (
            (cf_stats, cf_names),
            (nuisance_stats, nuisance_names),
            (permutation_stats, permutation_names),
        ):
            for name in names:
                stats.setdefault(name, None)
                stats.setdefault(f"{name}_ci_low", None)
                stats.setdefault(f"{name}_ci_high", None)

        output = dict(zip(GROUP_FIELDS, key))
        output.update(
            {
                "input_channel": _condition_input_channel([*rows, *expected_rows]),
                "counterfactual_complete_units": len(cf),
                "counterfactual_correctness_units": sum(
                    item.get("both_correct") is not None for item in cf
                ),
                "counterfactual_semantic_units": sum(
                    item.get("semantic_flip") is not None for item in cf
                ),
                "counterfactual_expected_units": (
                    expected_cf_possible if expected else cf_possible
                ),
                "counterfactual_coverage": (
                    len(cf) / expected_cf_possible if expected_cf_possible else None
                ),
                "counterfactual_ambiguous_units": cf_ambiguous,
                "nuisance_complete_units": len(nuisance),
                "nuisance_correctness_units": sum(
                    item.get("both_correct") is not None for item in nuisance
                ),
                "nuisance_semantic_units": sum(
                    item.get("invariance") is not None for item in nuisance
                ),
                "nuisance_expected_units": (
                    expected_nuisance_possible if expected else nuisance_possible
                ),
                "nuisance_coverage": (
                    len(nuisance) / expected_nuisance_possible
                    if expected_nuisance_possible
                    else None
                ),
                "nuisance_ambiguous_units": nuisance_ambiguous,
                "permutation_complete_units": permutation_coverage["complete_units"],
                "permutation_correctness_units": sum(
                    item.get("all_permutations_correct") is not None
                    for item in permutations
                ),
                "permutation_semantic_units": sum(
                    item.get("semantic_consistency") is not None
                    for item in permutations
                ),
                "permutation_expected_units": permutation_coverage["expected_units"],
                "permutation_coverage": (
                    permutation_coverage["complete_units"]
                    / permutation_coverage["expected_units"]
                    if permutation_coverage["expected_units"]
                    else None
                ),
                "permutation_incomplete_units": permutation_coverage[
                    "incomplete_units"
                ],
                "permutation_ambiguous_units": permutation_coverage["ambiguous_units"],
                "confidence_level": confidence_level,
                "bootstrap_replicates": bootstrap_replicates,
            }
        )
        output.update(
            {f"counterfactual_{name}": value for name, value in cf_stats.items()}
        )
        output.update(
            {f"nuisance_{name}": value for name, value in nuisance_stats.items()}
        )
        output.update(
            {f"permutation_{name}": value for name, value in permutation_stats.items()}
        )
        outputs.append(output)
        totals.update(
            {
                "counterfactual_complete_units": len(cf),
                "counterfactual_expected_units": (
                    expected_cf_possible if expected else cf_possible
                ),
                "nuisance_complete_units": len(nuisance),
                "nuisance_expected_units": (
                    expected_nuisance_possible if expected else nuisance_possible
                ),
                "permutation_complete_units": permutation_coverage["complete_units"],
                "permutation_expected_units": permutation_coverage["expected_units"],
                "permutation_incomplete_units": permutation_coverage[
                    "incomplete_units"
                ],
            }
        )
    return outputs, dict(totals)


def _requested_dose_sort(
    value: Any, effective_mean: float | None
) -> tuple[float, tuple[tuple[int, Any], ...]]:
    effective = (
        effective_mean
        if effective_mean is not None and math.isfinite(effective_mean)
        else math.inf
    )
    return effective, _sort_key([value])


def compute_dose_curves(
    observed: Sequence[Mapping[str, Any]],
    expected: Sequence[Mapping[str, Any]] = (),
    *,
    reference_condition: str = DEFAULT_REFERENCE_CONDITION,
    seed: int = DEFAULT_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reference_conditions = {reference_condition, "question_only"}
    reference_observed: dict[
        str, dict[tuple[Any, ...], dict[tuple[str, int | str], Mapping[str, Any]]]
    ] = {}
    reference_expected: dict[
        str, dict[tuple[Any, ...], dict[tuple[str, int | str], Mapping[str, Any]]]
    ] = {}
    ambiguous_reference: dict[str, int] = {}
    ambiguous_expected_reference: dict[str, int] = {}
    for name in sorted(reference_conditions):
        reference_observed[name], ambiguous_reference[name] = _condition_index(
            observed, name
        )
        reference_expected[name], ambiguous_expected_reference[name] = _condition_index(
            expected, name
        )
    all_rows = [*observed, *expected]
    candidate_conditions = {
        str(row.get("condition"))
        for row in all_rows
        if not _missing(row.get("condition"))
        and str(row.get("condition")) != reference_condition
        and (
            "oracle" in str(row.get("condition")).casefold()
            or _dose_value(row.get("requested_dose")) not in {0, 0.0, "0", "<missing>"}
        )
    }
    point_fields = STRATUM_FIELDS + ("condition", "input_channel", "requested_dose")
    observed_points: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    expected_points: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in observed:
        if str(row.get("condition")) in candidate_conditions:
            observed_points[_group_key(row, point_fields)].append(row)
    for row in expected:
        if str(row.get("condition")) in candidate_conditions:
            expected_points[_group_key(row, point_fields)].append(row)

    raw_points: list[dict[str, Any]] = []
    unmatched = 0
    for key in sorted(set(observed_points) | set(expected_points), key=_sort_key):
        rows = observed_points.get(key, [])
        expected_rows = expected_points.get(key, [])
        stratum = key[: len(STRATUM_FIELDS)]
        condition = key[len(STRATUM_FIELDS)]
        input_channel = key[len(STRATUM_FIELDS) + 1]
        actual_reference_condition = _dose_reference_condition(
            input_channel=input_channel,
            condition=condition,
            visual_reference_condition=reference_condition,
        )
        reference = reference_observed.get(actual_reference_condition, {}).get(
            stratum, {}
        )
        expected_reference = reference_expected.get(actual_reference_condition, {}).get(
            stratum, {}
        )
        pairs: list[dict[str, Any]] = []
        effective_by_cluster: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            effective = _finite_float(row.get("effective_dose"))
            if effective is not None:
                effective_by_cluster[_aggregation_unit_id(row)].append(effective)
            other = reference.get(_alignment_key(row))
            if other is None:
                unmatched += 1
                continue
            pairs.append(
                {
                    "cluster_id": _aggregation_unit_id(row),
                    "resampling_cluster_id": _cluster_id(row),
                    "left_correct": (
                        row.get("_correct")
                        if "_correct" in row
                        else _bool_value(row.get("correct"))
                    ),
                    "right_correct": (
                        other.get("_correct")
                        if "_correct" in other
                        else _bool_value(other.get("correct"))
                    ),
                    "left_margin": _finite_float(row.get("gold_margin")),
                    "right_margin": _finite_float(other.get("gold_margin")),
                }
            )
        expected_aligned = (
            sum(_alignment_key(row) in expected_reference for row in expected_rows)
            if expected
            else None
        )
        paired_units = _paired_units(pairs)
        stats = _bootstrap_paired_stat_map(
            paired_units,
            resampling_ids=_observation_unit_resampling_ids(paired_units),
            seed=_stable_seed(seed, "dose", actual_reference_condition, key),
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
        )
        for name in PAIRED_METRIC_NAMES:
            stats.setdefault(name, None)
            stats.setdefault(f"{name}_ci_low", None)
            stats.setdefault(f"{name}_ci_high", None)
        effective_values = [
            _mean(effective_by_cluster[cluster_id])
            for cluster_id in sorted(effective_by_cluster)
        ]
        output = dict(zip(point_fields, key))
        output.update(
            {
                "reference_condition": actual_reference_condition,
                "visual_reference_condition": reference_condition,
                "effective_dose_mean": (
                    _mean(effective_values) if effective_values else None
                ),
                "effective_dose_min": (
                    min(effective_values) if effective_values else None
                ),
                "effective_dose_max": (
                    max(effective_values) if effective_values else None
                ),
                "aligned_rows": len(pairs),
                "expected_aligned_rows": expected_aligned,
                "alignment_coverage": (
                    len(pairs) / expected_aligned
                    if expected_aligned
                    else (None if expected else (1.0 if pairs else None))
                ),
                "confidence_level": confidence_level,
                "bootstrap_replicates": bootstrap_replicates,
                **stats,
            }
        )
        output["oracle_gain"] = output.pop("accuracy_gain", None)
        output["oracle_gain_ci_low"] = output.pop("accuracy_gain_ci_low", None)
        output["oracle_gain_ci_high"] = output.pop("accuracy_gain_ci_high", None)
        raw_points.append(output)

    curves: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    curve_fields = STRATUM_FIELDS + (
        "condition",
        "input_channel",
        "reference_condition",
    )
    for point in raw_points:
        curves[tuple(point[field] for field in curve_fields)].append(point)
    outputs: list[dict[str, Any]] = []
    nonpositive = 0
    for curve_key in sorted(curves, key=_sort_key):
        points = curves[curve_key]
        points.sort(
            key=lambda point: _requested_dose_sort(
                point.get("requested_dose"), point.get("effective_dose_mean")
            )
        )
        finite_gains = [
            float(point["oracle_gain"])
            for point in points
            if _finite_float(point.get("oracle_gain")) is not None
        ]
        max_gain = max(finite_gains) if finite_gains else None
        threshold = 0.9 * max_gain if max_gain is not None and max_gain > 0 else None
        k90_point: dict[str, Any] | None = None
        if threshold is not None:
            k90_point = next(
                (
                    point
                    for point in points
                    if _finite_float(point.get("oracle_gain")) is not None
                    and float(point["oracle_gain"]) >= threshold
                ),
                None,
            )
        else:
            nonpositive += 1
        for index, point in enumerate(points):
            point.update(
                {
                    "dose_index": index,
                    "max_oracle_gain": max_gain,
                    "k90_threshold": threshold,
                    "k90_requested_dose": (
                        k90_point.get("requested_dose") if k90_point else None
                    ),
                    "k90_effective_dose": (
                        k90_point.get("effective_dose_mean") if k90_point else None
                    ),
                    "k90_status": (
                        "reached"
                        if k90_point
                        else (
                            "nonpositive_max_gain"
                            if max_gain is not None
                            else "no_aligned_data"
                        )
                    ),
                }
            )
            outputs.append(point)
    return outputs, {
        "curve_count": len(curves),
        "nonpositive_or_empty_curve_count": nonpositive,
        "unmatched_dose_rows": unmatched,
        "ambiguous_reference_alignment_keys": ambiguous_reference,
        "ambiguous_expected_reference_alignment_keys": ambiguous_expected_reference,
    }


def analyze_predictions(
    prediction_rows: Iterable[Mapping[str, Any]],
    expected_rows: Iterable[Mapping[str, Any]] | None = None,
    *,
    seed: int = DEFAULT_SEED,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    reference_condition: str = DEFAULT_REFERENCE_CONDITION,
    ece_bins: int = DEFAULT_ECE_BINS,
    minimum_confirmatory_resampling_units: int = (
        DEFAULT_MINIMUM_CONFIRMATORY_RESAMPLING_UNITS
    ),
    confirmatory_comparisons: Sequence[Sequence[str]] = (),
    input_issues: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be >= 0")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0, 1)")
    if ece_bins < 1:
        raise ValueError("ece_bins must be >= 1")
    if _positive_int(minimum_confirmatory_resampling_units) is None:
        raise ValueError(
            "minimum_confirmatory_resampling_units must be a positive integer"
        )
    if not reference_condition:
        raise ValueError("reference_condition cannot be empty")
    normalized_confirmatory_comparisons = _normalize_confirmatory_comparisons(
        confirmatory_comparisons
    )

    observed, expected, coverage = _prepare_rows(
        prediction_rows, expected_rows, external_issues=input_issues
    )
    summary = summarize_predictions(
        observed,
        expected,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        ece_bins=ece_bins,
    )
    comparisons, comparison_coverage = paired_comparisons(
        observed,
        expected,
        reference_condition=reference_condition,
        confirmatory_comparisons=normalized_confirmatory_comparisons,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )
    pair_metrics, pair_coverage = compute_pair_metrics(
        observed,
        expected,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )
    dose_curves, dose_coverage = compute_dose_curves(
        observed,
        expected,
        reference_condition=reference_condition,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
        confidence_level=confidence_level,
    )
    clevrer_summary_rows = [
        row
        for row in summary
        if str(row.get("dataset", "")).strip().casefold() == "clevrer"
    ]
    clevrer_confirmatory_rows = [
        row
        for row in comparisons
        if str(row.get("dataset", "")).strip().casefold() == "clevrer"
        and row.get("comparison_type") == "confirmatory"
    ]
    clevrer_primary_coverage = {
        "summary_groups": len(clevrer_summary_rows),
        "authenticated_summary_groups": sum(
            row.get("official_question_exact_set_authenticated") is True
            and all(
                _finite_float(row.get(field)) is not None
                for field in (
                    "primary_accuracy",
                    "primary_accuracy_ci_low",
                    "primary_accuracy_ci_high",
                )
            )
            for row in clevrer_summary_rows
        ),
        "confirmatory_comparison_rows": len(clevrer_confirmatory_rows),
        "authenticated_confirmatory_comparison_rows": sum(
            row.get("official_question_exact_set_comparison_authenticated") is True
            and all(
                _finite_float(row.get(field)) is not None
                for name in CLEVRER_PAIRED_METRIC_NAMES
                for field in (name, f"{name}_ci_low", f"{name}_ci_high")
            )
            for row in clevrer_confirmatory_rows
        ),
    }
    report = {
        "protocol": {
            "seed": seed,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_engine": {
                "summary_and_mean_metrics": (
                    "batched_cluster_count_sufficient_statistics_v1"
                    if np is not None
                    else "generic_python_cluster_resampling"
                ),
                "max_batch_replicates": BOOTSTRAP_MAX_BATCH_REPLICATES,
                "max_count_cells": BOOTSTRAP_MAX_COUNT_CELLS,
                "rng": "python_random.Random_replicate_major",
            },
            "confidence_level": confidence_level,
            "reference_condition": reference_condition,
            "confirmatory_comparisons": normalized_confirmatory_comparisons,
            "minimum_confirmatory_resampling_units": (
                minimum_confirmatory_resampling_units
            ),
            "dose_reference_by_input_channel": {
                "text_oracle": "question_only",
                "embedding_oracle": "question_only",
                "visual_plus_text": reference_condition,
                "default": reference_condition,
            },
            "ece_bins": ece_bins,
            "aggregation_unit": (
                "pair_id for original/counterfactual/nuisance; otherwise "
                "independent_unit_id when supplied, then base_id"
            ),
            "bootstrap_resampling_unit": (
                "resampling_unit_id (raw video/scene or paired-video family); "
                "fallback to the aggregation unit"
            ),
            "summary_accuracy_definitions": {
                "primary_accuracy": (
                    "official_question_exact_set_accuracy for CLEVRER; otherwise "
                    "cluster_macro_accuracy"
                ),
                "row_micro_accuracy": "mean correctness over valid scored rows",
                "cluster_macro_accuracy": "unweighted mean of within-cluster row accuracy",
                "accuracy": "backward-compatible alias of cluster_macro_accuracy",
                "cluster_all_rows_correct": "fraction of complete clusters with every row correct",
                "official_question_exact_set_accuracy": (
                    "CLEVRER primary estimand: fraction of authenticated official questions "
                    "whose complete candidate set is correct after semantic probability "
                    "aggregation across authenticated option permutations"
                ),
                "official_question_permutation_robustness_accuracy": (
                    "CLEVRER diagnostic: fraction of authenticated official questions "
                    "for which every candidate is correct under every option permutation"
                ),
                "clevrer_candidate_level_accuracy": (
                    "diagnostic only because candidate labels are dominated by 'No'"
                ),
            },
            "clevrer_semantic_aggregation_rule": (CLEVRER_SEMANTIC_AGGREGATION_RULE),
            "brier_definition": "unnormalized multiclass Brier over normalized option pseudo-probabilities",
            "ece_definition": "equal-cluster-weighted max-option confidence ECE",
        },
        "coverage": coverage,
        "analysis_coverage": {
            "summary_groups": len(summary),
            "comparison_rows": len(comparisons),
            "pair_metric_groups": len(pair_metrics),
            "dose_curve_rows": len(dose_curves),
            "clevrer_primary": clevrer_primary_coverage,
            "comparisons": comparison_coverage,
            "pairs_and_permutations": pair_coverage,
            "dose_curves": dose_coverage,
        },
    }
    return {
        "summary": summary,
        "comparisons": comparisons,
        "pair_metrics": pair_metrics,
        "dose_curves": dose_curves,
        "report": report,
    }


def _read_jsonl_tolerant(
    path: str | Path, *, source: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    input_path = Path(path)
    if input_path.suffix.casefold() == ".gz":
        handle_context = gzip.open(input_path, "rt", encoding="utf-8")
    else:
        handle_context = input_path.open("r", encoding="utf-8")
    with handle_context as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                issues.append(
                    {
                        "kind": f"malformed_{source}_jsonl",
                        "line": line_number,
                        "message": str(exc),
                    }
                )
                continue
            if not isinstance(value, dict):
                issues.append(
                    {
                        "kind": f"non_object_{source}_jsonl",
                        "line": line_number,
                    }
                )
                continue
            rows.append(value)
    return rows, issues


def _scalar_yaml_value(text: str) -> Any:
    value = text.strip()
    if not value:
        return None
    if value[0:1] in {'"', "'"} and value[-1:] == value[0:1]:
        return value[1:-1]
    lowered = value.casefold()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _normalize_confirmatory_comparisons(value: Any) -> list[list[str]]:
    if value in (None, "", []):
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            "confirmatory_comparisons must be a list of [left, right] pairs"
        )
    output: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for index, raw_pair in enumerate(value):
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            raise ValueError(
                f"confirmatory_comparisons[{index}] must contain exactly [left, right]"
            )
        left, right = (str(item).strip() for item in raw_pair)
        if not left or not right:
            raise ValueError(
                f"confirmatory_comparisons[{index}] contains an empty condition"
            )
        pair = (left, right)
        if pair in seen:
            raise ValueError(f"duplicate confirmatory comparison: {left} minus {right}")
        seen.add(pair)
        output.append([left, right])
    return output


def _yaml_confirmatory_comparisons(raw_lines: Sequence[str]) -> list[list[str]] | None:
    header = re.compile(r"^(\s*)confirmatory_comparisons\s*:\s*(.*?)\s*$")
    entry = re.compile(r"^\s*-\s*\[(.*)\]\s*$")
    for index, raw_line in enumerate(raw_lines):
        content = raw_line.split("#", 1)[0].rstrip()
        match = header.match(content)
        if not match:
            continue
        inline = match.group(2).strip()
        if inline:
            if inline == "[]":
                return []
            raise ValueError(
                "inline confirmatory_comparisons must be []; use one '- [left, right]' per line"
            )
        header_indent = len(match.group(1))
        pairs: list[list[str]] = []
        for candidate in raw_lines[index + 1 :]:
            candidate_content = candidate.split("#", 1)[0].rstrip()
            if not candidate_content.strip():
                continue
            indent = len(candidate_content) - len(candidate_content.lstrip())
            if indent <= header_indent:
                break
            item_match = entry.match(candidate_content)
            if not item_match:
                raise ValueError(
                    "confirmatory_comparisons entries must use '- [left_condition, right_condition]'"
                )
            pieces = [piece.strip() for piece in item_match.group(1).split(",")]
            if len(pieces) != 2:
                raise ValueError(
                    "each confirmatory comparison must contain exactly two conditions"
                )
            pairs.append([str(_scalar_yaml_value(piece)) for piece in pieces])
        return _normalize_confirmatory_comparisons(pairs)
    return None


def load_protocol_config(
    path: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load scalar settings from the protocol's ``analysis`` section.

    The protocol also contains a sampling seed.  Scoping this parser to the
    analysis section prevents YAML key order from silently selecting the wrong
    seed when both sections are present.
    """

    if path is None:
        return {}, {"path": None, "found": False, "used": False}
    source = Path(path)
    if not source.is_file():
        return {}, {"path": str(source), "found": False, "used": False}
    text = source.read_text(encoding="utf-8")
    wanted = {
        "bootstrap_replicates",
        "confidence_level",
        "seed",
        "reference_condition",
        "ece_bins",
        "minimum_confirmatory_resampling_units",
    }
    values: dict[str, Any] = {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, Mapping):
        candidate = payload.get("analysis", payload)
        if isinstance(candidate, Mapping):
            for key in wanted:
                item = candidate.get(key)
                if item is not None and not isinstance(item, (Mapping, list)):
                    values[key] = item
        raw_comparisons = payload.get("confirmatory_comparisons")
        if raw_comparisons is None and isinstance(candidate, Mapping):
            raw_comparisons = candidate.get("confirmatory_comparisons")
        if raw_comparisons is not None:
            values["confirmatory_comparisons"] = _normalize_confirmatory_comparisons(
                raw_comparisons
            )
    else:
        pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*?)\s*$")
        raw_lines = text.splitlines()
        analysis_start: int | None = None
        analysis_indent = 0
        for index, raw_line in enumerate(raw_lines):
            content = raw_line.split("#", 1)[0].rstrip()
            match = pattern.match(content)
            if match and match.group(1) == "analysis" and not match.group(2):
                analysis_start = index + 1
                analysis_indent = len(content) - len(content.lstrip())
                break
        scoped_lines = raw_lines
        if analysis_start is not None:
            scoped_lines = []
            for raw_line in raw_lines[analysis_start:]:
                content = raw_line.split("#", 1)[0].rstrip()
                if not content.strip():
                    continue
                indent = len(content) - len(content.lstrip())
                if indent <= analysis_indent:
                    break
                scoped_lines.append(content)
        for raw_line in scoped_lines:
            line = raw_line.split("#", 1)[0].rstrip()
            match = pattern.match(line)
            if match and match.group(1) in wanted:
                values[match.group(1)] = _scalar_yaml_value(match.group(2))
        comparisons = _yaml_confirmatory_comparisons(raw_lines)
        if comparisons is not None:
            values["confirmatory_comparisons"] = comparisons
    return values, {
        "path": str(source.resolve()),
        "found": True,
        "used": bool(values),
        "values": values,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_analysis_output_paths(args: argparse.Namespace) -> dict[str, Path]:
    """Reject output/input aliases and implicit replacement before analysis."""

    out_dir = Path(args.out_dir).expanduser()
    outputs = {name: out_dir / name for name in ANALYSIS_OUTPUT_FILENAMES}
    explicit_metadata = list(getattr(args, "score_metadata", None) or [])
    metadata_inputs = explicit_metadata or [
        _default_score_metadata_path(args.predictions)
    ]
    config_path = args.config if args.config is not None else DEFAULT_PROTOCOL_PATH
    resolved_inputs = {
        Path(value).expanduser().resolve()
        for value in (
            args.predictions,
            args.expected_trials,
            config_path,
            *metadata_inputs,
        )
        if value not in (None, "")
    }
    resolved_out_dir = out_dir.resolve()
    if resolved_out_dir in resolved_inputs:
        raise ValueError(
            f"analysis output directory aliases an authenticated input: {out_dir}"
        )

    resolved_outputs: dict[Path, str] = {}
    for name, path in outputs.items():
        resolved = path.resolve()
        if resolved in resolved_inputs:
            raise ValueError(f"analysis output aliases an input: {path}")
        previous = resolved_outputs.setdefault(resolved, name)
        if previous != name:
            raise ValueError(
                "analysis output paths collide after resolution: "
                f"{previous!r} and {name!r}"
            )
        if (path.exists() or path.is_symlink()) and not bool(
            getattr(args, "overwrite", False)
        ):
            raise FileExistsError(f"analysis output exists; pass --overwrite: {path}")
    return outputs


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]
) -> None:
    fields = list(preferred)
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(dict(row) for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_protocol(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    *,
    locked: bool = False,
) -> dict[str, Any]:
    defaults = {
        "bootstrap_replicates": DEFAULT_BOOTSTRAP_REPLICATES,
        "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
        "seed": DEFAULT_SEED,
        "reference_condition": DEFAULT_REFERENCE_CONDITION,
        "ece_bins": DEFAULT_ECE_BINS,
        "minimum_confirmatory_resampling_units": (
            DEFAULT_MINIMUM_CONFIRMATORY_RESAMPLING_UNITS
        ),
    }
    result: dict[str, Any] = {}
    for key, default in defaults.items():
        cli_value = getattr(args, key, None)
        if (
            locked
            and cli_value is not None
            and key in config
            and _group_value(cli_value) != _group_value(config[key])
        ):
            option = "--" + key.replace("_", "-")
            raise ValueError(
                f"{option} conflicts with the locked protocol; edit and re-freeze the "
                "protocol before confirmatory analysis"
            )
        result[key] = cli_value if cli_value is not None else config.get(key, default)
    raw_bootstrap_replicates = result["bootstrap_replicates"]
    if locked and _positive_int(raw_bootstrap_replicates) is None:
        raise ValueError(
            "locked analysis.bootstrap_replicates must be a positive integer "
            "for confirmatory confidence intervals"
        )
    result["bootstrap_replicates"] = int(raw_bootstrap_replicates)
    result["confidence_level"] = float(result["confidence_level"])
    result["seed"] = int(result["seed"])
    result["reference_condition"] = str(result["reference_condition"])
    result["ece_bins"] = int(result["ece_bins"])
    if _positive_int(result["minimum_confirmatory_resampling_units"]) is None:
        raise ValueError(
            "analysis.minimum_confirmatory_resampling_units must be a positive integer"
        )
    result["confirmatory_comparisons"] = _normalize_confirmatory_comparisons(
        config.get("confirmatory_comparisons", [])
    )
    return result


def _default_score_metadata_path(predictions_path: str | Path) -> Path:
    source = Path(predictions_path)
    return source.with_suffix(source.suffix + ".metadata.json")


def _authenticate_score_metadata(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    expected_rows: Sequence[Mapping[str, Any]] | None,
    expected_path: str | Path | None,
    metadata_paths: Sequence[str | Path],
    protocol_sha256: str,
    data_release_sha256: str,
    trial_build_attestation_sha256: str,
    locked_protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Authenticate score sidecars, rows, and exact manifest-shard coverage."""

    issues: list[dict[str, Any]] = []
    locked_model = validate_frozen_model_protocol(locked_protocol)
    locked_projector = protocol_section(locked_protocol, "projector")
    locked_sampling = protocol_section(locked_protocol, "sampling")
    for field in (
        "checkpoint_sha256",
        "metadata_sha256",
        "encoder_extraction_pipeline_identity_sha256",
        "llm_pretrained_identity_sha256",
        "evaluation_trial_matrix_closure_sha256",
        "evaluation_trial_set_root_sha256",
    ):
        value = str(locked_projector.get(field, "")).lower()
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(
                f"locked protocol projector.{field} must be a 64-character SHA256 "
                "before confirmatory analysis"
            )
    locked_trial_count = locked_projector.get("evaluation_trial_count")
    if (
        isinstance(locked_trial_count, bool)
        or not isinstance(locked_trial_count, int)
        or locked_trial_count < 1
    ):
        raise ValueError(
            "locked protocol projector.evaluation_trial_count must be a positive integer "
            "before confirmatory analysis"
        )
    sidecars_by_run: dict[str, dict[str, Any]] = {}
    score_partitions_by_run: dict[str, dict[str, int | str]] = {}
    score_partition_presence_by_run: dict[str, bool] = {}
    loaded_paths: list[str] = []
    global_signatures: set[str] = set()
    for raw_path in metadata_paths:
        path = Path(raw_path).resolve()
        loaded_paths.append(str(path))
        if not path.is_file():
            issues.append({"kind": "score_metadata_sidecar_missing", "path": str(path)})
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                sidecar = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            issues.append(
                {
                    "kind": "score_metadata_sidecar_unreadable",
                    "path": str(path),
                    "message": str(exc),
                }
            )
            continue
        if not isinstance(sidecar, Mapping):
            issues.append(
                {"kind": "score_metadata_sidecar_not_object", "path": str(path)}
            )
            continue
        sidecar = dict(sidecar)
        run_signature = sidecar.get("run_signature")
        global_signature = sidecar.get("global_signature")
        declared_run = str(sidecar.get("run_signature_sha256", ""))
        declared_global = str(sidecar.get("global_signature_sha256", ""))
        try:
            actual_run_digest = (
                canonical_sha256(run_signature)
                if isinstance(run_signature, Mapping)
                else None
            )
            actual_global_digest = (
                canonical_sha256(global_signature)
                if isinstance(global_signature, Mapping)
                else None
            )
        except (TypeError, ValueError) as exc:
            issues.append(
                {
                    "kind": "score_metadata_noncanonical_signature_payload",
                    "path": str(path),
                    "message": str(exc),
                }
            )
            continue
        if not isinstance(run_signature, Mapping) or actual_run_digest != declared_run:
            issues.append(
                {"kind": "score_metadata_invalid_run_signature", "path": str(path)}
            )
            continue
        if (
            not isinstance(global_signature, Mapping)
            or actual_global_digest != declared_global
        ):
            issues.append(
                {"kind": "score_metadata_invalid_global_signature", "path": str(path)}
            )
            continue
        if declared_run in sidecars_by_run:
            issues.append(
                {
                    "kind": "duplicate_score_metadata_run_signature",
                    "run_signature_sha256": declared_run,
                }
            )
            continue
        sidecars_by_run[declared_run] = {**sidecar, "_path": str(path)}
        global_signatures.add(declared_global)
        top_level_has_partition = "score_partition" in sidecar
        signed_has_partition = "score_partition" in run_signature
        score_partition_presence_by_run[declared_run] = (
            top_level_has_partition or signed_has_partition
        )
        if top_level_has_partition != signed_has_partition or (
            top_level_has_partition
            and sidecar.get("score_partition") != run_signature.get("score_partition")
        ):
            issues.append(
                {
                    "kind": "score_metadata_partition_top_level_mismatch",
                    "path": str(path),
                }
            )
        if signed_has_partition:
            try:
                score_partitions_by_run[declared_run] = validate_score_partition(
                    run_signature.get("score_partition")
                )
            except ValueError as exc:
                issues.append(
                    {
                        "kind": "score_metadata_invalid_partition",
                        "path": str(path),
                        "message": str(exc),
                    }
                )
        if (
            sidecar.get("result_integrity_schema_version")
            != RESULT_INTEGRITY_SCHEMA_VERSION
        ):
            issues.append(
                {"kind": "score_metadata_result_schema_mismatch", "path": str(path)}
            )
        if (
            run_signature.get("schema_version")
            != "information_upper_bound.scoring_run_signature.v2"
        ):
            issues.append(
                {"kind": "score_metadata_run_schema_mismatch", "path": str(path)}
            )
        if (
            global_signature.get("schema_version")
            != "information_upper_bound.scoring_global_signature.v2"
        ):
            issues.append(
                {"kind": "score_metadata_global_schema_mismatch", "path": str(path)}
            )
        expected_global_fields = {
            "scoring_protocol_version": SCORING_PROTOCOL_VERSION,
            "projector_checkpoint_sha256": str(
                locked_projector.get("checkpoint_sha256", "")
            ).lower(),
            "projector_metadata_sha256": str(
                locked_projector.get("metadata_sha256", "")
            ).lower(),
            "encoder_extraction_pipeline_identity_sha256": str(
                locked_projector.get("encoder_extraction_pipeline_identity_sha256", "")
            ).lower(),
            "llm_id": locked_model.get("llm_id"),
            "llm_revision_requested": locked_model.get("llm_revision"),
            "dtype": locked_model.get("dtype"),
            "max_length": locked_model.get("max_length"),
            "overflow_policy": locked_model.get("overflow_policy"),
            "media_sha256_required": bool(
                locked_sampling.get("require_media_sha256", False)
            ),
            "trial_matrix_closure_sha256": str(
                locked_projector.get("evaluation_trial_matrix_closure_sha256", "")
            ).lower(),
            "full_trial_set_root_sha256": str(
                locked_projector.get("evaluation_trial_set_root_sha256", "")
            ).lower(),
            "full_trial_count": locked_projector.get("evaluation_trial_count"),
        }
        global_field_mismatches = [
            name
            for name, expected in expected_global_fields.items()
            if global_signature.get(name) != expected
        ]
        pretrained_identity = global_signature.get("llm_pretrained_identity")
        if (
            not isinstance(pretrained_identity, Mapping)
            or str(pretrained_identity.get("identity_sha256", ""))
            != str(locked_projector.get("llm_pretrained_identity_sha256", "")).lower()
        ):
            global_field_mismatches.append("llm_pretrained_identity.identity_sha256")
        if global_field_mismatches:
            issues.append(
                {
                    "kind": "score_metadata_global_protocol_mismatch",
                    "path": str(path),
                    "fields": global_field_mismatches,
                }
            )
        expected_evaluation_features = {
            "feature_index_sha256": str(
                locked_projector.get("evaluation_feature_index_sha256", "")
            ).lower(),
            "feature_metadata_sha256": str(
                locked_projector.get("evaluation_feature_metadata_sha256", "")
            ).lower(),
            "feature_artifact_root_sha256": str(
                locked_projector.get("evaluation_feature_artifact_root_sha256", "")
            ).lower(),
        }
        evaluation_feature_mismatches = [
            name
            for name, expected in expected_evaluation_features.items()
            if run_signature.get(name) != expected
        ]
        if evaluation_feature_mismatches:
            issues.append(
                {
                    "kind": "score_metadata_evaluation_feature_lock_mismatch",
                    "path": str(path),
                    "fields": evaluation_feature_mismatches,
                }
            )
        if (
            str(run_signature.get("scoring_global_signature_sha256", ""))
            != declared_global
        ):
            issues.append(
                {
                    "kind": "score_metadata_run_global_signature_mismatch",
                    "path": str(path),
                }
            )
        shared_signature_fields = (
            "scoring_protocol_version",
            "protocol_config_sha256",
            "projector_checkpoint_sha256",
            "projector_metadata_sha256",
            "llm_id",
            "llm_revision_requested",
            "llm_pretrained_identity",
            "dtype",
            "max_length",
            "overflow_policy",
            "trial_matrix_closure_sha256",
            "full_trial_set_root_sha256",
            "full_trial_count",
        )
        inconsistent_shared_fields = [
            field
            for field in shared_signature_fields
            if run_signature.get(field) != global_signature.get(field)
        ]
        if inconsistent_shared_fields:
            issues.append(
                {
                    "kind": "score_metadata_run_global_payload_mismatch",
                    "path": str(path),
                    "fields": inconsistent_shared_fields,
                }
            )
        if (
            str(run_signature.get("protocol_config_sha256", "")) != protocol_sha256
            or str(global_signature.get("protocol_config_sha256", ""))
            != protocol_sha256
        ):
            issues.append(
                {
                    "kind": "score_metadata_protocol_sha256_mismatch",
                    "path": str(path),
                    "expected": protocol_sha256,
                }
            )
        if (
            str(global_signature.get("data_release_sha256", "")) != data_release_sha256
            or str(run_signature.get("data_release_sha256", "")) != data_release_sha256
        ):
            issues.append(
                {
                    "kind": "score_metadata_data_release_sha256_mismatch",
                    "path": str(path),
                    "expected": data_release_sha256,
                }
            )
        if (
            str(global_signature.get("trial_build_attestation_sha256", ""))
            != trial_build_attestation_sha256
            or str(run_signature.get("trial_build_attestation_sha256", ""))
            != trial_build_attestation_sha256
        ):
            issues.append(
                {
                    "kind": "score_metadata_trial_build_attestation_mismatch",
                    "path": str(path),
                    "expected": trial_build_attestation_sha256,
                }
            )
        raw_failures = sidecar.get("num_failures")
        failures_valid = (
            not isinstance(raw_failures, bool)
            and isinstance(raw_failures, int)
            and raw_failures == 0
        )
        if sidecar.get("status") != "complete" or not failures_valid:
            issues.append(
                {
                    "kind": "score_metadata_run_incomplete",
                    "path": str(path),
                    "status": sidecar.get("status"),
                    "num_failures": sidecar.get("num_failures"),
                }
            )
        sidecar_trial_set = sidecar.get("trial_set_identity")
        if not isinstance(
            sidecar_trial_set, Mapping
        ) or sidecar_trial_set != run_signature.get("trial_set_identity"):
            issues.append(
                {
                    "kind": "score_metadata_trial_set_identity_mismatch",
                    "path": str(path),
                }
            )
        if str(sidecar.get("trials_manifest_sha256", "")) != str(
            run_signature.get("trials_manifest_sha256", "")
        ):
            issues.append(
                {
                    "kind": "score_metadata_manifest_signature_mismatch",
                    "path": str(path),
                }
            )

    if len(global_signatures) > 1:
        issues.append(
            {
                "kind": "score_metadata_mixed_global_signatures",
                "signatures": sorted(global_signatures),
            }
        )

    partitioned_runs = {
        run_sha
        for run_sha, present in score_partition_presence_by_run.items()
        if present
    }
    if partitioned_runs and partitioned_runs != set(sidecars_by_run):
        issues.append(
            {
                "kind": "score_metadata_mixed_partition_presence",
                "partitioned_run_signatures": sorted(partitioned_runs),
                "unpartitioned_run_signatures": sorted(
                    set(sidecars_by_run) - partitioned_runs
                ),
            }
        )
    partition_worker_counts = {
        int(partition["worker_count"]) for partition in score_partitions_by_run.values()
    }
    partition_worker_indices = [
        int(partition["worker_index"]) for partition in score_partitions_by_run.values()
    ]
    if len(partition_worker_counts) > 1:
        issues.append(
            {
                "kind": "score_metadata_partition_worker_count_mismatch",
                "worker_counts": sorted(partition_worker_counts),
            }
        )
    if len(partition_worker_indices) != len(set(partition_worker_indices)):
        duplicate_indices = sorted(
            index
            for index, count in Counter(partition_worker_indices).items()
            if count > 1
        )
        issues.append(
            {
                "kind": "score_metadata_duplicate_partition_worker_index",
                "worker_indices": duplicate_indices,
            }
        )
    partition_worker_count: int | None = None
    if len(partition_worker_counts) == 1:
        partition_worker_count = next(iter(partition_worker_counts))
        expected_worker_indices = set(range(partition_worker_count))
        actual_worker_indices = set(partition_worker_indices)
        if actual_worker_indices != expected_worker_indices:
            issues.append(
                {
                    "kind": "score_metadata_incomplete_partition_worker_indices",
                    "worker_count": partition_worker_count,
                    "expected_worker_indices": sorted(expected_worker_indices),
                    "actual_worker_indices": sorted(actual_worker_indices),
                }
            )

    rows_by_run: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        run_sha = str(row.get("scoring_run_signature_sha256", ""))
        global_sha = str(row.get("scoring_global_signature_sha256", ""))
        sidecar = sidecars_by_run.get(run_sha)
        if sidecar is None:
            issues.append(
                {
                    "kind": "prediction_run_signature_has_no_sidecar",
                    "trial_id": row.get("trial_id", row.get("id")),
                    "run_signature_sha256": run_sha,
                }
            )
            continue
        if global_sha != str(sidecar.get("global_signature_sha256", "")):
            issues.append(
                {
                    "kind": "prediction_global_signature_sidecar_mismatch",
                    "trial_id": row.get("trial_id", row.get("id")),
                }
            )
        partition = score_partitions_by_run.get(run_sha)
        if partition is not None:
            trial_content_digest = str(row.get("trial_content_sha256", ""))
            if re.fullmatch(r"[0-9a-f]{64}", trial_content_digest) is None:
                issues.append(
                    {
                        "kind": "prediction_partition_content_sha256_invalid",
                        "trial_id": row.get("trial_id", row.get("id")),
                        "value": trial_content_digest,
                    }
                )
            else:
                actual_worker_index = score_worker_index(
                    trial_content_digest,
                    worker_count=int(partition["worker_count"]),
                )
                expected_worker_index = int(partition["worker_index"])
                if actual_worker_index != expected_worker_index:
                    issues.append(
                        {
                            "kind": "prediction_partition_owner_mismatch",
                            "trial_id": row.get("trial_id", row.get("id")),
                            "worker_count": int(partition["worker_count"]),
                            "expected_worker_index": expected_worker_index,
                            "actual_worker_index": actual_worker_index,
                        }
                    )
        rows_by_run[run_sha].append(row)
    for run_sha in sorted(set(sidecars_by_run) - set(rows_by_run)):
        issues.append(
            {
                "kind": "score_metadata_sidecar_has_no_prediction_rows",
                "run_signature_sha256": run_sha,
            }
        )

    expected_index: dict[str, Mapping[str, Any]] = {}
    if expected_rows is not None:
        for row in expected_rows:
            trial_id = str(row.get("trial_id", row.get("id", "")))
            if trial_id and trial_id not in expected_index:
                expected_index[trial_id] = row
    for run_sha, rows in rows_by_run.items():
        sidecar = sidecars_by_run[run_sha]
        bound_expected: list[Mapping[str, Any]] = []
        missing_expected = False
        for row in rows:
            trial_id = str(row.get("trial_id", row.get("id", "")))
            expected = expected_index.get(trial_id)
            if expected is None:
                missing_expected = True
            else:
                bound_expected.append(expected)
        if missing_expected or not bound_expected:
            issues.append(
                {
                    "kind": "score_metadata_trial_set_cannot_bind_expected_manifest",
                    "run_signature_sha256": run_sha,
                }
            )
            continue
        try:
            recomputed_trial_set = trial_set_identity(bound_expected)
        except (TypeError, ValueError) as exc:
            issues.append(
                {
                    "kind": "score_metadata_trial_set_recompute_error",
                    "run_signature_sha256": run_sha,
                    "message": str(exc),
                }
            )
            continue
        if recomputed_trial_set != sidecar.get("trial_set_identity"):
            issues.append(
                {
                    "kind": "score_metadata_trial_set_coverage_mismatch",
                    "run_signature_sha256": run_sha,
                    "sidecar": sidecar.get("trial_set_identity"),
                    "recomputed_from_rows": recomputed_trial_set,
                }
            )
        requested = sidecar.get("num_trials_requested")
        if (
            isinstance(requested, bool)
            or not isinstance(requested, int)
            or requested != len(rows)
        ):
            issues.append(
                {
                    "kind": "score_metadata_prediction_count_mismatch",
                    "run_signature_sha256": run_sha,
                    "requested": requested,
                    "prediction_rows": len(rows),
                }
            )

    # JSONL byte identity is intentionally not a scientific completeness gate.
    # The authenticated trial-set root remains stable under row reordering and
    # remounted media paths; every score shard is bound to that locked closure.

    authenticated = not issues and bool(sidecars_by_run) and len(global_signatures) == 1
    return {
        "authenticated": authenticated,
        "authenticated_sharded_run": authenticated and len(sidecars_by_run) > 1,
        "metadata_paths": loaded_paths,
        "sidecar_count": len(sidecars_by_run),
        "run_signatures": sorted(sidecars_by_run),
        "global_signatures": sorted(global_signatures),
        "partitioned_run": bool(partitioned_runs),
        "score_partition_worker_count": partition_worker_count,
        "score_partition_worker_indices": sorted(set(partition_worker_indices)),
        "result_integrity_schema_version": RESULT_INTEGRITY_SCHEMA_VERSION,
        "issues": issues,
    }, issues


def _require_complete_failures(
    report: Mapping[str, Any], *, expected_requested: bool
) -> list[str]:
    coverage = report.get("coverage") or {}
    failures: list[str] = []
    if not expected_requested:
        failures.append("expected_trials_manifest_not_provided")
    else:
        joined_coverage = _finite_float(coverage.get("joined_coverage"))
        expected_trials = coverage.get("expected_trials")
        if expected_trials == 0:
            failures.append("expected_trials_manifest_empty")
        elif joined_coverage is None or joined_coverage < 1.0:
            failures.append("expected_trial_join_coverage_below_one")

    prediction_input = coverage.get("prediction_input") or {}
    manifest_input = coverage.get("manifest_input") or {}
    if prediction_input.get("duplicate_trial_ids"):
        failures.append("duplicate_prediction_trial_ids")
    if manifest_input.get("duplicate_trial_ids"):
        failures.append("duplicate_manifest_trial_ids")
    if coverage.get("unexpected_prediction_count", 0):
        failures.append("unexpected_prediction_trials")
    if coverage.get("metadata_mismatch_count", 0):
        failures.append("prediction_manifest_metadata_mismatch")

    authentication = report.get("score_metadata_authentication") or {}
    if not authentication.get("authenticated"):
        failures.append("score_metadata_authentication_failed")

    issue_counts = (coverage.get("issues") or {}).get("counts") or {}
    failure_issue_kinds = {
        "malformed_prediction_jsonl",
        "non_object_prediction_jsonl",
        "prediction_non_object_row",
        "prediction_missing_trial_id",
        "malformed_manifest_jsonl",
        "non_object_manifest_jsonl",
        "manifest_non_object_row",
        "manifest_missing_trial_id",
        "manifest_missing_trial_content_sha256",
        "manifest_trial_content_hash_recompute_error",
        "stale_manifest_trial_content_sha256",
        "manifest_data_release_sha256_mismatch",
        "manifest_trial_build_attestation_invalid",
        "manifest_mixed_or_missing_trial_build_attestation",
        "manifest_trial_matrix_closure_recompute_error",
        "manifest_trial_matrix_closure_mismatch",
        "manifest_trial_matrix_closure_missing_expected_trials",
        "missing_group_field",
        "missing_input_channel",
        "missing_scoring_run_signature_sha256",
        "invalid_scoring_run_signature_sha256",
        "missing_scoring_global_signature_sha256",
        "invalid_scoring_global_signature_sha256",
        "mixed_scoring_global_signatures",
        "missing_or_invalid_result_content_sha256",
        "result_content_sha256_mismatch",
        "prediction_manifest_missing_binding_field",
        "invalid_choices",
        "invalid_prediction_label",
        "invalid_choice_probability_key_set",
        "invalid_choice_probability_value",
        "invalid_choice_probability_total",
        "invalid_choice_nll_key_set",
        "invalid_choice_nll_value",
        "choice_probability_nll_inconsistency",
        "prediction_nll_inconsistency",
        "gold_nll_inconsistency",
        "best_distractor_nll_inconsistency",
        "gold_margin_nll_inconsistency",
        "correct_nll_inconsistency",
        "correct_prediction_inconsistency",
        "prediction_text_choice_mismatch",
        "invalid_correct",
        "invalid_choice_probability",
        "invalid_gold_margin",
        "invalid_projected_original_visual_tokens",
        "invalid_projected_effective_visual_tokens",
        "projected_visual_token_truncation",
        "projected_visual_tokens_missing_visual_id",
        "inconsistent_projected_visual_token_budget",
        "clevrer_missing_independent_unit_id",
        "clevrer_missing_resampling_unit_id",
        "clevrer_invalid_official_candidate_count",
        "clevrer_missing_official_candidate_id",
        "clevrer_duplicate_official_candidate_id",
        "clevrer_incomplete_official_candidate_set",
    }
    for kind in sorted(failure_issue_kinds):
        if issue_counts.get(kind, 0):
            failures.append(kind)
    if issue_counts.get("mixed_scoring_run_signatures", 0) and not authentication.get(
        "authenticated_sharded_run"
    ):
        failures.append("mixed_scoring_run_signatures")

    analysis_protocol = report.get("protocol") or {}
    minimum_resampling_units = _positive_int(
        analysis_protocol.get(
            "minimum_confirmatory_resampling_units",
            DEFAULT_MINIMUM_CONFIRMATORY_RESAMPLING_UNITS,
        )
    )
    if minimum_resampling_units is None:
        failures.append("invalid_minimum_confirmatory_resampling_units")
        minimum_resampling_units = DEFAULT_MINIMUM_CONFIRMATORY_RESAMPLING_UNITS

    def count(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    comparison_coverage = (report.get("analysis_coverage") or {}).get(
        "comparisons"
    ) or {}
    clevrer_primary_coverage = (report.get("analysis_coverage") or {}).get(
        "clevrer_primary"
    ) or {}
    clevrer_summary_groups = count(clevrer_primary_coverage.get("summary_groups"))
    clevrer_authenticated_summary_groups = count(
        clevrer_primary_coverage.get("authenticated_summary_groups")
    )
    if clevrer_summary_groups and (
        clevrer_authenticated_summary_groups != clevrer_summary_groups
    ):
        failures.append("clevrer_primary_summary_not_authenticated")
    clevrer_comparison_rows = count(
        clevrer_primary_coverage.get("confirmatory_comparison_rows")
    )
    clevrer_authenticated_comparison_rows = count(
        clevrer_primary_coverage.get("authenticated_confirmatory_comparison_rows")
    )
    if clevrer_comparison_rows and (
        clevrer_authenticated_comparison_rows != clevrer_comparison_rows
    ):
        failures.append("clevrer_primary_comparison_not_authenticated")
    comparisons = analysis_protocol.get("confirmatory_comparisons") or []

    def add_comparison_failure(kind: str, name: str) -> None:
        value = f"{kind}::{name}"
        if value not in failures:
            failures.append(value)

    for raw_pair in comparisons:
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            failures.append("invalid_confirmatory_comparison_definition")
            continue
        left_condition, right_condition = (str(value) for value in raw_pair)
        name = f"{left_condition}_minus_{right_condition}"
        details = comparison_coverage.get(f"confirmatory::{name}")
        if not isinstance(details, Mapping) or details.get("not_computed"):
            add_comparison_failure("confirmatory_comparison_not_computed", name)
            continue

        estimands = details.get("estimands")
        if not isinstance(estimands, list) or not estimands:
            add_comparison_failure(
                "confirmatory_comparison_zero_expected_alignment_units", name
            )
            add_comparison_failure("confirmatory_comparison_zero_aligned_units", name)
        else:
            zero_expected = False
            zero_aligned = False
            below_minimum = False
            incomplete_expected = False
            incomplete_observed = False
            for estimand in estimands:
                if not isinstance(estimand, Mapping):
                    incomplete_expected = True
                    incomplete_observed = True
                    continue
                expected_left = count(estimand.get("expected_left_rows"))
                expected_aligned = count(estimand.get("expected_aligned_rows"))
                aligned = count(estimand.get("aligned_rows"))
                expected_units = count(
                    estimand.get("expected_unique_aligned_resampling_units")
                )
                aligned_units = count(estimand.get("unique_aligned_resampling_units"))
                if (
                    expected_left is None
                    or expected_aligned is None
                    or expected_aligned != expected_left
                ):
                    incomplete_expected = True
                if (
                    expected_aligned is None
                    or aligned is None
                    or aligned != expected_aligned
                ):
                    incomplete_observed = True
                if expected_units in (None, 0):
                    zero_expected = True
                if aligned_units in (None, 0):
                    zero_aligned = True
                if (
                    expected_units is None
                    or aligned_units is None
                    or expected_units < minimum_resampling_units
                    or aligned_units < minimum_resampling_units
                ):
                    below_minimum = True
            if zero_expected:
                add_comparison_failure(
                    "confirmatory_comparison_zero_expected_alignment_units", name
                )
            if zero_aligned:
                add_comparison_failure(
                    "confirmatory_comparison_zero_aligned_units", name
                )
            if below_minimum:
                add_comparison_failure(
                    "confirmatory_comparison_below_minimum_resampling_units", name
                )
            if incomplete_expected:
                add_comparison_failure(
                    "confirmatory_comparison_incomplete_expected_alignment", name
                )
            if incomplete_observed:
                add_comparison_failure(
                    "confirmatory_comparison_incomplete_observed_alignment", name
                )

        if any(
            count(details.get(field)) not in (None, 0)
            for field in (
                "ambiguous_expected_left_alignment_keys",
                "ambiguous_expected_right_alignment_keys",
            )
        ):
            add_comparison_failure(
                "confirmatory_comparison_ambiguous_expected_alignment", name
            )
        if any(
            count(details.get(field)) not in (None, 0)
            for field in (
                "ambiguous_left_alignment_keys",
                "ambiguous_right_alignment_keys",
            )
        ):
            add_comparison_failure(
                "confirmatory_comparison_ambiguous_observed_alignment", name
            )
        if count(details.get("expected_unmatched_left_rows")) not in (None, 0):
            add_comparison_failure(
                "confirmatory_comparison_unmatched_expected_alignment", name
            )
        if count(details.get("unmatched_left_rows")) not in (None, 0):
            add_comparison_failure(
                "confirmatory_comparison_unmatched_observed_alignment", name
            )
    return failures


def run_analysis_cli(args: argparse.Namespace) -> dict[str, Any]:
    output_paths = _validate_analysis_output_paths(args)
    predictions, prediction_issues = _read_jsonl_tolerant(
        args.predictions, source="prediction"
    )
    if args.expected_trials:
        expected, expected_issues = _read_jsonl_tolerant(
            args.expected_trials, source="manifest"
        )
    else:
        expected, expected_issues = None, []

    development = bool(
        getattr(args, "development", False) or getattr(args, "allow_incomplete", False)
    )
    if development and bool(getattr(args, "require_complete", False)):
        raise ValueError(
            "--require-complete cannot be combined with a development escape hatch"
        )
    strict = not development
    config_path = args.config if args.config is not None else str(DEFAULT_PROTOCOL_PATH)
    config, config_report = load_protocol_config(config_path)
    protocol_sha256: str | None = None
    data_release_sha256: str | None = None
    trial_build_attestation_sha256: str | None = None
    data_release_issues: list[dict[str, Any]] = []
    if strict:
        _locked_protocol, locked_metadata = load_protocol(config_path)
        protocol_sha256 = str(locked_metadata["sha256"])
        data_release_sha256 = str(
            validate_data_protocol(_locked_protocol)["data_release_sha256"]
        )
        attestation_values: set[str] = set()
        for row in expected or []:
            if str(row.get("data_release_sha256", "")) != data_release_sha256:
                data_release_issues.append(
                    {
                        "kind": "manifest_data_release_sha256_mismatch",
                        "trial_id": row.get("trial_id", row.get("id")),
                        "expected": data_release_sha256,
                        "actual": row.get("data_release_sha256"),
                    }
                )
            try:
                attestation = validate_trial_build_attestation(
                    row,
                    protocol=_locked_protocol,
                    require_confirmatory=True,
                )
            except (TypeError, ValueError) as exc:
                data_release_issues.append(
                    {
                        "kind": "manifest_trial_build_attestation_invalid",
                        "trial_id": row.get("trial_id", row.get("id")),
                        "message": str(exc),
                    }
                )
            else:
                attestation_values.add(str(attestation["attestation_sha256"]))
        if len(attestation_values) != 1:
            data_release_issues.append(
                {
                    "kind": "manifest_mixed_or_missing_trial_build_attestation",
                    "count": len(attestation_values),
                    "values": sorted(attestation_values),
                }
            )
        else:
            trial_build_attestation_sha256 = next(iter(attestation_values))
        locked_projector_section = protocol_section(_locked_protocol, "projector")
        if expected:
            try:
                expected_trial_set_identity = trial_set_identity(expected)
            except (TypeError, ValueError) as exc:
                data_release_issues.append(
                    {
                        "kind": "manifest_trial_matrix_closure_recompute_error",
                        "message": str(exc),
                    }
                )
            else:
                locked_root = str(
                    locked_projector_section.get("evaluation_trial_set_root_sha256", "")
                ).lower()
                locked_count = locked_projector_section.get("evaluation_trial_count")
                if (
                    expected_trial_set_identity["root_sha256"] != locked_root
                    or expected_trial_set_identity["trial_count"] != locked_count
                ):
                    data_release_issues.append(
                        {
                            "kind": "manifest_trial_matrix_closure_mismatch",
                            "expected_root_sha256": locked_root,
                            "actual_root_sha256": expected_trial_set_identity[
                                "root_sha256"
                            ],
                            "expected_trial_count": locked_count,
                            "actual_trial_count": expected_trial_set_identity[
                                "trial_count"
                            ],
                        }
                    )
        else:
            data_release_issues.append(
                {"kind": "manifest_trial_matrix_closure_missing_expected_trials"}
            )
        required_analysis_keys = {
            "bootstrap_replicates",
            "confidence_level",
            "seed",
            "reference_condition",
            "ece_bins",
            "minimum_confirmatory_resampling_units",
        }
        missing_locked = sorted(required_analysis_keys - set(config))
        if missing_locked:
            raise ValueError(
                f"locked protocol analysis section is missing required fields: {missing_locked}"
            )
    elif config_path is not None and Path(config_path).is_file():
        protocol_sha256 = sha256_file(config_path)
    protocol = _resolve_protocol(args, config, locked=strict)

    explicit_metadata = list(getattr(args, "score_metadata", None) or [])
    metadata_paths: list[str | Path] = explicit_metadata
    if strict and not metadata_paths:
        metadata_paths = [_default_score_metadata_path(args.predictions)]
    if strict:
        assert protocol_sha256 is not None
        assert data_release_sha256 is not None
        # Use an impossible sentinel when the manifest attestation audit failed;
        # sidecar authentication will then fail closed without hiding the root
        # manifest issue from the report.
        resolved_attestation_sha256 = trial_build_attestation_sha256 or "<invalid>"
        metadata_authentication, metadata_issues = _authenticate_score_metadata(
            prediction_rows=predictions,
            expected_rows=expected,
            expected_path=args.expected_trials,
            metadata_paths=metadata_paths,
            protocol_sha256=protocol_sha256,
            data_release_sha256=data_release_sha256,
            trial_build_attestation_sha256=resolved_attestation_sha256,
            locked_protocol=_locked_protocol,
        )
    else:
        metadata_authentication = {
            "authenticated": False,
            "authenticated_sharded_run": False,
            "metadata_paths": [str(Path(value).resolve()) for value in metadata_paths],
            "sidecar_count": 0,
            "run_signatures": [],
            "global_signatures": [],
            "result_integrity_schema_version": RESULT_INTEGRITY_SCHEMA_VERSION,
            "issues": [],
            "skipped_in_development": True,
        }
        metadata_issues = []
    result = analyze_predictions(
        predictions,
        expected,
        input_issues=[
            *prediction_issues,
            *expected_issues,
            *data_release_issues,
            *metadata_issues,
        ],
        **protocol,
    )
    result["report"]["analysis_mode"] = {
        "mode": "confirmatory_strict" if strict else "development",
        "confirmatory": strict,
        "escape_hatch": (
            "--development"
            if getattr(args, "development", False)
            else (
                "--allow-incomplete"
                if getattr(args, "allow_incomplete", False)
                else None
            )
        ),
        "warning": (
            None
            if strict
            else "DEVELOPMENT RESULT: incomplete or unauthenticated inputs may be present; "
            "do not report as confirmatory."
        ),
    }
    result["report"]["score_metadata_authentication"] = metadata_authentication
    result["report"]["inputs"] = {
        "predictions": str(Path(args.predictions).resolve()),
        "expected_trials": (
            str(Path(args.expected_trials).resolve()) if args.expected_trials else None
        ),
        "config": config_report,
        "protocol_config_sha256": protocol_sha256,
        "data_release_sha256": data_release_sha256,
        "trial_build_attestation_sha256": trial_build_attestation_sha256,
        "trial_matrix_closure_sha256": (
            protocol_section(_locked_protocol, "projector").get(
                "evaluation_trial_matrix_closure_sha256"
            )
            if strict
            else None
        ),
        "score_metadata": [str(Path(value).resolve()) for value in metadata_paths],
    }
    result["report"]["protocol_resolution"] = {
        "precedence": (
            "locked protocol; conflicting CLI overrides forbidden"
            if strict
            else "CLI > config > defaults (development only)"
        ),
        "resolved": protocol,
    }
    require_complete = strict
    strict_failures = (
        _require_complete_failures(
            result["report"], expected_requested=bool(args.expected_trials)
        )
        if require_complete
        else []
    )
    result["report"]["require_complete"] = {
        "enabled": require_complete,
        "passed": not strict_failures,
        "failures": strict_failures,
    }
    _write_csv(
        output_paths["summary.csv"],
        result["summary"],
        (*GROUP_FIELDS, "input_channel"),
    )
    _write_csv(
        output_paths["comparisons.csv"],
        result["comparisons"],
        (
            *STRATUM_FIELDS,
            "comparison_type",
            "contrast",
            "condition",
            "input_channel",
            "reference_condition",
            "requested_dose",
            "effective_dose",
        ),
    )
    _write_csv(
        output_paths["pair_metrics.csv"],
        result["pair_metrics"],
        (*GROUP_FIELDS, "input_channel"),
    )
    _write_csv(
        output_paths["dose_curves.csv"],
        result["dose_curves"],
        (
            *STRATUM_FIELDS,
            "condition",
            "input_channel",
            "reference_condition",
            "requested_dose",
            "effective_dose_mean",
            "dose_index",
        ),
    )
    _atomic_write(
        output_paths["report.json"],
        json.dumps(
            result["report"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )
    return result["report"]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze information-upper-bound prediction rows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser(
        "analyze", help="Write summary, paired, dose, and coverage reports."
    )
    analyze.add_argument("--predictions", "--results", required=True)
    analyze.add_argument("--expected-trials", "--trials")
    analyze.add_argument(
        "--score-metadata",
        action="append",
        help=(
            "score metadata sidecar; repeat once per concatenated shard. Defaults to "
            "<predictions>.metadata.json for one run"
        ),
    )
    analyze.add_argument("--out-dir", required=True)
    analyze.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing analysis artifacts after input/output alias checks.",
    )
    analyze.add_argument("--config")
    analyze.add_argument(
        "--bootstrap-replicates", "--bootstrap-samples", type=int, default=None
    )
    analyze.add_argument("--confidence-level", type=float, default=None)
    analyze.add_argument("--seed", type=int, default=None)
    analyze.add_argument("--reference-condition", default=None)
    analyze.add_argument("--ece-bins", type=int, default=None)
    analyze.add_argument(
        "--require-complete",
        action="store_true",
        help="Deprecated compatibility flag; confirmatory analysis is strict by default.",
    )
    analyze.add_argument(
        "--development",
        action="store_true",
        help="Allow incomplete/unauthenticated exploratory analysis and watermark the report.",
    )
    analyze.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Alias for --development; output is explicitly non-confirmatory.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "analyze":
        report = run_analysis_cli(args)
        if (
            report["require_complete"]["enabled"]
            and not report["require_complete"]["passed"]
        ):
            return 2
        return 0
    raise ValueError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
