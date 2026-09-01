from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any


RESAMPLING_UNIT_SELECTION_SCHEMA_VERSION = (
    "information_upper_bound.resampling_unit_selection.v1"
)
RESAMPLING_UNIT_RANKING_SCHEMA_VERSION = (
    "information_upper_bound.resampling_unit_ranking.v1"
)
RESAMPLING_UNIT_SET_SCHEMA_VERSION = "information_upper_bound.resampling_unit_set.v1"
RESAMPLING_UNIT_COUNTS_SCHEMA_VERSION = (
    "information_upper_bound.resampling_unit_counts.v1"
)
RESAMPLING_UNIT_SELECTION_ALGORITHM = "sha256_canonical_json_rank_v1"
RESAMPLING_UNIT_FIELD = "diagnostic.resampling_unit_id"
MAX_SELECTION_SEED = 2**63 - 1


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _unit_id(row: Mapping[str, Any]) -> str:
    diagnostic = row.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        raise ValueError("record has no diagnostic mapping")
    value = diagnostic.get("resampling_unit_id")
    unit_id = str(value).strip() if value is not None else ""
    if not unit_id:
        raise ValueError(
            "every record must have a non-empty diagnostic.resampling_unit_id"
        )
    return unit_id


def resampling_unit_set_sha256(unit_ids: Sequence[str]) -> str:
    return _canonical_sha256(
        {
            "schema_version": RESAMPLING_UNIT_SET_SCHEMA_VERSION,
            "resampling_unit_ids": sorted(unit_ids),
        }
    )


def _unit_counts_sha256(counts: Mapping[str, int]) -> str:
    return _canonical_sha256(
        {
            "schema_version": RESAMPLING_UNIT_COUNTS_SCHEMA_VERSION,
            "units": [
                {"resampling_unit_id": unit_id, "record_count": int(counts[unit_id])}
                for unit_id in sorted(counts)
            ],
        }
    )


def _ranked_units(
    unit_ids: Sequence[str],
    *,
    dataset: str,
    canonical_split: str,
    seed: int,
) -> list[tuple[str, str]]:
    ranked: list[tuple[str, str]] = []
    for unit_id in unit_ids:
        digest = _canonical_sha256(
            {
                "schema_version": RESAMPLING_UNIT_SELECTION_SCHEMA_VERSION,
                "dataset": dataset,
                "canonical_split": canonical_split,
                "seed": seed,
                "resampling_unit_id": unit_id,
            }
        )
        ranked.append((digest, unit_id))
    return sorted(ranked)


def _ranking_sha256(ranked: Sequence[tuple[str, str]]) -> str:
    return _canonical_sha256(
        {
            "schema_version": RESAMPLING_UNIT_RANKING_SCHEMA_VERSION,
            "ranked_units": [
                {"rank_sha256": digest, "resampling_unit_id": unit_id}
                for digest, unit_id in ranked
            ],
        }
    )


def select_resampling_units(
    rows: Sequence[dict[str, Any]],
    *,
    dataset: str,
    canonical_split: str,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Select complete, answer-blind resampling units by a portable hash rank.

    Ranking reads only the dataset, canonical split, seed, and stable unit ID.
    It never reads questions, answers, choices, oracle annotations, or row order.
    """

    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size < 1
    ):
        raise ValueError("resampling-unit sample size must be a positive integer")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= MAX_SELECTION_SEED
    ):
        raise ValueError(
            f"resampling-unit sample seed must be in [0, {MAX_SELECTION_SEED}]"
        )
    if not dataset.strip() or not canonical_split.strip():
        raise ValueError("dataset and canonical split must be non-empty")
    if not rows:
        raise ValueError("cannot sample resampling units from an empty adapter output")

    counts = Counter(_unit_id(row) for row in rows)
    unit_ids = sorted(counts)
    if sample_size > len(unit_ids):
        raise ValueError(
            "resampling-unit sample size exceeds the post-exclusion population: "
            f"requested={sample_size}, available={len(unit_ids)}"
        )
    ranked = _ranked_units(
        unit_ids,
        dataset=dataset,
        canonical_split=canonical_split,
        seed=seed,
    )
    selected_ids = {unit_id for _, unit_id in ranked[:sample_size]}
    selected = [row for row in rows if _unit_id(row) in selected_ids]
    selected_counts = Counter(_unit_id(row) for row in selected)
    if selected_counts != Counter(
        {unit_id: counts[unit_id] for unit_id in selected_ids}
    ):
        raise AssertionError("resampling-unit sampling failed to preserve unit closure")

    population_units = [
        {"resampling_unit_id": unit_id, "record_count": counts[unit_id]}
        for unit_id in unit_ids
    ]
    selected_unit_ids = sorted(selected_ids)
    population_root = resampling_unit_set_sha256(unit_ids)
    selected_root = resampling_unit_set_sha256(selected_unit_ids)
    population_counts_root = _unit_counts_sha256(counts)
    ranking_root = _ranking_sha256(ranked)
    options = {
        "schema_version": RESAMPLING_UNIT_SELECTION_SCHEMA_VERSION,
        "algorithm": RESAMPLING_UNIT_SELECTION_ALGORITHM,
        "unit_field": RESAMPLING_UNIT_FIELD,
        "sample_size": sample_size,
        "seed": seed,
        "population_record_count": len(rows),
        "population_unit_count": len(unit_ids),
        "population_unit_set_sha256": population_root,
        "population_unit_record_counts_sha256": population_counts_root,
        "selected_unit_set_sha256": selected_root,
        "selected_record_count": len(selected),
        "ranking_sha256": ranking_root,
    }
    report = {
        **options,
        "dataset": dataset,
        "canonical_split": canonical_split,
        "selected_unit_count": len(selected_ids),
        "population_units": population_units,
        "selected_unit_ids": selected_unit_ids,
    }
    return selected, options, report


def validate_resampling_unit_selection(
    *,
    report: Mapping[str, Any],
    options: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    dataset: str,
    canonical_split: str,
) -> None:
    """Authenticate a selection report and prove selected-unit row closure."""

    required_option_fields = {
        "schema_version",
        "algorithm",
        "unit_field",
        "sample_size",
        "seed",
        "population_record_count",
        "population_unit_count",
        "population_unit_set_sha256",
        "population_unit_record_counts_sha256",
        "selected_unit_set_sha256",
        "selected_record_count",
        "ranking_sha256",
    }
    required_report_fields = required_option_fields | {
        "dataset",
        "canonical_split",
        "selected_unit_count",
        "population_units",
        "selected_unit_ids",
    }
    if set(options) != required_option_fields:
        raise ValueError("resampling-unit selection options have malformed fields")
    if set(report) != required_report_fields:
        raise ValueError("resampling-unit selection report has malformed fields")
    if any(report.get(key) != options.get(key) for key in required_option_fields):
        raise ValueError(
            "resampling-unit selection report disagrees with adapter options"
        )
    if (
        report.get("schema_version") != RESAMPLING_UNIT_SELECTION_SCHEMA_VERSION
        or report.get("algorithm") != RESAMPLING_UNIT_SELECTION_ALGORITHM
        or report.get("unit_field") != RESAMPLING_UNIT_FIELD
    ):
        raise ValueError(
            "resampling-unit selection uses an unsupported schema/algorithm"
        )
    if (
        report.get("dataset") != dataset
        or report.get("canonical_split") != canonical_split
    ):
        raise ValueError("resampling-unit selection dataset/split mismatch")

    sample_size = report.get("sample_size")
    seed = report.get("seed")
    if (
        isinstance(sample_size, bool)
        or not isinstance(sample_size, int)
        or sample_size < 1
    ):
        raise ValueError("resampling-unit selection sample_size must be positive")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed <= MAX_SELECTION_SEED
    ):
        raise ValueError(
            "resampling-unit selection seed is outside the supported range"
        )

    raw_population = report.get("population_units")
    if not isinstance(raw_population, list) or not raw_population:
        raise ValueError("resampling-unit selection population_units must be non-empty")
    population_counts: dict[str, int] = {}
    normal_population: list[dict[str, Any]] = []
    for item in raw_population:
        if not isinstance(item, Mapping) or set(item) != {
            "resampling_unit_id",
            "record_count",
        }:
            raise ValueError("resampling-unit selection has malformed population unit")
        unit_id = str(item.get("resampling_unit_id", "")).strip()
        count = item.get("record_count")
        if (
            not unit_id
            or unit_id in population_counts
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
        ):
            raise ValueError("resampling-unit selection has invalid population unit")
        population_counts[unit_id] = count
        normal_population.append({"resampling_unit_id": unit_id, "record_count": count})
    normal_population.sort(key=lambda value: value["resampling_unit_id"])
    if raw_population != normal_population:
        raise ValueError("resampling-unit population must be sorted canonically")

    unit_ids = sorted(population_counts)
    if report.get("population_unit_count") != len(unit_ids):
        raise ValueError("resampling-unit population count mismatch")
    if report.get("population_record_count") != sum(population_counts.values()):
        raise ValueError("resampling-unit population record count mismatch")
    if report.get("population_unit_set_sha256") != resampling_unit_set_sha256(unit_ids):
        raise ValueError("resampling-unit population set digest mismatch")
    if report.get("population_unit_record_counts_sha256") != _unit_counts_sha256(
        population_counts
    ):
        raise ValueError("resampling-unit population record-count digest mismatch")
    if sample_size > len(unit_ids):
        raise ValueError("resampling-unit sample_size exceeds population")

    ranked = _ranked_units(
        unit_ids,
        dataset=dataset,
        canonical_split=canonical_split,
        seed=seed,
    )
    if report.get("ranking_sha256") != _ranking_sha256(ranked):
        raise ValueError("resampling-unit ranking digest mismatch")
    expected_selected_ids = sorted(unit_id for _, unit_id in ranked[:sample_size])
    if report.get("selected_unit_ids") != expected_selected_ids:
        raise ValueError(
            "resampling-unit selected IDs do not reproduce the hash ranking"
        )
    if report.get("selected_unit_count") != sample_size:
        raise ValueError("resampling-unit selected count mismatch")
    if report.get("selected_unit_set_sha256") != resampling_unit_set_sha256(
        expected_selected_ids
    ):
        raise ValueError("resampling-unit selected set digest mismatch")

    observed_counts = Counter(_unit_id(row) for row in selected_rows)
    expected_counts = Counter(
        {unit_id: population_counts[unit_id] for unit_id in expected_selected_ids}
    )
    if observed_counts != expected_counts:
        raise ValueError(
            "selected manifest rows do not preserve every record in each sampled unit"
        )
    if report.get("selected_record_count") != len(selected_rows):
        raise ValueError("resampling-unit selected record count mismatch")
