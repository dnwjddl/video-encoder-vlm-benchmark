from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from .unit_sampling import (
    RESAMPLING_UNIT_FIELD,
    RESAMPLING_UNIT_SELECTION_ALGORITHM,
    RESAMPLING_UNIT_SELECTION_SCHEMA_VERSION,
    resampling_unit_set_sha256,
)


CLEVRER_PILOT_SEED = 42
CLEVRER_PILOT_CONTRACTS = {
    "validation": {
        "canonical_split": "validation",
        "population_scene_start": 10000,
        "population_unit_count": 5000,
        "population_record_count": 70862,
        "sample_size": 500,
    },
    "train": {
        "canonical_split": "train",
        "population_scene_start": 0,
        "population_unit_count": 10000,
        "population_record_count": 141211,
        "sample_size": 2000,
    },
}


def _contract(role: Literal["validation", "train"]) -> Mapping[str, int | str]:
    return CLEVRER_PILOT_CONTRACTS[role]


def expected_clevrer_unit_ids(
    role: Literal["validation", "train"],
) -> list[str]:
    contract = _contract(role)
    start = int(contract["population_scene_start"])
    count = int(contract["population_unit_count"])
    return sorted(
        f"clevrer:scene:{scene_index}" for scene_index in range(start, start + count)
    )


def validate_clevrer_selection_options(
    selection: Mapping[str, Any],
    *,
    role: Literal["validation", "train"],
    locked_record_count: int,
) -> None:
    contract = _contract(role)
    expected_unit_ids = expected_clevrer_unit_ids(role)
    expected = {
        "schema_version": RESAMPLING_UNIT_SELECTION_SCHEMA_VERSION,
        "algorithm": RESAMPLING_UNIT_SELECTION_ALGORITHM,
        "unit_field": RESAMPLING_UNIT_FIELD,
        "sample_size": int(contract["sample_size"]),
        "seed": CLEVRER_PILOT_SEED,
        "population_record_count": int(contract["population_record_count"]),
        "population_unit_count": int(contract["population_unit_count"]),
        "population_unit_set_sha256": resampling_unit_set_sha256(expected_unit_ids),
        "selected_record_count": locked_record_count,
    }
    mismatches = {
        key: {"expected": value, "actual": selection.get(key)}
        for key, value in expected.items()
        if selection.get(key) != value
    }
    if mismatches:
        raise ValueError(f"CLEVRER {role} sampling contract mismatch: {mismatches}")


def validate_clevrer_selection_report(
    selection: Mapping[str, Any],
    *,
    role: Literal["validation", "train"],
    locked_record_count: int,
) -> None:
    validate_clevrer_selection_options(
        selection,
        role=role,
        locked_record_count=locked_record_count,
    )
    contract = _contract(role)
    if (
        selection.get("dataset") != "clevrer"
        or selection.get("canonical_split") != contract["canonical_split"]
        or selection.get("selected_unit_count") != contract["sample_size"]
    ):
        raise ValueError(
            f"CLEVRER {role} sampling report has wrong dataset/split/count"
        )
    population_units = selection.get("population_units")
    if not isinstance(population_units, list):
        raise ValueError(f"CLEVRER {role} sampling report has no population units")
    actual_ids = [
        str(item.get("resampling_unit_id", ""))
        for item in population_units
        if isinstance(item, Mapping)
    ]
    expected_ids = expected_clevrer_unit_ids(role)
    if len(actual_ids) != len(population_units) or actual_ids != expected_ids:
        raise ValueError(
            f"CLEVRER {role} sample was not drawn from the exact official scene-ID universe"
        )
