"""Deterministic execution partitions for strict score workers."""

from __future__ import annotations

from typing import Any, Mapping


SCORING_PARTITION_SCHEMA_VERSION = "information_upper_bound.scoring_partition.v1"
SCORING_PARTITION_ALGORITHM = "trial_content_sha256_mod_worker_count"


def validate_score_worker(*, worker_count: int, worker_index: int) -> None:
    """Validate zero-based execution worker coordinates."""

    if isinstance(worker_count, bool) or not isinstance(worker_count, int):
        raise ValueError("--worker-count must be an integer >= 1")
    if worker_count < 1:
        raise ValueError("--worker-count must be >= 1")
    if isinstance(worker_index, bool) or not isinstance(worker_index, int):
        raise ValueError("--worker-index must be an integer")
    if not 0 <= worker_index < worker_count:
        raise ValueError("--worker-index must be in [0, --worker-count)")


def score_worker_index(trial_content_digest: str, *, worker_count: int) -> int:
    """Map an authenticated trial digest to one stable execution worker."""

    return int(trial_content_digest, 16) % worker_count


def validate_score_partition(value: Any) -> dict[str, int | str]:
    """Return one canonical signed partition descriptor or raise."""

    if not isinstance(value, Mapping):
        raise ValueError("score_partition must be an object")
    expected_fields = {
        "schema_version",
        "algorithm",
        "worker_count",
        "worker_index",
    }
    if set(value) != expected_fields:
        raise ValueError(
            "score_partition must contain exactly schema_version, algorithm, "
            "worker_count, and worker_index"
        )
    if value.get("schema_version") != SCORING_PARTITION_SCHEMA_VERSION:
        raise ValueError(
            "score_partition schema_version must be "
            f"{SCORING_PARTITION_SCHEMA_VERSION!r}"
        )
    if value.get("algorithm") != SCORING_PARTITION_ALGORITHM:
        raise ValueError(
            f"score_partition algorithm must be {SCORING_PARTITION_ALGORITHM!r}"
        )
    worker_count = value.get("worker_count")
    worker_index = value.get("worker_index")
    validate_score_worker(worker_count=worker_count, worker_index=worker_index)
    return {
        "schema_version": SCORING_PARTITION_SCHEMA_VERSION,
        "algorithm": SCORING_PARTITION_ALGORITHM,
        "worker_count": int(worker_count),
        "worker_index": int(worker_index),
    }
