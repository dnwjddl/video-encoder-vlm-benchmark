"""Fail-closed authentication of a condition-expanded trial matrix.

The data lock authenticates base records while scoring consumes expanded
trials.  This module closes that gap without requiring the original base
manifest: every trial still carries the base record, so the option permutation
can be inverted, the locked base-record root can be reproduced, and the full
condition matrix can be regenerated deterministically.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
from typing import Any, Iterable, Iterator, Mapping

from .attestation import validate_trial_build_attestation
from .conditions import (
    load_condition_config,
    stream_trials,
    trial_content_sha256,
)
from .data_lock import (
    manifest_semantic_record_set_sha256,
    validate_data_lock,
    validate_trial_media_lock,
)
from .integrity import TRIAL_SET_SCHEMA_VERSION, canonical_sha256
from .io import iter_jsonl, sha256_file, write_jsonl
from .protocol import protocol_section
from .schema import normalize_answer, option_label


TRIAL_MATRIX_CLOSURE_SCHEMA_VERSION = "information_upper_bound.trial_matrix_closure.v1"
DEVELOPMENT_TRIAL_MATRIX_CLOSURE_SCHEMA_VERSION = (
    "information_upper_bound.development_trial_matrix_closure.v1"
)
BASE_ID_SET_SCHEMA_VERSION = "information_upper_bound.base_id_set.v1"

# These names are written by build-trials and therefore may not already exist
# in a base manifest.  Keeping the list public lets the builder reject an input
# that would make expansion non-invertible.
TRIAL_EXPANSION_FIELDS = frozenset(
    {
        "answer_text",
        "base_id",
        "clue_text",
        "condition",
        "trial_content_sha256",
        "trial_id",
        "visual_id",
        "visual_spec",
    }
)
TRIAL_BUILD_INJECTED_FIELDS = frozenset(
    {"data_release_sha256", "trial_build_attestation"}
)
GENERATED_TRIAL_FIELDS = frozenset(TRIAL_EXPANSION_FIELDS | TRIAL_BUILD_INJECTED_FIELDS)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _iter_rows(
    rows_or_manifest: Iterable[Mapping[str, Any]] | str | Path,
) -> Iterator[Mapping[str, Any]]:
    if isinstance(rows_or_manifest, (str, Path)):
        yield from iter_jsonl(rows_or_manifest)
    else:
        for index, row in enumerate(rows_or_manifest):
            if not isinstance(row, Mapping):
                raise ValueError(f"trial row {index} must be a mapping")
            yield row


def _normalized_option_permutations(value: Any) -> int | str:
    if str(value).strip().casefold() in {"all", "all_positions"}:
        return "all"
    if isinstance(value, bool):
        raise ValueError("locked option_permutations must be >= 1 or 'all'")
    try:
        converted = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("locked option_permutations must be >= 1 or 'all'") from exc
    if converted < 1 or str(value).strip() != str(converted):
        raise ValueError("locked option_permutations must be >= 1 or 'all'")
    return converted


def _validated_trial_identity(
    row: Mapping[str, Any], *, label: str, index: int
) -> tuple[str, str]:
    trial_id = str(row.get("trial_id", "")).strip()
    row_id = str(row.get("id", "")).strip()
    if not trial_id or row_id != trial_id:
        raise ValueError(f"{label} trial row {index} has inconsistent id/trial_id")
    declared = str(row.get("trial_content_sha256", "")).strip().lower()
    if _SHA256.fullmatch(declared) is None:
        raise ValueError(
            f"{label} trial {trial_id!r} has no valid trial_content_sha256"
        )
    recomputed = trial_content_sha256(row)
    if declared != recomputed:
        raise ValueError(f"{label} trial {trial_id!r} has a stale trial_content_sha256")
    if trial_id != f"trial::{declared}":
        raise ValueError(
            f"{label} trial {trial_id!r} is not derived from its content digest"
        )
    return trial_id, declared


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _open_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute(
        "CREATE TABLE actual (trial_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE expected (trial_id TEXT PRIMARY KEY, content_sha256 TEXT NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE bases ("
        "base_id TEXT PRIMARY KEY, digest TEXT NOT NULL, canonical_json TEXT NOT NULL)"
    )
    return connection


def _trial_set_identity_from_table(
    connection: sqlite3.Connection, table: str
) -> dict[str, Any]:
    """Reproduce integrity.trial_set_identity without materializing its entries."""

    if table not in {"actual", "expected"}:
        raise ValueError("unknown trial identity table")
    count = int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    if count < 1:
        raise ValueError("cannot compute a trial-set identity for an empty manifest")
    digest = hashlib.sha256()
    digest.update(b'{"entries":[')
    first = True
    for trial_id, content_sha256 in connection.execute(
        f"SELECT trial_id, content_sha256 FROM {table} ORDER BY trial_id"
    ):
        if not first:
            digest.update(b",")
        first = False
        digest.update(
            _canonical_json(
                {
                    "trial_id": str(trial_id),
                    "trial_content_sha256": str(content_sha256),
                }
            ).encode("utf-8")
        )
    digest.update(b'],"schema_version":')
    digest.update(_canonical_json(TRIAL_SET_SCHEMA_VERSION).encode("utf-8"))
    digest.update(b"}")
    return {
        "schema_version": TRIAL_SET_SCHEMA_VERSION,
        "trial_count": count,
        "root_sha256": digest.hexdigest(),
    }


def _base_record_from_trial(row: Mapping[str, Any]) -> dict[str, Any]:
    base_id = str(row.get("base_id", "")).strip()
    if not base_id:
        raise ValueError("trial row has no base_id for base-record reconstruction")
    choices = row.get("choices")
    if not isinstance(choices, list) or not 2 <= len(choices) <= 26:
        raise ValueError(f"trial {row.get('trial_id')!r} has invalid choices")
    normalized_choices = [str(value) for value in choices]
    condition = row.get("condition")
    if not isinstance(condition, Mapping):
        raise ValueError(f"trial {row.get('trial_id')!r} has no condition object")
    raw_permutation = condition.get("permutation")
    if not isinstance(raw_permutation, list) or any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in raw_permutation
    ):
        raise ValueError(f"trial {row.get('trial_id')!r} has invalid permutation")
    permutation = [int(value) for value in raw_permutation]
    if len(permutation) != len(normalized_choices) or sorted(permutation) != list(
        range(len(normalized_choices))
    ):
        raise ValueError(f"trial {row.get('trial_id')!r} has invalid permutation")

    try:
        new_gold_label = normalize_answer(row.get("answer"), normalized_choices)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"trial {row.get('trial_id')!r} has an invalid permuted answer"
        ) from exc
    new_gold_index = ord(new_gold_label) - ord("A")
    original_choices: list[str | None] = [None] * len(normalized_choices)
    for new_position, old_position in enumerate(permutation):
        original_choices[old_position] = normalized_choices[new_position]
    if any(
        value is None for value in original_choices
    ):  # defensive; bijection checked.
        raise ValueError(f"trial {row.get('trial_id')!r} permutation is not invertible")
    original_gold_index = permutation[new_gold_index]

    base = deepcopy(dict(row))
    for field in GENERATED_TRIAL_FIELDS:
        base.pop(field, None)
    base["id"] = base_id
    base["choices"] = [str(value) for value in original_choices]
    base["answer"] = option_label(original_gold_index)
    return base


def reconstruct_base_records(
    trial_rows_or_manifest: Iterable[Mapping[str, Any]] | str | Path,
) -> list[dict[str, Any]]:
    """Invert expanded trials and require one identical base record per base ID."""

    by_base: dict[str, tuple[str, dict[str, Any]]] = {}
    row_count = 0
    for row in _iter_rows(trial_rows_or_manifest):
        row_count += 1
        base = _base_record_from_trial(row)
        base_id = str(base["id"])
        digest = canonical_sha256(base)
        prior = by_base.get(base_id)
        if prior is not None and prior[0] != digest:
            raise ValueError(
                f"trial rows reconstruct inconsistent base record {base_id!r}"
            )
        by_base[base_id] = (digest, base)
    if row_count == 0:
        raise ValueError("trial-matrix closure requires a non-empty trial manifest")
    return [by_base[base_id][1] for base_id in sorted(by_base)]


def validate_trial_base_release(
    trial_rows_or_manifest: Iterable[Mapping[str, Any]] | str | Path,
    *,
    data_lock_path: str | Path,
) -> dict[str, Any]:
    """Authenticate every reconstructed base field against a data lock.

    Development training trials do not carry the confirmatory attestation
    needed for full trial-matrix closure.  They still contain an invertible
    copy of each base record, so reconstruct that exact base manifest and run
    the normal data-lock validator over it.  This binds questions, choices,
    answers, diagnostics, resampling-unit membership, and media bytes rather
    than checking only base IDs and media paths.
    """

    base_records = reconstruct_base_records(trial_rows_or_manifest)
    return _validate_reconstructed_base_release(
        base_records, data_lock_path=data_lock_path
    )


def _validate_reconstructed_base_release(
    base_records: list[dict[str, Any]], *, data_lock_path: str | Path
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(
        prefix="information-upper-bound-base-release-"
    ) as directory:
        manifest_path = Path(directory) / "reconstructed-base.jsonl"
        write_jsonl(manifest_path, base_records)
        return validate_data_lock(
            data_lock_path,
            manifest_path=manifest_path,
            verify_sources=False,
            verify_media=True,
        )


def _trial_set_identity_from_entries(entries: Mapping[str, str]) -> dict[str, Any]:
    if not entries:
        raise ValueError("cannot compute a trial-set identity for an empty manifest")
    values = [
        {"trial_id": trial_id, "trial_content_sha256": entries[trial_id]}
        for trial_id in sorted(entries)
    ]
    return {
        "schema_version": TRIAL_SET_SCHEMA_VERSION,
        "trial_count": len(values),
        "root_sha256": canonical_sha256(
            {"schema_version": TRIAL_SET_SCHEMA_VERSION, "entries": values}
        ),
    }


def validate_development_trial_matrix_closure(
    trial_rows_or_manifest: Iterable[Mapping[str, Any]] | str | Path,
    *,
    data_lock_path: str | Path,
    conditions_config_path: str | Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and exactly regenerate a locked development trial matrix.

    This is the training counterpart to confirmatory closure.  It requires a
    development attestation, an exact locked base release, and one unsharded
    deterministic expansion from the supplied condition and protocol files.
    """

    if isinstance(trial_rows_or_manifest, (str, Path)):
        source: Iterable[Mapping[str, Any]] | str | Path = trial_rows_or_manifest
    else:
        source = [dict(row) for row in _iter_rows(trial_rows_or_manifest)]
    base_records = reconstruct_base_records(source)
    data_lock = _validate_reconstructed_base_release(
        base_records, data_lock_path=data_lock_path
    )
    locked_release = str(data_lock["data_release_sha256"])

    conditions_sha256 = sha256_file(conditions_config_path)
    specs, condition_options = load_condition_config(conditions_config_path)
    sampling = protocol_section(protocol, "sampling")
    try:
        seed = int(sampling["seed"])
        option_permutations = _normalized_option_permutations(
            sampling["option_permutations"]
        )
        trial_shards = int(sampling["trial_shards"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "development trial protocol has invalid sampling fields"
        ) from exc
    if trial_shards != 1:
        raise ValueError("development trial-matrix closure requires trial_shards=1")
    if "seed" in condition_options and int(condition_options["seed"]) != seed:
        raise ValueError("development condition seed differs from its protocol")
    if "option_permutations" in condition_options and (
        _normalized_option_permutations(condition_options["option_permutations"])
        != option_permutations
    ):
        raise ValueError(
            "development condition option_permutations differs from its protocol"
        )

    actual_entries: dict[str, str] = {}
    common_attestation: dict[str, Any] | None = None
    common_attestation_canonical: str | None = None
    validated_attestation: dict[str, Any] | None = None
    for index, row in enumerate(_iter_rows(source)):
        trial_id, content_sha256 = _validated_trial_identity(
            row, label="development training", index=index
        )
        if trial_id in actual_entries:
            raise ValueError(
                f"development training matrix duplicates trial_id {trial_id!r}"
            )
        actual_entries[trial_id] = content_sha256
        raw_attestation = row.get("trial_build_attestation")
        if common_attestation is None:
            validated_attestation = validate_trial_build_attestation(
                row, protocol=protocol, require_confirmatory=False
            )
            if validated_attestation.get("mode") != "development":
                raise ValueError(
                    "projector training trials must carry a development attestation"
                )
            if validated_attestation.get("data_release_sha256") != locked_release:
                raise ValueError(
                    "development training trials name a different locked data release"
                )
            if (
                str(validated_attestation.get("condition_config_sha256", "")).lower()
                != conditions_sha256
            ):
                raise ValueError(
                    "development training trials use a different condition config"
                )
            if not isinstance(raw_attestation, Mapping):
                raise ValueError("trial has no trial_build_attestation object")
            common_attestation = deepcopy(dict(raw_attestation))
            common_attestation_canonical = _canonical_json(common_attestation)
        elif _canonical_json(raw_attestation) != common_attestation_canonical:
            raise ValueError(
                "development training matrix has mixed trial-build attestations"
            )
    if (
        not actual_entries
        or common_attestation is None
        or validated_attestation is None
    ):
        raise ValueError(
            "development trial-matrix closure requires a non-empty manifest"
        )

    regenerated_inputs = [
        {
            **base,
            "data_release_sha256": locked_release,
            "trial_build_attestation": deepcopy(common_attestation),
        }
        for base in base_records
    ]
    regenerated, _state = stream_trials(
        regenerated_inputs,
        specs,
        seed=seed,
        option_permutations=option_permutations,
    )
    expected_entries: dict[str, str] = {}
    for index, row in enumerate(regenerated):
        trial_id, content_sha256 = _validated_trial_identity(
            row, label="regenerated development training", index=index
        )
        if trial_id in expected_entries:
            raise ValueError(
                f"regenerated development matrix duplicates trial_id {trial_id!r}"
            )
        expected_entries[trial_id] = content_sha256
    if actual_entries != expected_entries:
        actual_ids = set(actual_entries)
        expected_ids = set(expected_entries)
        raise ValueError(
            "development training matrix is not the exact deterministic condition "
            "expansion; "
            f"missing={sorted(expected_ids - actual_ids)[:10]}, "
            f"extra={sorted(actual_ids - expected_ids)[:10]}"
        )

    identity = _trial_set_identity_from_entries(actual_entries)
    closure_payload = {
        "schema_version": DEVELOPMENT_TRIAL_MATRIX_CLOSURE_SCHEMA_VERSION,
        "status": "exact",
        "mode": "development",
        "data_release_sha256": locked_release,
        "conditions_sha256": conditions_sha256,
        "trial_build_attestation_sha256": validated_attestation["attestation_sha256"],
        "base_semantic_record_set_sha256": data_lock[
            "manifest_semantic_record_set_sha256"
        ],
        "base_id_set_root_sha256": _base_id_set_root(
            str(base["id"]) for base in base_records
        ),
        "base_records": len(base_records),
        "conditions": [spec.name for spec in specs],
        "trial_count": identity["trial_count"],
        "trial_set_root_sha256": identity["root_sha256"],
        "sampling": {
            "seed": seed,
            "option_permutations": option_permutations,
            "trial_shards": trial_shards,
        },
    }
    return {
        "data_lock": data_lock,
        "closure": {
            **closure_payload,
            "closure_sha256": canonical_sha256(closure_payload),
        },
    }


def _load_data_lock(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read data lock {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"data lock must be a JSON object: {source}")
    return value


def _locked_base_ids(lock: Mapping[str, Any]) -> set[str]:
    bindings = lock.get("media_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("data lock has no canonical media_bindings")
    identifiers: list[str] = []
    for index, binding in enumerate(bindings):
        if not isinstance(binding, Mapping) or not isinstance(
            binding.get("record_ids"), list
        ):
            raise ValueError(f"data lock media_bindings[{index}] is malformed")
        identifiers.extend(str(value).strip() for value in binding["record_ids"])
    if any(not value for value in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("data lock media_bindings contain empty/duplicate base IDs")
    return set(identifiers)


def _base_id_set_root(identifiers: Iterable[str]) -> str:
    return canonical_sha256(
        {
            "schema_version": BASE_ID_SET_SCHEMA_VERSION,
            "base_ids": sorted(str(value) for value in identifiers),
        }
    )


def validate_trial_matrix_closure(
    trial_rows_or_manifest: Iterable[Mapping[str, Any]] | str | Path,
    *,
    data_lock_path: str | Path,
    conditions_config_path: str | Path,
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate and exactly regenerate a confirmatory expanded trial matrix.

    The returned object contains no local paths or row-order-dependent values,
    so its digest can be carried by feature and projector metadata.
    """
    data_section = protocol_section(protocol, "data")
    locked_release = str(data_section.get("data_release_sha256", "")).lower()
    conditions_sha256 = str(data_section.get("conditions_sha256", "")).lower()
    if (
        _SHA256.fullmatch(locked_release) is None
        or _SHA256.fullmatch(conditions_sha256) is None
    ):
        raise ValueError("protocol data release/conditions are not SHA256-locked")
    if sha256_file(conditions_config_path) != conditions_sha256:
        raise ValueError("condition config differs from the locked protocol")

    sampling = protocol_section(protocol, "sampling")
    if isinstance(sampling.get("seed"), bool):
        raise ValueError("locked sampling.seed must be an integer")
    try:
        seed = int(sampling["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("locked sampling.seed must be an integer") from exc
    option_permutations = _normalized_option_permutations(
        sampling.get("option_permutations")
    )
    trial_shards = sampling.get("trial_shards")
    if (
        isinstance(trial_shards, bool)
        or not isinstance(trial_shards, int)
        or trial_shards < 1
    ):
        raise ValueError("locked sampling.trial_shards must be a positive integer")

    specs, condition_options = load_condition_config(conditions_config_path)
    if "seed" in condition_options and int(condition_options["seed"]) != seed:
        raise ValueError("condition config seed differs from locked sampling.seed")
    if "option_permutations" in condition_options and (
        _normalized_option_permutations(condition_options["option_permutations"])
        != option_permutations
    ):
        raise ValueError(
            "condition config option_permutations differs from locked sampling"
        )

    lock = _load_data_lock(data_lock_path)
    locked_record_count = lock.get("records")
    if (
        isinstance(locked_record_count, bool)
        or not isinstance(locked_record_count, int)
        or locked_record_count < 1
    ):
        raise ValueError("data lock has an invalid record count")
    locked_ids = _locked_base_ids(lock)
    if len(locked_ids) != locked_record_count:
        raise ValueError("data-lock record count differs from its exact base-ID set")

    with tempfile.TemporaryDirectory(prefix="information-upper-bound-closure-") as tmp:
        connection = _open_database(Path(tmp) / "trial-matrix.sqlite3")
        try:
            actual_count = 0
            common_attestation: dict[str, Any] | None = None
            common_attestation_canonical: str | None = None
            attestation_digest: str | None = None
            for index, row in enumerate(_iter_rows(trial_rows_or_manifest)):
                trial_id, content_sha256 = _validated_trial_identity(
                    row, label="actual", index=index
                )
                try:
                    connection.execute(
                        "INSERT INTO actual(trial_id, content_sha256) VALUES (?, ?)",
                        (trial_id, content_sha256),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"actual trial matrix duplicates trial_id {trial_id!r}"
                    ) from exc

                raw_attestation = row.get("trial_build_attestation")
                if common_attestation is None:
                    attestation = validate_trial_build_attestation(
                        row, protocol=protocol, require_confirmatory=True
                    )
                    attestation_digest = str(attestation["attestation_sha256"])
                    if not isinstance(
                        raw_attestation, Mapping
                    ):  # validator is explicit.
                        raise ValueError("trial has no trial_build_attestation object")
                    common_attestation = deepcopy(dict(raw_attestation))
                    common_attestation_canonical = _canonical_json(common_attestation)
                elif _canonical_json(raw_attestation) != common_attestation_canonical:
                    raise ValueError(
                        "trial matrix has non-identical trial-build attestations"
                    )
                if str(row.get("data_release_sha256", "")) != locked_release:
                    raise ValueError(
                        f"trial {trial_id!r} data release differs from the protocol"
                    )

                base = _base_record_from_trial(row)
                base_id = str(base["id"])
                base_json = _canonical_json(base)
                base_digest = hashlib.sha256(base_json.encode("utf-8")).hexdigest()
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO bases(base_id, digest, canonical_json) "
                    "VALUES (?, ?, ?)",
                    (base_id, base_digest, base_json),
                )
                if cursor.rowcount != 1:
                    prior = connection.execute(
                        "SELECT digest FROM bases WHERE base_id = ?", (base_id,)
                    ).fetchone()
                    if prior is None or str(prior[0]) != base_digest:
                        raise ValueError(
                            f"trial rows reconstruct inconsistent base record {base_id!r}"
                        )
                actual_count += 1
            if actual_count == 0 or common_attestation is None:
                raise ValueError(
                    "trial-matrix closure requires a non-empty trial manifest"
                )
            assert attestation_digest is not None
            connection.commit()

            base_count = int(
                connection.execute("SELECT COUNT(*) FROM bases").fetchone()[0]
            )
            actual_base_ids = {
                str(row[0]) for row in connection.execute("SELECT base_id FROM bases")
            }
            if actual_base_ids != locked_ids or base_count != locked_record_count:
                raise ValueError(
                    "reconstructed base-ID coverage differs from the locked release; "
                    f"missing={sorted(locked_ids - actual_base_ids)[:10]}, "
                    f"extra={sorted(actual_base_ids - locked_ids)[:10]}"
                )

            def base_records() -> Iterator[dict[str, Any]]:
                for (base_json,) in connection.execute(
                    "SELECT canonical_json FROM bases ORDER BY base_id"
                ):
                    value = json.loads(str(base_json))
                    if not isinstance(
                        value, dict
                    ):  # canonical rows are inserted above.
                        raise ValueError("reconstructed base record is not an object")
                    yield value

            # validate_trial_media_lock currently materializes its input.  Pass
            # exactly one small projection per reconstructed base, never the
            # condition-expanded trial stream.
            def media_rows() -> Iterator[dict[str, Any]]:
                for base in base_records():
                    diagnostic = base.get("diagnostic")
                    dataset = (
                        str(diagnostic.get("dataset", ""))
                        if isinstance(diagnostic, Mapping)
                        else ""
                    )
                    yield {
                        "id": base["id"],
                        "base_id": base["id"],
                        "data_release_sha256": locked_release,
                        "diagnostic": {"dataset": dataset},
                        "visual_spec": {"media_path": base.get("media_path")},
                    }

            media_lock = validate_trial_media_lock(data_lock_path, media_rows())
            if locked_release != str(media_lock.get("data_release_sha256", "")):
                raise ValueError("data lock release differs from the locked protocol")

            semantic_root = manifest_semantic_record_set_sha256(base_records())
            locked_semantic_root = str(
                lock.get("manifest_semantic_record_set_sha256", "")
            ).lower()
            if _SHA256.fullmatch(locked_semantic_root) is None:
                raise ValueError("data lock has no valid semantic base-record root")
            if semantic_root != locked_semantic_root:
                raise ValueError(
                    "reconstructed base semantic content differs from the locked release"
                )

            actual_identity = _trial_set_identity_from_table(connection, "actual")

            def regenerated_inputs() -> Iterator[dict[str, Any]]:
                for base in base_records():
                    yield {
                        **base,
                        "data_release_sha256": locked_release,
                        "trial_build_attestation": deepcopy(common_attestation),
                    }

            regenerated, _state = stream_trials(
                regenerated_inputs(),
                specs,
                seed=seed,
                option_permutations=option_permutations,
            )
            missing: list[tuple[str, str]] = []
            expected_count = 0
            for index, row in enumerate(regenerated):
                trial_id, content_sha256 = _validated_trial_identity(
                    row, label="regenerated", index=index
                )
                try:
                    connection.execute(
                        "INSERT INTO expected(trial_id, content_sha256) VALUES (?, ?)",
                        (trial_id, content_sha256),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ValueError(
                        f"regenerated trial matrix duplicates trial_id {trial_id!r}"
                    ) from exc
                deleted = connection.execute(
                    "DELETE FROM actual WHERE trial_id = ? AND content_sha256 = ?",
                    (trial_id, content_sha256),
                )
                if deleted.rowcount != 1 and len(missing) < 10:
                    missing.append((trial_id, content_sha256))
                expected_count += 1
            connection.commit()
            expected_identity = _trial_set_identity_from_table(connection, "expected")
            remaining = int(
                connection.execute("SELECT COUNT(*) FROM actual").fetchone()[0]
            )
            extra = [
                (str(trial_id), str(content_sha256))
                for trial_id, content_sha256 in connection.execute(
                    "SELECT trial_id, content_sha256 FROM actual ORDER BY trial_id LIMIT 10"
                )
            ]
            if (
                missing
                or remaining
                or actual_count != expected_count
                or actual_identity != expected_identity
            ):
                raise ValueError(
                    "trial matrix is not the exact deterministic condition expansion; "
                    f"missing={missing}, extra={extra}, extra_count={remaining}"
                )

            closure_payload = {
                "schema_version": TRIAL_MATRIX_CLOSURE_SCHEMA_VERSION,
                "status": "exact",
                "data_release_sha256": locked_release,
                "conditions_sha256": conditions_sha256,
                "trial_build_attestation_sha256": attestation_digest,
                "base_semantic_record_set_sha256": semantic_root,
                "base_id_set_root_sha256": _base_id_set_root(actual_base_ids),
                "base_records": base_count,
                "trial_count": actual_count,
                "trial_set_root_sha256": actual_identity["root_sha256"],
                "sampling": {
                    "seed": seed,
                    "option_permutations": option_permutations,
                    "trial_shards": trial_shards,
                },
            }
            return {
                **closure_payload,
                "closure_sha256": canonical_sha256(closure_payload),
            }
        finally:
            connection.close()


__all__ = [
    "BASE_ID_SET_SCHEMA_VERSION",
    "DEVELOPMENT_TRIAL_MATRIX_CLOSURE_SCHEMA_VERSION",
    "GENERATED_TRIAL_FIELDS",
    "TRIAL_BUILD_INJECTED_FIELDS",
    "TRIAL_EXPANSION_FIELDS",
    "TRIAL_MATRIX_CLOSURE_SCHEMA_VERSION",
    "reconstruct_base_records",
    "validate_development_trial_matrix_closure",
    "validate_trial_base_release",
    "validate_trial_matrix_closure",
]
