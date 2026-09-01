from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import tracemalloc
import unittest
from unittest.mock import patch

from information_upper_bound.attestation import TRIAL_BUILD_ATTESTATION_SCHEMA_VERSION
from information_upper_bound.conditions import (
    build_trials,
    load_condition_config,
    stream_trials,
)
from information_upper_bound.data_lock import manifest_semantic_record_set_sha256
from information_upper_bound.integrity import canonical_sha256, trial_set_identity
from information_upper_bound.io import sha256_file, write_json, write_jsonl
from information_upper_bound.protocol import trial_build_protocol_sha256
from information_upper_bound.schema import SCHEMA_VERSION
from information_upper_bound.trial_matrix import (
    GENERATED_TRIAL_FIELDS,
    reconstruct_base_records,
    validate_trial_matrix_closure,
)


def _base_record(media_path: str = "/portable/media.mp4") -> dict:
    return {
        "id": "base-1",
        "source": "official-fixture",
        "benchmark": "fixture",
        "task": "mcq",
        "media_type": "video",
        "media_path": media_path,
        "question": "What happens first?",
        "choices": ["opens", "closes"],
        "answer": "A",
        "diagnostic": {
            "schema_version": SCHEMA_VERSION,
            "dataset": "fixture",
            "split": "test",
            "information_family": "temporal_order",
            "question_family": "event_order",
            "reasoning_depth": 1,
            "resampling_unit_id": "fixture:video:1",
            "pair_id": "standalone:base-1",
            "pair_role": "standalone",
            "evidence_spans": [],
            "oracles": {
                "static_facts": [
                    {
                        "text": "The door opens.",
                        "access": "safe_visual_gt",
                        "source": "official_annotation",
                        "lineage": "official_adapter",
                    }
                ],
                "unordered_events": [],
                "ordered_events": [],
                "temporal_relations": [],
                "state_changes": [],
                "relations": [],
                "operator": None,
                "intermediate": [],
                "answer_derived": False,
            },
            "provenance": {"source_id": "base-1"},
        },
    }


def _protocol(*, release: str, conditions_sha256: str) -> dict:
    return {
        "schema_version": "1.0",
        "name": "trial-matrix-test",
        "data": {
            "data_release_sha256": release,
            "conditions_sha256": conditions_sha256,
            "required_datasets": ["fixture"],
            "coverage_contract": {
                "schema_version": "information_upper_bound.coverage_contract.v1",
                "datasets": {
                    "fixture": {
                        "required_adapter_runs": 1,
                        "required_adapter_run_ids": ["adapter-run::" + "a" * 64],
                        "required_splits": ["test"],
                        "required_information_families": ["temporal_order"],
                        "required_question_families": [],
                        "exact_question_family_set": False,
                        "required_source_roles": ["annotations"],
                        "minimum_records": 1,
                        "minimum_records_with_evidence": 0,
                        "minimum_records_with_safe_oracles": 1,
                    }
                },
            },
        },
        "sampling": {
            "seed": 42,
            "option_permutations": "all",
            "trial_shards": 1,
        },
        "dataset_roles": {
            "fixture": {
                "information_families": ["temporal_order"],
                "primary_use": "unit-test closure",
            }
        },
        "analysis": {"bootstrap_replicates": 1},
        "confirmatory_comparisons": [["full_video", "question_only"]],
        # Deliberately late-bound and excluded from the build-protocol digest.
        "projector": {"checkpoint_sha256": "template"},
    }


class TrialMatrixClosureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.conditions_path = self.root / "conditions.yaml"
        write_json(
            self.conditions_path,
            {
                "schema_version": "1.0",
                "options": {"seed": 42, "option_permutations": "all"},
                "conditions": [
                    {
                        "name": "question_only",
                        "input_channel": "question_only",
                        "visual_view": "none",
                        "doses": [0],
                    },
                    {
                        "name": "full_video",
                        "input_channel": "visual",
                        "visual_view": "full",
                        "doses": [0],
                    },
                    {
                        "name": "static_oracle",
                        "input_channel": "text_oracle",
                        "visual_view": "none",
                        "clue_fields": ["static_facts"],
                        "doses": [1, 2],
                    },
                ],
            },
        )
        self.release = "d" * 64
        self.protocol = _protocol(
            release=self.release,
            conditions_sha256=sha256_file(self.conditions_path),
        )
        self.base = _base_record()
        payload = {
            "schema_version": TRIAL_BUILD_ATTESTATION_SCHEMA_VERSION,
            "mode": "confirmatory",
            "data_release_sha256": self.release,
            "condition_config_sha256": sha256_file(self.conditions_path),
            "trial_build_protocol_sha256": trial_build_protocol_sha256(self.protocol),
            "sampling": {
                "seed": 42,
                "option_permutations": "all",
                "trial_shards": 1,
            },
        }
        self.attestation = {
            **payload,
            "attestation_sha256": canonical_sha256(payload),
        }
        self.trials = self._build(self.base)
        self.lock_path = self.root / "data-lock.json"
        write_json(
            self.lock_path,
            {
                "records": 1,
                "data_release_sha256": self.release,
                "manifest_semantic_record_set_sha256": (
                    manifest_semantic_record_set_sha256([self.base])
                ),
                "media_bindings": [
                    {
                        "record_ids": ["base-1"],
                    }
                ],
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, base: dict) -> list[dict]:
        specs, _options = load_condition_config(self.conditions_path)
        rows, _report = build_trials(
            [
                {
                    **deepcopy(base),
                    "data_release_sha256": self.release,
                    "trial_build_attestation": deepcopy(self.attestation),
                }
            ],
            specs,
            seed=42,
            option_permutations="all",
        )
        return rows

    def _closure(self, rows: list[dict], *, lock_path: Path | None = None) -> dict:
        with patch(
            "information_upper_bound.trial_matrix.validate_trial_media_lock",
            return_value={"data_release_sha256": self.release},
        ):
            return validate_trial_matrix_closure(
                rows,
                data_lock_path=lock_path or self.lock_path,
                conditions_config_path=self.conditions_path,
                protocol=self.protocol,
            )

    def test_honest_full_matrix_passes_and_reconstructs_from_nonidentity(self) -> None:
        report = self._closure(self.trials)
        self.assertEqual(report["status"], "exact")
        self.assertEqual(report["base_records"], 1)
        self.assertEqual(report["trial_count"], 8)
        self.assertRegex(report["trial_set_root_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            report["trial_set_root_sha256"],
            trial_set_identity(self.trials)["root_sha256"],
        )
        self.assertRegex(report["closure_sha256"], r"^[0-9a-f]{64}$")

        manifest_path = self.root / "trials.jsonl"
        write_jsonl(manifest_path, self.trials)
        self.assertEqual(self._closure(manifest_path), report)

        nonidentity = [
            row for row in self.trials if row["condition"]["permutation_index"] == 1
        ]
        reconstructed = reconstruct_base_records(nonidentity)
        self.assertEqual(reconstructed, [self.base])

    def test_closure_identity_is_row_order_and_media_path_independent(self) -> None:
        first = self._closure(list(reversed(self.trials)))
        relocated_base = _base_record("/different/mount/media.mp4")
        relocated_trials = self._build(relocated_base)
        relocated_lock = self.root / "relocated-lock.json"
        write_json(
            relocated_lock,
            {
                "records": 1,
                "data_release_sha256": self.release,
                "manifest_semantic_record_set_sha256": (
                    manifest_semantic_record_set_sha256([relocated_base])
                ),
                "media_bindings": [{"record_ids": ["base-1"]}],
            },
        )
        second = self._closure(relocated_trials, lock_path=relocated_lock)
        self.assertEqual(first, second)

    def test_missing_condition_dose_or_permutation_fails(self) -> None:
        subsets = {
            "condition": [
                row for row in self.trials if row["condition"]["name"] != "full_video"
            ],
            "dose": [
                row
                for row in self.trials
                if not (
                    row["condition"]["name"] == "static_oracle"
                    and row["condition"]["requested_dose"] == 1
                )
            ],
            "permutation": [
                row for row in self.trials if row["condition"]["permutation_index"] != 0
            ],
            "one-row-per-base": [self.trials[0]],
        }
        for name, rows in subsets.items():
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "exact deterministic"):
                    self._closure(rows)

    def test_duplicate_and_stale_trial_identities_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicates trial_id"):
            self._closure([*self.trials, deepcopy(self.trials[0])])
        stale = deepcopy(self.trials)
        stale[0]["question"] = "Tampered question"
        with self.assertRaisesRegex(ValueError, "stale trial_content_sha256"):
            self._closure(stale)

    def test_cross_row_base_inconsistency_and_semantic_mutation_fail(self) -> None:
        inconsistent = deepcopy(self.trials)
        inconsistent[0]["source"] = "different-source"
        with self.assertRaisesRegex(ValueError, "inconsistent base record"):
            self._closure(inconsistent)

        changed = deepcopy(self.trials)
        for row in changed:
            row["source"] = "different-source"
        with self.assertRaisesRegex(ValueError, "semantic content"):
            self._closure(changed)

    def test_locked_base_count_and_id_set_are_both_required(self) -> None:
        wrong_count = self.root / "wrong-count.json"
        write_json(
            wrong_count,
            {
                "records": 2,
                "data_release_sha256": self.release,
                "manifest_semantic_record_set_sha256": (
                    manifest_semantic_record_set_sha256([self.base])
                ),
                "media_bindings": [{"record_ids": ["base-1"]}],
            },
        )
        with self.assertRaisesRegex(ValueError, "record count"):
            self._closure(self.trials, lock_path=wrong_count)

        wrong_id = self.root / "wrong-id.json"
        write_json(
            wrong_id,
            {
                "records": 1,
                "data_release_sha256": self.release,
                "manifest_semantic_record_set_sha256": (
                    manifest_semantic_record_set_sha256([self.base])
                ),
                "media_bindings": [{"record_ids": ["other-base"]}],
            },
        )
        with self.assertRaisesRegex(ValueError, "base-ID coverage"):
            self._closure(self.trials, lock_path=wrong_id)

    def test_exported_reserved_fields_cover_every_noninvertible_output(self) -> None:
        self.assertTrue(
            {
                "answer_text",
                "base_id",
                "clue_text",
                "condition",
                "data_release_sha256",
                "trial_build_attestation",
                "trial_content_sha256",
                "trial_id",
                "visual_id",
                "visual_spec",
            }.issubset(GENERATED_TRIAL_FIELDS)
        )

    def test_large_synthetic_matrix_is_consumed_once_with_bounded_python_memory(
        self,
    ) -> None:
        # Four thousand expanded trials are large enough to catch accidental
        # listification/replay while keeping this regression suitable for the
        # ordinary unit-test suite.
        base_count = 500
        bases: list[dict] = []
        for index in range(base_count):
            base = _base_record(f"/portable/media-{index}.mp4")
            base_id = f"base-{index:05d}"
            base["id"] = base_id
            base["diagnostic"]["resampling_unit_id"] = f"fixture:video:{index}"
            base["diagnostic"]["pair_id"] = f"standalone:{base_id}"
            base["diagnostic"]["provenance"]["source_id"] = base_id
            bases.append(base)

        lock_path = self.root / "large-lock.json"
        write_json(
            lock_path,
            {
                "records": base_count,
                "data_release_sha256": self.release,
                "manifest_semantic_record_set_sha256": (
                    manifest_semantic_record_set_sha256(bases)
                ),
                "media_bindings": [
                    {"record_ids": sorted(base["id"] for base in bases)}
                ],
            },
        )
        specs, _options = load_condition_config(self.conditions_path)
        trial_iterator, _state = stream_trials(
            (
                {
                    **base,
                    "data_release_sha256": self.release,
                    "trial_build_attestation": deepcopy(self.attestation),
                }
                for base in bases
            ),
            specs,
            seed=42,
            option_permutations="all",
        )

        class SinglePass:
            def __init__(self, values):
                self.values = values
                self.iterations = 0

            def __iter__(self):
                self.iterations += 1
                if self.iterations != 1:
                    raise AssertionError("expanded trial iterable was replayed")
                return iter(self.values)

        single_pass = SinglePass(trial_iterator)
        media_base_count = 0

        def validate_media(_path, rows):
            nonlocal media_base_count
            media_base_count = sum(1 for _row in rows)
            return {"data_release_sha256": self.release}

        tracemalloc.start()
        try:
            with patch(
                "information_upper_bound.trial_matrix.validate_trial_media_lock",
                side_effect=validate_media,
            ):
                report = validate_trial_matrix_closure(
                    single_pass,
                    data_lock_path=lock_path,
                    conditions_config_path=self.conditions_path,
                    protocol=self.protocol,
                )
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(single_pass.iterations, 1)
        self.assertEqual(media_base_count, base_count)
        self.assertEqual(report["base_records"], base_count)
        self.assertEqual(report["trial_count"], base_count * 8)
        self.assertLess(peak, 64 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
