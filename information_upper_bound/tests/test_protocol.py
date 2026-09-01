from __future__ import annotations

import unittest

from information_upper_bound.cli import _shard_records
from information_upper_bound.protocol import (
    load_protocol,
    protocol_section,
    validate_data_protocol,
    validate_frozen_model_protocol,
    validate_release_coverage,
)
from information_upper_bound.run import parse_args as parse_score_args


class ProtocolTests(unittest.TestCase):
    def test_default_protocol_locks_sampling_and_frozen_model(self) -> None:
        protocol, metadata = load_protocol()
        sampling = protocol_section(protocol, "sampling")
        self.assertEqual(sampling["option_permutations"], "all")
        self.assertEqual(sampling["trial_shards"], 1)
        analysis = protocol_section(protocol, "analysis")
        self.assertEqual(
            analysis["cluster_key_priority"],
            ["resampling_unit_id", "pair_id", "independent_unit_id", "base_id"],
        )
        self.assertEqual(analysis["minimum_confirmatory_resampling_units"], 1)
        comparisons = {tuple(value) for value in protocol["confirmatory_comparisons"]}
        self.assertIn(("ordered_oracle", "ordered_timestamp_sham"), comparisons)
        self.assertIn(("reasoning_oracle", "reasoning_operator_sham"), comparisons)
        self.assertIn(("evidence_present", "random_position_mask"), comparisons)
        self.assertIn(("random_position_mask", "evidence_removed"), comparisons)
        with self.assertRaisesRegex(ValueError, "llm_revision"):
            validate_frozen_model_protocol(protocol)
        protocol["model"] = {
            **protocol["model"],
            "llm_revision": "a" * 40,
        }
        model = validate_frozen_model_protocol(protocol)
        self.assertTrue(model["llm_frozen"])
        self.assertEqual(model["llm_revision"], "a" * 40)
        self.assertEqual(model["dtype"], "bf16")
        self.assertEqual(model["max_length"], 4096)
        self.assertEqual(len(metadata["sha256"]), 64)

    def test_default_protocol_requires_a_real_data_lock_before_confirmatory_build(
        self,
    ) -> None:
        protocol, _ = load_protocol()
        with self.assertRaisesRegex(ValueError, "data.data_release_sha256"):
            validate_data_protocol(protocol)
        protocol["data"] = {
            **protocol["data"],
            "data_release_sha256": "b" * 64,
        }
        for dataset_index, dataset in enumerate(protocol["data"]["required_datasets"]):
            contract = protocol["data"]["coverage_contract"]["datasets"][dataset]
            contract["required_adapter_run_ids"] = [
                "adapter-run::" + format(dataset_index * 16 + run_index + 1, "064x")
                for run_index in range(contract["required_adapter_runs"])
            ]
        locked = validate_data_protocol(protocol)
        self.assertEqual(locked["data_release_sha256"], "b" * 64)
        self.assertEqual(len(locked["conditions_sha256"]), 64)
        self.assertEqual(
            set(locked["required_datasets"]), set(protocol["dataset_roles"])
        )

    def test_coverage_contract_rejects_partial_or_template_adapter_runs(self) -> None:
        protocol, _ = load_protocol()
        protocol["data"]["data_release_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "replace all templates"):
            validate_data_protocol(protocol)
        for dataset_index, dataset in enumerate(protocol["data"]["required_datasets"]):
            contract = protocol["data"]["coverage_contract"]["datasets"][dataset]
            contract["required_adapter_run_ids"] = [
                "adapter-run::" + format(dataset_index * 16 + run_index + 1, "064x")
                for run_index in range(contract["required_adapter_runs"])
            ]
        protocol["data"]["coverage_contract"]["datasets"]["mvp"][
            "required_adapter_run_ids"
        ] = ["adapter-run::" + "1" * 64]
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            validate_data_protocol(protocol)

    def test_release_coverage_rejects_missing_preregistered_partition(self) -> None:
        run_id = "adapter-run::" + "a" * 64
        contract = {
            "required_datasets": ["fixture"],
            "coverage_contract": {
                "datasets": {
                    "fixture": {
                        "required_adapter_run_ids": [run_id],
                        "required_splits": ["test"],
                        "required_information_families": ["temporal_order"],
                        "required_question_families": [
                            "fixture:first",
                            "fixture:second",
                        ],
                        "exact_question_family_set": True,
                        "required_source_roles": ["annotations"],
                        "minimum_records": 2,
                        "minimum_records_with_evidence": 0,
                        "minimum_records_with_safe_oracles": 0,
                    }
                }
            },
        }
        lock = {
            "adapter_runs": [
                {
                    "dataset": "fixture",
                    "adapter_run_id": run_id,
                    "canonical_split": "test",
                    "source_artifacts": [
                        {
                            "role": "annotations",
                            "relative_path": "release.json",
                            "sha256": "b" * 64,
                            "size_bytes": 1,
                        }
                    ],
                }
            ]
        }
        rows = [
            {
                "id": "one",
                "diagnostic": {
                    "dataset": "fixture",
                    "split": "test",
                    "information_family": "temporal_order",
                    "question_family": "fixture:first",
                    "evidence_spans": [],
                    "oracles": {},
                },
            }
        ]
        with self.assertRaisesRegex(ValueError, "question-family mismatch"):
            validate_release_coverage(contract, lock, rows)
        rows.append(
            {
                "id": "two",
                "diagnostic": {
                    **rows[0]["diagnostic"],
                    "question_family": "fixture:second",
                },
            }
        )
        self.assertTrue(validate_release_coverage(contract, lock, rows)["valid"])

    def test_frozen_flags_are_enforced(self) -> None:
        protocol, _ = load_protocol()
        protocol["model"] = {**protocol["model"], "llm_frozen": False}
        with self.assertRaisesRegex(ValueError, "llm_frozen"):
            validate_frozen_model_protocol(protocol)

    def test_score_cli_defers_locked_values_to_protocol(self) -> None:
        args = parse_score_args(
            [
                "--trials",
                "trials.jsonl",
                "--out",
                "predictions.jsonl",
                "--projector-ckpt",
                "projector.pt",
                "--projector-metadata",
                "metadata.json",
            ]
        )
        self.assertIsNone(args.llm_id)
        self.assertIsNone(args.dtype)
        self.assertIsNone(args.max_length)
        self.assertIsNone(args.overflow_policy)

    def test_sharding_is_disjoint_and_complete(self) -> None:
        rows = [{"id": f"item-{index}"} for index in range(100)]
        shards = [
            _shard_records(rows, shard_count=7, shard_index=index) for index in range(7)
        ]
        flattened = [row["id"] for shard in shards for row in shard]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {row["id"] for row in rows})
        self.assertEqual(
            shards,
            [
                _shard_records(rows, shard_count=7, shard_index=index)
                for index in range(7)
            ],
        )

    def test_resampling_families_never_cross_shards(self) -> None:
        rows = [
            {
                "id": f"scene-a-question-{index}",
                "diagnostic": {
                    "resampling_unit_id": "clevrer:scene:10000",
                    "pair_role": "standalone",
                    "independent_unit_id": f"clevrer:10000:{index}",
                },
            }
            for index in range(12)
        ] + [
            {
                "id": f"scene-b-question-{index}",
                "diagnostic": {
                    "resampling_unit_id": "clevrer:scene:10001",
                    "pair_role": "standalone",
                    "independent_unit_id": f"clevrer:10001:{index}",
                },
            }
            for index in range(9)
        ]
        shards = [
            _shard_records(rows, shard_count=5, shard_index=index) for index in range(5)
        ]
        family_shards: dict[str, set[int]] = {}
        for shard_index, shard in enumerate(shards):
            for row in shard:
                family = str(row["diagnostic"]["resampling_unit_id"])
                family_shards.setdefault(family, set()).add(shard_index)
        self.assertEqual(
            {family: len(indices) for family, indices in family_shards.items()},
            {"clevrer:scene:10000": 1, "clevrer:scene:10001": 1},
        )

    def test_legacy_pair_fallback_keeps_pair_members_together(self) -> None:
        rows = [
            {
                "id": f"member-{role}",
                "diagnostic": {"pair_id": "pair-family", "pair_role": role},
            }
            for role in ("original", "counterfactual")
        ]
        memberships = [
            len(_shard_records(rows, shard_count=7, shard_index=index))
            for index in range(7)
        ]
        self.assertEqual(sorted(memberships), [0, 0, 0, 0, 0, 0, 2])


if __name__ == "__main__":
    unittest.main()
