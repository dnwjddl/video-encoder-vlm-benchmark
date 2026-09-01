from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from information_upper_bound.data_lock import (
    ADAPTER_REPORT_SCHEMA_VERSION,
    ADAPTER_RUN_SCHEMA_VERSION,
    create_data_lock,
    main as data_lock_main,
    manifest_record_set_sha256,
    source_artifact_root_sha256,
    validate_data_lock,
    validate_trial_media_lock,
)
from information_upper_bound.attestation import TRIAL_BUILD_ATTESTATION_SCHEMA_VERSION
from information_upper_bound.conditions import (
    ConditionSpec,
    build_trials,
    trial_content_sha256,
)
from information_upper_bound.integrity import canonical_sha256
from information_upper_bound.io import sha256_file, write_json, write_jsonl
from information_upper_bound.protocol import trial_build_protocol_sha256
from information_upper_bound.schema import SCHEMA_VERSION
from information_upper_bound.trial_matrix import (
    validate_development_trial_matrix_closure,
    validate_trial_base_release,
)
from information_upper_bound.validate import validate_manifest


def _adapter_run_id(
    *,
    dataset: str,
    split: str,
    adapter_options: dict,
    source_artifacts: list[dict],
    record_ids: list[str],
) -> str:
    return "adapter-run::" + canonical_sha256(
        {
            "schema_version": ADAPTER_RUN_SCHEMA_VERSION,
            "dataset": dataset,
            "canonical_split": split,
            "adapter_options": adapter_options,
            "source_artifacts": source_artifacts,
            "record_ids": sorted(record_ids),
        }
    )


def _record(
    media_path: Path,
    *,
    record_id: str,
    adapter_run_id: str,
    annotation_path: Path,
    dataset: str = "fixture",
) -> dict:
    return {
        "id": record_id,
        "source": "official-fixture",
        "benchmark": "fixture",
        "task": "mcq",
        "media_type": "video",
        "media_path": str(media_path),
        "question": "What happens?",
        "choices": ["opens", "closes"],
        "answer": "A",
        "diagnostic": {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset,
            "split": "test",
            "information_family": "temporal_order",
            "question_family": "event_order",
            "reasoning_depth": 1,
            "pair_id": record_id,
            "pair_role": "standalone",
            "resampling_unit_id": "video-" + record_id,
            "adapter_run_id": adapter_run_id,
            "evidence_spans": [],
            "oracles": {
                "static_facts": [],
                "unordered_events": [],
                "ordered_events": [],
                "temporal_relations": [],
                "state_changes": [],
                "relations": [],
                "operator": None,
                "intermediate": [],
                "answer_derived": False,
            },
            "provenance": {
                "source_id": record_id,
                "annotation_file": str(annotation_path),
                "raw_location": str(annotation_path.parent / "raw-release"),
            },
        },
    }


class DataLockTests(unittest.TestCase):
    def _release(
        self,
        root: Path,
        *,
        run_rows: list[list[str]] | None = None,
        source_payloads: list[dict] | None = None,
    ) -> tuple[Path, list[Path], dict[str, Path], list[dict]]:
        run_rows = run_rows or [["official-row-1"]]
        source_payloads = source_payloads or [
            {"release": index + 1} for index in range(len(run_rows))
        ]
        rows: list[dict] = []
        reports: list[Path] = []
        media_by_id: dict[str, Path] = {}
        for run_index, record_ids in enumerate(run_rows):
            annotation = root / f"source_{run_index}.json"
            annotation.write_text(
                json.dumps(source_payloads[run_index], sort_keys=True), encoding="utf-8"
            )
            source_artifacts = [
                {
                    "role": "annotations",
                    "relative_path": annotation.name,
                    "sha256": sha256_file(annotation),
                    "size_bytes": annotation.stat().st_size,
                }
            ]
            adapter_options = {
                "include_track_geometry": False,
                "include_audio_oracles": False,
                "source_split": None,
                "task": f"run-{run_index}",
                "tasks": None,
                "category": None,
            }
            run_id = _adapter_run_id(
                dataset="fixture",
                split="test",
                adapter_options=adapter_options,
                source_artifacts=source_artifacts,
                record_ids=record_ids,
            )
            run_manifest_rows: list[dict] = []
            for record_id in record_ids:
                media = root / f"{record_id}.mp4"
                media.write_bytes(("media-bytes::" + record_id).encode("utf-8"))
                media_by_id[record_id] = media
                row = _record(
                    media,
                    record_id=record_id,
                    adapter_run_id=run_id,
                    annotation_path=annotation,
                )
                rows.append(row)
                run_manifest_rows.append(row)
            validation = validate_manifest(
                run_manifest_rows, require_media=True, strict_diagnostic=True
            )
            report = root / f"run_{run_index}.report.json"
            write_json(
                report,
                {
                    "schema_version": ADAPTER_REPORT_SCHEMA_VERSION,
                    "dataset": "fixture",
                    "adapter_run_id": run_id,
                    "records": len(run_manifest_rows),
                    "limited": False,
                    "limit": None,
                    "canonical_split": "test",
                    "require_media": True,
                    "confirmatory_eligible": True,
                    "debug_options": {
                        "allow_missing_media": False,
                        "allow_missing_grounding": False,
                        "allow_missing_cut_mapping": False,
                        "allow_uncut_cup_games": False,
                        "limited": False,
                    },
                    "adapter_options": adapter_options,
                    "manifest_record_set_sha256": manifest_record_set_sha256(
                        run_manifest_rows
                    ),
                    "source_artifacts": source_artifacts,
                    "source_artifact_root_sha256": source_artifact_root_sha256(
                        source_artifacts
                    ),
                    "source_checksums_sha256": {
                        str(annotation.resolve()): sha256_file(annotation)
                    },
                    "validation": validation,
                },
            )
            reports.append(report)
        manifest = root / "manifest.jsonl"
        write_jsonl(manifest, rows)
        return manifest, reports, media_by_id, rows

    def test_lock_authenticates_manifest_reports_sources_and_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, _media, _rows = self._release(root)
            value = create_data_lock(
                manifest_path=manifest, adapter_report_paths=reports
            )
            lock_path = root / "data_lock.json"
            write_json(lock_path, value)
            authenticated = validate_data_lock(
                lock_path,
                manifest_path=manifest,
                verify_sources=True,
                verify_media=True,
            )
            self.assertEqual(authenticated["records"], 1)
            self.assertEqual(value["datasets"], {"fixture": 1})
            self.assertEqual(authenticated["sha256"], value["data_release_sha256"])
            self.assertNotEqual(authenticated["sha256"], authenticated["file_sha256"])
            self.assertEqual(authenticated["adapter_runs"], value["adapter_runs"])

    def test_relocation_preserves_scientific_release_digest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            first = parent / "mount-a"
            second = parent / "mount-b"
            first.mkdir()
            second.mkdir()
            first_manifest, first_reports, _media, _rows = self._release(first)
            second_manifest, second_reports, _media, _rows = self._release(second)
            first_lock = create_data_lock(
                manifest_path=first_manifest, adapter_report_paths=first_reports
            )
            second_lock = create_data_lock(
                manifest_path=second_manifest, adapter_report_paths=second_reports
            )
            self.assertEqual(
                first_lock["manifest_semantic_record_set_sha256"],
                second_lock["manifest_semantic_record_set_sha256"],
            )
            self.assertEqual(
                first_lock["source_artifact_root_sha256"],
                second_lock["source_artifact_root_sha256"],
            )
            self.assertEqual(
                first_lock["media_binding_root_sha256"],
                second_lock["media_binding_root_sha256"],
            )
            self.assertEqual(
                first_lock["data_release_sha256"], second_lock["data_release_sha256"]
            )
            self.assertNotEqual(
                first_lock["lock_payload_sha256"], second_lock["lock_payload_sha256"]
            )

    def test_lock_rejects_debug_adapter_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, _media, _rows = self._release(root)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            report["confirmatory_eligible"] = False
            report["debug_options"]["allow_missing_media"] = True
            write_json(reports[0], report)
            with self.assertRaisesRegex(ValueError, "debug escape hatch"):
                create_data_lock(manifest_path=manifest, adapter_report_paths=reports)

    def test_lock_rejects_declared_confirmatory_eligibility_issues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, _media, _rows = self._release(root)
            report = json.loads(reports[0].read_text(encoding="utf-8"))
            report["confirmatory_eligibility_issues"] = ["noncanonical_split"]
            write_json(reports[0], report)
            with self.assertRaisesRegex(ValueError, "eligibility issues"):
                create_data_lock(manifest_path=manifest, adapter_report_paths=reports)

    def test_lock_rejects_source_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, _media, _rows = self._release(root)
            source = root / "source_0.json"
            source.write_text(json.dumps({"release": 999}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source changed"):
                create_data_lock(manifest_path=manifest, adapter_report_paths=reports)

    def test_media_mutation_is_rejected_when_media_verification_is_enabled(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, media_by_id, _rows = self._release(root)
            value = create_data_lock(
                manifest_path=manifest, adapter_report_paths=reports
            )
            lock_path = root / "data_lock.json"
            write_json(lock_path, value)
            media_by_id["official-row-1"].write_bytes(b"mutated-media")
            with self.assertRaisesRegex(ValueError, "media bytes"):
                validate_data_lock(lock_path, manifest_path=manifest, verify_media=True)

    def test_two_disjoint_reports_for_same_dataset_are_exactly_joined(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, _media, _rows = self._release(
                root, run_rows=[["row-a"], ["row-b"]]
            )
            lock = create_data_lock(
                manifest_path=manifest, adapter_report_paths=reports
            )
            self.assertEqual(lock["datasets"], {"fixture": 2})
            self.assertEqual(len(lock["adapter_runs"]), 2)
            with self.assertRaisesRegex(ValueError, "no strict report"):
                create_data_lock(
                    manifest_path=manifest, adapter_report_paths=reports[:1]
                )
            with self.assertRaisesRegex(ValueError, "duplicate/overlapping"):
                create_data_lock(
                    manifest_path=manifest,
                    adapter_report_paths=[reports[0], reports[0], reports[1]],
                )

    def test_wrong_report_group_content_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, _media, _rows = self._release(
                root, run_rows=[["row-a"], ["row-b"]]
            )
            first = json.loads(reports[0].read_text(encoding="utf-8"))
            second = json.loads(reports[1].read_text(encoding="utf-8"))
            second["manifest_record_set_sha256"] = first["manifest_record_set_sha256"]
            write_json(reports[1], second)
            with self.assertRaisesRegex(ValueError, "exact manifest group"):
                create_data_lock(manifest_path=manifest, adapter_report_paths=reports)

    def test_semantic_manifest_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, _media, rows = self._release(root)
            value = create_data_lock(
                manifest_path=manifest, adapter_report_paths=reports
            )
            lock_path = root / "data_lock.json"
            write_json(lock_path, value)
            changed = deepcopy(rows)
            changed[0]["question"] = "What happened before the door opened?"
            write_jsonl(manifest, changed)
            with self.assertRaisesRegex(ValueError, "semantic record content"):
                validate_data_lock(lock_path, manifest_path=manifest)

    def test_malformed_self_digested_lock_fails_structural_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, _media, _rows = self._release(root)
            lock = create_data_lock(
                manifest_path=manifest, adapter_report_paths=reports
            )
            lock["datasets"] = {"fixture": 123}
            lock["lock_payload_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in lock.items()
                    if key != "lock_payload_sha256"
                }
            )
            lock_path = root / "malformed-lock.json"
            write_json(lock_path, lock)
            with self.assertRaisesRegex(ValueError, "dataset counts"):
                validate_data_lock(lock_path, manifest_path=manifest)

    def test_expanded_trials_authenticate_locked_media_by_base_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, media_by_id, _rows = self._release(
                root, run_rows=[["row-a", "row-b"]]
            )
            lock = create_data_lock(
                manifest_path=manifest, adapter_report_paths=reports
            )
            lock_path = root / "data_lock.json"
            write_json(lock_path, lock)
            trials = [
                {
                    "id": f"trial-{record_id}-{dose}",
                    "base_id": record_id,
                    "data_release_sha256": lock["data_release_sha256"],
                    "diagnostic": {"dataset": "fixture"},
                    "visual_spec": {"media_path": str(media_by_id[record_id])},
                }
                for record_id in ("row-a", "row-b")
                for dose in (0, 1)
            ]
            verified = validate_trial_media_lock(lock_path, trials)
            self.assertEqual(
                verified["data_release_sha256"], lock["data_release_sha256"]
            )
            with self.assertRaisesRegex(ValueError, "base_id coverage"):
                validate_trial_media_lock(
                    lock_path,
                    [trial for trial in trials if trial["base_id"] == "row-a"],
                )
            media_by_id["row-b"].write_bytes(b"mutated-after-lock")
            with self.assertRaisesRegex(ValueError, "trial media bytes"):
                validate_trial_media_lock(lock_path, trials)

    def test_training_trials_authenticate_full_reconstructed_base_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, _media, rows = self._release(
                root, run_rows=[["row-a", "row-b"]]
            )
            lock = create_data_lock(
                manifest_path=manifest, adapter_report_paths=reports
            )
            lock_path = root / "data_lock.json"
            write_json(lock_path, lock)
            trials, _report = build_trials(
                rows,
                [
                    ConditionSpec(
                        name="full_video",
                        input_channel="visual",
                        visual_view="full",
                        doses=(0,),
                    )
                ],
                seed=42,
                option_permutations=1,
            )

            verified = validate_trial_base_release(trials, data_lock_path=lock_path)
            self.assertEqual(
                verified["manifest_semantic_record_set_sha256"],
                lock["manifest_semantic_record_set_sha256"],
            )
            self.assertEqual(verified["unit_summaries"], lock["unit_summaries"])

            changed_question = deepcopy(trials)
            changed_question[0]["question"] = "A substituted training question"
            with self.assertRaisesRegex(ValueError, "semantic record content"):
                validate_trial_base_release(changed_question, data_lock_path=lock_path)

            changed_membership = deepcopy(trials)
            changed_membership[0]["diagnostic"]["resampling_unit_id"] = (
                "substituted-unit"
            )
            with self.assertRaisesRegex(ValueError, "semantic record content"):
                validate_trial_base_release(
                    changed_membership, data_lock_path=lock_path
                )

    def test_development_training_matrix_has_exact_condition_closure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, _media, rows = self._release(
                root, run_rows=[["row-a", "row-b"]]
            )
            lock = create_data_lock(
                manifest_path=manifest, adapter_report_paths=reports
            )
            lock_path = root / "data_lock.json"
            write_json(lock_path, lock)
            conditions_path = root / "train_conditions.yaml"
            write_json(
                conditions_path,
                {
                    "schema_version": "1.0",
                    "options": {"seed": 42, "option_permutations": 1},
                    "conditions": [
                        {
                            "name": "full_video",
                            "input_channel": "visual",
                            "visual_view": "full",
                            "doses": [0],
                        }
                    ],
                },
            )
            protocol = {
                "schema_version": "1.0",
                "name": "training-closure-test",
                "sampling": {
                    "seed": 42,
                    "option_permutations": 1,
                    "trial_shards": 1,
                },
            }
            attestation_payload = {
                "schema_version": TRIAL_BUILD_ATTESTATION_SCHEMA_VERSION,
                "mode": "development",
                "data_release_sha256": lock["data_release_sha256"],
                "condition_config_sha256": sha256_file(conditions_path),
                "trial_build_protocol_sha256": trial_build_protocol_sha256(protocol),
                "sampling": dict(protocol["sampling"]),
            }
            attestation = {
                **attestation_payload,
                "attestation_sha256": canonical_sha256(attestation_payload),
            }
            trial_inputs = [
                {
                    **row,
                    "data_release_sha256": lock["data_release_sha256"],
                    "trial_build_attestation": deepcopy(attestation),
                }
                for row in rows
            ]
            trials, _report = build_trials(
                trial_inputs,
                [
                    ConditionSpec(
                        name="full_video",
                        input_channel="visual",
                        visual_view="full",
                        doses=(0,),
                    )
                ],
                seed=42,
                option_permutations=1,
            )
            authenticated = validate_development_trial_matrix_closure(
                trials,
                data_lock_path=lock_path,
                conditions_config_path=conditions_path,
                protocol=protocol,
            )
            closure = authenticated["closure"]
            self.assertEqual(closure["status"], "exact")
            self.assertEqual(closure["conditions"], ["full_video"])
            self.assertEqual(closure["trial_count"], len(rows))

            duplicated = [*trials, deepcopy(trials[0])]
            with self.assertRaisesRegex(ValueError, "duplicates trial_id"):
                validate_development_trial_matrix_closure(
                    duplicated,
                    data_lock_path=lock_path,
                    conditions_config_path=conditions_path,
                    protocol=protocol,
                )

            substituted_view = deepcopy(trials)
            substituted_view[0]["condition"]["visual_view"] = "single"
            substituted_view[0]["visual_spec"]["view"] = "single"
            changed_digest = trial_content_sha256(substituted_view[0])
            substituted_view[0]["trial_content_sha256"] = changed_digest
            substituted_view[0]["trial_id"] = f"trial::{changed_digest}"
            substituted_view[0]["id"] = f"trial::{changed_digest}"
            with self.assertRaisesRegex(ValueError, "exact deterministic"):
                validate_development_trial_matrix_closure(
                    substituted_view,
                    data_lock_path=lock_path,
                    conditions_config_path=conditions_path,
                    protocol=protocol,
                )

    def test_lock_data_output_cannot_alias_an_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, reports, _media, _rows = self._release(root)
            with self.assertRaisesRegex(ValueError, "must not alias"):
                data_lock_main(
                    [
                        "--manifest",
                        str(manifest),
                        "--adapter-report",
                        str(reports[0]),
                        "--out",
                        str(manifest),
                        "--overwrite",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
