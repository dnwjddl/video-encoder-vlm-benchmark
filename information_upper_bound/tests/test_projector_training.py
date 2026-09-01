from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import torch

from information_upper_bound.io import write_jsonl
from information_upper_bound.projector_training import FeatureTextDataset
from information_upper_bound.split_integrity import audit_projector_split_disjointness
from information_upper_bound.train_projector import (
    _validate_strict_output_directory,
    _validate_evaluation_visual_coverage,
    parse_args,
    save_checkpoint,
    strict_information_upper_bound_mode,
)


class ProjectorTrainingIntegrityTests(unittest.TestCase):
    @staticmethod
    def _row(record_id: str, unit: str, media_path: Path, **extra) -> dict:
        return {
            "id": record_id,
            "media_path": str(media_path),
            "question": "What happened?",
            "choices": ["open", "close"],
            "answer": "A",
            "diagnostic": {"resampling_unit_id": unit},
            **extra,
        }

    def test_split_audit_rejects_family_and_content_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_media = root / "train.mp4"
            eval_media = root / "eval.mp4"
            train_media.write_bytes(b"training-video")
            eval_media.write_bytes(b"evaluation-video")
            train = root / "train.jsonl"
            evaluation = root / "eval.jsonl"
            write_jsonl(train, [self._row("train", "shared-family", train_media)])
            write_jsonl(evaluation, [self._row("eval", "shared-family", eval_media)])
            with self.assertRaisesRegex(ValueError, "resampling_unit_id"):
                audit_projector_split_disjointness(train, evaluation)

            write_jsonl(evaluation, [self._row("eval", "held-out-family", train_media)])
            with self.assertRaisesRegex(ValueError, "source-media SHA256"):
                audit_projector_split_disjointness(train, evaluation)

            write_jsonl(evaluation, [self._row("eval", "held-out-family", eval_media)])
            report = audit_projector_split_disjointness(train, evaluation)
            self.assertEqual(report["overlapping_resampling_units"], 0)
            self.assertEqual(report["overlapping_media_sha256"], 0)

    def test_feature_dataset_joins_every_trial_by_shared_visual_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_path = root / "feature.pt"
            torch.save({"features": torch.ones(3, 4)}, feature_path)
            index = root / "index.jsonl"
            write_jsonl(
                index,
                [
                    {
                        "id": "visual::shared",
                        "visual_id": "visual::shared",
                        "feature_path": "feature.pt",
                    }
                ],
            )
            manifest = root / "manifest.jsonl"
            write_jsonl(
                manifest,
                [
                    {
                        "id": "trial::one",
                        "visual_id": "visual::shared",
                        "question": "First?",
                        "choices": ["yes", "no"],
                        "answer": "A",
                    },
                    {
                        "id": "trial::two",
                        "visual_id": "visual::shared",
                        "question": "Second?",
                        "choices": ["yes", "no"],
                        "answer": "B",
                    },
                ],
            )
            dataset = FeatureTextDataset(manifest, index)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                {dataset[index]["id"] for index in range(len(dataset))},
                {"trial::one", "trial::two"},
            )
            self.assertTrue(
                all(
                    tuple(dataset[index]["features"].shape) == (3, 4)
                    for index in range(2)
                )
            )
            with self.assertRaisesRegex(ValueError, "integrity fields"):
                FeatureTextDataset(manifest, index, require_integrity=True)

    def test_legacy_projector_cli_requires_no_strict_provenance_arguments(self) -> None:
        args = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--feature-index",
                "features.jsonl",
                "--out-dir",
                "checkpoints",
                "--encoder-name",
                "encoder",
            ]
        )
        self.assertFalse(strict_information_upper_bound_mode(args))
        self.assertIsNone(args.eval_manifest)
        self.assertIsNone(args.feature_metadata)
        self.assertIsNone(args.eval_feature_index)
        self.assertIsNone(args.eval_feature_metadata)

    def test_partial_strict_projector_cli_is_rejected(self) -> None:
        args = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--feature-index",
                "features.jsonl",
                "--feature-metadata",
                "metadata.json",
                "--out-dir",
                "checkpoints",
                "--encoder-name",
                "encoder",
            ]
        )
        with self.assertRaisesRegex(ValueError, "full provenance argument set"):
            strict_information_upper_bound_mode(args)

    def test_complete_strict_projector_cli_is_detected(self) -> None:
        args = parse_args(
            [
                "--manifest",
                "train.jsonl",
                "--feature-index",
                "train-features.jsonl",
                "--feature-metadata",
                "train-metadata.json",
                "--eval-manifest",
                "eval.jsonl",
                "--eval-feature-index",
                "eval-features.jsonl",
                "--eval-feature-metadata",
                "eval-metadata.json",
                "--eval-data-lock",
                "eval-data-lock.json",
                "--out-dir",
                "checkpoints",
                "--encoder-name",
                "encoder",
            ]
        )
        self.assertTrue(strict_information_upper_bound_mode(args))

    def test_strict_projector_refuses_implicit_output_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out_dir = root / "run"
            out_dir.mkdir()
            (out_dir / "metadata.json").write_text("old", encoding="utf-8")
            argv = [
                "--manifest",
                str(root / "train.jsonl"),
                "--feature-index",
                str(root / "train-features.jsonl"),
                "--feature-metadata",
                str(root / "train-metadata.json"),
                "--eval-manifest",
                str(root / "eval.jsonl"),
                "--eval-feature-index",
                str(root / "eval-features.jsonl"),
                "--eval-feature-metadata",
                str(root / "eval-metadata.json"),
                "--out-dir",
                str(out_dir),
                "--encoder-name",
                "encoder",
            ]
            args = parse_args(argv)
            with self.assertRaisesRegex(FileExistsError, "not empty"):
                _validate_strict_output_directory(args)
            overwrite = parse_args([*argv, "--overwrite"])
            _validate_strict_output_directory(overwrite)

    def test_strict_projector_rejects_metadata_output_input_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out_dir = root / "run"
            args = parse_args(
                [
                    "--manifest",
                    str(out_dir / "metadata.json"),
                    "--feature-index",
                    str(root / "train-features.jsonl"),
                    "--feature-metadata",
                    str(root / "train-metadata.json"),
                    "--eval-manifest",
                    str(root / "eval.jsonl"),
                    "--eval-feature-index",
                    str(root / "eval-features.jsonl"),
                    "--eval-feature-metadata",
                    str(root / "eval-metadata.json"),
                    "--out-dir",
                    str(out_dir),
                    "--encoder-name",
                    "encoder",
                ]
            )
            with self.assertRaisesRegex(ValueError, "aliases an authenticated input"):
                _validate_strict_output_directory(args)

    def test_evaluation_visual_coverage_must_be_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "eval.jsonl"
            write_jsonl(
                manifest,
                [
                    {"id": "text", "visual_id": None},
                    {"id": "one", "visual_id": "visual::one"},
                    {"id": "two", "visual_id": "visual::two"},
                    {"id": "two-repeat", "visual_id": "visual::two"},
                ],
            )
            report = _validate_evaluation_visual_coverage(
                manifest, {"visual::one", "visual::two"}
            )
            self.assertEqual(report["evaluation_rows"], 4)
            self.assertEqual(report["evaluation_unique_visual_ids"], 2)
            with self.assertRaisesRegex(ValueError, "does not exactly match"):
                _validate_evaluation_visual_coverage(
                    manifest, {"visual::one", "visual::extra"}
                )

    def test_protocol_lock_is_strict_only_and_publishes_eval_feature_hashes(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            projector = torch.nn.Linear(2, 2)
            legacy = root / "legacy"
            save_checkpoint(legacy, projector, 1, {"encoder_name": "encoder"})
            self.assertFalse(
                (legacy / "step_000001" / "protocol_projector_lock.json").exists()
            )

            strict = root / "strict"
            digest = "a" * 64
            metadata = {
                "projector_training_mode": "information_upper_bound_strict",
                "training_manifest_sha256": digest,
                "evaluation_manifest_sha256": digest,
                "training_feature_index_sha256": digest,
                "training_feature_artifact_root_sha256": digest,
                "training_feature_metadata_sha256": digest,
                "evaluation_feature_index_sha256": digest,
                "evaluation_feature_artifact_root_sha256": digest,
                "evaluation_feature_metadata_sha256": digest,
                "evaluation_trial_matrix_closure_sha256": digest,
                "evaluation_trial_set_root_sha256": digest,
                "evaluation_trial_count": 8,
                "encoder_extraction_pipeline_identity_sha256": digest,
                "llm_pretrained_identity_sha256": digest,
                "dtype": "bf16",
                "max_length": 4096,
                "seed": 42,
            }
            save_checkpoint(strict, projector, 2, metadata)
            lock_path = strict / "step_000002" / "protocol_projector_lock.json"
            lock = __import__("json").loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(
                lock["schema_version"],
                "information_upper_bound.projector_lock.v3",
            )
            self.assertEqual(lock["evaluation_feature_index_sha256"], digest)
            self.assertEqual(lock["evaluation_feature_artifact_root_sha256"], digest)
            self.assertEqual(lock["evaluation_feature_metadata_sha256"], digest)
            self.assertEqual(lock["evaluation_trial_matrix_closure_sha256"], digest)
            self.assertEqual(lock["evaluation_trial_set_root_sha256"], digest)
            self.assertEqual(lock["evaluation_trial_count"], 8)


if __name__ == "__main__":
    unittest.main()
