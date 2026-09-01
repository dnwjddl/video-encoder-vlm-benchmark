from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from PIL import Image

from information_upper_bound.integrity import (
    RESULT_DESIGN_FIELDS,
    canonical_sha256,
    decoded_frames_identity,
    feature_artifact_root,
    resolved_pretrained_identity,
    scored_result_sha256,
    tensor_identity,
)
from information_upper_bound.extract_features import (
    extraction_pipeline_identity,
    main as extract_main,
)
from information_upper_bound.protocol import (
    validate_dataset_roles,
    validate_locked_projector_protocol,
)
from information_upper_bound.run import (
    _validate_existing_output,
    _validate_existing_output_signature,
    _validate_evaluation_feature_lock,
    _validate_score_output_paths,
)
from information_upper_bound.io import sha256_file, write_jsonl


class _Config:
    _commit_hash = None


class _Model:
    config = _Config()


class _Encoder:
    def __init__(self, source: Path) -> None:
        self.pretrained_source = str(source)
        self.model = _Model()
        self.processor = None
        self.dtype = torch.bfloat16


class IntegrityTests(unittest.TestCase):
    def test_default_confirmatory_extraction_requires_data_release_lock(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires --data-lock"):
            extract_main(
                [
                    "--manifest",
                    "unused.jsonl",
                    "--encoder",
                    "unused",
                    "--encoder-config",
                    "unused.yaml",
                    "--out-dir",
                    "unused",
                ]
            )

    def test_confirmatory_extraction_authenticates_trial_media_before_model_load(
        self,
    ) -> None:
        release = "a" * 64
        protocol = {"sampling": {"seed": 42}}
        trial = {"id": "trial::placeholder", "visual_spec": {}}
        attestation = {
            "attestation_sha256": "b" * 64,
            "data_release_sha256": release,
        }
        with (
            patch(
                "information_upper_bound.extract_features.load_protocol",
                return_value=(protocol, {"sha256": "c" * 64}),
            ),
            patch(
                "information_upper_bound.extract_features.validate_data_protocol",
                return_value={"data_release_sha256": release},
            ),
            patch(
                "information_upper_bound.extract_features.validate_trial_matrix_closure",
                side_effect=ValueError("trial media lock data release differs"),
            ) as matrix_closure,
        ):
            with self.assertRaisesRegex(ValueError, "media lock data release differs"):
                extract_main(
                    [
                        "--manifest",
                        "trials.jsonl",
                        "--data-lock",
                        "release.lock.json",
                        "--encoder",
                        "unused",
                        "--encoder-config",
                        "unused.yaml",
                        "--out-dir",
                        "unused",
                    ]
                )
        self.assertEqual(trial["id"], "trial::placeholder")
        self.assertEqual(attestation["data_release_sha256"], release)
        matrix_closure.assert_called_once()

    def test_development_extraction_can_skip_confirmatory_media_hash_policy(
        self,
    ) -> None:
        protocol = {
            "sampling": {
                "seed": 42,
                "require_media_sha256": True,
                "require_strong_encoder_identity": True,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "information_upper_bound.extract_features.load_protocol",
                    return_value=(protocol, {"sha256": "c" * 64}),
                ),
                patch(
                    "information_upper_bound.extract_features.iter_jsonl",
                    side_effect=[iter([{"id": "base"}]), iter([{"id": "base"}])],
                ),
                patch(
                    "information_upper_bound.extract_features.resolve_encoder",
                    side_effect=RuntimeError("reached encoder resolution"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "reached encoder resolution"):
                    extract_main(
                        [
                            "--manifest",
                            "base.jsonl",
                            "--development",
                            "--encoder",
                            "unused",
                            "--encoder-config",
                            "unused.yaml",
                            "--out-dir",
                            str(Path(directory) / "features"),
                        ]
                    )

    def test_extraction_enforces_optional_locked_encoder_identity(self) -> None:
        protocol = {
            "model": {
                "visual_encoder_name": "encoder-a",
                "visual_encoder_id": "owner/expected",
                "visual_encoder_revision": "a" * 40,
            },
            "sampling": {"seed": 42},
        }
        resolved = SimpleNamespace(
            model_id="owner/different",
            revision="a" * 40,
        )
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch(
                    "information_upper_bound.extract_features.load_protocol",
                    return_value=(protocol, {"sha256": "c" * 64}),
                ),
                patch(
                    "information_upper_bound.extract_features.iter_jsonl",
                    return_value=iter([{"id": "base"}]),
                ),
                patch(
                    "information_upper_bound.extract_features.resolve_encoder",
                    return_value=resolved,
                ),
            ):
                with self.assertRaisesRegex(ValueError, "visual_encoder_id"):
                    extract_main(
                        [
                            "--manifest",
                            "base.jsonl",
                            "--development",
                            "--encoder",
                            "encoder-a",
                            "--encoder-config",
                            "unused.yaml",
                            "--out-dir",
                            str(Path(directory) / "features"),
                        ]
                    )

    def test_extraction_refuses_implicit_output_replacement_and_input_alias(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out_dir = root / "features"
            out_dir.mkdir()
            (out_dir / "index.jsonl").write_text("old\n", encoding="utf-8")
            common = [
                "--manifest",
                str(root / "manifest.jsonl"),
                "--development",
                "--encoder",
                "unused",
                "--encoder-config",
                str(root / "encoders.yaml"),
                "--out-dir",
                str(out_dir),
            ]
            with self.assertRaisesRegex(FileExistsError, "pass --overwrite"):
                extract_main(common)

            alias_args = [
                "--manifest",
                str(out_dir / "index.jsonl"),
                "--development",
                "--encoder",
                "unused",
                "--encoder-config",
                str(root / "encoders.yaml"),
                "--out-dir",
                str(out_dir),
                "--overwrite",
            ]
            with self.assertRaisesRegex(ValueError, "aliases an input"):
                extract_main(alias_args)

    def test_local_pretrained_identity_binds_file_bytes_not_absolute_root(self) -> None:
        with (
            tempfile.TemporaryDirectory() as first_dir,
            tempfile.TemporaryDirectory() as second_dir,
        ):
            first = Path(first_dir) / "model"
            second = Path(second_dir) / "renamed"
            first.mkdir()
            second.mkdir()
            (first / "weights.bin").write_bytes(b"same-weights")
            (second / "weights.bin").write_bytes(b"same-weights")
            first_identity = resolved_pretrained_identity(
                requested_id=str(first), resolved_source=first, model=_Model()
            )
            second_identity = resolved_pretrained_identity(
                requested_id=str(second), resolved_source=second, model=_Model()
            )
            self.assertEqual(
                first_identity["source"]["tree_sha256"],
                second_identity["source"]["tree_sha256"],
            )
            self.assertEqual(
                first_identity["identity_sha256"], second_identity["identity_sha256"]
            )
            (second / "weights.bin").write_bytes(b"changed-weights")
            changed = resolved_pretrained_identity(
                requested_id=str(second), resolved_source=second, model=_Model()
            )
            self.assertNotEqual(
                first_identity["source"]["tree_sha256"],
                changed["source"]["tree_sha256"],
            )

    def test_encoder_revision_must_match_resolved_weight_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "encoder"
            source.mkdir()
            (source / "weights.bin").write_bytes(b"encoder-weights")
            encoder = _Encoder(source)
            base_config = {"model_id": str(source), "revision": None}
            identity = extraction_pipeline_identity(
                encoder=encoder,
                encoder_config=base_config,
                requested_dtype="bf16",
                requested_backend="decord",
            )
            revision = identity["encoder_pretrained_identity"]["source"]["tree_sha256"]
            pinned = extraction_pipeline_identity(
                encoder=encoder,
                encoder_config={**base_config, "revision": revision},
                requested_dtype="bf16",
                requested_backend="decord",
            )
            self.assertEqual(
                pinned["encoder_pretrained_identity"]["source"]["tree_sha256"], revision
            )
            with self.assertRaisesRegex(ValueError, "does not match --model-revision"):
                extraction_pipeline_identity(
                    encoder=encoder,
                    encoder_config={**base_config, "revision": "f" * 64},
                    requested_dtype="bf16",
                    requested_backend="decord",
                )

    def test_tensor_digest_binds_shape_dtype_and_values(self) -> None:
        base = tensor_identity(torch.arange(4, dtype=torch.float32).reshape(2, 2))
        changed_shape = tensor_identity(
            torch.arange(4, dtype=torch.float32).reshape(1, 4)
        )
        changed_dtype = tensor_identity(
            torch.arange(4, dtype=torch.float64).reshape(2, 2)
        )
        changed_value = tensor_identity(torch.tensor([[0.0, 1.0], [2.0, 4.0]]))
        self.assertNotEqual(base["sha256"], changed_shape["sha256"])
        self.assertNotEqual(base["sha256"], changed_dtype["sha256"])
        self.assertNotEqual(base["sha256"], changed_value["sha256"])

    def test_decoded_pixel_digest_binds_frame_order(self) -> None:
        red = Image.new("RGB", (2, 2), (255, 0, 0))
        blue = Image.new("RGB", (2, 2), (0, 0, 255))
        forward = decoded_frames_identity([red, blue])
        reverse = decoded_frames_identity([blue, red])
        self.assertNotEqual(forward["sha256"], reverse["sha256"])

    def test_feature_artifact_root_is_order_and_path_independent(self) -> None:
        def row(visual_id: str, path: str) -> dict:
            return {
                "visual_id": visual_id,
                "feature_content_hash": visual_id * 8,
                "feature_file_sha256": "a" * 64,
                "feature_tensor_identity": {
                    "dtype": "torch.float32",
                    "shape": [2, 2],
                    "numel": 4,
                    "sha256": "b" * 64,
                },
                "feature_artifact_identity_sha256": "c" * 64,
                "feature_path": path,
            }

        first = [row("one", "/machine/a.pt"), row("two", "/machine/b.pt")]
        second = [row("two", "/other/b.pt"), row("one", "/other/a.pt")]
        self.assertEqual(feature_artifact_root(first), feature_artifact_root(second))

    def test_locked_projector_authenticates_training_provenance(self) -> None:
        names = (
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
        values = {name: f"{index + 1:x}" * 64 for index, name in enumerate(names)}
        values["training_data_release_sha256"] = "e" * 64
        closure_payload = {
            "schema_version": "information_upper_bound.trial_matrix_closure.v1",
            "status": "exact",
            "trial_set_root_sha256": values["evaluation_trial_set_root_sha256"],
            "trial_count": 3,
        }
        values["evaluation_trial_matrix_closure_sha256"] = canonical_sha256(
            closure_payload
        )
        values.update(
            {
                "training_dtype": "bf16",
                "training_max_length": 4096,
                "training_seed": 42,
                "evaluation_trial_count": 3,
            }
        )
        protocol = {"projector": values}
        metadata = {
            name: value
            for name, value in values.items()
            if name
            not in {
                "checkpoint_sha256",
                "metadata_sha256",
                "training_data_release_sha256",
            }
        }
        metadata.update(
            {
                "dtype": "bf16",
                "max_length": 4096,
                "seed": 42,
                "evaluation_trial_matrix_closure": {
                    **closure_payload,
                    "closure_sha256": values["evaluation_trial_matrix_closure_sha256"],
                },
                "training_data_lock": {
                    "data_release_sha256": values["training_data_release_sha256"]
                },
            }
        )
        locked = validate_locked_projector_protocol(
            protocol,
            checkpoint_sha256=values["checkpoint_sha256"],
            metadata_sha256=values["metadata_sha256"],
            projector_metadata=metadata,
        )
        self.assertEqual(locked, values)
        metadata["training_manifest_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_locked_projector_protocol(
                protocol,
                checkpoint_sha256=values["checkpoint_sha256"],
                metadata_sha256=values["metadata_sha256"],
                projector_metadata=metadata,
            )
        metadata["training_manifest_sha256"] = values["training_manifest_sha256"]
        protocol["projector"]["evaluation_trial_count"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            validate_locked_projector_protocol(
                protocol,
                checkpoint_sha256=values["checkpoint_sha256"],
                metadata_sha256=values["metadata_sha256"],
                projector_metadata=metadata,
            )

    def test_evaluation_feature_files_are_bound_to_the_final_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "features.metadata.json"
            metadata_path.write_text("{}", encoding="utf-8")
            store = SimpleNamespace(
                index_sha256="a" * 64,
                artifact_root_sha256="b" * 64,
            )
            locked = {
                "evaluation_feature_index_sha256": "a" * 64,
                "evaluation_feature_metadata_sha256": sha256_file(metadata_path),
                "evaluation_feature_artifact_root_sha256": "b" * 64,
                "evaluation_trial_matrix_closure_sha256": "d" * 64,
                "evaluation_trial_set_root_sha256": "e" * 64,
                "evaluation_trial_count": 2,
            }
            closure_payload = {
                "schema_version": "information_upper_bound.trial_matrix_closure.v1",
                "status": "exact",
                "trial_set_root_sha256": "e" * 64,
                "trial_count": 2,
            }
            locked["evaluation_trial_matrix_closure_sha256"] = canonical_sha256(
                closure_payload
            )
            feature_metadata = {
                "manifest_sha256": "c" * 64,
                "trial_matrix_closure": {
                    **closure_payload,
                    "closure_sha256": locked["evaluation_trial_matrix_closure_sha256"],
                },
                "trial_matrix_closure_sha256": locked[
                    "evaluation_trial_matrix_closure_sha256"
                ],
            }
            _validate_evaluation_feature_lock(
                locked_projector=locked,
                feature_store=store,
                feature_metadata=feature_metadata,
                feature_metadata_path=metadata_path,
                # Raw JSONL bytes may differ after row reordering or media
                # remounting; the portable closure is authoritative.
                trials_manifest_sha256="f" * 64,
            )
            store.index_sha256 = "d" * 64
            with self.assertRaisesRegex(ValueError, "evaluation features"):
                _validate_evaluation_feature_lock(
                    locked_projector=locked,
                    feature_store=store,
                    feature_metadata=feature_metadata,
                    feature_metadata_path=metadata_path,
                    trials_manifest_sha256="f" * 64,
                )

    def test_dataset_role_validation_rejects_unknown_information_family(self) -> None:
        valid = {
            "dataset_roles": {
                "example": {
                    "information_families": ["temporal_order"],
                    "primary_use": "temporal control",
                }
            }
        }
        self.assertIn("example", validate_dataset_roles(valid))
        valid["dataset_roles"]["example"]["information_families"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "unknown information families"):
            validate_dataset_roles(valid)

    def test_resume_rejects_row_from_another_feature_root_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            write_jsonl(
                output,
                [
                    {
                        "trial_id": "trial::one",
                        "scoring_run_signature_sha256": "a" * 64,
                    }
                ],
            )
            _validate_existing_output_signature(output, "a" * 64)
            with self.assertRaisesRegex(ValueError, "cannot resume"):
                _validate_existing_output_signature(output, "b" * 64)

    def test_resume_authenticates_manifest_design_signatures_and_result_digest(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "predictions.jsonl"
            run_sha = "a" * 64
            global_sha = "b" * 64
            design = {name: None for name in RESULT_DESIGN_FIELDS}
            design.update(
                {
                    "trial_id": "trial::authenticated",
                    "base_id": "base",
                    "trial_content_sha256": "c" * 64,
                    "data_release_sha256": "d" * 64,
                    "choices": ["no", "yes"],
                    "answer": "A",
                    "answer_text": "no",
                }
            )
            row = {
                **design,
                "prediction": "A",
                "prediction_text": "no",
                "correct": True,
                "choice_nll": {"A": 0.1, "B": 1.1},
                "choice_probability": {"A": 0.7310586, "B": 0.2689414},
                "gold_nll": 0.1,
                "best_distractor_nll": 1.1,
                "gold_margin": 1.0,
                "scoring_global_signature_sha256": global_sha,
                "scoring_run_signature_sha256": run_sha,
            }
            row["result_content_sha256"] = scored_result_sha256(row)
            write_jsonl(output, [row])
            completed = _validate_existing_output(
                output,
                expected_designs={"trial::authenticated": design},
                run_signature_sha256=run_sha,
                global_signature_sha256=global_sha,
            )
            self.assertEqual(completed, {"trial::authenticated"})

            row["gold_margin"] = 2.0
            write_jsonl(output, [row])
            with self.assertRaisesRegex(ValueError, "invalid result digest"):
                _validate_existing_output(
                    output,
                    expected_designs={"trial::authenticated": design},
                    run_signature_sha256=run_sha,
                    global_signature_sha256=global_sha,
                )

            row["result_content_sha256"] = scored_result_sha256(row)
            changed_design = {**design, "answer": "B"}
            write_jsonl(output, [row])
            with self.assertRaisesRegex(
                ValueError, "differs from the current trial manifest"
            ):
                _validate_existing_output(
                    output,
                    expected_designs={"trial::authenticated": changed_design},
                    run_signature_sha256=run_sha,
                    global_signature_sha256=global_sha,
                )

    def test_score_output_and_sidecar_cannot_alias_any_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {
                "trials_path": root / "trials.jsonl",
                "projector_checkpoint": root / "projector.pt",
                "projector_metadata_path": root / "projector.json",
                "feature_index_path": root / "features.jsonl",
                "feature_metadata_path": root / "features.metadata.json",
                "protocol_config_path": root / "protocol.yaml",
            }
            with self.assertRaisesRegex(ValueError, "aliases trial manifest"):
                _validate_score_output_paths(
                    output_path=common["trials_path"],
                    metadata_path=root / "sidecar.json",
                    **common,
                )
            output = root / "predictions.jsonl"
            with self.assertRaisesRegex(ValueError, "aliases protocol config"):
                _validate_score_output_paths(
                    output_path=output,
                    metadata_path=output.with_suffix(".jsonl.metadata.json"),
                    **{
                        **common,
                        "protocol_config_path": output.with_suffix(
                            ".jsonl.metadata.json"
                        ),
                    },
                )


if __name__ == "__main__":
    unittest.main()
