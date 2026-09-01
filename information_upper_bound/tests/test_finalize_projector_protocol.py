from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from information_upper_bound.finalize_projector_protocol import (
    PROJECTOR_LOCK_SCHEMA_VERSION,
    main as finalize_main,
    merge_projector_lock,
)
from information_upper_bound.io import sha256_file
from information_upper_bound.integrity import canonical_sha256
from information_upper_bound.pilot_protocol import DEFAULT_TEMPLATE
from information_upper_bound.protocol import validate_locked_projector_protocol
from information_upper_bound.train_projector import save_checkpoint


class FinalizeProjectorProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.lock_payload = {
            "checkpoint_sha256": "a" * 64,
            "metadata_sha256": "b" * 64,
            "evaluation_trial_count": 42,
        }
        self.protocol = {
            "schema_version": "1.0",
            "model": {"llm_id": "example/model"},
            "data": {"data_release_sha256": "c" * 64},
            "projector": {
                "checkpoint_sha256": "REPLACE_CHECKPOINT",
                "metadata_sha256": "REPLACE_METADATA",
                "evaluation_trial_count": "REPLACE_COUNT",
            },
            "sampling": {"seed": 42},
            "dataset_roles": {
                "example": {
                    "information_families": ["temporal_order"],
                    "primary_use": "test fixture",
                }
            },
        }

    def test_merge_changes_only_projector_and_requires_exact_fields(self) -> None:
        lock = {"schema_version": PROJECTOR_LOCK_SCHEMA_VERSION, **self.lock_payload}
        merged = merge_projector_lock(self.protocol, lock)
        self.assertEqual(merged["projector"], self.lock_payload)
        self.assertEqual(
            {key: value for key, value in merged.items() if key != "projector"},
            {key: value for key, value in self.protocol.items() if key != "projector"},
        )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            merge_projector_lock(self.protocol, {**lock, "extra": "value"})

    def test_command_writes_new_authenticated_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.yaml"
            protocol_path.write_text(
                yaml.safe_dump(self.protocol, sort_keys=False), encoding="utf-8"
            )
            lock_path = root / "lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "schema_version": PROJECTOR_LOCK_SCHEMA_VERSION,
                        **self.lock_payload,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_path = root / "projector.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            metadata_path = root / "metadata.json"
            metadata_path.write_text("{}", encoding="utf-8")
            output_path = root / "protocol.final.yaml"
            validated_lock = dict(self.lock_payload)
            with (
                patch(
                    "information_upper_bound.finalize_projector_protocol.validate_data_protocol"
                ),
                patch(
                    "information_upper_bound.finalize_projector_protocol.validate_frozen_model_protocol"
                ),
                patch(
                    "information_upper_bound.finalize_projector_protocol.validate_locked_projector_protocol",
                    return_value=validated_lock,
                ) as validate_projector,
                redirect_stdout(io.StringIO()),
            ):
                result = finalize_main(
                    [
                        "--protocol",
                        str(protocol_path),
                        "--projector-lock",
                        str(lock_path),
                        "--projector-ckpt",
                        str(checkpoint_path),
                        "--projector-metadata",
                        str(metadata_path),
                        "--out",
                        str(output_path),
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(
                yaml.safe_load(output_path.read_text(encoding="utf-8"))["projector"],
                self.lock_payload,
            )
            self.assertEqual(validate_projector.call_count, 2)

    def test_failed_candidate_validation_does_not_publish_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.yaml"
            protocol_path.write_text(
                yaml.safe_dump(self.protocol, sort_keys=False), encoding="utf-8"
            )
            lock_path = root / "lock.json"
            lock_path.write_text(
                json.dumps(
                    {
                        "schema_version": PROJECTOR_LOCK_SCHEMA_VERSION,
                        **self.lock_payload,
                    }
                ),
                encoding="utf-8",
            )
            checkpoint_path = root / "projector.pt"
            checkpoint_path.write_bytes(b"checkpoint")
            metadata_path = root / "metadata.json"
            metadata_path.write_text("{}", encoding="utf-8")
            output_path = root / "protocol.final.yaml"
            output_path.write_text("previous-output\n", encoding="utf-8")
            with (
                patch(
                    "information_upper_bound.finalize_projector_protocol.validate_data_protocol"
                ),
                patch(
                    "information_upper_bound.finalize_projector_protocol.validate_frozen_model_protocol"
                ),
                patch(
                    "information_upper_bound.finalize_projector_protocol.validate_locked_projector_protocol",
                    side_effect=[self.lock_payload, RuntimeError("candidate rejected")],
                ),
                self.assertRaisesRegex(RuntimeError, "candidate rejected"),
            ):
                finalize_main(
                    [
                        "--protocol",
                        str(protocol_path),
                        "--projector-lock",
                        str(lock_path),
                        "--projector-ckpt",
                        str(checkpoint_path),
                        "--projector-metadata",
                        str(metadata_path),
                        "--out",
                        str(output_path),
                        "--overwrite",
                    ]
                )
            self.assertEqual(
                output_path.read_text(encoding="utf-8"), "previous-output\n"
            )

    def test_real_strict_lock_matches_clevrer_projector_template(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digests = iter(f"{index:064x}" for index in range(1, 20))
            evaluation_trial_set_root = next(digests)
            evaluation_trial_count = 468_000
            closure_payload = {
                "status": "exact",
                "trial_set_root_sha256": evaluation_trial_set_root,
                "trial_count": evaluation_trial_count,
            }
            evaluation_trial_matrix_closure = {
                **closure_payload,
                "closure_sha256": canonical_sha256(closure_payload),
            }
            metadata = {
                "projector_training_mode": "information_upper_bound_strict",
                "training_manifest_sha256": next(digests),
                "evaluation_manifest_sha256": next(digests),
                "training_feature_index_sha256": next(digests),
                "training_feature_artifact_root_sha256": next(digests),
                "training_feature_metadata_sha256": next(digests),
                "evaluation_feature_index_sha256": next(digests),
                "evaluation_feature_artifact_root_sha256": next(digests),
                "evaluation_feature_metadata_sha256": next(digests),
                "evaluation_trial_matrix_closure_sha256": (
                    evaluation_trial_matrix_closure["closure_sha256"]
                ),
                "evaluation_trial_set_root_sha256": evaluation_trial_set_root,
                "evaluation_trial_count": evaluation_trial_count,
                "evaluation_trial_matrix_closure": (evaluation_trial_matrix_closure),
                "encoder_extraction_pipeline_identity_sha256": next(digests),
                "llm_pretrained_identity_sha256": next(digests),
                "dtype": "bf16",
                "max_length": 4096,
                "seed": 42,
                "training_data_lock": {"data_release_sha256": next(digests)},
            }
            save_checkpoint(root, torch.nn.Linear(4, 8), 1, metadata)
            step = root / "step_000001"
            projector_lock = json.loads(
                (step / "protocol_projector_lock.json").read_text(encoding="utf-8")
            )
            template = yaml.safe_load(DEFAULT_TEMPLATE.read_text(encoding="utf-8"))
            merged = merge_projector_lock(template, projector_lock)
            locked = validate_locked_projector_protocol(
                merged,
                checkpoint_sha256=sha256_file(step / "projector.pt"),
                metadata_sha256=sha256_file(step / "metadata.json"),
                projector_metadata={**metadata, "step": 1},
            )
            self.assertEqual(locked["evaluation_trial_count"], 468_000)
            self.assertEqual(
                set(merged["projector"]),
                set(template["projector"]),
            )


if __name__ == "__main__":
    unittest.main()
