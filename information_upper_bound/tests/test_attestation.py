from __future__ import annotations

import unittest

from information_upper_bound.attestation import validate_trial_build_attestation
from information_upper_bound.integrity import canonical_sha256
from information_upper_bound.protocol import trial_build_protocol_sha256


def _row(mode: str = "confirmatory") -> dict:
    release = "a" * 64 if mode == "confirmatory" else None
    payload = {
        "schema_version": "information_upper_bound.trial_build_attestation.v2",
        "mode": mode,
        "data_release_sha256": release,
        "condition_config_sha256": "b" * 64,
        "trial_build_protocol_sha256": "c" * 64,
        "sampling": {"seed": 42, "option_permutations": "all", "trial_shards": 1},
    }
    return {
        "data_release_sha256": release,
        "trial_build_attestation": {
            **payload,
            "attestation_sha256": canonical_sha256(payload),
        },
    }


class TrialBuildAttestationTests(unittest.TestCase):
    def test_valid_confirmatory_attestation(self) -> None:
        result = validate_trial_build_attestation(_row())
        self.assertEqual(result["mode"], "confirmatory")

    def test_development_attestation_is_rejected_by_confirmatory_consumer(self) -> None:
        with self.assertRaisesRegex(ValueError, "development trial"):
            validate_trial_build_attestation(_row("development"))
        self.assertEqual(
            validate_trial_build_attestation(
                _row("development"), require_confirmatory=False
            )["mode"],
            "development",
        )

    def test_attestation_mutation_is_rejected(self) -> None:
        row = _row()
        row["trial_build_attestation"]["sampling"]["seed"] = 7
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            validate_trial_build_attestation(row)

    def test_row_release_must_match_attestation(self) -> None:
        row = _row()
        row["data_release_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_trial_build_attestation(row)

    def test_late_bound_projector_lock_does_not_create_trial_hash_cycle(self) -> None:
        protocol = {
            "schema_version": "1.0",
            "model": {"llm_id": "frozen/test"},
            "sampling": {
                "seed": 42,
                "option_permutations": "all",
                "trial_shards": 1,
            },
            "projector": {"metadata_sha256": "unavailable-before-trials"},
        }
        row = _row()
        payload = dict(row["trial_build_attestation"])
        payload.pop("attestation_sha256")
        payload["trial_build_protocol_sha256"] = trial_build_protocol_sha256(protocol)
        row["trial_build_attestation"] = {
            **payload,
            "attestation_sha256": canonical_sha256(payload),
        }
        validate_trial_build_attestation(
            row, protocol=protocol, require_confirmatory=False
        )

        finalized = {
            **protocol,
            "projector": {"metadata_sha256": "a" * 64},
        }
        validate_trial_build_attestation(
            row, protocol=finalized, require_confirmatory=False
        )
        changed_design = {**finalized, "model": {"llm_id": "different/model"}}
        with self.assertRaisesRegex(ValueError, "preregistered protocol content"):
            validate_trial_build_attestation(
                row, protocol=changed_design, require_confirmatory=False
            )


if __name__ == "__main__":
    unittest.main()
