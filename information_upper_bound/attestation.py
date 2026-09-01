"""Validation for trial-build attestations carried inside every trial row."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .integrity import canonical_sha256
from .protocol import (
    protocol_section,
    trial_build_protocol_sha256,
    validate_data_protocol,
)


TRIAL_BUILD_ATTESTATION_SCHEMA_VERSION = (
    "information_upper_bound.trial_build_attestation.v2"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_trial_build_attestation(
    row: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any] | None = None,
    require_confirmatory: bool = True,
) -> dict[str, Any]:
    """Authenticate build mode, data release, configs, and sampling for one trial."""

    raw = row.get("trial_build_attestation")
    if not isinstance(raw, Mapping):
        raise ValueError("trial has no trial_build_attestation object")
    attestation = dict(raw)
    if attestation.get("schema_version") != TRIAL_BUILD_ATTESTATION_SCHEMA_VERSION:
        raise ValueError("trial has an unsupported trial-build attestation schema")
    declared_digest = str(attestation.pop("attestation_sha256", "")).lower()
    if _SHA256.fullmatch(declared_digest) is None:
        raise ValueError("trial-build attestation has no valid attestation_sha256")
    if canonical_sha256(attestation) != declared_digest:
        raise ValueError("trial-build attestation digest mismatch")
    expected_fields = {
        "schema_version",
        "mode",
        "data_release_sha256",
        "condition_config_sha256",
        "trial_build_protocol_sha256",
        "sampling",
    }
    if set(attestation) != expected_fields:
        raise ValueError(
            "trial-build attestation fields are incomplete or unknown: "
            f"missing={sorted(expected_fields - set(attestation))}, "
            f"extra={sorted(set(attestation) - expected_fields)}"
        )
    for field in ("condition_config_sha256", "trial_build_protocol_sha256"):
        if _SHA256.fullmatch(str(attestation.get(field, "")).lower()) is None:
            raise ValueError(f"trial-build attestation has no valid {field}")
    mode = str(attestation.get("mode", "")).strip()
    if mode not in {"confirmatory", "development"}:
        raise ValueError(
            "trial-build attestation mode must be confirmatory or development"
        )
    if require_confirmatory and mode != "confirmatory":
        raise ValueError(
            "development trial manifests are forbidden in confirmatory execution"
        )

    release = attestation.get("data_release_sha256")
    row_release = row.get("data_release_sha256")
    if release != row_release:
        raise ValueError("trial data_release_sha256 differs from its build attestation")
    if mode == "confirmatory" and _SHA256.fullmatch(str(release or "").lower()) is None:
        raise ValueError("confirmatory trial has no valid data_release_sha256")

    sampling = attestation.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("trial-build attestation sampling must be an object")
    if set(sampling) != {"seed", "option_permutations", "trial_shards"}:
        raise ValueError(
            "trial-build attestation sampling fields are incomplete or unknown"
        )

    if protocol is not None:
        sampling_protocol = protocol_section(protocol, "sampling")
        expected_sampling = {
            "seed": int(sampling_protocol["seed"]),
            "option_permutations": sampling_protocol["option_permutations"],
            "trial_shards": int(sampling_protocol["trial_shards"]),
        }
        if dict(sampling) != expected_sampling:
            raise ValueError("trial-build sampling differs from the locked protocol")
        expected_build_protocol_sha256 = trial_build_protocol_sha256(protocol)
        if str(attestation["trial_build_protocol_sha256"]).lower() != (
            expected_build_protocol_sha256
        ):
            raise ValueError(
                "trial was built with different preregistered protocol content"
            )
        if require_confirmatory:
            data_protocol = validate_data_protocol(protocol)
            if release != data_protocol["data_release_sha256"]:
                raise ValueError("trial data release differs from the locked protocol")
            if (
                str(attestation.get("condition_config_sha256", "")).lower()
                != data_protocol["conditions_sha256"]
            ):
                raise ValueError(
                    "trial condition matrix differs from the locked protocol"
                )
    return {**attestation, "attestation_sha256": declared_digest}
