from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import importlib
import json
from pathlib import Path
import sys

from .attestation import TRIAL_BUILD_ATTESTATION_SCHEMA_VERSION
from .conditions import DEFAULT_CONDITION_PATH, load_condition_config, stream_trials
from .data_lock import validate_data_lock
from .integrity import canonical_sha256
from .io import read_jsonl, sha256_file, write_json, write_jsonl
from .protocol import (
    DEFAULT_PROTOCOL_PATH,
    load_protocol,
    protocol_section,
    trial_build_protocol_sha256,
    validate_data_protocol,
    validate_release_coverage,
)
from .trial_matrix import GENERATED_TRIAL_FIELDS
from .validate import validate_manifest


DELEGATED_COMMANDS = {
    "adapt": "information_upper_bound.adapters.cli",
    "lock-data": "information_upper_bound.data_lock",
    "extract": "information_upper_bound.extract_features",
    "score": "information_upper_bound.run",
    "analyze": "information_upper_bound.metrics",
}


def _refuse_overwrite(path: Path, *, overwrite: bool) -> None:
    if (path.exists() or path.is_symlink()) and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {path}")


def _reject_path_aliases(**named_paths: str | Path | None) -> None:
    resolved: dict[Path, str] = {}
    for name, raw_path in named_paths.items():
        if raw_path in (None, ""):
            continue
        path = Path(str(raw_path)).expanduser().resolve()
        previous = resolved.get(path)
        if previous is not None:
            raise ValueError(
                f"path collision: {previous} and {name} resolve to the same path: {path}"
            )
        resolved[path] = name


def _parse_option_permutations(value: object) -> int | str:
    normalized = str(value).strip().casefold()
    if normalized in {"all", "all_positions"}:
        return "all"
    try:
        count = int(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "option permutations must be a positive integer or 'all'"
        ) from exc
    if count < 1:
        raise ValueError("option permutations must be a positive integer or 'all'")
    return count


def _shard_records(
    records: list[dict[str, object]], *, shard_count: int, shard_index: int
) -> list[dict[str, object]]:
    if shard_count < 1:
        raise ValueError("--shard-count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("--shard-index must be in [0, --shard-count)")
    if shard_count == 1:
        return records
    selected: list[dict[str, object]] = []
    for record in records:
        diagnostic = record.get("diagnostic")
        diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
        pair_role = str(diagnostic.get("pair_role", "")).casefold()
        candidates = [diagnostic.get("resampling_unit_id")]
        if pair_role in {"original", "counterfactual", "nuisance"}:
            candidates.append(diagnostic.get("pair_id"))
        candidates.extend([diagnostic.get("independent_unit_id"), record.get("id")])
        shard_key = next(
            (str(value) for value in candidates if value not in (None, "")),
            "<missing>",
        )
        bucket = (
            int.from_bytes(
                hashlib.sha256(shard_key.encode("utf-8")).digest()[:8], "big"
            )
            % shard_count
        )
        if bucket == shard_index:
            selected.append(record)
    return selected


def _validate_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="python -m information_upper_bound validate")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-media", action="store_true")
    parser.add_argument("--allow-incomplete-diagnostic", action="store_true")
    parser.add_argument("--fail-on-warning", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    target = Path(args.out)
    _reject_path_aliases(manifest=args.manifest, validation_output=target)
    _refuse_overwrite(target, overwrite=args.overwrite)
    report = validate_manifest(
        read_jsonl(args.manifest),
        require_media=args.require_media,
        strict_diagnostic=not args.allow_incomplete_diagnostic,
    )
    report["manifest"] = str(Path(args.manifest).resolve())
    report["manifest_sha256"] = sha256_file(args.manifest)
    write_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    warning_count = int((report.get("issue_counts") or {}).get("warning", 0))
    if not report["valid"] or (args.fail_on_warning and warning_count):
        raise SystemExit(2)


def _build_trials_command(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m information_upper_bound build-trials"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--data-lock",
        help=(
            "Strict lock-data JSON authenticating the final manifest and adapter reports; "
            "required unless --development is explicit."
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONDITION_PATH))
    parser.add_argument(
        "--protocol-config",
        default=str(DEFAULT_PROTOCOL_PATH),
        help="locked protocol supplying sampling seed and option counterbalancing",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--report-out")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--option-permutations",
        metavar="{all,N}",
        help="use every answer position per item ('all') or N deterministic permutations",
    )
    parser.add_argument("--allow-validation-errors", action="store_true")
    parser.add_argument(
        "--development",
        action="store_true",
        help="Allow an unlocked trial build for infrastructure/debug work only.",
    )
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    target = Path(args.out)
    report_target = (
        Path(args.report_out)
        if args.report_out
        else target.with_suffix(target.suffix + ".report.json")
    )
    _reject_path_aliases(
        manifest=args.manifest,
        conditions=args.config,
        protocol=args.protocol_config,
        data_lock=args.data_lock,
        trials_output=target,
        report_output=report_target,
    )
    _refuse_overwrite(target, overwrite=args.overwrite)
    _refuse_overwrite(report_target, overwrite=args.overwrite)
    records = read_jsonl(args.manifest)
    contaminated = [
        str(record.get("id", ""))
        for record in records
        if any(name in record for name in GENERATED_TRIAL_FIELDS)
    ]
    if contaminated:
        raise ValueError(
            "base manifests may not predeclare reserved trial-expansion fields; "
            f"first contaminated ids: {contaminated[:5]}"
        )
    validation = validate_manifest(records, strict_diagnostic=True)
    if not validation["valid"] and not args.allow_validation_errors:
        preview = (validation.get("issues") or [])[:5]
        raise ValueError(f"base manifest failed validation; first issues: {preview}")
    specs, options = load_condition_config(args.config)
    protocol, protocol_metadata = load_protocol(args.protocol_config)
    data_lock_metadata: dict[str, object] | None = None
    data_protocol: dict[str, object] | None = None
    coverage_validation: dict[str, object] | None = None
    if not args.development:
        if args.allow_validation_errors:
            raise ValueError(
                "--allow-validation-errors is development-only; add --development explicitly"
            )
        if not args.data_lock:
            raise ValueError(
                "confirmatory trial construction requires --data-lock; run the lock-data "
                "command after merging strict adapter outputs"
            )
        data_protocol = validate_data_protocol(protocol)
        condition_config_sha256 = sha256_file(args.config)
        if condition_config_sha256 != data_protocol["conditions_sha256"]:
            raise ValueError(
                "condition config SHA256 does not match locked protocol data.conditions_sha256"
            )
        data_lock_metadata = validate_data_lock(
            args.data_lock,
            manifest_path=args.manifest,
            # Source bytes were verified while creating the lock. Rechecking
            # media and semantic manifest content here preserves portability
            # when the locked release is remounted elsewhere.
            verify_sources=False,
            verify_media=True,
        )
        if (
            data_lock_metadata["data_release_sha256"]
            != data_protocol["data_release_sha256"]
        ):
            raise ValueError(
                "--data-lock release identity does not match locked protocol "
                "data.data_release_sha256"
            )
        locked_datasets = set((data_lock_metadata.get("datasets") or {}).keys())
        required_datasets = set(data_protocol["required_datasets"])
        if locked_datasets != required_datasets:
            raise ValueError(
                "data-lock dataset coverage does not match protocol data.required_datasets: "
                f"locked={sorted(locked_datasets)}, required={sorted(required_datasets)}"
            )
        coverage_validation = validate_release_coverage(
            data_protocol,
            data_lock_metadata,
            records,
        )
    elif args.data_lock:
        data_lock_metadata = validate_data_lock(
            args.data_lock,
            manifest_path=args.manifest,
            verify_sources=False,
            verify_media=False,
        )
    resolved_data_release_sha256 = (
        str(data_lock_metadata["data_release_sha256"])
        if data_lock_metadata is not None
        else None
    )
    sampling = protocol_section(protocol, "sampling")
    if not {"seed", "option_permutations", "trial_shards"}.issubset(sampling):
        raise ValueError(
            "locked protocol sampling must define seed, option_permutations, and trial_shards"
        )
    condition_seed = options.get("seed")
    protocol_seed = sampling.get("seed")
    if condition_seed is not None and int(condition_seed) != int(protocol_seed):
        raise ValueError(
            "conditions options.seed conflicts with locked protocol sampling.seed; "
            "make them equal before freezing the run"
        )
    if args.seed is not None and int(args.seed) != int(protocol_seed):
        raise ValueError(
            "--seed conflicts with locked protocol sampling.seed; update the protocol first"
        )
    seed = int(
        args.seed
        if args.seed is not None
        else protocol_seed
        if protocol_seed is not None
        else condition_seed
        if condition_seed is not None
        else 42
    )
    condition_permutations = options.get("option_permutations")
    protocol_permutations = sampling.get("option_permutations")
    if condition_permutations is not None:
        if _parse_option_permutations(
            condition_permutations
        ) != _parse_option_permutations(protocol_permutations):
            raise ValueError(
                "conditions option_permutations conflicts with the locked protocol; "
                "make them equal before freezing the run"
            )
    if args.option_permutations is not None and _parse_option_permutations(
        args.option_permutations
    ) != _parse_option_permutations(protocol_permutations):
        raise ValueError(
            "--option-permutations conflicts with the locked protocol; update the protocol first"
        )
    permutations = _parse_option_permutations(
        args.option_permutations
        if args.option_permutations is not None
        else protocol_permutations
        if protocol_permutations is not None
        else condition_permutations
        if condition_permutations is not None
        else 1
    )
    protocol_shard_count = int(sampling["trial_shards"])
    if args.shard_count is not None and args.shard_count != protocol_shard_count:
        raise ValueError(
            "--shard-count conflicts with locked protocol sampling.trial_shards; "
            "update the protocol first"
        )
    shard_count = (
        args.shard_count if args.shard_count is not None else protocol_shard_count
    )
    trial_build_payload = {
        "schema_version": TRIAL_BUILD_ATTESTATION_SCHEMA_VERSION,
        "mode": "development" if args.development else "confirmatory",
        "data_release_sha256": resolved_data_release_sha256,
        "condition_config_sha256": sha256_file(args.config),
        "trial_build_protocol_sha256": trial_build_protocol_sha256(protocol),
        "sampling": {
            "seed": seed,
            "option_permutations": permutations,
            "trial_shards": shard_count,
        },
    }
    trial_build_attestation = {
        **trial_build_payload,
        "attestation_sha256": canonical_sha256(trial_build_payload),
    }
    records = [
        {
            **record,
            "data_release_sha256": resolved_data_release_sha256,
            "trial_build_attestation": trial_build_attestation,
        }
        for record in records
    ]
    selected_records = _shard_records(
        records, shard_count=shard_count, shard_index=args.shard_index
    )
    trials, state = stream_trials(
        selected_records,
        specs,
        seed=seed,
        option_permutations=permutations,
    )
    write_jsonl(target, trials)
    report = state.report()
    report.update(
        {
            "input_base_records": len(records),
            "selected_base_records": len(selected_records),
            "shard_count": shard_count,
            "shard_index": args.shard_index,
            "base_manifest": str(Path(args.manifest).resolve()),
            "base_manifest_sha256": sha256_file(args.manifest),
            "condition_config": str(Path(args.config).resolve()),
            "condition_config_sha256": sha256_file(args.config),
            "protocol_config": protocol_metadata,
            "execution_mode": "development" if args.development else "confirmatory",
            "data_lock": data_lock_metadata,
            "coverage_validation": coverage_validation,
            "trial_build_attestation": trial_build_attestation,
            "resolved_sampling": {
                "seed": seed,
                "option_permutations": permutations,
                "trial_shards": shard_count,
            },
            "base_validation": {
                "valid": validation["valid"],
                "issue_counts": validation["issue_counts"],
            },
        }
    )
    write_json(report_target, report)
    console_report = {
        key: value for key, value in report.items() if key not in {"skipped"}
    }
    console_report["report_out"] = str(report_target.resolve())
    print(json.dumps(console_report, ensure_ascii=False, indent=2))


def _delegate(command: str, argv: list[str]) -> None:
    module_name = DELEGATED_COMMANDS[command]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"command {command!r} is unavailable because {module_name} could not be imported"
        ) from exc
    if not hasattr(module, "main"):
        raise RuntimeError(f"{module_name} does not expose main(argv)")
    delegated_argv = ["analyze", *argv] if command == "analyze" else argv
    exit_code = module.main(delegated_argv)
    if isinstance(exit_code, int) and exit_code:
        raise SystemExit(exit_code)


def main(argv: list[str] | None = None) -> None:
    values = list(sys.argv[1:] if argv is None else argv)
    commands = [
        "adapt",
        "lock-data",
        "validate",
        "build-trials",
        "extract",
        "score",
        "analyze",
    ]
    if not values or values[0] in {"-h", "--help"}:
        print(
            "Frozen VideoLLM information upper-bound suite\n\n"
            "usage: python -m information_upper_bound <command> [options]\n\n"
            "commands:\n"
            "  adapt         convert an official dataset annotation release\n"
            "  lock-data     authenticate the merged manifest and adapter provenance\n"
            "  validate      audit schema, pairing, leakage, media, and coverage\n"
            "  build-trials  expand base items into controlled conditions/doses\n"
            "  extract       cache timestamp-aware frozen encoder features\n"
            "  score         score all trials with one frozen VideoLLM/projector\n"
            "  analyze       compute paired metrics and cluster-bootstrap CIs\n"
        )
        return
    command, rest = values[0], values[1:]
    if command not in commands:
        raise SystemExit(
            f"unknown command {command!r}; choose from {', '.join(commands)}"
        )
    if command == "validate":
        _validate_command(rest)
    elif command == "build-trials":
        _build_trials_command(rest)
    else:
        _delegate(command, rest)


if __name__ == "__main__":
    main()
