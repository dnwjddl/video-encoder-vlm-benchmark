from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .schema import PairRole, ValidationIssue, answer_text, validate_record


def _issue(
    level: str, path: str, message: str, record_id: str | None = None
) -> ValidationIssue:
    return ValidationIssue(level=level, path=path, message=message, record_id=record_id)


def validate_manifest(
    records: Iterable[Mapping[str, Any]],
    *,
    require_media: bool = False,
    strict_diagnostic: bool = True,
) -> dict[str, Any]:
    rows = list(records)
    issues: list[ValidationIssue] = []
    ids: Counter[str] = Counter()
    pair_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    media_splits: dict[str, set[str]] = defaultdict(set)
    source_video_splits: dict[str, set[str]] = defaultdict(set)
    independent_unit_splits: dict[str, set[str]] = defaultdict(set)
    resampling_unit_splits: dict[str, set[str]] = defaultdict(set)
    independent_unit_resampling_units: dict[str, set[str]] = defaultdict(set)
    answer_positions: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    evidence_record_counts: Counter[str] = Counter()
    evidence_span_counts: Counter[str] = Counter()

    for row in rows:
        issues.extend(
            validate_record(
                row,
                require_media=require_media,
                strict_diagnostic=strict_diagnostic,
            )
        )
        record_id = str(row.get("id", ""))
        ids[record_id] += 1
        diagnostic = row.get("diagnostic") or {}
        pair_id = str(diagnostic.get("pair_id", ""))
        if pair_id:
            pair_groups[pair_id].append(row)
        split = str(diagnostic.get("split", ""))
        dataset_namespace = str(
            diagnostic.get("dataset", row.get("benchmark", "unknown"))
        )
        resampling_unit_id = diagnostic.get("resampling_unit_id")
        if resampling_unit_id not in (None, "") and split:
            resampling_unit_splits[f"{dataset_namespace}::{resampling_unit_id}"].add(
                split
            )
        independent_unit_id = diagnostic.get("independent_unit_id")
        if independent_unit_id not in (None, "") and split:
            namespaced_independent_id = f"{dataset_namespace}::{independent_unit_id}"
            independent_unit_splits[namespaced_independent_id].add(split)
            if resampling_unit_id not in (None, ""):
                independent_unit_resampling_units[namespaced_independent_id].add(
                    str(resampling_unit_id)
                )
        media_path = str(row.get("media_path", ""))
        if media_path and split:
            media_splits[str(Path(media_path).resolve())].add(split)
        provenance = diagnostic.get("provenance") or {}
        if isinstance(provenance, Mapping):
            source_video_id = next(
                (
                    provenance.get(key)
                    for key in ("source_video_id", "raw_video_id", "video_id")
                    if provenance.get(key) not in (None, "")
                ),
                None,
            )
            if source_video_id is not None and split:
                dataset_namespace = str(
                    diagnostic.get("dataset", row.get("benchmark", "unknown"))
                )
                source_video_splits[f"{dataset_namespace}::{source_video_id}"].add(
                    split
                )
        answer = row.get("answer")
        if isinstance(answer, str):
            answer_positions[answer.strip().upper()] += 1
        family_counts[str(diagnostic.get("information_family", "unknown"))] += 1
        dataset_name = str(diagnostic.get("dataset", row.get("benchmark", "unknown")))
        dataset_counts[dataset_name] += 1
        evidence_spans = diagnostic.get("evidence_spans")
        if isinstance(evidence_spans, list) and evidence_spans:
            evidence_record_counts[dataset_name] += 1
            evidence_span_counts[dataset_name] += len(evidence_spans)

    for record_id, count in ids.items():
        if record_id and count > 1:
            issues.append(
                _issue("error", "id", f"duplicate id occurs {count} times", record_id)
            )

    for media_path, splits in media_splits.items():
        if len(splits) > 1:
            issues.append(
                _issue(
                    "error",
                    "diagnostic.split",
                    f"same media appears in multiple splits {sorted(splits)}: {media_path}",
                )
            )

    for source_video_id, splits in source_video_splits.items():
        if len(splits) > 1:
            issues.append(
                _issue(
                    "error",
                    "diagnostic.provenance.source_video_id",
                    f"source video appears in multiple splits {sorted(splits)}: {source_video_id}",
                )
            )

    for independent_unit_id, splits in independent_unit_splits.items():
        if len(splits) > 1:
            issues.append(
                _issue(
                    "error",
                    "diagnostic.independent_unit_id",
                    f"independent analysis unit crosses splits {sorted(splits)}: "
                    f"{independent_unit_id}",
                )
            )

    for resampling_unit_id, splits in resampling_unit_splits.items():
        if len(splits) > 1:
            issues.append(
                _issue(
                    "error",
                    "diagnostic.resampling_unit_id",
                    f"resampling family crosses splits {sorted(splits)}: "
                    f"{resampling_unit_id}",
                )
            )

    for (
        independent_unit_id,
        resampling_ids,
    ) in independent_unit_resampling_units.items():
        if len(resampling_ids) > 1:
            issues.append(
                _issue(
                    "error",
                    "diagnostic.resampling_unit_id",
                    "one official aggregation unit maps to multiple resampling families "
                    f"{sorted(resampling_ids)}: {independent_unit_id}",
                )
            )

    pair_stats: Counter[str] = Counter()
    for pair_id, group in pair_groups.items():
        roles: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in group:
            roles[str((row.get("diagnostic") or {}).get("pair_role"))].append(row)
        if PairRole.STANDALONE.value in roles and len(group) != 1:
            issues.append(
                _issue(
                    "error",
                    "diagnostic.pair_id",
                    "standalone pair has multiple rows",
                    pair_id,
                )
            )
        originals = roles.get(PairRole.ORIGINAL.value, [])
        counterfactuals = roles.get(PairRole.COUNTERFACTUAL.value, [])
        nuisances = roles.get(PairRole.NUISANCE.value, [])
        pair_resampling_ids = {
            str((row.get("diagnostic") or {}).get("resampling_unit_id"))
            for row in group
            if (row.get("diagnostic") or {}).get("resampling_unit_id") not in (None, "")
        }
        if len(pair_resampling_ids) > 1:
            issues.append(
                _issue(
                    "error",
                    "diagnostic.resampling_unit_id",
                    f"pair family maps to multiple resampling families {sorted(pair_resampling_ids)}",
                    pair_id,
                )
            )

        if counterfactuals or nuisances:
            if len(originals) != 1:
                issues.append(
                    _issue(
                        "error",
                        "diagnostic.pair_role",
                        f"paired family must contain exactly one original, got {len(originals)}",
                        pair_id,
                    )
                )
                continue
            original = originals[0]
            original_question = " ".join(str(original.get("question", "")).split())
            original_choices = [
                " ".join(str(value).split()) for value in original.get("choices") or []
            ]
            try:
                original_answer = answer_text(original)
            except (TypeError, ValueError):
                continue
            for row in counterfactuals:
                pair_stats["counterfactual_rows"] += 1
                if " ".join(str(row.get("question", "")).split()) != original_question:
                    issues.append(
                        _issue(
                            "warning",
                            "question",
                            "counterfactual question differs from original",
                            str(row.get("id")),
                        )
                    )
                if [
                    " ".join(str(value).split()) for value in row.get("choices") or []
                ] != original_choices:
                    issues.append(
                        _issue(
                            "warning",
                            "choices",
                            "counterfactual options differ from original",
                            str(row.get("id")),
                        )
                    )
                try:
                    if answer_text(row) == original_answer:
                        issues.append(
                            _issue(
                                "error",
                                "answer",
                                "counterfactual must change the semantic answer",
                                str(row.get("id")),
                            )
                        )
                except (TypeError, ValueError):
                    pass
            for row in nuisances:
                pair_stats["nuisance_rows"] += 1
                try:
                    if answer_text(row) != original_answer:
                        issues.append(
                            _issue(
                                "error",
                                "answer",
                                "nuisance variant must preserve the semantic answer",
                                str(row.get("id")),
                            )
                        )
                except (TypeError, ValueError):
                    pass

        split_values = {
            str((row.get("diagnostic") or {}).get("split", "")) for row in group
        }
        if len(split_values) > 1:
            issues.append(
                _issue(
                    "error",
                    "diagnostic.split",
                    f"pair family crosses splits {sorted(split_values)}",
                    pair_id,
                )
            )

    level_counts = Counter(issue.level for issue in issues)
    return {
        "valid": level_counts["error"] == 0,
        "num_records": len(rows),
        "num_pairs": len(pair_groups),
        "issue_counts": dict(sorted(level_counts.items())),
        "issues": [issue.to_dict() for issue in issues],
        "coverage": {
            "dataset": dict(sorted(dataset_counts.items())),
            "information_family": dict(sorted(family_counts.items())),
            "answer_position": dict(sorted(answer_positions.items())),
            "pair_rows": dict(sorted(pair_stats.items())),
            "independent_units": len(independent_unit_splits),
            "resampling_units": len(resampling_unit_splits),
            "records_with_evidence_by_dataset": dict(
                sorted(evidence_record_counts.items())
            ),
            "evidence_spans_by_dataset": dict(sorted(evidence_span_counts.items())),
            "records_with_evidence": sum(evidence_record_counts.values()),
            "evidence_spans": sum(evidence_span_counts.values()),
            "media_files_present": sum(
                bool(row.get("media_path")) and Path(str(row["media_path"])).is_file()
                for row in rows
            ),
        },
    }
