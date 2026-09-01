from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "1.1"
MAX_CHOICES = 26
ORACLE_LIST_FIELDS = (
    "static_facts",
    "unordered_events",
    "ordered_events",
    "temporal_relations",
    "state_changes",
    "relations",
    "intermediate",
)
ORACLE_KEYS = set(ORACLE_LIST_FIELDS) | {"operator", "answer_derived"}
SAFE_ORACLE_ACCESS = {"safe_visual_gt", "operator_only"}
UNSAFE_ORACLE_ACCESS = {"target", "target_semantic", "answer", "answer_key"}
ALLOWED_ORACLE_LINEAGE = {
    "official_adapter",
    "audited_human_annotation",
    "audited_simulator_gt",
}


class InformationFamily(str, Enum):
    STATIC = "static"
    LOCAL_MOTION = "local_motion"
    TEMPORAL_ORDER = "temporal_order"
    METRIC_TEMPORAL = "metric_temporal"
    BINDING_TRACKING = "binding_tracking"
    CAUSAL_COMPOSITIONAL = "causal_compositional"
    LONG_RANGE_SELECTION = "long_range_selection"


class PairRole(str, Enum):
    STANDALONE = "standalone"
    ORIGINAL = "original"
    COUNTERFACTUAL = "counterfactual"
    NUISANCE = "nuisance"


class SpanUnit(str, Enum):
    SECONDS = "seconds"
    NORMALIZED = "normalized"


@dataclass(frozen=True)
class EvidenceSpan:
    start: float
    end: float
    unit: str = SpanUnit.SECONDS.value
    role: str = "necessary"
    event_id: str | None = None

    def validate(self, *, path: str) -> list["ValidationIssue"]:
        issues: list[ValidationIssue] = []
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            issues.append(ValidationIssue("error", path, "span bounds must be finite"))
            return issues
        if self.start < 0:
            issues.append(ValidationIssue("error", f"{path}.start", "must be >= 0"))
        if self.end <= self.start:
            issues.append(
                ValidationIssue("error", f"{path}.end", "must be greater than start")
            )
        if self.unit not in {unit.value for unit in SpanUnit}:
            issues.append(
                ValidationIssue(
                    "error",
                    f"{path}.unit",
                    f"must be one of {[unit.value for unit in SpanUnit]}",
                )
            )
        if self.unit == SpanUnit.NORMALIZED.value and self.end > 1:
            issues.append(
                ValidationIssue(
                    "error", f"{path}.end", "normalized bounds must be <= 1"
                )
            )
        return issues

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class ValidationIssue:
    level: str
    path: str
    message: str
    record_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def option_label(index: int) -> str:
    if not 0 <= index < MAX_CHOICES:
        raise ValueError(f"Option index {index} is outside [0, {MAX_CHOICES})")
    return chr(ord("A") + index)


def normalize_answer(answer: Any, choices: list[str]) -> str:
    """Normalize an answer to an option letter without guessing index bases.

    Integer labels are intentionally rejected here: individual dataset adapters
    must declare whether their source is zero- or one-indexed. This prevents a
    silent, scientifically damaging off-by-one conversion.
    """

    if isinstance(answer, bool):
        answer = str(answer)
    if isinstance(answer, int):
        raise ValueError("integer answers require an adapter-declared index base")
    text = str(answer).strip()
    match = re.fullmatch(r"\(?\s*([A-Za-z])\s*\)?[.):]?", text)
    if match:
        label = match.group(1).upper()
        if ord(label) - ord("A") < len(choices):
            return label
    normalized = " ".join(text.casefold().split())
    matches = [
        option_label(index)
        for index, choice in enumerate(choices)
        if " ".join(str(choice).casefold().split()) == normalized
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("answer text matches multiple duplicate choices")
    raise ValueError(
        f"answer {answer!r} does not match a valid option label or choice text"
    )


def answer_text(record: Mapping[str, Any]) -> str:
    choices = [str(value) for value in record.get("choices") or []]
    label = normalize_answer(record.get("answer"), choices)
    return choices[ord(label) - ord("A")]


def _required_text(
    obj: Mapping[str, Any],
    key: str,
    issues: list[ValidationIssue],
    *,
    prefix: str = "",
) -> None:
    value = obj.get(key)
    path = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, str) or not value.strip():
        issues.append(ValidationIssue("error", path, "must be a non-empty string"))


def _as_span(value: Any) -> EvidenceSpan:
    if not isinstance(value, Mapping):
        raise TypeError("span must be an object")
    start = value.get("start", value.get("start_sec"))
    end = value.get("end", value.get("end_sec"))
    unit = value.get("unit", "seconds")
    return EvidenceSpan(
        start=float(start),
        end=float(end),
        unit=str(unit),
        role=str(value.get("role", "necessary")),
        event_id=str(value["event_id"]) if value.get("event_id") is not None else None,
    )


def parse_evidence_spans(record: Mapping[str, Any]) -> list[EvidenceSpan]:
    diagnostic = record.get("diagnostic") or {}
    values = diagnostic.get("evidence_spans") or []
    return [_as_span(value) for value in values]


def _validate_oracles(oracles: Any, issues: list[ValidationIssue]) -> None:
    if not isinstance(oracles, Mapping):
        issues.append(
            ValidationIssue("error", "diagnostic.oracles", "must be an object")
        )
        return
    unknown = sorted(set(oracles) - ORACLE_KEYS)
    if unknown:
        issues.append(
            ValidationIssue(
                "error",
                "diagnostic.oracles",
                f"contains unsupported fields: {unknown}",
            )
        )

    def validate_fact(field: str, value: Any, index: int) -> None:
        path = f"diagnostic.oracles.{field}[{index}]"
        if not isinstance(value, Mapping):
            issues.append(
                ValidationIssue(
                    "error",
                    path,
                    "must be an object with explicit access, source, and lineage",
                )
            )
            return
        access = value.get("access")
        allowed_access = SAFE_ORACLE_ACCESS | UNSAFE_ORACLE_ACCESS
        if access not in allowed_access:
            issues.append(
                ValidationIssue(
                    "error",
                    f"{path}.access",
                    f"must be one of {sorted(allowed_access)}",
                )
            )
        if field == "operator" and access != "operator_only":
            issues.append(
                ValidationIssue(
                    "error",
                    f"{path}.access",
                    "operator facts must use access='operator_only'",
                )
            )
        if field != "operator" and access == "operator_only":
            issues.append(
                ValidationIssue(
                    "error",
                    f"{path}.access",
                    "operator_only access is valid only in diagnostic.oracles.operator",
                )
            )
        for key in ("source", "lineage"):
            if not isinstance(value.get(key), str) or not str(value.get(key)).strip():
                issues.append(
                    ValidationIssue(
                        "error", f"{path}.{key}", "must be a non-empty string"
                    )
                )
        lineage = value.get("lineage")
        if isinstance(lineage, str) and lineage not in ALLOWED_ORACLE_LINEAGE:
            issues.append(
                ValidationIssue(
                    "error",
                    f"{path}.lineage",
                    f"must be one of {sorted(ALLOWED_ORACLE_LINEAGE)}",
                )
            )

    for field in ORACLE_LIST_FIELDS:
        value = oracles.get(field)
        if not isinstance(value, list):
            issues.append(
                ValidationIssue(
                    "error", f"diagnostic.oracles.{field}", "must be a list"
                )
            )
            continue
        for index, fact in enumerate(value):
            validate_fact(field, fact, index)
    operator = oracles.get("operator")
    if operator is not None and not isinstance(operator, list):
        issues.append(
            ValidationIssue(
                "error",
                "diagnostic.oracles.operator",
                "must be null or a list of explicit fact objects",
            )
        )
    elif isinstance(operator, list):
        for index, fact in enumerate(operator):
            validate_fact("operator", fact, index)

    if "answer_derived" not in oracles:
        issues.append(
            ValidationIssue(
                "error",
                "diagnostic.oracles.answer_derived",
                "must be explicitly present and false",
            )
        )
    elif oracles.get("answer_derived") is not False:
        issues.append(
            ValidationIssue(
                "error",
                "diagnostic.oracles.answer_derived",
                "must be false; test clues may not be derived from the gold answer",
            )
        )


def validate_record(
    record: Mapping[str, Any],
    *,
    require_media: bool = False,
    strict_diagnostic: bool = True,
) -> list[ValidationIssue]:
    """Validate one diagnostic manifest record.

    The function returns all issues rather than failing at the first one so a
    large public dataset can be audited in a single pass.
    """

    issues: list[ValidationIssue] = []
    record_id = str(record.get("id")) if record.get("id") is not None else None
    for key in ("id", "media_type", "media_path", "question"):
        _required_text(record, key, issues)

    media_type = record.get("media_type")
    if media_type not in {"video", "image"}:
        issues.append(
            ValidationIssue("error", "media_type", "must be 'video' or 'image'")
        )
    media_path = record.get("media_path")
    if (
        require_media
        and isinstance(media_path, str)
        and media_path
        and not Path(media_path).is_file()
    ):
        issues.append(ValidationIssue("error", "media_path", "file does not exist"))

    choices = record.get("choices")
    if not isinstance(choices, list) or not 2 <= len(choices) <= MAX_CHOICES:
        issues.append(
            ValidationIssue(
                "error", "choices", f"must contain 2..{MAX_CHOICES} options"
            )
        )
        choices_list: list[str] = []
    else:
        choices_list = [str(value).strip() for value in choices]
        if any(not value for value in choices_list):
            issues.append(
                ValidationIssue("error", "choices", "options must be non-empty")
            )
        if len({" ".join(value.casefold().split()) for value in choices_list}) != len(
            choices_list
        ):
            issues.append(
                ValidationIssue(
                    "error", "choices", "contains duplicate normalized options"
                )
            )
    if choices_list:
        try:
            normalize_answer(record.get("answer"), choices_list)
        except (TypeError, ValueError) as exc:
            issues.append(ValidationIssue("error", "answer", str(exc)))

    diagnostic = record.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        level = "error" if strict_diagnostic else "warning"
        issues.append(ValidationIssue(level, "diagnostic", "must be an object"))
    else:
        if diagnostic.get("schema_version") != SCHEMA_VERSION:
            issues.append(
                ValidationIssue(
                    "error",
                    "diagnostic.schema_version",
                    f"must equal {SCHEMA_VERSION!r}",
                )
            )
        for key in (
            "dataset",
            "split",
            "information_family",
            "question_family",
            "pair_id",
            "pair_role",
        ):
            _required_text(diagnostic, key, issues, prefix="diagnostic")
        family = diagnostic.get("information_family")
        if family not in {value.value for value in InformationFamily}:
            issues.append(
                ValidationIssue(
                    "error",
                    "diagnostic.information_family",
                    f"must be one of {[value.value for value in InformationFamily]}",
                )
            )
        pair_role = diagnostic.get("pair_role")
        if pair_role not in {value.value for value in PairRole}:
            issues.append(
                ValidationIssue(
                    "error",
                    "diagnostic.pair_role",
                    f"must be one of {[value.value for value in PairRole]}",
                )
            )
        depth = diagnostic.get("reasoning_depth")
        if not isinstance(depth, int) or isinstance(depth, bool) or not 0 <= depth <= 3:
            issues.append(
                ValidationIssue(
                    "error",
                    "diagnostic.reasoning_depth",
                    "must be an integer in [0, 3]",
                )
            )
        spans = diagnostic.get("evidence_spans", [])
        if not isinstance(spans, list):
            issues.append(
                ValidationIssue("error", "diagnostic.evidence_spans", "must be a list")
            )
        else:
            for index, raw_span in enumerate(spans):
                try:
                    span = _as_span(raw_span)
                except (TypeError, ValueError) as exc:
                    issues.append(
                        ValidationIssue(
                            "error", f"diagnostic.evidence_spans[{index}]", str(exc)
                        )
                    )
                    continue
                issues.extend(span.validate(path=f"diagnostic.evidence_spans[{index}]"))
        _validate_oracles(diagnostic.get("oracles", {}), issues)
        media_clip = diagnostic.get("media_clip")
        if media_clip is not None:
            if not isinstance(media_clip, Mapping):
                issues.append(
                    ValidationIssue(
                        "error", "diagnostic.media_clip", "must be an object"
                    )
                )
            else:
                try:
                    unit = str(media_clip.get("unit", "seconds"))
                    if unit == "seconds":
                        clip_start = float(media_clip.get("start", 0.0))
                        clip_end = float(media_clip["end"])
                        if not math.isfinite(clip_start) or not math.isfinite(clip_end):
                            raise ValueError("bounds must be finite")
                        if clip_start < 0 or clip_end <= clip_start:
                            raise ValueError("requires 0 <= start < end")
                    elif unit == "frames":
                        clip_start = media_clip.get(
                            "start_frame", media_clip.get("start", 0)
                        )
                        clip_end = media_clip.get(
                            "end_frame_exclusive", media_clip.get("end")
                        )
                        expected_total = media_clip.get("expected_total_frames")
                        if any(
                            isinstance(value, bool) or not isinstance(value, int)
                            for value in (clip_start, clip_end, expected_total)
                        ):
                            raise ValueError(
                                "frame bounds and expected_total_frames must be integers"
                            )
                        if clip_start < 0 or clip_end <= clip_start:
                            raise ValueError(
                                "requires 0 <= start_frame < end_frame_exclusive"
                            )
                        if expected_total < clip_end:
                            raise ValueError(
                                "expected_total_frames must cover end_frame_exclusive"
                            )
                    else:
                        raise ValueError("unit must be 'seconds' or 'frames'")
                except (KeyError, TypeError, ValueError) as exc:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "diagnostic.media_clip",
                            f"invalid media clip: {exc}",
                        )
                    )
        provenance = diagnostic.get("provenance")
        resampling_unit_id = diagnostic.get("resampling_unit_id")
        if not isinstance(resampling_unit_id, str) or not resampling_unit_id.strip():
            issues.append(
                ValidationIssue(
                    "error",
                    "diagnostic.resampling_unit_id",
                    "must identify the raw source video, scene, or paired-video family",
                )
            )
        independent_unit_id = diagnostic.get("independent_unit_id")
        if independent_unit_id is not None and (
            not isinstance(independent_unit_id, str) or not independent_unit_id.strip()
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "diagnostic.independent_unit_id",
                    "must be a non-empty string when provided",
                )
            )
        adapter_run_id = diagnostic.get("adapter_run_id")
        if adapter_run_id is not None and (
            not isinstance(adapter_run_id, str)
            or not adapter_run_id.startswith("adapter-run::")
            or len(adapter_run_id) != len("adapter-run::") + 64
            or any(
                character not in "0123456789abcdef"
                for character in adapter_run_id[len("adapter-run::") :].lower()
            )
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "diagnostic.adapter_run_id",
                    "must be adapter-run::<64 hexadecimal SHA256> when provided",
                )
            )
        official_candidate_id = diagnostic.get("official_candidate_id")
        official_candidate_count = diagnostic.get("official_candidate_count")
        if official_candidate_id is not None and (
            not isinstance(official_candidate_id, str)
            or not official_candidate_id.strip()
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "diagnostic.official_candidate_id",
                    "must be a non-empty string when supplied",
                )
            )
        if official_candidate_count is not None and (
            isinstance(official_candidate_count, bool)
            or not isinstance(official_candidate_count, int)
            or official_candidate_count < 1
        ):
            issues.append(
                ValidationIssue(
                    "error",
                    "diagnostic.official_candidate_count",
                    "must be a positive integer when supplied",
                )
            )
        if (official_candidate_id is None) != (official_candidate_count is None):
            issues.append(
                ValidationIssue(
                    "error",
                    "diagnostic.official_candidate_id",
                    "official candidate ID and count must be supplied together",
                )
            )
        if not isinstance(provenance, Mapping):
            issues.append(
                ValidationIssue("error", "diagnostic.provenance", "must be an object")
            )
        elif not provenance.get("source_id"):
            issues.append(
                ValidationIssue(
                    "error",
                    "diagnostic.provenance.source_id",
                    "must identify the raw annotation",
                )
            )

    return [
        ValidationIssue(
            issue.level, issue.path, issue.message, issue.record_id or record_id
        )
        for issue in issues
    ]


def has_errors(issues: Iterable[ValidationIssue]) -> bool:
    return any(issue.level == "error" for issue in issues)


def diagnostic_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    diagnostic = record.get("diagnostic")
    if not isinstance(diagnostic, Mapping):
        return {}
    keys = (
        "dataset",
        "split",
        "information_family",
        "question_family",
        "reasoning_depth",
        "resampling_unit_id",
        "pair_id",
        "pair_role",
        "independent_unit_id",
        "official_candidate_id",
        "official_candidate_count",
    )
    return {key: diagnostic.get(key) for key in keys}
