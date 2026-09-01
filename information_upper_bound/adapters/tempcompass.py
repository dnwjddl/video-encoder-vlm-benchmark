from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from information_upper_bound.schema import InformationFamily, PairRole

from .common import (
    AdapterError,
    BaseAdapter,
    empty_oracles,
    load_json,
    make_record,
    normalize_text_answer,
    require_list,
    require_mapping,
    require_text,
    resolve_media,
    stable_id,
)


CHOICE_RE = re.compile(r"^\s*([A-Z])[.)]\s*(.+?)\s*$")
ANSWER_RE = re.compile(r"^\s*([A-Z])[.)]\s*(.+?)\s*$")

FAMILY_MAP = {
    "action": InformationFamily.LOCAL_MOTION.value,
    "direction": InformationFamily.LOCAL_MOTION.value,
    "speed": InformationFamily.METRIC_TEMPORAL.value,
    "order": InformationFamily.TEMPORAL_ORDER.value,
    "attribute_change": InformationFamily.BINDING_TRACKING.value,
}


def parse_multichoice(
    text: Any, answer: Any, *, path: str
) -> tuple[str, list[str], str]:
    raw = require_text(text, path=f"{path}.question")
    question_lines: list[str] = []
    labeled: list[tuple[str, str]] = []
    for line in raw.splitlines():
        match = CHOICE_RE.match(line)
        if match:
            labeled.append((match.group(1), match.group(2).strip()))
        elif line.strip():
            if labeled:
                raise AdapterError(
                    f"{path}.question: non-choice text appears after options"
                )
            question_lines.append(line.strip())
    if not question_lines or len(labeled) < 2:
        raise AdapterError(
            f"{path}.question: expected prompt followed by at least two labeled options"
        )
    expected = [chr(ord("A") + index) for index in range(len(labeled))]
    actual = [label for label, _ in labeled]
    if actual != expected:
        raise AdapterError(
            f"{path}.question: option labels must be contiguous {expected}, got {actual}"
        )
    choices = [choice for _, choice in labeled]

    answer_text = require_text(answer, path=f"{path}.answer")
    match = ANSWER_RE.match(answer_text)
    if match:
        label, supplied_text = match.groups()
        if label not in actual:
            raise AdapterError(
                f"{path}.answer: label {label} is outside the available options"
            )
        expected_text = choices[ord(label) - ord("A")]
        if " ".join(supplied_text.casefold().split()) != " ".join(
            expected_text.casefold().split()
        ):
            raise AdapterError(
                f"{path}.answer: labeled answer text {supplied_text!r} does not match option {label}"
            )
        normalized_answer = label
    else:
        normalized_answer = normalize_text_answer(
            answer_text, choices, path=f"{path}.answer"
        )
    return "\n".join(question_lines), choices, normalized_answer


def _fact_text(dimension: str, value: Mapping[str, Any]) -> str:
    subject = str(value.get("subject") or "the scene")
    if dimension == "action":
        return f"{subject} performs {value.get('action')}."
    if dimension == "direction":
        return f"{subject} moves with direction: {value.get('direction')}."
    if dimension == "speed":
        return f"{subject} has speed: {value.get('speed')}."
    if dimension == "order":
        return (
            f"For {subject}, {value.get('event1')} occurs before {value.get('event2')}."
        )
    if dimension == "attribute_change":
        return f"{subject} changes state: {value.get('attribute_change')}."
    raise AdapterError(f"unsupported TempCompass dimension: {dimension}")


def _oracles_from_meta(
    meta: Mapping[str, Any], dimension: str, *, path: str
) -> dict[str, Any]:
    oracles = empty_oracles()
    eval_dim = require_mapping(meta.get("eval_dim"), path=f"{path}.eval_dim")
    raw = eval_dim.get(dimension)
    if raw is None:
        # A small number of official multi-choice rows have no corresponding
        # meta_info fact for that dimension. The QA row remains evaluable, but
        # there is no source-grounded oracle to expose.
        return oracles
    value = require_mapping(raw, path=f"{path}.eval_dim.{dimension}")
    fact = {
        **value,
        "text": _fact_text(dimension, value),
        "source": "meta_info.eval_dim",
        # eval_dim is designed to encode the discriminating target semantics.
        # Preserve it for provenance, but conditions.oracle_facts excludes this
        # access class from every safe clue intervention.
        "access": "target_semantic",
    }
    if dimension == "action":
        oracles["unordered_events"] = [fact]
    elif dimension == "direction":
        oracles["relations"] = [fact]
    elif dimension == "speed":
        oracles["temporal_relations"] = [fact]
    elif dimension == "order":
        oracles["temporal_relations"] = [fact]
    elif dimension == "attribute_change":
        oracles["state_changes"] = [fact]
    return oracles


class TempCompassAdapter(BaseAdapter):
    """Adapter for the official TempCompass multi-choice release."""

    name = "tempcompass"

    def _paths(self) -> tuple[Path, Path | None]:
        annotation = self.config.annotation_path
        if annotation.is_dir():
            question_path = annotation / "questions" / "multi-choice.json"
            if not question_path.is_file():
                question_path = annotation / "multi-choice.json"
            meta_path = annotation / "meta_info.json"
        else:
            question_path = annotation
            option = self.config.options.get("meta_info_path")
            meta_path = Path(option).expanduser().resolve() if option else None
        if not question_path.is_file():
            raise AdapterError(
                f"TempCompass multi-choice annotation not found: {question_path}"
            )
        if meta_path is not None and not meta_path.is_file():
            raise AdapterError(f"TempCompass meta_info file not found: {meta_path}")
        return question_path, meta_path

    def iter_records(self) -> Iterable[dict[str, Any]]:
        question_path, meta_path = self._paths()
        questions = require_mapping(load_json(question_path), path=str(question_path))
        metadata = (
            require_mapping(load_json(meta_path), path=str(meta_path))
            if meta_path
            else {}
        )

        # A reverse video is only a diagnostic counterfactual when the release
        # supplies the *same* MCQ and option semantics for both clips and the
        # gold answer changes.  The official release also contains rows whose
        # reverse-side question/options were regenerated, and a few whose
        # textual answer happens to remain unchanged.  Treating those as
        # controlled pairs would confound the intervention with a prompt change.
        pairable: set[tuple[str, str, int]] = set()
        for base_key, raw_dimensions in questions.items():
            if base_key.endswith("_reverse") or f"{base_key}_reverse" not in questions:
                continue
            original_dimensions = require_mapping(
                raw_dimensions, path=f"questions.{base_key}"
            )
            reverse_dimensions = require_mapping(
                questions[f"{base_key}_reverse"], path=f"questions.{base_key}_reverse"
            )
            for dimension in sorted(set(original_dimensions) & set(reverse_dimensions)):
                if dimension not in FAMILY_MAP:
                    continue
                originals = require_list(
                    original_dimensions[dimension],
                    path=f"questions.{base_key}.{dimension}",
                )
                reverses = require_list(
                    reverse_dimensions[dimension],
                    path=f"questions.{base_key}_reverse.{dimension}",
                )
                for index, (original_raw, reverse_raw) in enumerate(
                    zip(originals, reverses)
                ):
                    original_source_id = f"{base_key}:{dimension}:{index}"
                    reverse_source_id = f"{base_key}_reverse:{dimension}:{index}"
                    original_excluded = self.is_excluded(original_source_id)
                    reverse_excluded = self.is_excluded(reverse_source_id)
                    if original_excluded != reverse_excluded:
                        raise AdapterError(
                            "TempCompass exclusions must close over an original/reverse "
                            f"question pair; declare both {original_source_id!r} and "
                            f"{reverse_source_id!r}"
                        )
                    if original_excluded:
                        continue
                    original = require_mapping(
                        original_raw, path=f"questions.{base_key}.{dimension}[{index}]"
                    )
                    reverse = require_mapping(
                        reverse_raw,
                        path=f"questions.{base_key}_reverse.{dimension}[{index}]",
                    )
                    original_parsed = parse_multichoice(
                        original.get("question"),
                        original.get("answer"),
                        path=f"questions.{base_key}.{dimension}[{index}]",
                    )
                    reverse_parsed = parse_multichoice(
                        reverse.get("question"),
                        reverse.get("answer"),
                        path=f"questions.{base_key}_reverse.{dimension}[{index}]",
                    )
                    original_semantic_answer = original_parsed[1][
                        ord(original_parsed[2]) - ord("A")
                    ]
                    reverse_semantic_answer = reverse_parsed[1][
                        ord(reverse_parsed[2]) - ord("A")
                    ]
                    if (
                        original_parsed[:2] == reverse_parsed[:2]
                        and original_semantic_answer != reverse_semantic_answer
                    ):
                        pairable.add((base_key, dimension, index))

        for video_key in sorted(questions):
            dimensions = require_mapping(
                questions[video_key], path=f"questions.{video_key}"
            )
            is_reverse = video_key.endswith("_reverse")
            base_key = video_key[: -len("_reverse")] if is_reverse else video_key
            media_path = resolve_media(
                self.config.media_root,
                (f"{video_key}.mp4", Path("videos") / f"{video_key}.mp4"),
                require=self.config.require_media,
                search_basename=f"{video_key}.mp4",
            )
            meta = None
            if metadata:
                if video_key not in metadata:
                    raise AdapterError(
                        f"meta_info.json has no entry for video {video_key}"
                    )
                meta = require_mapping(
                    metadata[video_key], path=f"meta_info.{video_key}"
                )

            for dimension in sorted(dimensions):
                if dimension not in FAMILY_MAP:
                    raise AdapterError(
                        f"questions.{video_key}: unsupported dimension {dimension!r}"
                    )
                items = require_list(
                    dimensions[dimension], path=f"questions.{video_key}.{dimension}"
                )
                for index, raw_item in enumerate(items):
                    item_path = f"questions.{video_key}.{dimension}[{index}]"
                    source_id = f"{video_key}:{dimension}:{index}"
                    if self.skip_excluded(source_id, raw_location=item_path):
                        continue
                    item = require_mapping(raw_item, path=item_path)
                    question, choices, answer = parse_multichoice(
                        item.get("question"), item.get("answer"), path=item_path
                    )
                    record_id = stable_id(
                        self.name, video_key, dimension, index, question
                    )
                    has_pair = (base_key, dimension, index) in pairable
                    if has_pair:
                        pair_id = f"tempcompass:{base_key}:{dimension}:{index}"
                        pair_role = (
                            PairRole.COUNTERFACTUAL.value
                            if is_reverse
                            else PairRole.ORIGINAL.value
                        )
                    else:
                        pair_id = f"standalone:{record_id}"
                        pair_role = PairRole.STANDALONE.value
                    oracles = (
                        _oracles_from_meta(
                            meta, dimension, path=f"meta_info.{video_key}"
                        )
                        if meta is not None
                        else empty_oracles()
                    )
                    yield make_record(
                        record_id=record_id,
                        source="TempCompass",
                        benchmark="tempcompass",
                        task="mcq",
                        media_path=media_path,
                        question=question,
                        choices=choices,
                        answer=answer,
                        dataset=self.name,
                        split=self.config.split,
                        information_family=FAMILY_MAP[dimension],
                        question_family=f"tempcompass:{dimension}",
                        reasoning_depth=1 if dimension != "order" else 2,
                        resampling_unit_id=f"tempcompass:video_family:{base_key}",
                        pair_id=pair_id,
                        pair_role=pair_role,
                        evidence_spans=[],
                        oracles=oracles,
                        provenance={
                            "source_id": source_id,
                            "video_key": video_key,
                            "dimension": dimension,
                            "annotation_file": str(question_path),
                            "meta_info_file": str(meta_path) if meta_path else None,
                            "answer_index_base": "letter-and-text",
                        },
                    )
