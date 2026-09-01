from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from information_upper_bound.schema import InformationFamily, PairRole

from .common import (
    AdapterError,
    BaseAdapter,
    empty_oracles,
    load_json,
    make_record,
    normalize_text_answer,
    parse_candidates,
    require_list,
    require_mapping,
    require_text,
    resolve_media,
    source_id_component,
    stable_id,
)


TASKS = {
    "action_antonym": (InformationFamily.LOCAL_MOTION.value, 1),
    "action_count": (InformationFamily.METRIC_TEMPORAL.value, 1),
    "action_localization": (InformationFamily.METRIC_TEMPORAL.value, 1),
    "action_sequence": (InformationFamily.TEMPORAL_ORDER.value, 2),
    "egocentric_sequence": (InformationFamily.TEMPORAL_ORDER.value, 2),
    "moving_direction": (InformationFamily.LOCAL_MOTION.value, 1),
    "object_count": (InformationFamily.BINDING_TRACKING.value, 1),
    "object_shuffle": (InformationFamily.BINDING_TRACKING.value, 2),
    "scene_transition": (InformationFamily.TEMPORAL_ORDER.value, 1),
    "unexpected_action": (InformationFamily.METRIC_TEMPORAL.value, 2),
}


def _number(value: Any, *, path: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{path}: expected a numeric timestamp") from exc
    if not math.isfinite(out):
        raise AdapterError(f"{path}: timestamp must be finite")
    return out


def _evidence(row: Mapping[str, Any], task: str, *, path: str) -> list[dict[str, Any]]:
    if task not in {"action_localization", "action_sequence"}:
        return []
    preferred = (
        (
            ("accurate_start", "accurate_end"),
            ("start", "end"),
        )
        if task == "action_localization"
        else (("start", "end"),)
    )
    for start_key, end_key in preferred:
        if row.get(start_key) not in (None, "") and row.get(end_key) not in (None, ""):
            start = _number(row[start_key], path=f"{path}.{start_key}")
            end = _number(row[end_key], path=f"{path}.{end_key}")
            if start < 0 or end <= start:
                raise AdapterError(
                    f"{path}: invalid evidence interval [{start}, {end}]"
                )
            return [
                {
                    "start": start,
                    "end": end,
                    "unit": "seconds",
                    "role": "dataset_temporal_annotation",
                }
            ]
    raise AdapterError(f"{path}: {task} row is missing its official temporal interval")


class TVBenchAdapter(BaseAdapter):
    """Adapter for the ten JSON tasks in the official TVBench release."""

    name = "tvbench"

    def _annotation_files(self) -> list[tuple[str, Path]]:
        source = self.config.annotation_path
        if source.is_file():
            task = str(self.config.options.get("task") or source.stem)
            if task not in TASKS:
                raise AdapterError(
                    f"TVBench task must be one of {sorted(TASKS)}, got {task!r}"
                )
            return [(task, source)]
        json_root = source / "json" if (source / "json").is_dir() else source
        files = [(task, json_root / f"{task}.json") for task in sorted(TASKS)]
        selected = self.config.options.get("tasks")
        if selected:
            wanted = {str(value) for value in selected}
            unknown = wanted - set(TASKS)
            if unknown:
                raise AdapterError(f"unknown TVBench tasks: {sorted(unknown)}")
            files = [(task, path) for task, path in files if task in wanted]
        missing = [str(path) for _, path in files if not path.is_file()]
        if missing:
            raise AdapterError(f"missing TVBench task annotations: {missing}")
        return files

    def iter_records(self) -> Iterable[dict[str, Any]]:
        for task, annotation_path in self._annotation_files():
            raw_rows = require_list(
                load_json(annotation_path), path=str(annotation_path)
            )
            information_family, depth = TASKS[task]
            for index, raw_row in enumerate(raw_rows):
                row_path = f"{annotation_path}[{index}]"
                raw_source_id = (
                    raw_row.get("question_id") if isinstance(raw_row, Mapping) else None
                )
                source_id = source_id_component(raw_source_id, fallback=f"row:{index}")
                exclusion_id = f"{task}:{source_id}"
                if self.skip_excluded(exclusion_id, raw_location=row_path):
                    continue
                row = require_mapping(raw_row, path=row_path)
                raw_video = require_text(row.get("video"), path=f"{row_path}.video")
                choices = parse_candidates(
                    row.get("candidates"), path=f"{row_path}.candidates"
                )
                answer = normalize_text_answer(
                    row.get("answer"), choices, path=f"{row_path}.answer"
                )
                question = require_text(
                    row.get("question"), path=f"{row_path}.question"
                )
                media_path = resolve_media(
                    self.config.media_root,
                    (
                        raw_video,
                        Path(task) / raw_video,
                        Path("video") / task / raw_video,
                        Path("videos") / task / raw_video,
                    ),
                    require=self.config.require_media,
                    search_basename=Path(raw_video).name,
                )
                record_id = stable_id(self.name, task, source_id, raw_video, question)
                yield make_record(
                    record_id=record_id,
                    source="TVBench",
                    benchmark=f"tvbench:{task}",
                    task="mcq",
                    media_path=media_path,
                    question=question,
                    choices=choices,
                    answer=answer,
                    dataset=self.name,
                    split=self.config.split,
                    information_family=information_family,
                    question_family=f"tvbench:{task}",
                    reasoning_depth=depth,
                    resampling_unit_id=f"tvbench:video:{raw_video}",
                    pair_id=f"standalone:{record_id}",
                    pair_role=PairRole.STANDALONE.value,
                    evidence_spans=_evidence(row, task, path=row_path),
                    oracles=empty_oracles(),
                    provenance={
                        "source_id": f"{task}:{source_id}",
                        "raw_video": raw_video,
                        "annotation_file": str(annotation_path),
                        "source_split": "train",
                        "canonical_split_note": "TVBench publishes evaluation rows under HF split 'train'",
                        "answer_index_base": "text",
                        "extra_fields": {
                            key: row[key]
                            for key in (
                                "video_length",
                                "start",
                                "end",
                                "accurate_start",
                                "accurate_end",
                                "is_seq",
                                "question_id",
                            )
                            if key in row
                        },
                    },
                )
