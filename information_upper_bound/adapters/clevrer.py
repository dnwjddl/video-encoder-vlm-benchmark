from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from information_upper_bound.schema import InformationFamily, PairRole

from .common import (
    AdapterError,
    BaseAdapter,
    empty_oracles,
    load_json,
    make_record,
    optional_source_id_component,
    require_list,
    require_mapping,
    require_text,
    resolve_media,
    stable_id,
)


DEPTH = {
    "explanatory": 2,
    "predictive": 2,
    "counterfactual": 3,
}


def _scene_annotation_path(root: Path, scene_index: int) -> Path:
    expected_name = f"sim_{scene_index:05d}.json"
    if root.is_file():
        if root.name != expected_name:
            raise AdapterError(
                f"CLEVRER scene annotation file for scene {scene_index} must be named "
                f"{expected_name}, got {root.name}"
            )
        return root
    candidate = (root / expected_name).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise AdapterError(
            f"CLEVRER scene annotation escapes sidecar root: {candidate}"
        ) from exc
    if not candidate.is_file():
        raise AdapterError(
            f"CLEVRER scene annotation missing for scene {scene_index}: {candidate}"
        )
    return candidate


def _object_description(value: Mapping[str, Any]) -> str:
    return f"{value['color']} {value['material']} {value['shape']}"


def _scene_oracles(path: Path, *, scene_index: int) -> dict[str, Any]:
    scene = require_mapping(load_json(path), path=str(path))
    ground_truth = require_mapping(
        scene.get("ground_truth"), path=f"{path}.ground_truth"
    )
    raw_objects = require_list(
        ground_truth.get("objects"), path=f"{path}.ground_truth.objects"
    )
    raw_collisions = require_list(
        ground_truth.get("collisions"), path=f"{path}.ground_truth.collisions"
    )
    objects: dict[int, dict[str, Any]] = {}
    for index, raw_object in enumerate(raw_objects):
        object_path = f"{path}.ground_truth.objects[{index}]"
        item = require_mapping(raw_object, path=object_path)
        object_id = item.get("id")
        if isinstance(object_id, bool) or not isinstance(object_id, int):
            raise AdapterError(f"{object_path}.id: expected an integer")
        if object_id in objects:
            raise AdapterError(f"{object_path}.id: duplicate object id {object_id}")
        if object_id < 0:
            raise AdapterError(f"{object_path}.id: must be non-negative")
        objects[object_id] = {
            "entity_id": object_id,
            "color": require_text(item.get("color"), path=f"{object_path}.color"),
            "material": require_text(
                item.get("material"), path=f"{object_path}.material"
            ),
            "shape": require_text(item.get("shape"), path=f"{object_path}.shape"),
        }
    if not objects:
        raise AdapterError(f"{path}.ground_truth.objects: must not be empty")

    oracles = empty_oracles()
    for object_id in sorted(objects):
        item = objects[object_id]
        oracles["static_facts"].append(
            {
                **item,
                "text": f"Object {object_id} is a {_object_description(item)}.",
                "source": "ground_truth.objects",
            }
        )

    unordered_collision_events: list[dict[str, Any]] = []
    ordered_collision_events: list[dict[str, Any]] = []
    seen_collisions: set[tuple[int, int, int]] = set()
    for index, raw_collision in enumerate(raw_collisions):
        collision_path = f"{path}.ground_truth.collisions[{index}]"
        collision = require_mapping(raw_collision, path=collision_path)
        frame = collision.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise AdapterError(
                f"{collision_path}.frame: expected a non-negative integer"
            )
        pair = require_list(collision.get("object"), path=f"{collision_path}.object")
        if len(pair) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) for value in pair
        ):
            raise AdapterError(
                f"{collision_path}.object: expected two integer object ids"
            )
        first_id, second_id = pair
        if first_id == second_id:
            raise AdapterError(
                f"{collision_path}.object: collision objects must be distinct"
            )
        missing = [object_id for object_id in pair if object_id not in objects]
        if missing:
            raise AdapterError(
                f"{collision_path}.object: references unknown object ids {missing}"
            )
        collision_key = (frame, *sorted((first_id, second_id)))
        if collision_key in seen_collisions:
            raise AdapterError(f"{collision_path}: duplicate collision {collision_key}")
        seen_collisions.add(collision_key)
        first = objects[first_id]
        second = objects[second_id]
        event_id = stable_id(
            "clevrer_collision", scene_index, index, first_id, second_id
        )
        identity = {
            "event_id": event_id,
            "event_type": "collision",
            "object_ids": [first_id, second_id],
            "objects": [dict(first), dict(second)],
            "source": "ground_truth.collisions",
        }
        unordered_collision_events.append(
            {
                **identity,
                "text": (
                    f"Object {first_id} ({_object_description(first)}) collides with "
                    f"object {second_id} ({_object_description(second)})."
                ),
            }
        )
        ordered_collision_events.append(
            {
                **identity,
                "frame": frame,
                "unit": "frame_index",
                "text": (
                    f"Object {first_id} ({_object_description(first)}) collides with "
                    f"object {second_id} ({_object_description(second)})."
                ),
            }
        )
    # Atomic collision semantics are deliberately ordered by opaque identity,
    # not by frame, so this condition does not smuggle temporal order.
    oracles["unordered_events"] = sorted(
        unordered_collision_events, key=lambda value: value["event_id"]
    )
    oracles["ordered_events"] = sorted(
        ordered_collision_events,
        key=lambda value: (
            value["frame"],
            min(value["object_ids"]),
            max(value["object_ids"]),
        ),
    )
    return oracles


class CLEVRERAdapter(BaseAdapter):
    """Choice-level binary adapter for CLEVRER's multi-label questions.

    CLEVRER marks every candidate as ``correct`` or ``wrong``; it is not a
    single-answer MCQ. Each candidate therefore becomes one Yes/No row, while
    provenance retains the original question group for exact-question scoring.
    """

    name = "clevrer"

    def iter_records(self) -> Iterable[dict[str, Any]]:
        scenes = require_list(
            load_json(self.config.annotation_path),
            path=str(self.config.annotation_path),
        )
        source_split = str(
            self.config.options.get("source_split") or self.config.annotation_path.stem
        )
        scene_option = self.config.options.get("scene_annotations_path")
        scene_root = Path(scene_option).expanduser().resolve() if scene_option else None
        if scene_root is not None and not (scene_root.is_file() or scene_root.is_dir()):
            raise AdapterError(
                f"CLEVRER scene annotations path does not exist: {scene_root}"
            )
        for scene_offset, raw_scene in enumerate(scenes):
            scene_path = f"{self.config.annotation_path}[{scene_offset}]"
            scene = require_mapping(raw_scene, path=scene_path)
            scene_index = scene.get("scene_index")
            if isinstance(scene_index, bool) or not isinstance(scene_index, int):
                raise AdapterError(f"{scene_path}.scene_index: expected an integer")
            if scene_index < 0:
                raise AdapterError(f"{scene_path}.scene_index: must be non-negative")
            scene_annotation_path = (
                _scene_annotation_path(scene_root, scene_index)
                if scene_root is not None
                else None
            )
            scene_oracles = (
                _scene_oracles(scene_annotation_path, scene_index=scene_index)
                if scene_annotation_path is not None
                else empty_oracles()
            )
            filename = require_text(
                scene.get("video_filename"), path=f"{scene_path}.video_filename"
            )
            media_path = resolve_media(
                self.config.media_root,
                (
                    filename,
                    Path(source_split) / filename,
                    Path("videos") / source_split / filename,
                    Path(f"video_{source_split}") / filename,
                ),
                require=self.config.require_media,
                search_basename=Path(filename).name,
            )
            questions = require_list(
                scene.get("questions"), path=f"{scene_path}.questions"
            )
            seen_qids: set[str] = set()
            for question_offset, raw_question in enumerate(questions):
                path = f"{scene_path}.questions[{question_offset}]"
                question = require_mapping(raw_question, path=path)
                raw_qid = question.get("question_id")
                if (
                    isinstance(raw_qid, bool)
                    or not isinstance(raw_qid, int)
                    or raw_qid < 0
                ):
                    raise AdapterError(
                        f"{path}.question_id: expected a non-negative integer"
                    )
                qid = str(raw_qid)
                if qid in seen_qids:
                    raise AdapterError(
                        f"{path}.question_id: duplicate within scene {scene_index}"
                    )
                seen_qids.add(qid)
                question_type = require_text(
                    question.get("question_type"), path=f"{path}.question_type"
                ).casefold()
                raw_choices = question.get("choices")
                if question_type == "descriptive":
                    if raw_choices not in (None, []):
                        raise AdapterError(
                            f"{path}: descriptive question unexpectedly contains choices"
                        )
                    continue
                if question_type not in DEPTH:
                    raise AdapterError(
                        f"{path}: unsupported CLEVRER question type {question_type!r}"
                    )
                choices = require_list(raw_choices, path=f"{path}.choices")
                if not choices:
                    raise AdapterError(
                        f"{path}.choices: multi-label question has no candidates"
                    )
                question_text = require_text(
                    question.get("question"), path=f"{path}.question"
                )
                question_program = require_list(
                    question.get("program", []), path=f"{path}.program"
                )
                question_group = f"clevrer:{scene_index}:{qid}"
                choice_exclusion_ids: list[str] = []
                for choice_offset, raw_choice in enumerate(choices):
                    raw_choice_id = (
                        raw_choice.get("choice_id")
                        if isinstance(raw_choice, Mapping)
                        else None
                    )
                    fallback_id = f"row:{scene_offset}:question:{question_offset}:choice:{choice_offset}"
                    choice_id_component = optional_source_id_component(raw_choice_id)
                    choice_exclusion_ids.append(
                        f"{scene_index}:{qid}:{choice_id_component}"
                        if choice_id_component is not None
                        else fallback_id
                    )
                excluded_choices = [
                    source_id
                    for source_id in choice_exclusion_ids
                    if self.is_excluded(source_id)
                ]
                if excluded_choices and len(excluded_choices) != len(choices):
                    missing = sorted(set(choice_exclusion_ids) - set(excluded_choices))
                    raise AdapterError(
                        "CLEVRER exclusions must close over every binary candidate in "
                        f"official question {question_group}; also declare {missing}"
                    )
                seen_choice_ids: set[str] = set()
                for choice_offset, raw_choice in enumerate(choices):
                    choice_path = f"{path}.choices[{choice_offset}]"
                    raw_choice_id = (
                        raw_choice.get("choice_id")
                        if isinstance(raw_choice, Mapping)
                        else None
                    )
                    fallback_id = f"row:{scene_offset}:question:{question_offset}:choice:{choice_offset}"
                    choice_id_component = optional_source_id_component(raw_choice_id)
                    exclusion_id = choice_exclusion_ids[choice_offset]
                    if self.skip_excluded(exclusion_id, raw_location=choice_path):
                        continue
                    choice = require_mapping(raw_choice, path=choice_path)
                    raw_choice_id = choice.get("choice_id")
                    if (
                        isinstance(raw_choice_id, bool)
                        or not isinstance(raw_choice_id, int)
                        or raw_choice_id < 0
                    ):
                        raise AdapterError(
                            f"{choice_path}.choice_id: expected a non-negative integer"
                        )
                    choice_id = str(raw_choice_id)
                    if choice_id in seen_choice_ids:
                        raise AdapterError(
                            f"{choice_path}.choice_id: duplicate within question"
                        )
                    seen_choice_ids.add(choice_id)
                    statement = require_text(
                        choice.get("choice"), path=f"{choice_path}.choice"
                    )
                    label = require_text(
                        choice.get("answer"), path=f"{choice_path}.answer"
                    ).casefold()
                    if label not in {"correct", "wrong"}:
                        raise AdapterError(
                            f"{choice_path}.answer: expected official label 'correct' or 'wrong', got {label!r}"
                        )
                    choice_program = require_list(
                        choice.get("program", []), path=f"{choice_path}.program"
                    )
                    record_id = stable_id(self.name, scene_index, qid, choice_id)
                    oracles = {
                        key: list(value) if isinstance(value, list) else value
                        for key, value in scene_oracles.items()
                    }
                    if question_program or choice_program:
                        oracles["operator"] = [
                            {
                                "question_program": list(question_program),
                                "choice_program": list(choice_program),
                                "composition": "choice_program + question_program",
                                "access": "operator_only",
                            }
                        ]
                    binary_question = (
                        f"{question_text}\n"
                        f"Candidate statement: {statement}\n"
                        "Is this candidate statement correct?"
                    )
                    yield make_record(
                        record_id=record_id,
                        source="CLEVRER",
                        benchmark=f"clevrer:{question_type}",
                        task="binary_mcq",
                        media_path=media_path,
                        question=binary_question,
                        choices=["No", "Yes"],
                        answer="B" if label == "correct" else "A",
                        dataset=self.name,
                        split=self.config.split,
                        information_family=InformationFamily.CAUSAL_COMPOSITIONAL.value,
                        question_family=f"clevrer:{question_type}",
                        reasoning_depth=DEPTH[question_type],
                        resampling_unit_id=f"clevrer:scene:{scene_index}",
                        pair_id=f"standalone:{record_id}",
                        pair_role=PairRole.STANDALONE.value,
                        evidence_spans=[],
                        oracles=oracles,
                        provenance={
                            "source_id": f"{scene_index}:{qid}:{choice_id}",
                            "scene_index": scene_index,
                            "question_id": qid,
                            "choice_id": choice_id,
                            "question_group": question_group,
                            "question_type": question_type,
                            "question_subtype": question.get("question_subtype"),
                            "original_question": question_text,
                            "candidate_statement": statement,
                            "official_choice_label": label,
                            "annotation_file": str(self.config.annotation_path),
                            "scene_annotations_file": (
                                str(scene_annotation_path)
                                if scene_annotation_path
                                else None
                            ),
                            "scene_annotations_schema": (
                                "processed_proposals.ground_truth"
                                if scene_annotation_path
                                else None
                            ),
                            "source_split": source_split,
                            "answer_index_base": "official correct/wrong converted to No/Yes",
                            "aggregation": (
                                "primary official-question exact-set accuracy; candidate metrics "
                                "are diagnostic only"
                            ),
                        },
                        extra_diagnostic={
                            "independent_unit_id": question_group,
                            "official_candidate_id": choice_id,
                            "official_candidate_count": len(choices),
                        },
                    )
