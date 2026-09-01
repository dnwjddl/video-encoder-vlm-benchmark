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
    normalize_integer_answer,
    optional_source_id_component,
    parse_candidates,
    require_list,
    require_mapping,
    require_text,
    resolve_media,
    stable_id,
)


class EgoSchemaAdapter(BaseAdapter):
    """Adapter for EgoSchema's public 500-answer offline subset."""

    name = "egoschema"

    def _paths(self) -> tuple[Path, Path]:
        source = self.config.annotation_path
        if source.is_dir():
            questions_path = source / "questions.json"
            answers_path = source / "subset_answers.json"
        else:
            questions_path = source
            option = self.config.options.get("answers_path")
            if not option:
                raise AdapterError(
                    "EgoSchema file input requires answers_path=subset_answers.json"
                )
            answers_path = Path(option).expanduser().resolve()
        if not questions_path.is_file():
            raise AdapterError(f"EgoSchema questions file not found: {questions_path}")
        if not answers_path.is_file():
            raise AdapterError(
                f"EgoSchema public subset answer file not found: {answers_path}"
            )
        return questions_path, answers_path

    def iter_records(self) -> Iterable[dict[str, Any]]:
        questions_path, answers_path = self._paths()
        raw_questions = require_list(
            load_json(questions_path), path=str(questions_path)
        )
        answers = require_mapping(load_json(answers_path), path=str(answers_path))
        questions: dict[str, dict[str, Any]] = {}
        excluded_question_ids: set[str] = set()
        for index, raw_question in enumerate(raw_questions):
            path = f"{questions_path}[{index}]"
            raw_uid = (
                raw_question.get("q_uid") if isinstance(raw_question, Mapping) else None
            )
            uid_component = optional_source_id_component(raw_uid)
            source_id = uid_component or f"row:{index}"
            if self.skip_excluded(source_id, raw_location=path):
                if uid_component is not None:
                    excluded_question_ids.add(uid_component)
                continue
            question = require_mapping(raw_question, path=path)
            uid = require_text(question.get("q_uid"), path=f"{path}.q_uid")
            if uid in questions:
                raise AdapterError(f"{path}.q_uid: duplicate question ID {uid}")
            questions[uid] = question
        unknown_answers = set(answers) - set(questions) - excluded_question_ids
        if unknown_answers:
            raise AdapterError(
                "subset_answers.json references unknown questions: "
                + ", ".join(sorted(unknown_answers)[:10])
            )

        for uid in sorted(answers):
            if uid in excluded_question_ids:
                continue
            question = questions[uid]
            choices = parse_candidates(
                [question.get(f"option {index}") for index in range(5)],
                path=f"questions.{uid}.options",
            )
            answer = normalize_integer_answer(
                answers[uid], choices, base=0, path=f"subset_answers.{uid}"
            )
            media_path = resolve_media(
                self.config.media_root,
                (f"{uid}.mp4", Path("videos") / f"{uid}.mp4"),
                require=self.config.require_media,
                search_basename=f"{uid}.mp4",
            )
            record_id = stable_id(self.name, uid)
            yield make_record(
                record_id=record_id,
                source="EgoSchema",
                benchmark="egoschema:public_500",
                task="mcq",
                media_path=media_path,
                question=require_text(
                    question.get("question"), path=f"questions.{uid}.question"
                ),
                choices=choices,
                answer=answer,
                dataset=self.name,
                split=self.config.split,
                information_family=InformationFamily.LONG_RANGE_SELECTION.value,
                question_family="egoschema:long_range_compositional",
                reasoning_depth=2,
                resampling_unit_id=f"egoschema:video:{uid}",
                pair_id=f"standalone:{record_id}",
                pair_role=PairRole.STANDALONE.value,
                evidence_spans=[],
                oracles=empty_oracles(),
                provenance={
                    "source_id": uid,
                    "q_uid": uid,
                    "google_drive_id": question.get("google_drive_id"),
                    "questions_file": str(questions_path),
                    "answers_file": str(answers_path),
                    "source_split": "public_500",
                    "answer_index_base": 0,
                    "evidence_annotation_available": False,
                    "note": "Official EgoSchema does not release gold temporal evidence spans.",
                },
            )
