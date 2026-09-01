from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from information_upper_bound.schema import InformationFamily, PairRole

from .common import (
    AdapterError,
    BaseAdapter,
    empty_oracles,
    load_json_rows,
    make_record,
    normalize_text_answer,
    parse_candidates,
    require_text,
    resolve_media,
    source_id_component,
    stable_id,
)


PAIR_RE = re.compile(r"^(?P<pair>.+)_(?P<role>[12])$")


def _normalized(text: Any) -> str:
    return " ".join(str(text).casefold().split())


def _information_family(category: str) -> str:
    return {
        "human_object_interactions": InformationFamily.BINDING_TRACKING.value,
        "robot_object_interactions": InformationFamily.BINDING_TRACKING.value,
        "intuitive_physics": InformationFamily.CAUSAL_COMPOSITIONAL.value,
        "temporal_reasoning": InformationFamily.TEMPORAL_ORDER.value,
    }.get(category, InformationFamily.CAUSAL_COMPOSITIONAL.value)


class MVPAdapter(BaseAdapter):
    """Strict pair adapter for facebook/minimal_video_pairs."""

    name = "mvp"

    def iter_records(self) -> Iterable[dict[str, Any]]:
        rows = load_json_rows(self.config.annotation_path)
        default_category = str(
            self.config.options.get("category")
            or (
                self.config.annotation_path.parent.name
                if self.config.annotation_path.is_file()
                else "unknown"
            )
        )
        prepared: list[dict[str, Any]] = []
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

        exclusion_groups: dict[str, list[tuple[str, bool]]] = defaultdict(list)
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            raw_video_id = row.get("video_id")
            if not isinstance(raw_video_id, str):
                continue
            match = PAIR_RE.fullmatch(raw_video_id.strip())
            if match is None:
                continue
            exclusion_id = source_id_component(raw_video_id, fallback=f"row:{index}")
            exclusion_groups[match.group("pair")].append(
                (exclusion_id, self.is_excluded(exclusion_id))
            )
        for pair_key, statuses in exclusion_groups.items():
            excluded_ids = [source_id for source_id, excluded in statuses if excluded]
            if excluded_ids and len(excluded_ids) != len(statuses):
                missing = [
                    source_id for source_id, excluded in statuses if not excluded
                ]
                raise AdapterError(
                    "MVP exclusions must close over both videos in pair "
                    f"{pair_key!r}; also declare {missing}"
                )

        for index, row in enumerate(rows):
            path = f"{self.config.annotation_path}[{index}]"
            raw_video_id = row.get("video_id") if isinstance(row, Mapping) else None
            exclusion_id = source_id_component(raw_video_id, fallback=f"row:{index}")
            if self.skip_excluded(exclusion_id, raw_location=path):
                continue
            video_id = require_text(row.get("video_id"), path=f"{path}.video_id")
            match = PAIR_RE.fullmatch(video_id)
            if not match:
                raise AdapterError(
                    f"{path}.video_id: expected official trailing _1 or _2 pair role"
                )
            pair_key = match.group("pair")
            role_number = int(match.group("role"))
            raw_candidates = row.get("candidates")
            if hasattr(raw_candidates, "tolist"):
                raw_candidates = raw_candidates.tolist()
            choices = parse_candidates(raw_candidates, path=f"{path}.candidates")
            if len(choices) != 2:
                raise AdapterError(f"{path}.candidates: MVP pairs must be binary")
            answer = normalize_text_answer(
                row.get("answer"), choices, path=f"{path}.answer"
            )
            raw_video = require_text(row.get("video_path"), path=f"{path}.video_path")
            media_path = resolve_media(
                self.config.media_root,
                (raw_video, Path("videos") / raw_video),
                require=self.config.require_media,
                search_basename=None,
            )
            item = {
                "row_index": index,
                "video_id": video_id,
                "pair_key": pair_key,
                "role_number": role_number,
                "question": require_text(row.get("question"), path=f"{path}.question"),
                "choices": choices,
                "answer": answer,
                "answer_text": choices[ord(answer) - ord("A")],
                "raw_video": raw_video,
                "media_path": media_path,
                "source_dataset": str(row.get("source") or "unknown"),
                "category": str(row.get("category") or default_category),
            }
            prepared.append(item)
            groups[pair_key].append(item)

        for pair_key, group in groups.items():
            if len(group) != 2:
                raise AdapterError(
                    f"MVP pair {pair_key!r} must contain exactly two rows, got {len(group)}"
                )
            by_role = {item["role_number"]: item for item in group}
            if set(by_role) != {1, 2}:
                raise AdapterError(
                    f"MVP pair {pair_key!r} must contain roles _1 and _2 exactly once"
                )
            first, second = by_role[1], by_role[2]
            if first["category"] != second["category"]:
                raise AdapterError(
                    f"MVP pair {pair_key!r} crosses categories: "
                    f"{first['category']!r} vs {second['category']!r}"
                )
            if _normalized(first["question"]) != _normalized(second["question"]):
                raise AdapterError(
                    f"MVP pair {pair_key!r} does not have an identical question"
                )
            if [_normalized(value) for value in first["choices"]] != [
                _normalized(value) for value in second["choices"]
            ]:
                raise AdapterError(
                    f"MVP pair {pair_key!r} does not have identical ordered candidates"
                )
            if first["media_path"] == second["media_path"]:
                raise AdapterError(
                    f"MVP pair {pair_key!r} resolves both roles to the same video"
                )
            if _normalized(first["answer_text"]) == _normalized(second["answer_text"]):
                raise AdapterError(
                    f"MVP pair {pair_key!r} must reverse the semantic answer"
                )

        for item in sorted(
            prepared, key=lambda value: (value["pair_key"], value["role_number"])
        ):
            category = item["category"]
            pair_id = f"mvp:{category}:{item['pair_key']}"
            record_id = stable_id(self.name, category, item["video_id"])
            yield make_record(
                record_id=record_id,
                source="MVP",
                benchmark=f"mvp:{category}",
                task="mcq",
                media_path=item["media_path"],
                question=item["question"],
                choices=item["choices"],
                answer=item["answer"],
                dataset=self.name,
                split=self.config.split,
                information_family=_information_family(category),
                question_family=f"mvp:{category}",
                reasoning_depth=2,
                resampling_unit_id=f"mvp:video_family:{category}:{item['pair_key']}",
                pair_id=pair_id,
                pair_role=(
                    PairRole.ORIGINAL.value
                    if item["role_number"] == 1
                    else PairRole.COUNTERFACTUAL.value
                ),
                evidence_spans=[],
                oracles=empty_oracles(),
                provenance={
                    "source_id": item["video_id"],
                    "official_pair_key": item["pair_key"],
                    "official_pair_role": item["role_number"],
                    "raw_video": item["raw_video"],
                    "source_dataset": item["source_dataset"],
                    "category": category,
                    "annotation_file": str(self.config.annotation_path),
                    "answer_index_base": "text",
                    "evidence_annotation_available": False,
                },
            )
