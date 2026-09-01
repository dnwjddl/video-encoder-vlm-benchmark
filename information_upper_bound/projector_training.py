from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from information_upper_bound.integrity import tensor_identity
from information_upper_bound.io import iter_jsonl, sha256_file
from vlmevalbench.data import make_prompt, split_prompt_on_visual


@dataclass
class FeatureExample:
    record: dict[str, Any]
    feature_path: str
    feature_index_row: dict[str, Any] | None = None


class FeatureTextDataset(Dataset):
    _INTEGRITY_FIELDS = (
        "schema_version",
        "visual_id",
        "view_content_hash",
        "feature_content_hash",
        "encoder_config",
        "extraction_identity",
        "media_content_identity",
        "decoded_frame_identity",
        "sampling",
        "feature_tensor_identity",
        "feature_artifact_identity_sha256",
    )

    def __init__(
        self,
        manifest_path: str | Path,
        feature_index_path: str | Path,
        *,
        require_integrity: bool = False,
    ) -> None:
        records = list(iter_jsonl(manifest_path))
        feature_index = list(iter_jsonl(feature_index_path))
        index_root = Path(feature_index_path).resolve().parent
        features_by_visual: dict[str, dict[str, Any]] = {}
        for item in feature_index:
            key = str(item.get("visual_id", item.get("id", ""))).strip()
            if not key:
                raise ValueError("Feature index row has no visual_id/id.")
            if key in features_by_visual:
                raise ValueError(f"Duplicate visual feature key: {key}")
            if require_integrity:
                required = set(self._INTEGRITY_FIELDS) | {
                    "feature_file_sha256",
                    "shape",
                }
                missing = sorted(required - set(item))
                if missing:
                    raise ValueError(
                        f"Feature index row {key!r} lacks required integrity fields: {missing}"
                    )
            features_by_visual[key] = item
        self.examples: list[FeatureExample] = []
        missing_records = 0
        seen_record_ids: set[str] = set()
        used_visuals: set[str] = set()
        for record in records:
            record_id = str(record.get("id", "")).strip()
            if not record_id or record_id in seen_record_ids:
                raise ValueError(
                    f"Manifest has empty/duplicate record id: {record_id!r}"
                )
            seen_record_ids.add(record_id)
            visual_key = str(record.get("visual_id") or record_id)
            item = features_by_visual.get(visual_key)
            if item is None:
                missing_records += 1
                continue
            feature_path = Path(str(item["feature_path"]))
            if not feature_path.is_absolute():
                feature_path = (index_root / feature_path).resolve()
            self.examples.append(
                FeatureExample(
                    record=record,
                    feature_path=str(feature_path),
                    feature_index_row=dict(item) if require_integrity else None,
                )
            )
            used_visuals.add(visual_key)
        if not self.examples:
            raise ValueError("No examples matched between manifest and feature index.")
        if missing_records:
            print(
                f"Warning: {missing_records} manifest rows had no matching visual feature."
            )
        unused_features = len(set(features_by_visual) - used_visuals)
        if unused_features:
            print(
                f"Warning: {unused_features} feature rows had no matching manifest record."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        expected = example.feature_index_row
        if expected is not None:
            actual_file_sha256 = sha256_file(example.feature_path)
            if actual_file_sha256 != str(expected["feature_file_sha256"]):
                raise ValueError(
                    f"Training feature file digest mismatch: {example.feature_path}"
                )
        loaded = torch.load(example.feature_path, map_location="cpu")
        if not isinstance(loaded, Mapping) or "features" not in loaded:
            raise ValueError(
                f"Training feature artifact is malformed: {example.feature_path}"
            )
        if expected is not None:
            mismatches = [
                field
                for field in self._INTEGRITY_FIELDS
                if loaded.get(field) != expected[field]
            ]
            if mismatches:
                raise ValueError(
                    f"Training feature index/artifact metadata mismatch: {mismatches}"
                )
            if (
                tensor_identity(loaded["features"])
                != expected["feature_tensor_identity"]
            ):
                raise ValueError(
                    f"Training feature tensor digest mismatch: {example.feature_path}"
                )
            if list(loaded["features"].shape) != list(expected["shape"]):
                raise ValueError(
                    f"Training feature tensor shape mismatch: {example.feature_path}"
                )
        prompt, answer = make_prompt(example.record)
        prefix, suffix = split_prompt_on_visual(prompt)
        return {
            "id": example.record["id"],
            "features": loaded["features"].float(),
            "prefix": prefix,
            "suffix": suffix,
            "answer": answer,
            "record": example.record,
        }


def collate_feature_text(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_tokens = max(item["features"].shape[0] for item in batch)
    feat_dim = batch[0]["features"].shape[-1]
    features = torch.zeros(len(batch), max_tokens, feat_dim, dtype=torch.float32)
    feature_mask = torch.zeros(len(batch), max_tokens, dtype=torch.bool)
    for idx, item in enumerate(batch):
        n = item["features"].shape[0]
        features[idx, :n] = item["features"]
        feature_mask[idx, :n] = True
    return {
        "ids": [item["id"] for item in batch],
        "features": features,
        "feature_mask": feature_mask,
        "prefixes": [item["prefix"] for item in batch],
        "suffixes": [item["suffix"] for item in batch],
        "answers": [item["answer"] for item in batch],
        "records": [item["record"] for item in batch],
    }


def build_inputs_embeds_and_labels(
    *,
    tokenizer,
    llm,
    visual_embeds: torch.Tensor,
    feature_mask: torch.Tensor,
    prefixes: list[str],
    suffixes: list[str],
    answers: list[str],
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = visual_embeds.device
    embed_layer = llm.get_input_embeddings()
    eos = tokenizer.eos_token or ""
    all_embeds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for idx in range(visual_embeds.shape[0]):
        prefix_ids = (
            tokenizer(prefixes[idx], add_special_tokens=True, return_tensors="pt")
            .input_ids[0]
            .to(device)
        )
        suffix_ids = (
            tokenizer(suffixes[idx], add_special_tokens=False, return_tensors="pt")
            .input_ids[0]
            .to(device)
        )
        answer_text = str(answers[idx]).strip()
        answer_ids = (
            tokenizer(
                " " + answer_text + eos, add_special_tokens=False, return_tensors="pt"
            )
            .input_ids[0]
            .to(device)
        )

        vis = visual_embeds[idx][feature_mask[idx]]
        text_len = len(prefix_ids) + len(suffix_ids) + len(answer_ids)
        vis_budget = max(max_length - text_len, 1)
        if vis.shape[0] > vis_budget:
            vis = vis[:vis_budget]

        prefix_emb = embed_layer(prefix_ids)
        suffix_emb = embed_layer(suffix_ids)
        answer_emb = embed_layer(answer_ids)
        vis = vis.to(dtype=prefix_emb.dtype)
        embeds = torch.cat([prefix_emb, vis, suffix_emb, answer_emb], dim=0)

        labels = torch.full((embeds.shape[0],), -100, device=device, dtype=torch.long)
        labels[-len(answer_ids) :] = answer_ids

        if embeds.shape[0] > max_length:
            embeds = embeds[-max_length:]
            labels = labels[-max_length:]

        all_embeds.append(embeds)
        all_labels.append(labels)

    max_seq = max(x.shape[0] for x in all_embeds)
    hidden = all_embeds[0].shape[-1]
    dtype = all_embeds[0].dtype
    padded_embeds = torch.zeros(
        len(all_embeds), max_seq, hidden, device=device, dtype=dtype
    )
    attention_mask = torch.zeros(
        len(all_embeds), max_seq, device=device, dtype=torch.long
    )
    padded_labels = torch.full(
        (len(all_embeds), max_seq), -100, device=device, dtype=torch.long
    )

    for idx, (embeds, labels) in enumerate(zip(all_embeds, all_labels)):
        seq_len = embeds.shape[0]
        padded_embeds[idx, :seq_len] = embeds
        attention_mask[idx, :seq_len] = 1
        padded_labels[idx, :seq_len] = labels

    return padded_embeds, attention_mask, padded_labels
