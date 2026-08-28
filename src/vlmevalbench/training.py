from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from vlmevalbench.data import make_prompt, read_jsonl, records_by_id, split_prompt_on_visual


@dataclass
class FeatureExample:
    record: dict[str, Any]
    feature_path: str


class FeatureTextDataset(Dataset):
    def __init__(self, manifest_path: str | Path, feature_index_path: str | Path) -> None:
        records = records_by_id(read_jsonl(manifest_path))
        feature_index = read_jsonl(feature_index_path)
        self.examples: list[FeatureExample] = []
        missing = 0
        for item in feature_index:
            record = records.get(str(item["id"]))
            if record is None:
                missing += 1
                continue
            self.examples.append(FeatureExample(record=record, feature_path=item["feature_path"]))
        if not self.examples:
            raise ValueError("No examples matched between manifest and feature index.")
        if missing:
            print(f"Warning: {missing} feature rows had no matching manifest record.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        loaded = torch.load(example.feature_path, map_location="cpu")
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
        prefix_ids = tokenizer(prefixes[idx], add_special_tokens=True, return_tensors="pt").input_ids[0].to(device)
        suffix_ids = tokenizer(suffixes[idx], add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)
        answer_text = str(answers[idx]).strip()
        answer_ids = tokenizer(" " + answer_text + eos, add_special_tokens=False, return_tensors="pt").input_ids[0].to(device)

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
    padded_embeds = torch.zeros(len(all_embeds), max_seq, hidden, device=device, dtype=dtype)
    attention_mask = torch.zeros(len(all_embeds), max_seq, device=device, dtype=torch.long)
    padded_labels = torch.full((len(all_embeds), max_seq), -100, device=device, dtype=torch.long)

    for idx, (embeds, labels) in enumerate(zip(all_embeds, all_labels)):
        seq_len = embeds.shape[0]
        padded_embeds[idx, :seq_len] = embeds
        attention_mask[idx, :seq_len] = 1
        padded_labels[idx, :seq_len] = labels

    return padded_embeds, attention_mask, padded_labels
