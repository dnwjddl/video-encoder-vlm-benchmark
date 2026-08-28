#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from vlmevalbench.data import format_choices, normalize_choice_answer, read_jsonl, split_prompt_on_visual, write_jsonl
from vlmevalbench.projector import MLPProjector
from vlmevalbench.training import build_inputs_embeds_and_labels
from vlmevalbench.utils import get_dtype, load_json


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_hf_source(model_id: str, *, local_files_only: bool) -> str:
    if not local_files_only or Path(model_id).exists():
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=model_id, repo_type="model", local_files_only=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained projector on unified MCQ benchmark JSONL.")
    parser.add_argument("--bench-manifest", required=True)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--projector-ckpt", required=True)
    parser.add_argument("--projector-metadata", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--llm-id", default=None)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=4096)
    return parser.parse_args()


def load_feature_map(feature_index: str) -> dict[str, str]:
    return {str(item["id"]): item["feature_path"] for item in read_jsonl(feature_index)}


def make_mcq_prompt(record: dict) -> str:
    question = str(record.get("question") or "").strip()
    choices = record.get("choices") or []
    choice_text = format_choices(choices)
    media_word = "video" if record.get("media_type", "video") == "video" else "image"
    return (
        f"You are given a {media_word}. Answer the question using only the option letter.\n"
        "<VISUAL>\n"
        f"Question: {question}\n"
        f"Options:\n{choice_text}\n"
        "Answer:"
    )


@torch.no_grad()
def score_choice(
    *,
    tokenizer,
    llm,
    projector,
    features: torch.Tensor,
    prompt: str,
    choice_label: str,
    device: torch.device,
    dtype: torch.dtype,
    max_length: int,
) -> float:
    prefix, suffix = split_prompt_on_visual(prompt)
    features = features.unsqueeze(0).to(device=device, dtype=dtype)
    feature_mask = torch.ones(features.shape[:2], device=device, dtype=torch.bool)
    visual_embeds = projector(features)
    inputs_embeds, attention_mask, labels = build_inputs_embeds_and_labels(
        tokenizer=tokenizer,
        llm=llm,
        visual_embeds=visual_embeds,
        feature_mask=feature_mask,
        prefixes=[prefix],
        suffixes=[suffix],
        answers=[choice_label],
        max_length=max_length,
    )
    outputs = llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
    valid = labels.ne(-100).sum().clamp_min(1)
    return float(outputs.loss.detach().float().item() * valid.item())


def main() -> None:
    args = parse_args()
    metadata = load_json(args.projector_metadata)
    llm_id = args.llm_id or metadata["llm_id"]
    dtype = get_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    local_files_only = env_flag("VLMEB_LOCAL_FILES_ONLY")
    llm_source = resolve_hf_source(llm_id, local_files_only=local_files_only)

    tokenizer = AutoTokenizer.from_pretrained(
        llm_source,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        llm_source,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    ).to(device)
    llm.eval()
    llm.requires_grad_(False)

    projector = MLPProjector(
        input_dim=int(metadata["input_dim"]),
        output_dim=int(metadata["output_dim"]),
        hidden_dim=metadata.get("projector_hidden_dim"),
        depth=int(metadata.get("projector_depth", 2)),
    ).to(device=device, dtype=dtype)
    state = torch.load(args.projector_ckpt, map_location="cpu")
    projector.load_state_dict(state)
    projector.eval()

    records = read_jsonl(args.bench_manifest)
    feature_map = load_feature_map(args.feature_index)
    predictions = []
    grouped: dict[str, list[bool]] = {}

    for record in tqdm(records, desc="eval"):
        record_id = str(record["id"])
        choices = record.get("choices") or []
        if not choices:
            continue
        if record_id not in feature_map:
            continue
        features = torch.load(feature_map[record_id], map_location="cpu")["features"].float()
        prompt = make_mcq_prompt(record)

        scores = {}
        for idx in range(len(choices)):
            label = chr(ord("A") + idx)
            scores[label] = score_choice(
                tokenizer=tokenizer,
                llm=llm,
                projector=projector,
                features=features,
                prompt=prompt,
                choice_label=label,
                device=device,
                dtype=dtype,
                max_length=args.max_length,
            )
        pred = min(scores, key=scores.get)
        gold = normalize_choice_answer(record.get("answer"), [str(choice) for choice in choices])
        correct = pred == gold
        benchmark = str(record.get("benchmark") or record.get("source") or "unknown")
        task_type = str(record.get("task_type") or benchmark)
        grouped.setdefault("ALL", []).append(correct)
        grouped.setdefault(benchmark, []).append(correct)
        predictions.append(
            {
                "id": record_id,
                "source": record.get("source", "unknown"),
                "benchmark": benchmark,
                "task_type": task_type,
                "prediction": pred,
                "answer": gold,
                "correct": correct,
                "scores": scores,
            }
        )

    write_jsonl(args.out_jsonl, predictions)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "num_examples", "accuracy"])
        writer.writeheader()
        for benchmark, values in sorted(grouped.items()):
            writer.writerow(
                {
                    "benchmark": benchmark,
                    "num_examples": len(values),
                    "accuracy": sum(values) / max(len(values), 1),
                }
            )
    print(f"Wrote predictions to {args.out_jsonl}")
    print(f"Wrote benchmark summary to {args.out_csv}")


if __name__ == "__main__":
    main()
