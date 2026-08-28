#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from vlmevalbench.data import format_choices, normalize_choice_answer, read_jsonl, write_jsonl
from vlmevalbench.utils import get_dtype


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
    parser = argparse.ArgumentParser(description="Evaluate MCQ questions with text only, no visual encoder/projector.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--llm-id", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--require-media", action="store_true")
    return parser.parse_args()


def make_prompt(record: dict[str, Any]) -> str:
    question = str(record.get("question") or "").strip()
    choices = record.get("choices") or []
    return (
        "No video or image is provided. Answer the multiple-choice question using only the text below. "
        "Return only the option letter.\n"
        f"Question: {question}\n"
        f"Options:\n{format_choices(choices)}\n"
        "Answer:"
    )


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device=device) for key, value in batch.items()}


@torch.no_grad()
def score_answer(
    *,
    tokenizer,
    llm,
    prompt: str,
    answer: str,
    device: torch.device,
    max_length: int,
) -> float:
    answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]
    answer_len = int(answer_ids.shape[1])
    prompt_budget = max(max_length - answer_len, 8)
    prompt_batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=prompt_budget)
    input_ids = torch.cat([prompt_batch["input_ids"], answer_ids], dim=1).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = input_ids.clone()
    labels[:, : prompt_batch["input_ids"].shape[1]] = -100
    outputs = llm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    valid = labels.ne(-100).sum().clamp_min(1)
    return float(outputs.loss.detach().float().item() * valid.item())


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {"ALL": rows}
    for row in rows:
        groups.setdefault(str(row.get("benchmark") or "unknown"), []).append(row)

    summary = []
    for group, items in sorted(groups.items()):
        correct = sum(bool(item["correct"]) for item in items)
        summary.append(
            {
                "group": group,
                "num_examples": len(items),
                "correct": correct,
                "accuracy": correct / max(len(items), 1),
            }
        )
    return summary


def main() -> None:
    args = parse_args()
    dtype = get_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    local_files_only = env_flag("VLMEB_LOCAL_FILES_ONLY")
    llm_source = resolve_hf_source(args.llm_id, local_files_only=local_files_only)

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

    records = [record for record in read_jsonl(args.manifest) if record.get("choices")]
    if args.require_media:
        records = [record for record in records if record.get("media_path") and Path(str(record["media_path"])).exists()]
    if args.limit:
        records = records[: args.limit]

    predictions = []
    for record in tqdm(records, desc="text_only_mcq"):
        choices = [str(choice) for choice in record.get("choices") or []]
        gold = normalize_choice_answer(record.get("answer"), choices)
        prompt = make_prompt(record)
        scores = {}
        for idx in range(len(choices)):
            label = chr(ord("A") + idx)
            scores[label] = score_answer(
                tokenizer=tokenizer,
                llm=llm,
                prompt=prompt,
                answer=label,
                device=device,
                max_length=args.max_length,
            )
        pred = min(scores, key=scores.get)
        predictions.append(
            {
                "id": str(record["id"]),
                "source": record.get("source", "unknown"),
                "benchmark": record.get("benchmark", record.get("source", "unknown")),
                "task_type": record.get("task_type", record.get("benchmark", "unknown")),
                "prediction": pred,
                "answer": gold,
                "correct": pred == gold,
                "scores": scores,
            }
        )

    write_jsonl(args.out_jsonl, predictions)
    summary = summarize(predictions)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_csv).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else ["group"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote text-only predictions to {args.out_jsonl}")
    print(f"Wrote text-only summary to {args.out_csv}")


if __name__ == "__main__":
    main()
