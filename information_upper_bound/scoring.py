from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from vlmevalbench.projector import MLPProjector
from information_upper_bound.projector_training import build_inputs_embeds_and_labels
from vlmevalbench.utils import get_dtype
from information_upper_bound.integrity import (
    resolved_pretrained_identity,
    validate_locked_pretrained_revision,
)


SCORING_PROTOCOL_VERSION = "information_upper_bound.mcq_nll.v3.integrity_locked"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_hf_source(
    model_id: str,
    *,
    revision: str | None,
    local_files_only: bool,
) -> str:
    if not local_files_only or Path(model_id).exists():
        return model_id
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id=model_id,
        repo_type="model",
        revision=revision,
        local_files_only=True,
    )


def make_prompt(trial: Mapping[str, Any]) -> tuple[str, str | None, str]:
    """Return prefix, optional visual insertion marker, and suffix.

    Every condition uses the same answer-scoring instruction and raw prompt
    format. Only the explicitly declared input channels differ.
    """

    condition = trial.get("condition") or {}
    channel = str(condition.get("input_channel", ""))
    media_word = "video" if trial.get("media_type", "video") == "video" else "image"
    question = str(trial.get("question", "")).strip()
    choices = [str(value) for value in trial.get("choices") or []]
    choice_text = "\n".join(
        f"{chr(ord('A') + index)}. {choice}" for index, choice in enumerate(choices)
    )
    clue = str(trial.get("clue_text", "")).strip()

    # Keep the language surface identical across interventions.  Otherwise a
    # visual-vs-question-only gain is confounded by telling the LLM whether a
    # video or continuous evidence exists.  This prefix also matches the
    # repository's projector-training MCQ prompt.
    prefix = f"You are given a {media_word}. Answer the question using only the option letter.\n"
    if channel in {"visual", "visual_plus_text"}:
        marker = "<VISUAL>"
        clue_block = f"\n{clue}\n" if channel == "visual_plus_text" else "\n"
    elif channel == "embedding_oracle":
        marker = "<VISUAL>"
        clue_block = "\n"
    elif channel in {"question_only", "text_oracle"}:
        marker = None
        clue_block = f"{clue}\n" if channel == "text_oracle" else ""
    else:
        raise ValueError(f"Unknown input channel: {channel!r}")

    suffix = f"{clue_block}Question: {question}\nOptions:\n{choice_text}\nAnswer:"
    return prefix, marker, suffix


@dataclass(frozen=True)
class ScoreResult:
    prediction: str
    prediction_text: str
    choice_nll: dict[str, float]
    choice_probability: dict[str, float]
    gold_nll: float
    best_distractor_nll: float
    gold_margin: float
    correct: bool
    prompt_tokens: int
    original_visual_tokens: int
    effective_visual_tokens: int
    token_source: str


class FrozenMultipleChoiceScorer:
    """One long-lived frozen LLM/projector scorer for every input condition."""

    def __init__(
        self,
        *,
        projector_checkpoint: str | Path,
        projector_metadata: Mapping[str, Any],
        llm_id: str | None = None,
        llm_revision: str | None = None,
        device: str = "cuda",
        dtype: str = "bf16",
        max_length: int = 4096,
        overflow_policy: str = "error",
    ) -> None:
        if overflow_policy not in {"error", "truncate_visual"}:
            raise ValueError("overflow_policy must be 'error' or 'truncate_visual'")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.metadata = dict(projector_metadata)
        self.llm_id = str(llm_id or self.metadata["llm_id"])
        self.llm_revision = str(llm_revision).strip() if llm_revision else None
        self.max_length = int(max_length)
        self.overflow_policy = overflow_policy
        self.dtype = get_dtype(dtype)
        self.device = torch.device(
            device if torch.cuda.is_available() or device == "cpu" else "cpu"
        )
        local_files_only = _env_flag("VLMEB_LOCAL_FILES_ONLY")
        llm_source = _resolve_hf_source(
            self.llm_id,
            revision=self.llm_revision,
            local_files_only=local_files_only,
        )
        revision_kwargs = (
            {"revision": self.llm_revision}
            if self.llm_revision is not None and not Path(llm_source).exists()
            else {}
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_source,
            trust_remote_code=True,
            use_fast=True,
            local_files_only=local_files_only,
            **revision_kwargs,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_source,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
            **revision_kwargs,
        ).to(self.device)
        self.llm.eval()
        self.llm.requires_grad_(False)
        self.pretrained_identity = resolved_pretrained_identity(
            requested_id=self.llm_id,
            resolved_source=llm_source,
            model=self.llm,
            auxiliaries={"tokenizer": self.tokenizer},
        )
        if self.pretrained_identity["identity_strength"] == "weak_mutable_identifier":
            raise ValueError(
                "could not resolve a strong LLM/tokenizer revision identity; pin model.llm_revision "
                "in the protocol or use a content-addressed local snapshot"
            )
        if self.llm_revision is not None:
            try:
                validate_locked_pretrained_revision(
                    self.pretrained_identity,
                    self.llm_revision,
                    component_name="LLM/tokenizer",
                )
            except ValueError as exc:
                raise ValueError(
                    "loaded LLM/tokenizer content identity does not match the locked llm_revision"
                ) from exc

        output_dim = int(self.metadata["output_dim"])
        hidden_size = int(self.llm.config.hidden_size)
        if output_dim != hidden_size:
            raise ValueError(
                f"projector output_dim={output_dim} does not match LLM hidden_size={hidden_size}"
            )
        self.projector = MLPProjector(
            input_dim=int(self.metadata["input_dim"]),
            output_dim=output_dim,
            hidden_dim=self.metadata.get("projector_hidden_dim"),
            depth=int(self.metadata.get("projector_depth", 2)),
        ).to(device=self.device, dtype=self.dtype)
        state = torch.load(projector_checkpoint, map_location="cpu")
        self.projector.load_state_dict(state)
        self.projector.eval()
        self.projector.requires_grad_(False)

    def _answer_ids(self, label: str) -> torch.Tensor:
        eos = self.tokenizer.eos_token or ""
        return self.tokenizer(
            " " + label + eos,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]

    def _preflight(
        self,
        *,
        prefix: str,
        suffix: str,
        choice_labels: Sequence[str],
        visual_tokens: int,
    ) -> tuple[int, int]:
        prefix_ids = self.tokenizer(
            prefix, add_special_tokens=True, return_tensors="pt"
        )["input_ids"][0]
        suffix_ids = self.tokenizer(
            suffix, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        answer_max = max(len(self._answer_ids(label)) for label in choice_labels)
        prompt_tokens = len(prefix_ids) + len(suffix_ids)
        text_total = prompt_tokens + answer_max
        if text_total > self.max_length:
            raise ValueError(
                f"text prompt requires {text_total} tokens, exceeding max_length={self.max_length}; "
                "text truncation is disabled because it would change the question"
            )
        budget = max(self.max_length - text_total, 0)
        if visual_tokens > 0 and budget < 1:
            raise ValueError(
                f"text leaves no room for a visual token at max_length={self.max_length}"
            )
        if visual_tokens > budget and self.overflow_policy == "error":
            raise ValueError(
                f"visual sequence has {visual_tokens} tokens but only {budget} fit; "
                "raise max_length or use --overflow-policy truncate_visual"
            )
        return prompt_tokens, min(visual_tokens, budget)

    @staticmethod
    def _sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        losses = F.cross_entropy(
            shift_logits.transpose(1, 2),
            shift_labels,
            ignore_index=-100,
            reduction="none",
        )
        return (losses * shift_labels.ne(-100)).sum(dim=1)

    @torch.inference_mode()
    def score(
        self, trial: Mapping[str, Any], features: torch.Tensor | None
    ) -> ScoreResult:
        choices = [str(value) for value in trial.get("choices") or []]
        if not 2 <= len(choices) <= 26:
            raise ValueError("trial must contain 2..26 choices")
        labels = [chr(ord("A") + index) for index in range(len(choices))]
        gold = str(trial.get("answer", "")).strip().upper()
        if gold not in labels:
            raise ValueError(f"gold answer {gold!r} is not one of {labels}")
        prefix, marker, suffix = make_prompt(trial)
        channel = str((trial.get("condition") or {}).get("input_channel", ""))
        is_embedding_oracle = channel == "embedding_oracle"
        needs_external_visual = channel in {"visual", "visual_plus_text"}
        if needs_external_visual != (features is not None):
            raise ValueError(
                f"condition expects external visual={needs_external_visual}, but features were "
                f"{'provided' if features is not None else 'not provided'}"
            )

        oracle_ids: torch.Tensor | None = None
        if is_embedding_oracle:
            clue_text = str(trial.get("clue_text", "")).strip()
            if not clue_text:
                raise ValueError("embedding_oracle requires non-empty clue_text")
            oracle_ids = self.tokenizer(
                clue_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
        original_visual_tokens = (
            int(features.shape[0])
            if features is not None
            else int(len(oracle_ids))
            if oracle_ids is not None
            else 0
        )
        prompt_tokens, effective_visual_tokens = self._preflight(
            prefix=prefix,
            suffix=suffix,
            choice_labels=labels,
            visual_tokens=original_visual_tokens,
        )

        if is_embedding_oracle:
            assert oracle_ids is not None
            nll = self._score_embedding_oracle(
                prefix=prefix,
                suffix=suffix,
                labels=labels,
                oracle_ids=oracle_ids[:effective_visual_tokens],
            )
            token_source = "frozen_llm_input_embeddings"
        elif features is None:
            nll = self._score_text(prefix + suffix, labels)
            token_source = "none"
        else:
            if features.ndim != 2:
                raise ValueError(
                    f"expected visual features [N,D], got {tuple(features.shape)}"
                )
            if features.shape[1] != int(self.metadata["input_dim"]):
                raise ValueError(
                    f"feature dim={features.shape[1]} does not match projector input_dim="
                    f"{self.metadata['input_dim']}"
                )
            nll = self._score_visual(
                prefix=prefix,
                suffix=suffix,
                labels=labels,
                features=features[:effective_visual_tokens],
            )
            token_source = "projected_visual_features"

        nll_values = nll.detach().float().cpu()
        probabilities = torch.softmax(-nll_values, dim=0)
        choice_nll = {
            label: float(nll_values[index]) for index, label in enumerate(labels)
        }
        choice_probability = {
            label: float(probabilities[index]) for index, label in enumerate(labels)
        }
        prediction_index = int(nll_values.argmin().item())
        prediction = labels[prediction_index]
        gold_index = labels.index(gold)
        distractor_nll = min(
            float(nll_values[index])
            for index in range(len(labels))
            if index != gold_index
        )
        gold_nll = float(nll_values[gold_index])
        return ScoreResult(
            prediction=prediction,
            prediction_text=choices[prediction_index],
            choice_nll=choice_nll,
            choice_probability=choice_probability,
            gold_nll=gold_nll,
            best_distractor_nll=distractor_nll,
            gold_margin=distractor_nll - gold_nll,
            correct=prediction == gold,
            prompt_tokens=prompt_tokens,
            original_visual_tokens=original_visual_tokens,
            effective_visual_tokens=effective_visual_tokens,
            token_source=token_source,
        )

    def _score_text(self, prompt: str, labels: Sequence[str]) -> torch.Tensor:
        input_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        for label in labels:
            prompt_ids = self.tokenizer(
                prompt, add_special_tokens=True, return_tensors="pt"
            )["input_ids"][0]
            answer_ids = self._answer_ids(label)
            ids = torch.cat([prompt_ids, answer_ids], dim=0)
            if len(ids) > self.max_length:
                raise ValueError(
                    f"text prompt plus answer requires {len(ids)} tokens, exceeding "
                    f"max_length={self.max_length}; text truncation is disabled"
                )
            row_labels = torch.full_like(ids, -100)
            row_labels[-len(answer_ids) :] = answer_ids
            input_rows.append(ids)
            label_rows.append(row_labels)
        max_len = max(len(row) for row in input_rows)
        pad_id = int(self.tokenizer.pad_token_id)
        input_ids = torch.full(
            (len(labels), max_len), pad_id, dtype=torch.long, device=self.device
        )
        attention_mask = torch.zeros(
            (len(labels), max_len), dtype=torch.long, device=self.device
        )
        targets = torch.full(
            (len(labels), max_len), -100, dtype=torch.long, device=self.device
        )
        for index, (ids, row_labels) in enumerate(zip(input_rows, label_rows)):
            length = len(ids)
            input_ids[index, :length] = ids.to(self.device)
            attention_mask[index, :length] = 1
            targets[index, :length] = row_labels.to(self.device)
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        return self._sequence_nll(outputs.logits, targets)

    def _score_visual(
        self,
        *,
        prefix: str,
        suffix: str,
        labels: Sequence[str],
        features: torch.Tensor,
    ) -> torch.Tensor:
        repeated = (
            features.unsqueeze(0)
            .expand(len(labels), -1, -1)
            .to(device=self.device, dtype=self.dtype)
        )
        feature_mask = torch.ones(
            repeated.shape[:2], device=self.device, dtype=torch.bool
        )
        visual_embeds = self.projector(repeated)
        inputs_embeds, attention_mask, targets = build_inputs_embeds_and_labels(
            tokenizer=self.tokenizer,
            llm=self.llm,
            visual_embeds=visual_embeds,
            feature_mask=feature_mask,
            prefixes=[prefix] * len(labels),
            suffixes=[suffix] * len(labels),
            answers=list(labels),
            max_length=self.max_length,
        )
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return self._sequence_nll(outputs.logits, targets)

    def _score_embedding_oracle(
        self,
        *,
        prefix: str,
        suffix: str,
        labels: Sequence[str],
        oracle_ids: torch.Tensor,
    ) -> torch.Tensor:
        if oracle_ids.numel() == 0:
            raise ValueError(
                "embedding oracle has zero tokens after applying the context budget"
            )
        token_ids = oracle_ids.to(self.device)
        oracle_embeds = self.llm.get_input_embeddings()(token_ids)
        repeated = oracle_embeds.unsqueeze(0).expand(len(labels), -1, -1)
        feature_mask = torch.ones(
            repeated.shape[:2], device=self.device, dtype=torch.bool
        )
        inputs_embeds, attention_mask, targets = build_inputs_embeds_and_labels(
            tokenizer=self.tokenizer,
            llm=self.llm,
            visual_embeds=repeated,
            feature_mask=feature_mask,
            prefixes=[prefix] * len(labels),
            suffixes=[suffix] * len(labels),
            answers=list(labels),
            max_length=self.max_length,
        )
        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return self._sequence_nll(outputs.logits, targets)
