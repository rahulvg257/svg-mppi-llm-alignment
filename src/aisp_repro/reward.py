from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from aisp_repro.config import EmbeddingConfig, RewardConfig
from aisp_repro.utils import resolve_device


@dataclass
class RewardModelMetadata:
    model_id: str
    revision: str | None
    commit_hash: str | None
    device: str


class SequenceRewardModel:
    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
        )
        self.model.to(self.device)
        self.model.eval()
        self.metadata = RewardModelMetadata(
            model_id=config.model_id,
            revision=config.revision,
            commit_hash=getattr(self.model.config, "_commit_hash", None),
            device=str(self.device),
        )

    @torch.inference_mode()
    def score(self, prompts: list[str], responses: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(prompts), self.config.batch_size):
            batch_prompts = prompts[start : start + self.config.batch_size]
            batch_responses = responses[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                batch_prompts,
                batch_responses,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded)
            logits = outputs.logits
            if logits.ndim == 2 and logits.shape[1] == 1:
                batch_scores = logits.squeeze(-1)
            elif logits.ndim == 2 and logits.shape[1] == 2:
                batch_scores = logits[:, 1] - logits[:, 0]
            else:
                raise ValueError(f"Unexpected reward logits shape: {tuple(logits.shape)}")
            scores.extend(batch_scores.detach().cpu().tolist())
        return [float(score) for score in scores]


@dataclass
class EmbeddingModelMetadata:
    model_id: str
    revision: str | None
    commit_hash: str | None
    device: str


class SentenceEmbedder:
    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
        )
        self.model.to(self.device)
        self.model.eval()
        self.metadata = EmbeddingModelMetadata(
            model_id=config.model_id,
            revision=config.revision,
            commit_hash=getattr(self.model.config, "_commit_hash", None),
            device=str(self.device),
        )

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> torch.Tensor:
        batches: list[torch.Tensor] = []
        for start in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[start : start + self.config.batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            outputs = self.model(**encoded, return_dict=True)
            if getattr(outputs, "pooler_output", None) is not None:
                pooled = outputs.pooler_output
            else:
                pooled = outputs.last_hidden_state[:, 0, :]
            batches.append(pooled.detach().cpu())
        return torch.cat(batches, dim=0)

