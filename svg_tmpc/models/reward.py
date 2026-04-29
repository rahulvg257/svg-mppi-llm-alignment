# svg_tmpc/models/reward.py
"""HuggingFace sequence-classification reward-model wrapper."""

from __future__ import annotations

from typing import Iterable, List, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

_DTYPE_FALLBACK = torch.float32


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


class RewardModel:
    """Scalar reward model over (prompt, response) pairs.

    Uses a HuggingFace AutoModelForSequenceClassification (e.g. the OpenAssistant
    DeBERTa-v3 reward model) and returns the first logit as the scalar score.
    """

    def __init__(
        self,
        model_name: str = "OpenAssistant/reward-model-deberta-v3-large-v2",
        device: str = "auto",
        max_length: int = 512,
        batch_size: int = 4,
    ) -> None:
        self.model_name = model_name
        self.device = _resolve_device(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=_DTYPE_FALLBACK
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, prompt: str, response: str) -> float:
        """Score a single (prompt, response) pair and return a Python float."""
        return float(self.score_batch([prompt], [response])[0])

    @torch.no_grad()
    def score_batch(
        self,
        prompts: Sequence[str],
        responses: Sequence[str],
    ) -> List[float]:
        """Score a batch of (prompt, response) pairs."""
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have the same length")

        scores: List[float] = []
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = list(prompts[i : i + self.batch_size])
            batch_responses = list(responses[i : i + self.batch_size])
            enc = self.tokenizer(
                batch_prompts,
                batch_responses,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            # Most reward models output a single scalar logit; if multi-class, take index 0.
            if logits.dim() == 2 and logits.size(-1) > 1:
                logits = logits[:, 0]
            scores.extend(logits.squeeze(-1).float().cpu().tolist())
        return scores

    @torch.no_grad()
    def score_texts(self, prompt: str, responses: Iterable[str]) -> List[float]:
        responses = list(responses)
        return self.score_batch([prompt] * len(responses), responses)
