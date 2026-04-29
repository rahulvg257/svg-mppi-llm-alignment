# svg_tmpc/metrics/coherence.py
"""Perplexity-based coherence metric using the backbone LM."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def perplexity(
    texts: Sequence[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> float:
    """Mean per-token cross-entropy perplexity over the supplied texts.

    Texts that are empty or tokenize to fewer than 2 tokens are skipped (a single
    token has no next-token loss). Returns ``float('inf')`` if every text is skipped.
    """
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    for text in texts:
        if not text:
            continue
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = encoding.input_ids.to(device)
        if input_ids.size(1) < 2:
            continue
        labels = input_ids.clone()
        outputs = model(input_ids=input_ids, labels=labels)
        n_tokens = input_ids.size(1) - 1
        total_loss += float(outputs.loss.item()) * n_tokens
        total_tokens += n_tokens
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)
