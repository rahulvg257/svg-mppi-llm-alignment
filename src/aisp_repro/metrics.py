from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from aisp_repro.reward import SentenceEmbedder


def diversity_score(token_ids: list[int]) -> float:
    filtered = [token for token in token_ids if token >= 0]
    if len(filtered) < 4:
        return 0.0

    score = 1.0
    for n in range(2, 5):
        grams = [tuple(filtered[index : index + n]) for index in range(len(filtered) - n + 1)]
        if not grams:
            return 0.0
        score *= len(set(grams)) / len(grams)
    return float(score)


def coherence_score(embedder: SentenceEmbedder, prompts: list[str], responses: list[str]) -> list[float]:
    prompt_embeddings = embedder.encode(prompts)
    response_embeddings = embedder.encode(responses)
    prompt_embeddings = F.normalize(prompt_embeddings, dim=-1)
    response_embeddings = F.normalize(response_embeddings, dim=-1)
    similarities = torch.sum(prompt_embeddings * response_embeddings, dim=-1)
    return [float(value) for value in similarities.tolist()]

