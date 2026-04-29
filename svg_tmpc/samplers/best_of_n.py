# svg_tmpc/samplers/best_of_n.py
"""Best-of-N: sample N independent baseline continuations and keep the highest-reward one."""

from __future__ import annotations

from typing import List

from svg_tmpc.models.backbone import Backbone
from svg_tmpc.models.reward import RewardModel
from svg_tmpc.samplers.base import BaseSampler
from svg_tmpc.samplers.baseline import BaselineSampler


class BestOfNSampler(BaseSampler):
    name = "best_of_n"

    def __init__(
        self,
        backbone: Backbone,
        reward_model: RewardModel,
        N: int = 8,
        do_sample: bool = True,
        top_p: float = 0.9,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
    ) -> None:
        if reward_model is None:
            raise ValueError("Best-of-N requires a reward model")
        super().__init__(backbone, reward_model)
        if N < 1:
            raise ValueError(f"N must be >= 1, got {N}")
        self.N = N
        self.inner = BaselineSampler(
            backbone=backbone,
            reward_model=reward_model,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

    def generate(self, prompt: str, max_new_tokens: int) -> List[str]:
        candidates: List[str] = []
        for _ in range(self.N):
            candidates.append(self.inner.generate(prompt, max_new_tokens)[0])

        scores = self.reward_model.score_texts(prompt, candidates)
        best_index = max(range(len(candidates)), key=lambda i: scores[i])
        ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
        ordered = [c for c, _ in ranked]
        # Ensure the top-scoring candidate is index 0 (already true after sorting).
        assert ordered[0] == candidates[best_index]
        return ordered
