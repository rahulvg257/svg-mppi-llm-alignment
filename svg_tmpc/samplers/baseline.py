# svg_tmpc/samplers/baseline.py
"""Unmodified HuggingFace .generate() decoding (greedy or nucleus)."""

from __future__ import annotations

from typing import List, Optional

from svg_tmpc.models.backbone import Backbone, GenerationConfig
from svg_tmpc.models.reward import RewardModel
from svg_tmpc.samplers.base import BaseSampler


class BaselineSampler(BaseSampler):
    name = "baseline"

    def __init__(
        self,
        backbone: Backbone,
        reward_model: Optional[RewardModel] = None,
        do_sample: bool = True,
        top_p: float = 0.9,
        temperature: float = 0.8,
        repetition_penalty: float = 1.0,
    ) -> None:
        super().__init__(backbone, reward_model)
        self.do_sample = do_sample
        self.top_p = top_p
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty

    def _gen_config(self, max_new_tokens: int) -> GenerationConfig:
        return GenerationConfig(
            do_sample=self.do_sample,
            top_p=self.top_p,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            max_new_tokens=max_new_tokens,
        )

    def generate(self, prompt: str, max_new_tokens: int) -> List[str]:
        return [self.backbone.generate(prompt, self._gen_config(max_new_tokens))]
