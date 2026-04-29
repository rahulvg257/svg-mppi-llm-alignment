# svg_tmpc/samplers/base.py
"""Abstract base class for all decoding strategies in svg_tmpc."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from svg_tmpc.models.backbone import Backbone
from svg_tmpc.models.reward import RewardModel


class BaseSampler(ABC):
    """All samplers share a backbone LM and an optional reward model."""

    name: str = "base"

    def __init__(self, backbone: Backbone, reward_model: Optional[RewardModel] = None) -> None:
        self.backbone = backbone
        self.reward_model = reward_model

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int) -> List[str]:
        """Generate one or more continuations for the given prompt.

        Returning a list (rather than a single string) lets samplers like Best-of-N
        expose intermediate candidates if desired; runners take element [0] as the
        canonical response unless they care about all candidates.
        """

    def generate_one(self, prompt: str, max_new_tokens: int) -> str:
        return self.generate(prompt, max_new_tokens)[0]

    def generate_many(self, prompts: List[str], max_new_tokens: int) -> List[str]:
        return [self.generate_one(p, max_new_tokens) for p in prompts]
