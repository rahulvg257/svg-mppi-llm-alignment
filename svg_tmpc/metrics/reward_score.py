# svg_tmpc/metrics/reward_score.py
"""Average-reward metric using a HuggingFace reward model."""

from __future__ import annotations

from typing import Sequence

from svg_tmpc.models.reward import RewardModel


def average_reward(
    prompts: Sequence[str],
    responses: Sequence[str],
    reward_model: RewardModel,
) -> float:
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    if not prompts:
        return 0.0
    scores = reward_model.score_batch(list(prompts), list(responses))
    return sum(scores) / len(scores)
