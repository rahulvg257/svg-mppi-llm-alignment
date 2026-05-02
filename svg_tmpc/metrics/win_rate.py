# svg_tmpc/metrics/win_rate.py
"""Reward-model-judged win rate of one method's responses against a baseline."""

from __future__ import annotations

from typing import Sequence

from svg_tmpc.models.reward import RewardModel


def win_rate(
    method_responses: Sequence[str],
    baseline_responses: Sequence[str],
    prompts: Sequence[str],
    reward_model: RewardModel,
) -> float:
    """Fraction of prompts where the method scores strictly higher than the baseline."""
    if not (len(method_responses) == len(baseline_responses) == len(prompts)):
        raise ValueError("method_responses, baseline_responses, and prompts must align")
    if not prompts:
        return 0.0

    method_scores = reward_model.score_batch(list(prompts), list(method_responses))
    baseline_scores = reward_model.score_batch(list(prompts), list(baseline_responses))
    wins = sum(1 for m, b in zip(method_scores, baseline_scores) if m > b)
    return wins / len(prompts)
