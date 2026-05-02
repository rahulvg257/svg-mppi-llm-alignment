# svg_tmpc/samplers/tsallis_mppi.py
"""Tsallis-MPPI: variational-inference weights via Tsallis divergence.

Same per-step machinery as TMPC, but the rollout-reward to weight transform is the
q-exponential (Tsallis) instead of standard softmax. q -> 1 recovers TMPC; q > 1
sparsifies the weighting toward the highest-reward rollouts; q < 1 broadens it.
"""

from __future__ import annotations

import torch

from svg_tmpc.models.backbone import Backbone
from svg_tmpc.models.reward import RewardModel
from svg_tmpc.samplers.tmpc import TMPCSampler, mppi_weights


def tsallis_weights(rewards: torch.Tensor, lambda_: float, q: float) -> torch.Tensor:
    """Tsallis (q-exponential) importance weights.

    For q == 1 this is exactly softmax(r/lambda) (numerically). For q != 1 we
    compute u_k = max(0, 1 + (q - 1) * (r_k - r_max) / lambda) ** (1 / (q - 1)),
    centring on r_max for numerical stability, then normalize. The shift by r_max
    keeps the q-exponential's argument in a bounded range without changing the
    rank order of the resulting weights.
    """
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be positive, got {lambda_}")
    if abs(q - 1.0) < 1e-6:
        return mppi_weights(rewards, lambda_)

    centered = rewards - rewards.max()
    t = 1.0 + (q - 1.0) * centered / lambda_
    t = torch.clamp(t, min=1e-12)
    exponent = 1.0 / (q - 1.0)
    u = t.pow(exponent)
    Z = u.sum()
    if not torch.isfinite(Z) or Z < 1e-12:
        return torch.ones_like(rewards) / rewards.numel()
    return u / Z


class TsallisMPPISampler(TMPCSampler):
    name = "tsallis_mppi"

    def __init__(
        self,
        backbone: Backbone,
        reward_model: RewardModel,
        K: int = 16,
        H: int = 8,
        sigma: float = 0.1,
        lambda_: float = 1.0,
        q: float = 1.5,
    ) -> None:
        super().__init__(backbone, reward_model, K=K, H=H, sigma=sigma, lambda_=lambda_)
        self.q = q

    def _compute_weights(self, rewards: torch.Tensor) -> torch.Tensor:
        return tsallis_weights(rewards, self.lambda_, self.q)
