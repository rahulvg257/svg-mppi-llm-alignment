# svg_tmpc/samplers/tmpc.py
"""TMPC: representation-editing decoding via vanilla MPPI weights.

At every autoregressive step we (a) read the post-final-norm hidden state for the
current position, (b) draw K Gaussian perturbations, (c) roll each one out for
H tokens through the LM head, (d) score the rollouts with a fixed reward model,
(e) form softmax (MPPI) weights over rewards, and (f) replace the hidden state
with the weighted mixture before sampling the next token.
"""

from __future__ import annotations

from typing import List

import torch

from svg_tmpc.models.backbone import Backbone
from svg_tmpc.models.reward import RewardModel
from svg_tmpc.samplers.base import BaseSampler


def mppi_weights(rewards: torch.Tensor, lambda_: float) -> torch.Tensor:
    """Standard MPPI weights: softmax(rewards / lambda)."""
    if lambda_ <= 0:
        raise ValueError(f"lambda_ must be positive, got {lambda_}")
    return torch.softmax(rewards / lambda_, dim=-1)


class TMPCSampler(BaseSampler):
    name = "tmpc"

    def __init__(
        self,
        backbone: Backbone,
        reward_model: RewardModel,
        K: int = 16,
        H: int = 8,
        sigma: float = 0.1,
        lambda_: float = 1.0,
    ) -> None:
        if reward_model is None:
            raise ValueError("TMPC requires a reward model")
        if K < 1:
            raise ValueError(f"K must be >= 1, got {K}")
        if H < 1:
            raise ValueError(f"H must be >= 1, got {H}")
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        super().__init__(backbone, reward_model)
        self.K = K
        self.H = H
        self.sigma = sigma
        self.lambda_ = lambda_

    def _compute_weights(self, rewards: torch.Tensor) -> torch.Tensor:
        return mppi_weights(rewards, self.lambda_)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int) -> List[str]:
        backbone = self.backbone
        device = backbone.device
        eos_id = backbone.eos_token_id

        input_ids = backbone.encode(prompt)
        prompt_len = input_ids.size(1)
        generated_token_ids: List[int] = []

        for _ in range(max_new_tokens):
            hidden_states = backbone.get_hidden_state(input_ids)
            h_t = hidden_states[0, -1, :]

            if self.sigma > 0:
                noise = torch.randn(
                    self.K, h_t.size(0), device=device, dtype=h_t.dtype
                ) * self.sigma
            else:
                noise = torch.zeros(self.K, h_t.size(0), device=device, dtype=h_t.dtype)
            h_perturbed = h_t.unsqueeze(0) + noise

            rollout_ids = backbone.greedy_decode(
                hidden_state=h_perturbed,
                max_new_tokens=self.H,
                prefix_ids=input_ids,
            )

            rewards = self._score_rollouts(prompt, input_ids, rollout_ids, prompt_len)

            weights = self._compute_weights(rewards).to(h_perturbed.dtype).to(device)
            h_star = (weights.unsqueeze(-1) * h_perturbed).sum(dim=0)

            next_logits = backbone.lm_head_forward(h_star)
            next_token = int(torch.argmax(next_logits, dim=-1).item())

            generated_token_ids.append(next_token)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_token]], device=device, dtype=input_ids.dtype)],
                dim=1,
            )
            if eos_id is not None and next_token == eos_id:
                break

        text = backbone.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        return [text]

    def _score_rollouts(
        self,
        prompt: str,
        input_ids: torch.Tensor,
        rollout_ids: torch.Tensor,
        prompt_len: int,
    ) -> torch.Tensor:
        already_generated = input_ids[0, prompt_len:]
        responses: List[str] = []
        for k in range(rollout_ids.size(0)):
            full_response_ids = torch.cat([already_generated, rollout_ids[k]], dim=0)
            response_text = self.backbone.tokenizer.decode(
                full_response_ids.tolist(), skip_special_tokens=True
            )
            responses.append(response_text)
        scores = self.reward_model.score_texts(prompt, responses)
        return torch.tensor(scores, dtype=torch.float32)
