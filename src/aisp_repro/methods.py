from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from aisp_repro.config import AISPConfig, BestOfNConfig, TsallisAISPConfig
from aisp_repro.modeling import ChatGenerationModel, GeneratedSample
from aisp_repro.reward import SequenceRewardModel
from aisp_repro.utils import softmax_log_weights


@dataclass
class MethodResult:
    method_name: str
    sample: GeneratedSample
    reward: float
    metadata: dict[str, Any]
    candidates: list[dict[str, Any]]


def run_greedy(
    model: ChatGenerationModel,
    reward_model: SequenceRewardModel,
    reward_prompt: str,
    messages: list[dict[str, str]],
) -> MethodResult:
    sample = model.generate_greedy(messages)
    reward = reward_model.score([reward_prompt], [sample.text])[0]
    return MethodResult(
        method_name="greedy",
        sample=sample,
        reward=reward,
        metadata={"sample_budget": 1},
        candidates=[
            {
                "candidate_index": 0,
                "response": sample.text,
                "reward": reward,
            }
        ],
    )


def run_best_of_n(
    model: ChatGenerationModel,
    reward_model: SequenceRewardModel,
    reward_prompt: str,
    messages: list[dict[str, str]],
    config: BestOfNConfig,
    matched_budget: int,
    seed: int,
) -> MethodResult:
    num_samples = config.num_samples or matched_budget
    candidates = model.sample_top_p(
        messages=messages,
        num_samples=num_samples,
        batch_size=config.batch_size,
        top_p=config.top_p,
        temperature=config.temperature,
        seed=seed,
    )
    rewards = reward_model.score(
        [reward_prompt] * len(candidates),
        [candidate.text for candidate in candidates],
    )
    best_index = max(range(len(candidates)), key=lambda index: rewards[index])
    best_candidate = candidates[best_index]
    return MethodResult(
        method_name="best_of_n",
        sample=best_candidate,
        reward=float(rewards[best_index]),
        metadata={
            "sample_budget": num_samples,
            "temperature": config.temperature,
            "top_p": config.top_p,
        },
        candidates=[
            {
                "candidate_index": index,
                "response": candidate.text,
                "reward": float(rewards[index]),
                "token_count": len(candidate.token_ids),
            }
            for index, candidate in enumerate(candidates)
        ],
    )


def _sample_trajectories(
    mean: torch.Tensor,
    sigma_sq: float,
    num_samples: int,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    sigma = sigma_sq ** 0.5
    noise = torch.randn(
        num_samples,
        mean.shape[0],
        mean.shape[1],
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    return mean.unsqueeze(0) + sigma * noise.to(device)


def _penalty_scale(alpha: float | None) -> float:
    return 1.0 if alpha is None else (1.0 - alpha)


def _trajectory_penalty(
    control_mean: torch.Tensor,
    trajectories: torch.Tensor,
    *,
    sigma_sq: float,
    alpha: float | None,
) -> torch.Tensor:
    dot_products = torch.einsum("td,ntd->n", control_mean, trajectories)
    return (_penalty_scale(alpha) / sigma_sq) * dot_products


def _raw_cost_tensor(
    reward_tensor: torch.Tensor,
    control_mean: torch.Tensor,
    trajectories: torch.Tensor,
    *,
    sigma_sq: float,
    lambda_: float,
    alpha: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    penalty_tensor = _trajectory_penalty(
        control_mean,
        trajectories,
        sigma_sq=sigma_sq,
        alpha=alpha,
    )
    raw_cost = -reward_tensor + lambda_ * penalty_tensor
    return raw_cost, penalty_tensor


def _min_max_normalize(costs: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, float, float]:
    min_value = float(costs.min().detach().cpu().item())
    max_value = float(costs.max().detach().cpu().item())
    scale = max(max_value - min_value, eps)
    normalized = (costs - min_value) / scale
    return normalized, min_value, max_value


def _tsallis_weight_update(
    raw_costs: torch.Tensor,
    *,
    r: float,
    gamma: float | None,
    elite_fraction: float,
    normalize_costs: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if r <= 1.0:
        raise ValueError("Tsallis parameter r must be greater than 1.")

    if normalize_costs:
        costs_for_weight, raw_min, raw_max = _min_max_normalize(raw_costs)
    else:
        costs_for_weight = raw_costs
        raw_min = float(raw_costs.min().detach().cpu().item())
        raw_max = float(raw_costs.max().detach().cpu().item())

    if gamma is None:
        quantile = min(max(elite_fraction, 1e-6), 1.0)
        gamma_value = float(torch.quantile(costs_for_weight.detach().cpu(), quantile).item())
        gamma_source = "elite_fraction"
    else:
        gamma_value = float(gamma)
        gamma_source = "explicit"

    gamma_value = max(gamma_value, 1e-6)
    exponent = 1.0 / (r - 1.0)
    support = torch.clamp(1.0 - (costs_for_weight / gamma_value), min=0.0)
    unnormalized = support.pow(exponent)
    weight_sum = float(unnormalized.sum().detach().cpu().item())

    if weight_sum <= 1e-12:
        weights = torch.zeros_like(unnormalized)
        best_index = int(torch.argmin(raw_costs).detach().cpu().item())
        weights[best_index] = 1.0
        fallback = "argmin_cost"
    else:
        weights = unnormalized / unnormalized.sum()
        fallback = None

    log_weights = torch.log(weights.clamp_min(1e-12))
    diagnostics = {
        "raw_cost_min": raw_min,
        "raw_cost_max": raw_max,
        "costs_normalized_to_unit_interval": normalize_costs,
        "gamma": gamma_value,
        "gamma_source": gamma_source,
        "elite_fraction": elite_fraction,
        "r": r,
        "support_size": int((support > 0).sum().detach().cpu().item()),
        "fallback": fallback,
    }
    return log_weights, weights, diagnostics


def run_aisp(
    model: ChatGenerationModel,
    reward_model: SequenceRewardModel,
    reward_prompt: str,
    messages: list[dict[str, str]],
    config: AISPConfig,
    seed: int,
) -> MethodResult:
    tau = model.config.tau or model.config.max_new_tokens
    hidden_size = model.hidden_size
    device = model.device
    final_sample_count = config.final_sample_count or config.n
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    control_mean = torch.zeros((tau, hidden_size), device=device, dtype=torch.float32)
    best_sample: GeneratedSample | None = None
    best_reward = float("-inf")
    candidate_records: list[dict[str, Any]] = []
    iteration_summaries: list[dict[str, Any]] = []
    first_shape_log: dict[str, Any] | None = None

    for iteration in range(config.kappa):
        trajectories = _sample_trajectories(
            mean=control_mean,
            sigma_sq=config.sigma_sq,
            num_samples=config.n,
            generator=generator,
            device=device,
        )
        rollouts = model.greedy_rollout_with_prelogit_perturbations(messages, trajectories)
        rewards = reward_model.score(
            [reward_prompt] * len(rollouts),
            [rollout.text for rollout in rollouts],
        )
        reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        penalty_tensor = _trajectory_penalty(
            control_mean,
            trajectories,
            sigma_sq=config.sigma_sq,
            alpha=config.alpha,
        )
        score_tensor = reward_tensor / config.lambda_ - penalty_tensor
        log_weights, weights = softmax_log_weights(score_tensor)
        control_mean = torch.einsum("n,ntd->td", weights, trajectories)

        for sample_index, rollout in enumerate(rollouts):
            reward = float(rewards[sample_index])
            if reward > best_reward:
                best_reward = reward
                best_sample = rollout
            candidate_records.append(
                {
                    "phase": "iteration",
                    "iteration": iteration + 1,
                    "candidate_index": sample_index,
                    "reward": reward,
                    "log_weight": float(log_weights[sample_index].detach().cpu().item()),
                    "weight": float(weights[sample_index].detach().cpu().item()),
                    "response": rollout.text,
                    "token_count": len(rollout.token_ids),
                }
            )
            if first_shape_log is None:
                first_shape_log = rollout.diagnostics

        iteration_summaries.append(
            {
                "iteration": iteration + 1,
                "mean_reward": float(reward_tensor.mean().detach().cpu().item()),
                "max_reward": float(reward_tensor.max().detach().cpu().item()),
                "best_reward_so_far": float(best_reward),
                "weight_entropy": float((-weights * log_weights).sum().detach().cpu().item()),
            }
        )

    final_trajectories = _sample_trajectories(
        mean=control_mean,
        sigma_sq=config.sigma_sq,
        num_samples=final_sample_count,
        generator=generator,
        device=device,
    )
    final_rollouts = model.greedy_rollout_with_prelogit_perturbations(messages, final_trajectories)
    final_rewards = reward_model.score(
        [reward_prompt] * len(final_rollouts),
        [rollout.text for rollout in final_rollouts],
    )
    final_best_index = max(range(len(final_rollouts)), key=lambda index: final_rewards[index])
    final_best_sample = final_rollouts[final_best_index]
    final_best_reward = float(final_rewards[final_best_index])

    for sample_index, rollout in enumerate(final_rollouts):
        candidate_records.append(
            {
                "phase": "final_samples",
                "iteration": config.kappa + 1,
                "candidate_index": sample_index,
                "reward": float(final_rewards[sample_index]),
                "response": rollout.text,
                "token_count": len(rollout.token_ids),
            }
        )

    if final_best_reward > best_reward:
        best_reward = final_best_reward
        best_sample = final_best_sample

    assert best_sample is not None

    return MethodResult(
        method_name="aisp",
        sample=best_sample,
        reward=best_reward,
        metadata={
            "sample_budget": config.n * config.kappa,
            "n": config.n,
            "kappa": config.kappa,
            "sigma_sq": config.sigma_sq,
            "lambda": config.lambda_,
            "alpha": config.alpha,
            "tau": tau,
            "final_sample_count": final_sample_count,
            "iteration_summaries": iteration_summaries,
            "shape_log": first_shape_log or {},
        },
        candidates=candidate_records,
    )


def run_tsallis_aisp(
    model: ChatGenerationModel,
    reward_model: SequenceRewardModel,
    reward_prompt: str,
    messages: list[dict[str, str]],
    config: TsallisAISPConfig,
    seed: int,
) -> MethodResult:
    tau = model.config.tau or model.config.max_new_tokens
    hidden_size = model.hidden_size
    device = model.device
    final_sample_count = config.final_sample_count or config.n
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    control_mean = torch.zeros((tau, hidden_size), device=device, dtype=torch.float32)
    best_sample: GeneratedSample | None = None
    best_reward = float("-inf")
    candidate_records: list[dict[str, Any]] = []
    iteration_summaries: list[dict[str, Any]] = []
    first_shape_log: dict[str, Any] | None = None

    for iteration in range(config.kappa):
        trajectories = _sample_trajectories(
            mean=control_mean,
            sigma_sq=config.sigma_sq,
            num_samples=config.n,
            generator=generator,
            device=device,
        )
        rollouts = model.greedy_rollout_with_prelogit_perturbations(messages, trajectories)
        rewards = reward_model.score(
            [reward_prompt] * len(rollouts),
            [rollout.text for rollout in rollouts],
        )
        reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
        raw_costs, penalty_tensor = _raw_cost_tensor(
            reward_tensor,
            control_mean,
            trajectories,
            sigma_sq=config.sigma_sq,
            lambda_=config.lambda_,
            alpha=config.alpha,
        )
        normalized_costs, _, _ = _min_max_normalize(raw_costs)
        log_weights, weights, tsallis_meta = _tsallis_weight_update(
            raw_costs,
            r=config.r,
            gamma=config.gamma,
            elite_fraction=config.elite_fraction,
            normalize_costs=config.normalize_costs,
        )
        control_mean = torch.einsum("n,ntd->td", weights, trajectories)

        for sample_index, rollout in enumerate(rollouts):
            reward = float(rewards[sample_index])
            if reward > best_reward:
                best_reward = reward
                best_sample = rollout
            candidate_records.append(
                {
                    "phase": "iteration",
                    "iteration": iteration + 1,
                    "candidate_index": sample_index,
                    "reward": reward,
                    "raw_cost": float(raw_costs[sample_index].detach().cpu().item()),
                    "normalized_cost": float(normalized_costs[sample_index].detach().cpu().item()),
                    "penalty_term": float(penalty_tensor[sample_index].detach().cpu().item()),
                    "log_weight": float(log_weights[sample_index].detach().cpu().item()),
                    "weight": float(weights[sample_index].detach().cpu().item()),
                    "response": rollout.text,
                    "token_count": len(rollout.token_ids),
                }
            )
            if first_shape_log is None:
                first_shape_log = rollout.diagnostics

        iteration_summaries.append(
            {
                "iteration": iteration + 1,
                "mean_reward": float(reward_tensor.mean().detach().cpu().item()),
                "max_reward": float(reward_tensor.max().detach().cpu().item()),
                "mean_raw_cost": float(raw_costs.mean().detach().cpu().item()),
                "min_raw_cost": float(raw_costs.min().detach().cpu().item()),
                "best_reward_so_far": float(best_reward),
                "weight_entropy": float((-weights * log_weights).sum().detach().cpu().item()),
                **tsallis_meta,
            }
        )

    final_trajectories = _sample_trajectories(
        mean=control_mean,
        sigma_sq=config.sigma_sq,
        num_samples=final_sample_count,
        generator=generator,
        device=device,
    )
    final_rollouts = model.greedy_rollout_with_prelogit_perturbations(messages, final_trajectories)
    final_rewards = reward_model.score(
        [reward_prompt] * len(final_rollouts),
        [rollout.text for rollout in final_rollouts],
    )
    final_best_index = max(range(len(final_rollouts)), key=lambda index: final_rewards[index])
    final_best_sample = final_rollouts[final_best_index]
    final_best_reward = float(final_rewards[final_best_index])

    for sample_index, rollout in enumerate(final_rollouts):
        candidate_records.append(
            {
                "phase": "final_samples",
                "iteration": config.kappa + 1,
                "candidate_index": sample_index,
                "reward": float(final_rewards[sample_index]),
                "response": rollout.text,
                "token_count": len(rollout.token_ids),
            }
        )

    if final_best_reward > best_reward:
        best_reward = final_best_reward
        best_sample = final_best_sample

    assert best_sample is not None

    return MethodResult(
        method_name="tsallis_aisp",
        sample=best_sample,
        reward=best_reward,
        metadata={
            "sample_budget": config.n * config.kappa,
            "n": config.n,
            "kappa": config.kappa,
            "sigma_sq": config.sigma_sq,
            "lambda": config.lambda_,
            "alpha": config.alpha,
            "r": config.r,
            "gamma": config.gamma,
            "elite_fraction": config.elite_fraction,
            "normalize_costs": config.normalize_costs,
            "tau": tau,
            "final_sample_count": final_sample_count,
            "iteration_summaries": iteration_summaries,
            "shape_log": first_shape_log or {},
            "approximation_note": "Tsallis-AISP is an extension of AISP that replaces the softmax importance weights with Tsallis-MPPI-style weights over iteration-normalized sampled costs; this update is not defined in the AISP paper itself.",
        },
        candidates=candidate_records,
    )
