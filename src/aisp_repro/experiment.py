from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from aisp_repro.config import AISPConfig, BestOfNConfig, ExperimentConfig, TsallisAISPConfig
from aisp_repro.data import PromptExample, load_prompt_examples
from aisp_repro.judge import build_pairwise_judge_rows
from aisp_repro.methods import MethodResult, run_aisp, run_best_of_n, run_greedy, run_tsallis_aisp
from aisp_repro.metrics import coherence_score, diversity_score
from aisp_repro.modeling import ChatGenerationModel
from aisp_repro.reporting import save_reward_plot, write_markdown_report
from aisp_repro.reward import SentenceEmbedder, SequenceRewardModel
from aisp_repro.utils import (
    ensure_dir,
    flatten_dict,
    now_stamp,
    peak_memory_mb,
    process_memory_mb,
    reset_peak_memory,
    safe_mean,
    set_seed,
    timed,
    to_serializable,
    write_json,
    write_jsonl,
)


def create_run_dir(config: ExperimentConfig) -> Path:
    run_name = config.output.run_name or f"{config.phase}-{now_stamp()}"
    return ensure_dir(Path(config.output.base_dir) / run_name)


def _method_runtime_metadata(model: ChatGenerationModel, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "runtime_seconds": elapsed_seconds,
        "process_memory_mb": process_memory_mb(),
        "device_memory_mb": peak_memory_mb(model.device),
    }


def _finalize_record(
    example: PromptExample,
    result: MethodResult,
    method_runtime: dict[str, Any],
    coherence_value: float,
) -> dict[str, Any]:
    return {
        "example_id": example.example_id,
        "dataset_index": example.dataset_index,
        "dataset_config_name": example.config_name,
        "split": example.split,
        "method": result.method_name,
        "prompt_messages": example.messages,
        "reward_prompt_text": example.reward_prompt_text,
        "response": result.sample.text,
        "response_token_ids": result.sample.token_ids,
        "reward": result.reward,
        "diversity": diversity_score(result.sample.token_ids),
        "coherence": coherence_value,
        "method_metadata": result.metadata,
        "runtime_metadata": method_runtime,
        "generation_diagnostics": result.sample.diagnostics,
    }


def _candidate_rows(example: PromptExample, method: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.append(
            {
                "example_id": example.example_id,
                "dataset_index": example.dataset_index,
                "method": method,
                **candidate,
            }
        )
    return rows


def _search_method_budgets(config: ExperimentConfig) -> dict[str, int]:
    budgets: dict[str, int] = {}
    if config.aisp.enabled:
        budgets["aisp"] = config.aisp.n * config.aisp.kappa
    if config.tsallis_aisp.enabled:
        budgets["tsallis_aisp"] = config.tsallis_aisp.n * config.tsallis_aisp.kappa
    return budgets


def _matched_sample_budget(config: ExperimentConfig) -> int:
    search_budgets = _search_method_budgets(config)
    if not search_budgets:
        return config.bon.num_samples or 1
    return max(search_budgets.values())


def _run_methods_for_example(
    config: ExperimentConfig,
    example: PromptExample,
    model: ChatGenerationModel,
    reward_model: SequenceRewardModel,
    example_seed: int,
) -> tuple[list[MethodResult], list[dict[str, Any]]]:
    results: list[MethodResult] = []
    candidate_rows: list[dict[str, Any]] = []
    matched_budget = _matched_sample_budget(config)

    if config.greedy.enabled:
        reset_peak_memory(model.device)
        with timed() as timer:
            greedy_result = run_greedy(model, reward_model, example.reward_prompt_text, example.messages)
        greedy_result.metadata.update(_method_runtime_metadata(model, timer["elapsed_seconds"]))
        results.append(greedy_result)
        candidate_rows.extend(_candidate_rows(example, "greedy", greedy_result.candidates))

    if config.bon.enabled:
        reset_peak_memory(model.device)
        with timed() as timer:
            bon_result = run_best_of_n(
                model=model,
                reward_model=reward_model,
                reward_prompt=example.reward_prompt_text,
                messages=example.messages,
                config=config.bon,
                matched_budget=matched_budget,
                seed=example_seed + 100,
            )
        bon_result.metadata.update(_method_runtime_metadata(model, timer["elapsed_seconds"]))
        results.append(bon_result)
        candidate_rows.extend(_candidate_rows(example, "best_of_n", bon_result.candidates))

    if config.aisp.enabled:
        reset_peak_memory(model.device)
        with timed() as timer:
            aisp_result = run_aisp(
                model=model,
                reward_model=reward_model,
                reward_prompt=example.reward_prompt_text,
                messages=example.messages,
                config=config.aisp,
                seed=example_seed + 200,
            )
        aisp_result.metadata.update(_method_runtime_metadata(model, timer["elapsed_seconds"]))
        results.append(aisp_result)
        candidate_rows.extend(_candidate_rows(example, "aisp", aisp_result.candidates))

    if config.tsallis_aisp.enabled:
        reset_peak_memory(model.device)
        with timed() as timer:
            tsallis_result = run_tsallis_aisp(
                model=model,
                reward_model=reward_model,
                reward_prompt=example.reward_prompt_text,
                messages=example.messages,
                config=config.tsallis_aisp,
                seed=example_seed + 300,
            )
        tsallis_result.metadata.update(_method_runtime_metadata(model, timer["elapsed_seconds"]))
        results.append(tsallis_result)
        candidate_rows.extend(_candidate_rows(example, "tsallis_aisp", tsallis_result.candidates))

    return results, candidate_rows


def run_experiment(config: ExperimentConfig, tuning_results_path: str | None = None) -> Path:
    set_seed(config.seed)
    run_dir = create_run_dir(config)
    examples = load_prompt_examples(config.dataset, config.phase)

    if tuning_results_path:
        with Path(tuning_results_path).open("r", encoding="utf-8") as handle:
            tuning_payload = json.load(handle)
        tuned_bon = {
            key: tuning_payload["best_bon"][key]
            for key in ("top_p", "temperature")
            if key in tuning_payload["best_bon"]
        }
        tuned_aisp = {
            key: tuning_payload["best_aisp"][key]
            for key in ("sigma_sq", "lambda_", "alpha")
            if key in tuning_payload["best_aisp"]
        }
        tuned_tsallis = {
            key: tuning_payload["best_tsallis_aisp"][key]
            for key in ("sigma_sq", "lambda_", "alpha", "r", "elite_fraction", "gamma")
            if "best_tsallis_aisp" in tuning_payload and key in tuning_payload["best_tsallis_aisp"]
        }
        config.bon = replace(config.bon, **tuned_bon)
        config.aisp = replace(config.aisp, **tuned_aisp)
        if tuned_tsallis:
            config.tsallis_aisp = replace(config.tsallis_aisp, **tuned_tsallis)

    model = ChatGenerationModel(config.model)
    reward_model = SequenceRewardModel(config.reward)
    embedder = SentenceEmbedder(config.embedding)

    final_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for example_index, example in enumerate(examples):
        example_seed = config.seed + example_index * 1000
        method_results, method_candidate_rows = _run_methods_for_example(
            config=config,
            example=example,
            model=model,
            reward_model=reward_model,
            example_seed=example_seed,
        )
        candidate_rows.extend(method_candidate_rows)
        prompt_texts = [example.reward_prompt_text] * len(method_results)
        response_texts = [result.sample.text for result in method_results]
        coherence_values = coherence_score(embedder, prompt_texts, response_texts)

        for result, coherence_value in zip(method_results, coherence_values):
            runtime_metadata = {
                key: value
                for key, value in result.metadata.items()
                if key in {"runtime_seconds", "process_memory_mb", "device_memory_mb"}
            }
            final_rows.append(_finalize_record(example, result, runtime_metadata, coherence_value))

    final_df = pd.DataFrame(final_rows)
    aggregate_df = (
        final_df.groupby("method", as_index=False)
        .agg(
            average_reward=("reward", "mean"),
            average_diversity=("diversity", "mean"),
            average_coherence=("coherence", "mean"),
            average_runtime_seconds=("runtime_metadata", lambda rows: safe_mean([row["runtime_seconds"] for row in rows])),
        )
        .sort_values("average_reward", ascending=False)
    )

    outputs_path = run_dir / "per_example_outputs.jsonl"
    candidates_path = run_dir / "candidate_outputs.jsonl"
    aggregate_json_path = run_dir / "aggregate_metrics.json"
    aggregate_csv_path = run_dir / "aggregate_metrics.csv"
    sanity_path = run_dir / "sanity_checks.json"
    report_path = run_dir / "report.md"
    plot_path = run_dir / "reward_plot.png"
    judge_pairs_path = run_dir / "judge_pairs.jsonl"

    write_jsonl(outputs_path, final_rows)
    if config.output.save_candidates:
        write_jsonl(candidates_path, candidate_rows)
    write_jsonl(judge_pairs_path, build_pairwise_judge_rows(final_rows))
    write_json(aggregate_json_path, aggregate_df.to_dict(orient="records"))
    if config.output.save_csv:
        aggregate_df.to_csv(aggregate_csv_path, index=False)
        final_df.to_csv(run_dir / "per_example_outputs.csv", index=False)

    sanity_checks = build_sanity_checks(
        config=config,
        final_rows=final_rows,
        candidate_path=candidates_path,
        model=model,
        reward_model=reward_model,
    )
    write_json(sanity_path, sanity_checks)

    if config.output.save_plot:
        save_reward_plot(aggregate_df, plot_path)

    assumptions = [
        "The course-project requirements in the user message were treated as the accessible proposal spec because macOS blocked direct reads from the Downloads folder.",
        "The evaluation subset is a fixed shuffled slice of `Anthropic/hh-rlhf` default/test rather than the full HH benchmark due local compute limits.",
        "Coherence uses SimCSE sentence embeddings from `princeton-nlp/sup-simcse-bert-base-uncased`.",
    ]
    deviations = [
        "The experiment configs use smaller budgets and shorter generations than the paper defaults so the pipeline can run end-to-end on a local MPS setup.",
        "The reward model is an HH-RLHF-compatible off-the-shelf Hugging Face sequence classifier instead of the paper's larger reward models.",
        "Diversity is computed over generated token ids from the base model tokenizer, which is a faithful token-sequence interpretation of the paper's n-gram formula.",
    ]
    if config.tsallis_aisp.enabled:
        deviations.append(
            "Tsallis-AISP is an explicit extension motivated by the Tsallis-MPPI update rule from Wang et al. (2021), not a method defined in the original AISP paper."
        )
    write_markdown_report(
        report_path,
        aggregate_df=aggregate_df,
        config_snapshot={
            "experiment": config.to_dict(),
            "generation_model": to_serializable(model.metadata),
            "reward_model": to_serializable(reward_model.metadata),
            "embedding_model": to_serializable(embedder.metadata),
        },
        assumptions=assumptions,
        deviations=deviations,
        win_rate_available=config.judge.enabled,
    )

    return run_dir


def build_sanity_checks(
    *,
    config: ExperimentConfig,
    final_rows: list[dict[str, Any]],
    candidate_path: Path,
    model: ChatGenerationModel,
    reward_model: SequenceRewardModel,
) -> dict[str, Any]:
    grouped = {}
    for row in final_rows:
        grouped.setdefault(row["method"], []).append(row)

    search_rows = grouped.get("aisp", []) or grouped.get("tsallis_aisp", [])
    aisp_rows = grouped.get("aisp", [])
    greedy_rows = grouped.get("greedy", [])

    small_noise_close = None
    if greedy_rows:
        prompt_messages = greedy_rows[0]["prompt_messages"]
        tau = model.config.tau or model.config.max_new_tokens
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.seed)
        zero_perturbation = torch.zeros((1, tau, model.hidden_size), dtype=torch.float32)
        tiny_perturbation = 1e-12 * torch.randn(
            1,
            tau,
            model.hidden_size,
            generator=generator,
            dtype=torch.float32,
        )
        zero_noise_sample = model.greedy_rollout_with_prelogit_perturbations(
            prompt_messages,
            zero_perturbation,
        )[0]
        tiny_noise_sample = model.greedy_rollout_with_prelogit_perturbations(
            prompt_messages,
            tiny_perturbation,
        )[0]
        greedy_tokens = zero_noise_sample.token_ids
        tiny_noise_tokens = tiny_noise_sample.token_ids
        prefix = min(len(tiny_noise_tokens), len(greedy_tokens), 32)
        if prefix == 0:
            agreement = 1.0
        else:
            agreement = sum(
                int(tiny_noise_tokens[index] == greedy_tokens[index]) for index in range(prefix)
            ) / prefix
        small_noise_close = {
            "approximate_prefix_token_agreement": agreement,
            "exact_match": tiny_noise_tokens == greedy_tokens,
            "passes": agreement >= 0.9,
            "note": "Compared zero-perturbation rollout to a rollout with Gaussian pre-logit perturbations at variance 1e-24 on the first prompt.",
        }

    reward_determinism = None
    if final_rows:
        first_row = final_rows[0]
        prompt = first_row["reward_prompt_text"]
        response = first_row["response"]
        first_score = reward_model.score([prompt], [response])[0]
        second_score = reward_model.score([prompt], [response])[0]
        reward_determinism = {
            "passes": abs(first_score - second_score) < 1e-8,
            "score_first": first_score,
            "score_second": second_score,
        }

    shape_row_method = None
    aisp_shape_log = {}
    if search_rows:
        aisp_shape_log = search_rows[0]["generation_diagnostics"]
        shape_row_method = search_rows[0]["method"]

    matched_budget = _matched_sample_budget(config)
    search_budgets = _search_method_budgets(config)

    return {
        "prelogit_and_perturbation_shapes_logged": bool(aisp_shape_log),
        "shape_log": aisp_shape_log,
        "shape_log_method": shape_row_method,
        "small_noise_behaves_close_to_greedy_proxy": small_noise_close,
        "bon_uses_matched_sample_budget": {
            "matched": (config.bon.num_samples or matched_budget) == matched_budget,
            "bon_samples": config.bon.num_samples or matched_budget,
            "search_method_budgets": search_budgets,
        },
        "reward_scoring_sequence_level_and_deterministic": reward_determinism,
        "machine_readable_outputs_saved": {
            "per_example_jsonl": True,
            "candidate_jsonl": candidate_path.exists() if config.output.save_candidates else False,
        },
        "perturbation_applied_to_prelogits_not_embeddings": {
            "passes": aisp_shape_log.get("perturbation_target") == "pre_logit_hidden_state_before_lm_head"
            and not aisp_shape_log.get("perturbs_input_embeddings", True),
            "details": {
                "perturbation_target": aisp_shape_log.get("perturbation_target"),
                "perturbs_input_embeddings": aisp_shape_log.get("perturbs_input_embeddings"),
            },
        },
        "logged_model_device": str(model.device),
    }


def run_tuning(config: ExperimentConfig) -> Path:
    set_seed(config.seed)
    run_dir = create_run_dir(config)
    examples = load_prompt_examples(config.dataset, "tune")
    model = ChatGenerationModel(config.model)
    reward_model = SequenceRewardModel(config.reward)

    bon_trials = []
    aisp_trials = []
    tsallis_trials = []
    matched_budget = _matched_sample_budget(config)

    for temperature, top_p in itertools.product(
        config.tuning.bon_grid["temperature"],
        config.tuning.bon_grid["top_p"],
    ):
        trial_config = replace(config.bon, temperature=float(temperature), top_p=float(top_p))
        rewards = []
        for example_index, example in enumerate(examples):
            result = run_best_of_n(
                model=model,
                reward_model=reward_model,
                reward_prompt=example.reward_prompt_text,
                messages=example.messages,
                config=trial_config,
                matched_budget=matched_budget,
                seed=config.seed + example_index * 1000 + 100,
            )
            rewards.append(result.reward)
        bon_trials.append(
            {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "average_reward": safe_mean(rewards),
            }
        )

    if config.aisp.enabled:
        for sigma_sq, lambda_, alpha in itertools.product(
            config.tuning.aisp_grid["sigma_sq"],
            config.tuning.aisp_grid["lambda_"],
            config.tuning.aisp_grid["alpha"],
        ):
            trial_config = replace(
                config.aisp,
                sigma_sq=float(sigma_sq),
                lambda_=float(lambda_),
                alpha=float(alpha),
            )
            rewards = []
            for example_index, example in enumerate(examples):
                result = run_aisp(
                    model=model,
                    reward_model=reward_model,
                    reward_prompt=example.reward_prompt_text,
                    messages=example.messages,
                    config=trial_config,
                    seed=config.seed + example_index * 1000 + 200,
                )
                rewards.append(result.reward)
            aisp_trials.append(
                {
                    "sigma_sq": float(sigma_sq),
                    "lambda_": float(lambda_),
                    "alpha": float(alpha),
                    "average_reward": safe_mean(rewards),
                }
            )

    if config.tsallis_aisp.enabled:
        for sigma_sq, lambda_, alpha, r, elite_fraction in itertools.product(
            config.tuning.tsallis_aisp_grid["sigma_sq"],
            config.tuning.tsallis_aisp_grid["lambda_"],
            config.tuning.tsallis_aisp_grid["alpha"],
            config.tuning.tsallis_aisp_grid["r"],
            config.tuning.tsallis_aisp_grid["elite_fraction"],
        ):
            trial_config = replace(
                config.tsallis_aisp,
                sigma_sq=float(sigma_sq),
                lambda_=float(lambda_),
                alpha=float(alpha),
                r=float(r),
                elite_fraction=float(elite_fraction),
            )
            rewards = []
            for example_index, example in enumerate(examples):
                result = run_tsallis_aisp(
                    model=model,
                    reward_model=reward_model,
                    reward_prompt=example.reward_prompt_text,
                    messages=example.messages,
                    config=trial_config,
                    seed=config.seed + example_index * 1000 + 300,
                )
                rewards.append(result.reward)
            tsallis_trials.append(
                {
                    "sigma_sq": float(sigma_sq),
                    "lambda_": float(lambda_),
                    "alpha": float(alpha),
                    "r": float(r),
                    "elite_fraction": float(elite_fraction),
                    "average_reward": safe_mean(rewards),
                }
            )

    best_bon = max(bon_trials, key=lambda item: item["average_reward"])
    best_aisp = max(aisp_trials, key=lambda item: item["average_reward"]) if aisp_trials else None
    best_tsallis = max(tsallis_trials, key=lambda item: item["average_reward"]) if tsallis_trials else None
    payload = {
        "best_bon": {
            "temperature": best_bon["temperature"],
            "top_p": best_bon["top_p"],
        },
        "best_aisp": (
            {
                "sigma_sq": best_aisp["sigma_sq"],
                "lambda_": best_aisp["lambda_"],
                "alpha": best_aisp["alpha"],
            }
            if best_aisp
            else {}
        ),
        "best_tsallis_aisp": (
            {
                "sigma_sq": best_tsallis["sigma_sq"],
                "lambda_": best_tsallis["lambda_"],
                "alpha": best_tsallis["alpha"],
                "r": best_tsallis["r"],
                "elite_fraction": best_tsallis["elite_fraction"],
            }
            if best_tsallis
            else {}
        ),
        "bon_trials": bon_trials,
        "aisp_trials": aisp_trials,
        "tsallis_aisp_trials": tsallis_trials,
    }
    output_path = run_dir / "tuning_results.json"
    write_json(output_path, payload)
    pd.DataFrame(bon_trials).to_csv(run_dir / "bon_tuning.csv", index=False)
    if aisp_trials:
        pd.DataFrame(aisp_trials).to_csv(run_dir / "aisp_tuning.csv", index=False)
    if tsallis_trials:
        pd.DataFrame(tsallis_trials).to_csv(run_dir / "tsallis_aisp_tuning.csv", index=False)
    return output_path
