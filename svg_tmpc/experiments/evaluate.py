# svg_tmpc/experiments/evaluate.py
"""Load saved generations and compute the four-metric summary table."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from svg_tmpc.metrics.coherence import perplexity
from svg_tmpc.metrics.diversity import distinct_n, self_bleu
from svg_tmpc.metrics.reward_score import average_reward
from svg_tmpc.metrics.win_rate import win_rate
from svg_tmpc.models.backbone import Backbone
from svg_tmpc.models.reward import RewardModel
from svg_tmpc.utils.logging import configure_logging, get_logger

_BASELINE_KEY = "baseline"


def load_results(output_dir: str, results_filename: str = "responses.json") -> Dict[str, Any]:
    path = os.path.join(output_dir, results_filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No results file at {path!r}")
    with open(path, "r") as f:
        return json.load(f)


def compute_all_metrics(
    results: Dict[str, Dict[str, Any]],
    prompts: List[str],
    backbone: Backbone,
    reward_model: RewardModel,
) -> pd.DataFrame:
    """Build a DataFrame with one row per method and the four primary metrics."""
    logger = get_logger("svg_tmpc.evaluate")
    baseline_responses = (
        results[_BASELINE_KEY]["responses"] if _BASELINE_KEY in results else None
    )

    rows = []
    for method, payload in results.items():
        responses = list(payload["responses"])
        logger.info("Computing metrics for method=%s (%d responses)", method, len(responses))

        d1 = distinct_n(responses, n=1)
        d2 = distinct_n(responses, n=2)
        sb = self_bleu(responses)
        ppl = perplexity(responses, backbone.model, backbone.tokenizer)
        avg_r = average_reward(prompts, responses, reward_model)

        if baseline_responses is None or method == _BASELINE_KEY:
            wr = float("nan") if baseline_responses is None else 0.5
        else:
            wr = win_rate(responses, baseline_responses, prompts, reward_model)

        rows.append(
            {
                "method": method,
                "distinct_1": d1,
                "distinct_2": d2,
                "self_bleu": sb,
                "perplexity": ppl,
                "avg_reward": avg_r,
                "win_rate_vs_baseline": wr,
            }
        )

    return pd.DataFrame(rows).set_index("method")


def print_summary_table(df: pd.DataFrame) -> None:
    formatted = df.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].map(lambda v: f"{v:.4f}" if isinstance(v, float) else v)
    print(formatted.to_string())


def save_metrics(df: pd.DataFrame, output_dir: str, filename: str = "metrics.csv") -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="svg-tmpc-eval", description="Evaluate saved generations."
    )
    parser.add_argument("--output-dir", "-o", required=True, help="Directory containing responses.json")
    parser.add_argument(
        "--results-filename",
        default="responses.json",
        help="Filename of the saved generations within output-dir.",
    )
    parser.add_argument(
        "--metrics-filename",
        default="metrics.csv",
        help="Filename for the metrics CSV written into output-dir.",
    )
    parser.add_argument(
        "--backbone",
        default=None,
        help="Override backbone model name (otherwise read from saved config).",
    )
    parser.add_argument(
        "--reward-model",
        default=None,
        help="Override reward model name (otherwise read from saved config).",
    )
    parser.add_argument("--device", default=None, help="Override device.")
    parser.add_argument("--dtype", default=None, help="Override dtype.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(level=args.log_level)
    logger = get_logger("svg_tmpc.evaluate")

    payload = load_results(args.output_dir, args.results_filename)
    config = payload.get("config", {}) or {}
    results = payload["results"]

    prompts: Optional[List[str]] = None
    for method_payload in results.values():
        prompts = list(method_payload.get("prompts", []))
        if prompts:
            break
    if not prompts:
        raise RuntimeError("No prompts found in results payload")

    backbone_name = args.backbone or config.get("backbone", {}).get("model_name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    rm_name = args.reward_model or config.get("reward", {}).get(
        "model_name", "OpenAssistant/reward-model-deberta-v3-large-v2"
    )
    device = args.device or config.get("device", "auto")
    dtype = args.dtype or config.get("dtype", "float16")

    logger.info("Loading models for evaluation: backbone=%s rm=%s", backbone_name, rm_name)
    backbone = Backbone(model_name=backbone_name, device=device, dtype=dtype)
    reward_model = RewardModel(
        model_name=rm_name,
        device=device,
        max_length=int(config.get("reward", {}).get("max_length", 512)),
        batch_size=int(config.get("reward", {}).get("batch_size", 4)),
    )

    df = compute_all_metrics(results, prompts, backbone, reward_model)
    print_summary_table(df)
    out_path = save_metrics(df, args.output_dir, args.metrics_filename)
    logger.info("Wrote metrics to %s", out_path)


if __name__ == "__main__":
    main()
