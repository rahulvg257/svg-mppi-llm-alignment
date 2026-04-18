from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from aisp_repro.utils import ensure_dir, write_json


def save_reward_plot(aggregate_df: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    ordered = aggregate_df.sort_values("average_reward", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    palette = ["#2a9d8f", "#e76f51", "#264653", "#f4a261", "#8ab17d"]
    colors = [palette[index % len(palette)] for index in range(len(ordered))]
    ax.bar(ordered["method"], ordered["average_reward"], color=colors)
    ax.set_title("Average Reward by Method")
    ax.set_ylabel("Average reward")
    ax.set_xlabel("Method")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def write_markdown_report(
    output_path: str | Path,
    *,
    aggregate_df: pd.DataFrame,
    config_snapshot: dict[str, Any],
    assumptions: list[str],
    deviations: list[str],
    win_rate_available: bool,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    winner = aggregate_df.sort_values("average_reward", ascending=False).iloc[0]["method"]
    methods_present = set(aggregate_df["method"].tolist())
    reward_lookup = {
        row["method"]: row["average_reward"] for row in aggregate_df.to_dict(orient="records")
    }

    implemented_lines = [
        "- Deterministic greedy decoding for TinyLlama-1.1B-Chat.",
        "- Top-p Best-of-N with a matched sample budget against the strongest enabled AISP-style search method.",
    ]
    if "aisp" in methods_present:
        implemented_lines.append(
            "- Faithful AISP with Gaussian perturbations in pre-logit space, greedy rollout decoding, adaptive importance-sampling updates, best-response tracking, and final-sample evaluation from `q(V | U_kappa, sigma^2)`."
        )
    if "tsallis_aisp" in methods_present:
        implemented_lines.append(
            "- Tsallis-AISP as an explicit extension: same pre-logit perturbation rollouts as AISP, but with Tsallis-MPPI-inspired deformed-exponential weighting over iteration-normalized sampled costs."
        )
    implemented_lines.append(
        "- HH-RLHF loading, reward scoring, diversity/coherence metrics, artifact saving, and plotting."
    )

    comparison_lines: list[str] = []
    bon_reward = reward_lookup.get("best_of_n")
    for method_name, label in (("aisp", "AISP"), ("tsallis_aisp", "Tsallis-AISP")):
        method_reward = reward_lookup.get(method_name)
        if method_reward is None or bon_reward is None:
            continue
        verdict = "beat" if method_reward > bon_reward else "did not beat"
        comparison_lines.append(f"- {label} {verdict} BoN under the matched sample budget.")

    lines: list[str] = [
        "# AISP Reproduction Report",
        "",
        "## What Was Implemented",
        "",
        *implemented_lines,
        "",
        "## Aggregate Results",
        "",
        aggregate_df.to_markdown(index=False),
        "",
        "## Assumptions",
        "",
    ]
    lines.extend([f"- {item}" for item in assumptions])
    lines.extend(
        [
            "",
            "## Deviations From The Paper",
            "",
        ]
    )
    lines.extend([f"- {item}" for item in deviations])
    lines.extend(
        [
            "",
            "## Outcome",
            "",
            f"- Best average-reward method on this run: `{winner}`.",
            *(comparison_lines or ["- No BoN comparison was available for AISP-style methods on this run."]),
            f"- External judge win-rate harness available: {'yes' if win_rate_available else 'no, left disabled'}",
            "",
            "## Next Step For Tsallis-AISP",
            "",
            "- Tune Tsallis-specific hyperparameters `r`, `elite_fraction`, and `sigma_sq` on the same held-out subset used for Gaussian AISP.",
            "- If TinyLlama remains brittle, try shorter perturbation horizons or norm-constrained pre-logit perturbations before adding more algorithmic complexity.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")

    write_json(output_path.with_suffix(".config.json"), config_snapshot)
