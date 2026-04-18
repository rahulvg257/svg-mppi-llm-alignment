from __future__ import annotations

from typing import Any


def build_pairwise_judge_rows(final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}
    for row in final_rows:
        grouped.setdefault(row["example_id"], {})[row["method"]] = row

    judge_rows: list[dict[str, Any]] = []
    for example_id, methods in grouped.items():
        bon_row = methods.get("best_of_n")
        if not bon_row:
            continue
        for method_name in ("aisp", "tsallis_aisp"):
            search_row = methods.get(method_name)
            if not search_row:
                continue
            judge_rows.append(
                {
                    "example_id": example_id,
                    "prompt_messages": search_row["prompt_messages"],
                    "reward_prompt_text": search_row["reward_prompt_text"],
                    "candidate_a_method": method_name,
                    "candidate_a_response": search_row["response"],
                    "candidate_b_method": "best_of_n",
                    "candidate_b_response": bon_row["response"],
                    "candidate_a_reward": search_row["reward"],
                    "candidate_b_reward": bon_row["reward"],
                    "judge_verdict": None,
                    "judge_enabled": False,
                }
            )
    return judge_rows
