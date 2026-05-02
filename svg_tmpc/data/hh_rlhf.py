# svg_tmpc/data/hh_rlhf.py
"""Anthropic HH-RLHF prompt loader."""

from __future__ import annotations

from typing import List

from datasets import load_dataset

_ASSISTANT_DELIM = "\n\nAssistant:"
_HUMAN_DELIM = "\n\nHuman:"


def _extract_first_human_turn(chosen: str) -> str:
    """Pull the text from the first ``Human:`` turn up to the first ``Assistant:`` cue.

    HH-RLHF samples are stored as multi-turn dialogues in a single string. We extract
    the *first* user turn so each prompt is a clean, self-contained question and append
    the trailing ``Assistant:`` cue so causal LMs naturally continue as the assistant.
    """
    text = chosen.strip()
    if text.startswith(_HUMAN_DELIM.strip()):
        text = text[len(_HUMAN_DELIM.strip()) :]
    elif text.startswith("Human:"):
        text = text[len("Human:") :]

    cut = text.find(_ASSISTANT_DELIM)
    if cut == -1:
        cut = text.find("Assistant:")
        human_text = text[:cut].strip() if cut != -1 else text.strip()
    else:
        human_text = text[:cut].strip()

    return f"Human: {human_text}{_ASSISTANT_DELIM}"


def load_prompts(split: str = "test", n: int = 500) -> List[str]:
    """Load up to ``n`` cleaned prompts from the HH-RLHF dataset."""
    if n <= 0:
        return []
    dataset = load_dataset("Anthropic/hh-rlhf", split=split)
    prompts: List[str] = []
    seen = set()
    for row in dataset:
        chosen = row.get("chosen", "")
        if not chosen:
            continue
        prompt = _extract_first_human_turn(chosen)
        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(prompt)
        if len(prompts) >= n:
            break
    return prompts
