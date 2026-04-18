from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset

from aisp_repro.config import DatasetConfig


ROLE_PATTERN = re.compile(r"(?:^|\n\n)(Human|Assistant):")


@dataclass
class PromptExample:
    example_id: str
    dataset_index: int
    config_name: str
    split: str
    messages: list[dict[str, str]]
    reward_prompt_text: str
    chosen_response: str
    rejected_response: str


def parse_hh_transcript(transcript: str) -> list[dict[str, str]]:
    text = transcript.strip()
    matches = list(ROLE_PATTERN.finditer(text))
    if not matches:
        raise ValueError("Could not parse HH-RLHF transcript format.")

    messages: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        role = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        mapped_role = "user" if role == "Human" else "assistant"
        messages.append({"role": mapped_role, "content": content})
    return messages


def split_prompt_and_response(transcript: str) -> tuple[list[dict[str, str]], str]:
    messages = parse_hh_transcript(transcript)
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("Expected transcript to end with an assistant message.")
    return messages[:-1], messages[-1]["content"]


def messages_to_hh_prompt(messages: list[dict[str, str]]) -> str:
    chunks: list[str] = []
    for message in messages:
        if message["role"] == "user":
            chunks.append(f"\n\nHuman: {message['content']}")
        elif message["role"] == "assistant":
            chunks.append(f"\n\nAssistant: {message['content']}")
        else:
            raise ValueError(f"Unsupported role in HH prompt conversion: {message['role']}")
    chunks.append("\n\nAssistant:")
    return "".join(chunks)


def _slice_dataset(dataset: Dataset, start: int, size: int) -> Dataset:
    end = min(start + size, len(dataset))
    return dataset.select(range(start, end))


def load_prompt_examples(config: DatasetConfig, phase: str) -> list[PromptExample]:
    dataset = load_dataset(
        config.dataset_id,
        config.config_name,
        split=config.split,
        cache_dir=config.cache_dir,
    )
    dataset = dataset.shuffle(seed=config.shuffle_seed)

    if phase == "debug":
        subset = _slice_dataset(dataset, config.tuning_start, config.debug_size)
    elif phase == "tune":
        subset = _slice_dataset(dataset, config.tuning_start, config.tuning_size)
    elif phase == "eval":
        subset = _slice_dataset(dataset, config.eval_start, config.eval_size)
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    examples: list[PromptExample] = []
    for local_idx, row in enumerate(subset):
        chosen_messages, chosen_response = split_prompt_and_response(row["chosen"])
        _, rejected_response = split_prompt_and_response(row["rejected"])
        reward_prompt_text = messages_to_hh_prompt(chosen_messages)
        examples.append(
            PromptExample(
                example_id=f"{config.config_name}-{config.split}-{local_idx}",
                dataset_index=int(local_idx),
                config_name=config.config_name,
                split=config.split,
                messages=chosen_messages,
                reward_prompt_text=reward_prompt_text,
                chosen_response=chosen_response,
                rejected_response=rejected_response,
            )
        )
    return examples
