from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

from aisp_repro.config import ModelConfig
from aisp_repro.utils import resolve_device, resolve_torch_dtype


@dataclass
class GeneratedSample:
    text: str
    token_ids: list[int]
    finish_reason: str
    diagnostics: dict[str, Any]


@dataclass
class GenerationModelMetadata:
    model_id: str
    revision: str | None
    commit_hash: str | None
    device: str
    dtype: str
    hidden_size: int
    vocab_size: int


class ChatGenerationModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.dtype = resolve_torch_dtype(config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=config.cache_dir,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.output_head = self.model.get_output_embeddings()

        hidden_size = int(getattr(self.model.config, "hidden_size"))
        vocab_size = int(getattr(self.model.config, "vocab_size"))
        self.metadata = GenerationModelMetadata(
            model_id=config.model_id,
            revision=config.revision,
            commit_hash=getattr(self.model.config, "_commit_hash", None),
            device=str(self.device),
            dtype=str(self.dtype).replace("torch.", ""),
            hidden_size=hidden_size,
            vocab_size=vocab_size,
        )

    @property
    def hidden_size(self) -> int:
        return self.metadata.hidden_size

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    def format_messages(self, messages: list[dict[str, str]]) -> str:
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("Tokenizer does not expose a chat template; native formatting is required.")
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def encode_messages(self, messages: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        prompt_text = self.format_messages(messages)
        encoded = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.config.max_prompt_tokens,
            return_tensors="pt",
        )
        return encoded

    def _decode_new_tokens(self, full_sequence: torch.Tensor, prompt_length: int) -> list[int]:
        tokens = full_sequence[prompt_length:].tolist()
        if self.eos_token_id in tokens:
            eos_index = tokens.index(self.eos_token_id)
            tokens = tokens[:eos_index]
        return [int(token) for token in tokens]

    @torch.inference_mode()
    def generate_greedy(self, messages: list[dict[str, str]]) -> GeneratedSample:
        encoded = self.encode_messages(messages)
        prompt_length = int(encoded["input_ids"].shape[1])
        outputs = self.model.generate(
            input_ids=encoded["input_ids"].to(self.device),
            attention_mask=encoded["attention_mask"].to(self.device),
            do_sample=False,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        generated_ids = self._decode_new_tokens(outputs[0].detach().cpu(), prompt_length)
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return GeneratedSample(
            text=text,
            token_ids=generated_ids,
            finish_reason="greedy",
            diagnostics={"prompt_length_tokens": prompt_length},
        )

    @torch.inference_mode()
    def sample_top_p(
        self,
        messages: list[dict[str, str]],
        num_samples: int,
        batch_size: int,
        top_p: float,
        temperature: float,
        seed: int,
    ) -> list[GeneratedSample]:
        encoded = self.encode_messages(messages)
        prompt_length = int(encoded["input_ids"].shape[1])
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        samples: list[GeneratedSample] = []

        for offset in range(0, num_samples, batch_size):
            current_batch_size = min(batch_size, num_samples - offset)
            torch.manual_seed(seed + offset)
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=current_batch_size,
                max_new_tokens=self.config.max_new_tokens,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.eos_token_id,
            )
            for batch_index in range(outputs.shape[0]):
                generated_ids = self._decode_new_tokens(outputs[batch_index].detach().cpu(), prompt_length)
                text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                samples.append(
                    GeneratedSample(
                        text=text,
                        token_ids=generated_ids,
                        finish_reason="sampled",
                        diagnostics={
                            "prompt_length_tokens": prompt_length,
                            "temperature": temperature,
                            "top_p": top_p,
                        },
                    )
                )
        return samples

    @torch.inference_mode()
    def greedy_rollout_with_prelogit_perturbations(
        self,
        messages: list[dict[str, str]],
        perturbations: torch.Tensor,
    ) -> list[GeneratedSample]:
        encoded = self.encode_messages(messages)
        prompt_ids = encoded["input_ids"][0].detach().cpu()
        tau = int(perturbations.shape[1])
        num_rollouts = int(perturbations.shape[0])
        sequences = [prompt_ids.clone() for _ in range(num_rollouts)]
        prompt_length = int(prompt_ids.shape[0])
        finished = [False] * num_rollouts
        first_step_diagnostics: dict[str, Any] = {}

        for step in range(self.config.max_new_tokens):
            active_indices = [index for index, done in enumerate(finished) if not done]
            if not active_indices:
                break

            active_sequences = [sequences[index] for index in active_indices]
            padded = pad_sequence(active_sequences, batch_first=True, padding_value=self.pad_token_id)
            attention_mask = padded.ne(self.pad_token_id).long()
            padded = padded.to(self.device)
            attention_mask = attention_mask.to(self.device)

            model_outputs = self.model.model(
                input_ids=padded,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = model_outputs.last_hidden_state
            last_indices = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.shape[0], device=self.device)
            prelogits = hidden_states[batch_indices, last_indices]
            base_prelogits = prelogits.detach().clone()

            if step < tau:
                step_perturbation = perturbations[active_indices, step].to(self.device, dtype=prelogits.dtype)
                prelogits = prelogits + step_perturbation
            else:
                step_perturbation = torch.zeros_like(prelogits)

            logits = self.output_head(prelogits)
            next_tokens = torch.argmax(logits, dim=-1).detach().cpu()

            if step == 0:
                first_step_diagnostics = {
                    "prelogit_shape": list(base_prelogits.shape),
                    "perturbation_shape": list(step_perturbation.shape),
                    "lm_head_input_shape": list(prelogits.shape),
                    "perturbation_target": "pre_logit_hidden_state_before_lm_head",
                    "perturbs_input_embeddings": False,
                }

            for relative_index, rollout_index in enumerate(active_indices):
                next_token = next_tokens[relative_index]
                sequences[rollout_index] = torch.cat([sequences[rollout_index], next_token.view(1)], dim=0)
                if int(next_token.item()) == self.eos_token_id:
                    finished[rollout_index] = True

        outputs: list[GeneratedSample] = []
        for rollout_index, sequence in enumerate(sequences):
            token_ids = self._decode_new_tokens(sequence, prompt_length)
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
            finish_reason = "eos" if finished[rollout_index] else "max_new_tokens"
            outputs.append(
                GeneratedSample(
                    text=text,
                    token_ids=token_ids,
                    finish_reason=finish_reason,
                    diagnostics={
                        "prompt_length_tokens": prompt_length,
                        "tau": tau,
                        **first_step_diagnostics,
                    },
                )
            )
        return outputs
