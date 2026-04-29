# svg_tmpc/models/backbone.py
"""TinyLlama backbone wrapper exposing hidden-state hooks for representation editing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


@dataclass
class GenerationConfig:
    do_sample: bool = True
    top_p: float = 0.9
    temperature: float = 0.8
    repetition_penalty: float = 1.0
    max_new_tokens: int = 64


class Backbone:
    """Wrapper around an autoregressive causal-LM with hooks for hidden-state edits.

    The wrapper captures the output of the final RMSNorm/LayerNorm before the LM head
    so that samplers can read the pre-logit hidden state for the *current* token,
    perturb it, and run rollouts directly through the LM head.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: str = "auto",
        dtype: str = "float16",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = _resolve_device(device)
        torch_dtype = _DTYPES.get(dtype.lower(), torch.float32)
        if self.device.type == "cpu":
            torch_dtype = torch.float32

        self.tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

        self._final_norm = self._locate_final_norm()
        self._lm_head = self.model.get_output_embeddings()
        self._captured_hidden: Optional[torch.Tensor] = None
        self._hook_handle = self._final_norm.register_forward_hook(self._capture_hook)

    def _locate_final_norm(self) -> torch.nn.Module:
        # Llama-family models expose model.model.norm as the final RMSNorm before lm_head.
        inner = getattr(self.model, "model", None)
        if inner is not None and hasattr(inner, "norm"):
            return inner.norm
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            return self.model.transformer.ln_f
        raise RuntimeError(
            f"Could not locate final norm layer for model {self.model_name!r}; "
            "extend Backbone._locate_final_norm to support this architecture."
        )

    def _capture_hook(self, _module: torch.nn.Module, _inputs, output) -> None:
        # output shape: (batch, seq, hidden)
        self._captured_hidden = output.detach()

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    def encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool = True) -> str:
        if token_ids.dim() == 2:
            token_ids = token_ids[0]
        return self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=skip_special_tokens)

    @torch.no_grad()
    def get_hidden_state(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the post-final-norm hidden states for every position in input_ids."""
        if input_ids.device != self.device:
            input_ids = input_ids.to(self.device)
        self._captured_hidden = None
        self.model(input_ids=input_ids, use_cache=False)
        if self._captured_hidden is None:
            raise RuntimeError("Final norm hook did not fire; capture failed.")
        return self._captured_hidden

    @torch.no_grad()
    def lm_head_forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Apply the LM head directly to a hidden-state tensor and return logits."""
        return self._lm_head(hidden_state)

    @torch.no_grad()
    def greedy_decode(
        self,
        hidden_state: torch.Tensor,
        max_new_tokens: int,
        prefix_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Greedy rollout starting from a single pre-logit hidden vector.

        ``hidden_state`` is the (already final-normed) pre-logit vector of shape
        (batch, hidden) or (hidden,). ``prefix_ids`` is the conditioning context that
        produced ``hidden_state``; it is used to seed the KV cache for subsequent steps.
        """
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)
        batch = hidden_state.size(0)

        first_logits = self._lm_head(hidden_state)
        next_token = torch.argmax(first_logits, dim=-1, keepdim=True)
        generated = [next_token]

        if prefix_ids is None:
            prefix_ids = torch.empty((batch, 0), dtype=torch.long, device=self.device)
        elif prefix_ids.size(0) != batch:
            prefix_ids = prefix_ids.expand(batch, -1).contiguous()

        seed_input = torch.cat([prefix_ids, next_token], dim=1)

        outputs = self.model(input_ids=seed_input, use_cache=True, return_dict=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated.append(next_token)

        for _ in range(max_new_tokens - 2):
            outputs = self.model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(next_token)
            if self.eos_token_id is not None and bool((next_token == self.eos_token_id).all()):
                break

        return torch.cat(generated, dim=1)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        gen_config: GenerationConfig,
    ) -> str:
        """Standard HuggingFace .generate() wrapper used by the baseline sampler."""
        input_ids = self.encode(prompt)
        output_ids = self.model.generate(
            input_ids=input_ids,
            do_sample=gen_config.do_sample,
            top_p=gen_config.top_p,
            temperature=gen_config.temperature,
            repetition_penalty=gen_config.repetition_penalty,
            max_new_tokens=gen_config.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = output_ids[0, input_ids.size(1):]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def batch_generate(self, prompts: List[str], gen_config: GenerationConfig) -> List[str]:
        return [self.generate(p, gen_config) for p in prompts]

    def close(self) -> None:
        if getattr(self, "_hook_handle", None) is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
