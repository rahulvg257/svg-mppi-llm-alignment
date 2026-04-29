# svg_tmpc/experiments/runner.py
"""Orchestration of all four decoding strategies on a shared prompt set."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

from svg_tmpc.data.hh_rlhf import load_prompts
from svg_tmpc.models.backbone import Backbone
from svg_tmpc.models.reward import RewardModel
from svg_tmpc.samplers.base import BaseSampler
from svg_tmpc.samplers.baseline import BaselineSampler
from svg_tmpc.samplers.best_of_n import BestOfNSampler
from svg_tmpc.samplers.tmpc import TMPCSampler
from svg_tmpc.samplers.tsallis_mppi import TsallisMPPISampler
from svg_tmpc.utils.logging import configure_logging, get_logger
from svg_tmpc.utils.seeding import set_global_seed

ALL_METHODS = ("baseline", "best_of_n", "tmpc", "tsallis_mppi")


@dataclass
class RunConfig:
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        return cls(raw=raw)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.raw
        for k in keys:
            if not isinstance(node, dict) or k not in node:
                return default
            node = node[k]
        return node


class ExperimentRunner:
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self.config = RunConfig.from_yaml(config_path)
        self.logger = get_logger("svg_tmpc.runner")

        self.backbone: Optional[Backbone] = None
        self.reward_model: Optional[RewardModel] = None
        self.prompts: List[str] = []
        self.samplers: Dict[str, BaseSampler] = {}

    def setup(self) -> None:
        seed = int(self.config.get("seed", default=42))
        set_global_seed(seed)
        self.logger.info("Seeded RNGs with seed=%d", seed)

        device = str(self.config.get("device", default="auto"))
        dtype = str(self.config.get("dtype", default="float16"))

        backbone_name = str(self.config.get("backbone", "model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"))
        trust_remote = bool(self.config.get("backbone", "trust_remote_code", default=False))
        device_map = self.config.get("backbone", "device_map", default=None)
        load_in_8bit = bool(self.config.get("backbone", "load_in_8bit", default=False))
        load_in_4bit = bool(self.config.get("backbone", "load_in_4bit", default=False))
        bnb_4bit_quant_type = str(self.config.get("backbone", "bnb_4bit_quant_type", default="nf4"))
        bnb_4bit_use_double_quant = bool(
            self.config.get("backbone", "bnb_4bit_use_double_quant", default=False)
        )
        self.logger.info(
            "Loading backbone %s on device=%s dtype=%s device_map=%s 8bit=%s 4bit=%s",
            backbone_name, device, dtype, device_map, load_in_8bit, load_in_4bit,
        )
        self.backbone = Backbone(
            model_name=backbone_name,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote,
            device_map=device_map,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        )

        rm_name = str(self.config.get("reward", "model_name", default="OpenAssistant/reward-model-deberta-v3-large-v2"))
        rm_max_length = int(self.config.get("reward", "max_length", default=512))
        rm_batch = int(self.config.get("reward", "batch_size", default=4))
        self.logger.info("Loading reward model %s", rm_name)
        self.reward_model = RewardModel(
            model_name=rm_name,
            device=device,
            max_length=rm_max_length,
            batch_size=rm_batch,
        )

        split = str(self.config.get("data", "split", default="test"))
        n_prompts = int(self.config.get("data", "n_prompts", default=50))
        self.logger.info("Loading %d prompts from HH-RLHF[%s]", n_prompts, split)
        self.prompts = load_prompts(split=split, n=n_prompts)
        self.logger.info("Loaded %d prompts", len(self.prompts))

        self.samplers = self._build_samplers()

    def _build_samplers(self) -> Dict[str, BaseSampler]:
        assert self.backbone is not None and self.reward_model is not None
        do_sample = bool(self.config.get("decoding", "do_sample", default=True))
        top_p = float(self.config.get("decoding", "top_p", default=0.9))
        temperature = float(self.config.get("decoding", "temperature", default=0.8))
        repetition_penalty = float(self.config.get("decoding", "repetition_penalty", default=1.0))

        samplers: Dict[str, BaseSampler] = {}

        samplers["baseline"] = BaselineSampler(
            backbone=self.backbone,
            reward_model=self.reward_model,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        samplers["best_of_n"] = BestOfNSampler(
            backbone=self.backbone,
            reward_model=self.reward_model,
            N=int(self.config.get("best_of_n", "N", default=8)),
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )

        samplers["tmpc"] = TMPCSampler(
            backbone=self.backbone,
            reward_model=self.reward_model,
            K=int(self.config.get("tmpc", "K", default=16)),
            H=int(self.config.get("tmpc", "H", default=8)),
            sigma=float(self.config.get("tmpc", "sigma", default=0.1)),
            lambda_=float(self.config.get("tmpc", "lambda_", default=1.0)),
        )

        samplers["tsallis_mppi"] = TsallisMPPISampler(
            backbone=self.backbone,
            reward_model=self.reward_model,
            K=int(self.config.get("tsallis_mppi", "K", default=16)),
            H=int(self.config.get("tsallis_mppi", "H", default=8)),
            sigma=float(self.config.get("tsallis_mppi", "sigma", default=0.1)),
            lambda_=float(self.config.get("tsallis_mppi", "lambda_", default=1.0)),
            q=float(self.config.get("tsallis_mppi", "q", default=1.5)),
        )

        return samplers

    def _enabled(self, method: str) -> bool:
        flag = self.config.get("methods", method, default=True)
        return bool(flag)

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply dotted-path overrides into the in-memory config and rebuild samplers.

        Used by the sweep harness to swap hyperparameters between runs without
        reloading the backbone or reward model. ``backbone.*`` and ``reward.*``
        overrides do *not* trigger a model reload and will be ignored at the model
        level; sweep over those by launching separate runs instead.
        """
        for dotted, value in overrides.items():
            keys = dotted.split(".")
            node = self.config.raw
            for k in keys[:-1]:
                if k not in node or not isinstance(node[k], dict):
                    node[k] = {}
                node = node[k]
            node[keys[-1]] = value
        if self.backbone is not None and self.reward_model is not None:
            self.samplers = self._build_samplers()

    def run_method(self, method_name: str) -> Dict[str, Any]:
        if self.backbone is None:
            raise RuntimeError("ExperimentRunner.setup() must be called before run_method")
        if method_name not in self.samplers:
            raise KeyError(f"Unknown method {method_name!r}; expected one of {list(self.samplers)}")

        sampler = self.samplers[method_name]
        max_new_tokens = int(self.config.get("backbone", "max_new_tokens", default=64))

        responses: List[str] = []
        self.logger.info("Running method=%s over %d prompts", method_name, len(self.prompts))
        for prompt in tqdm(self.prompts, desc=f"{method_name}", leave=False):
            response = sampler.generate_one(prompt, max_new_tokens)
            responses.append(response)

        return {
            "method": method_name,
            "prompts": list(self.prompts),
            "responses": responses,
        }

    def run_all(self) -> Dict[str, Dict[str, Any]]:
        results: Dict[str, Dict[str, Any]] = {}
        for method in ALL_METHODS:
            if not self._enabled(method):
                self.logger.info("Skipping disabled method=%s", method)
                continue
            results[method] = self.run_method(method)
        return results

    def save_results(self, results: Dict[str, Dict[str, Any]], output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        filename = str(self.config.get("output", "results_filename", default="responses.json"))
        path = os.path.join(output_dir, filename)
        payload = {
            "config_path": self.config_path,
            "config": self.config.raw,
            "results": results,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.logger.info("Wrote results to %s", path)
        return path


def main() -> None:
    parser = argparse.ArgumentParser(prog="svg-tmpc-run", description="Run all decoding methods.")
    parser.add_argument(
        "--config",
        "-c",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "default.yaml"),
        help="Path to YAML config (defaults to configs/default.yaml shipped with the package).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Override output directory (otherwise read from config.output.dir).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity (DEBUG/INFO/WARNING/ERROR).",
    )
    args = parser.parse_args()

    configure_logging(level=args.log_level)

    runner = ExperimentRunner(args.config)
    runner.setup()
    results = runner.run_all()
    output_dir = args.output_dir or str(runner.config.get("output", "dir", default="outputs"))
    runner.save_results(results, output_dir)


if __name__ == "__main__":
    main()
