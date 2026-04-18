from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DatasetConfig:
    dataset_id: str = "Anthropic/hh-rlhf"
    config_name: str = "default"
    split: str = "test"
    shuffle_seed: int = 17
    debug_size: int = 2
    tuning_size: int = 3
    eval_size: int = 6
    tuning_start: int = 0
    eval_start: int = 8
    prompt_source_field: str = "chosen"
    cache_dir: str = ".hf_cache/datasets"


@dataclass
class ModelConfig:
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    revision: str | None = None
    cache_dir: str = ".hf_cache/models"
    device: str = "auto"
    torch_dtype: str = "float16"
    max_new_tokens: int = 128
    max_prompt_tokens: int = 1024
    tau: int | None = None


@dataclass
class RewardConfig:
    model_id: str = "sileod/deberta-v3-large-tasksource-rlhf-reward-model"
    revision: str | None = None
    cache_dir: str = ".hf_cache/models"
    device: str = "cpu"
    max_length: int = 1024
    batch_size: int = 4


@dataclass
class EmbeddingConfig:
    model_id: str = "princeton-nlp/sup-simcse-bert-base-uncased"
    revision: str | None = None
    cache_dir: str = ".hf_cache/models"
    device: str = "cpu"
    max_length: int = 256
    batch_size: int = 8


@dataclass
class GreedyConfig:
    enabled: bool = True


@dataclass
class BestOfNConfig:
    enabled: bool = True
    num_samples: int | None = None
    top_p: float = 0.9
    temperature: float = 0.8
    batch_size: int = 4


@dataclass
class AISPConfig:
    enabled: bool = True
    n: int = 32
    kappa: int = 32
    sigma_sq: float = 0.5
    lambda_: float = 0.3
    alpha: float | None = 0.999
    final_sample_count: int | None = None


@dataclass
class TsallisAISPConfig:
    enabled: bool = False
    n: int = 32
    kappa: int = 32
    sigma_sq: float = 0.5
    lambda_: float = 0.3
    alpha: float | None = 0.999
    final_sample_count: int | None = None
    r: float = 2.5
    gamma: float | None = None
    elite_fraction: float = 0.5
    normalize_costs: bool = True


@dataclass
class TuningConfig:
    bon_grid: dict[str, list[float]] = field(
        default_factory=lambda: {
            "temperature": [0.7, 0.8, 1.0],
            "top_p": [0.8, 0.9, 0.95],
        }
    )
    aisp_grid: dict[str, list[float]] = field(
        default_factory=lambda: {
            "sigma_sq": [0.1, 0.3, 0.5],
            "lambda_": [0.1, 0.3, 0.5],
            "alpha": [0.99, 0.999],
        }
    )
    tsallis_aisp_grid: dict[str, list[float]] = field(
        default_factory=lambda: {
            "sigma_sq": [0.1, 0.3, 0.5],
            "lambda_": [0.1, 0.3, 0.5],
            "alpha": [0.99, 0.999],
            "r": [2.0, 2.5, 3.0],
            "elite_fraction": [0.4, 0.5, 0.6],
        }
    )


@dataclass
class OutputConfig:
    base_dir: str = "outputs"
    run_name: str | None = None
    save_candidates: bool = True
    save_csv: bool = True
    save_plot: bool = True


@dataclass
class JudgeConfig:
    enabled: bool = False


@dataclass
class ExperimentConfig:
    seed: int = 7
    phase: str = "debug"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    greedy: GreedyConfig = field(default_factory=GreedyConfig)
    bon: BestOfNConfig = field(default_factory=BestOfNConfig)
    aisp: AISPConfig = field(default_factory=AISPConfig)
    tsallis_aisp: TsallisAISPConfig = field(default_factory=TsallisAISPConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dataclass_from_dict(cls: type[Any], values: dict[str, Any] | None) -> Any:
    values = values or {}
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    return ExperimentConfig(
        seed=raw.get("seed", 7),
        phase=raw.get("phase", "debug"),
        dataset=_dataclass_from_dict(DatasetConfig, raw.get("dataset")),
        model=_dataclass_from_dict(ModelConfig, raw.get("model")),
        reward=_dataclass_from_dict(RewardConfig, raw.get("reward")),
        embedding=_dataclass_from_dict(EmbeddingConfig, raw.get("embedding")),
        greedy=_dataclass_from_dict(GreedyConfig, raw.get("greedy")),
        bon=_dataclass_from_dict(BestOfNConfig, raw.get("bon")),
        aisp=_dataclass_from_dict(AISPConfig, raw.get("aisp")),
        tsallis_aisp=_dataclass_from_dict(TsallisAISPConfig, raw.get("tsallis_aisp")),
        tuning=_dataclass_from_dict(TuningConfig, raw.get("tuning")),
        output=_dataclass_from_dict(OutputConfig, raw.get("output")),
        judge=_dataclass_from_dict(JudgeConfig, raw.get("judge")),
    )
