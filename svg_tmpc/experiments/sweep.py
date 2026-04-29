# svg_tmpc/experiments/sweep.py
"""Hyperparameter sweep harness for svg_tmpc.

Loads the backbone + reward model + prompts once, then iterates a Cartesian grid
of configuration overrides, generating and evaluating each grid point. Per-run
artefacts are written to ``<output-dir>/run_NNN/`` and a flat ``sweep_summary.csv``
aggregates one row per (run, method).

Sweep spec format (YAML):

    grid:
      tmpc.sigma: [0.05, 0.1, 0.2]
      tsallis_mppi.q: [1.2, 1.5, 2.0]

Keys are dotted paths into the base config (see ``configs/default.yaml``).
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
from typing import Any, Dict, Iterator, List

import pandas as pd
import yaml
from tqdm import tqdm

from svg_tmpc.experiments.evaluate import compute_all_metrics
from svg_tmpc.experiments.runner import ExperimentRunner
from svg_tmpc.utils.logging import configure_logging, get_logger

_NON_RELOADABLE_PREFIXES = ("backbone.", "reward.")
_RELOADABLE_BACKBONE_KEYS = {"backbone.max_new_tokens"}


def load_sweep_spec(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        spec = yaml.safe_load(f) or {}
    if "grid" not in spec or not isinstance(spec["grid"], dict):
        raise ValueError(f"Sweep spec {path!r} must define a top-level 'grid' mapping")
    if not spec["grid"]:
        raise ValueError(f"Sweep spec {path!r} has an empty 'grid'")
    return spec


def _grid_iter(grid: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    keys = list(grid.keys())
    for combo in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, combo))


def _warn_non_reloadable(grid: Dict[str, List[Any]], logger) -> None:
    for key in grid:
        if any(key.startswith(p) for p in _NON_RELOADABLE_PREFIXES):
            if key in _RELOADABLE_BACKBONE_KEYS:
                continue
            logger.warning(
                "Override %s is in a section that does NOT reload the model between "
                "sweep runs; the value will be applied to the config but ignored at the "
                "model level. Run separate sweeps for different backbones / RMs.",
                key,
            )


def run_sweep(
    base_config_path: str,
    sweep_spec: Dict[str, Any],
    output_dir: str,
) -> pd.DataFrame:
    logger = get_logger("svg_tmpc.sweep")

    grid = sweep_spec["grid"]
    _warn_non_reloadable(grid, logger)
    combos = list(_grid_iter(grid))
    logger.info("Sweep has %d configurations across %d axes", len(combos), len(grid))

    runner = ExperimentRunner(base_config_path)
    runner.setup()

    base_raw = copy.deepcopy(runner.config.raw)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "sweep_spec.yaml"), "w") as f:
        yaml.safe_dump(sweep_spec, f, sort_keys=False)
    with open(os.path.join(output_dir, "base_config.yaml"), "w") as f:
        yaml.safe_dump(base_raw, f, sort_keys=False)

    rows: List[Dict[str, Any]] = []
    for i, overrides in enumerate(tqdm(combos, desc="sweep", leave=True)):
        runner.config.raw = copy.deepcopy(base_raw)
        runner.apply_overrides(overrides)

        run_dir = os.path.join(output_dir, f"run_{i:03d}")
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "overrides.json"), "w") as f:
            json.dump(overrides, f, indent=2)

        results = runner.run_all()
        runner.save_results(results, run_dir)

        df = compute_all_metrics(results, list(runner.prompts), runner.backbone, runner.reward_model)
        df.to_csv(os.path.join(run_dir, "metrics.csv"))

        for method, metrics_row in df.iterrows():
            row: Dict[str, Any] = {"run_index": i, "method": method}
            row.update(overrides)
            row.update(metrics_row.to_dict())
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(output_dir, "sweep_summary.csv")
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote sweep summary to %s", summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="svg-tmpc-sweep",
        description="Run a Cartesian grid of svg_tmpc configurations.",
    )
    parser.add_argument("--config", "-c", required=True, help="Base config YAML.")
    parser.add_argument("--sweep", "-s", required=True, help="Sweep spec YAML.")
    parser.add_argument("--output-dir", "-o", default="sweeps/")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(level=args.log_level)
    spec = load_sweep_spec(args.sweep)
    df = run_sweep(args.config, spec, args.output_dir)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
