from __future__ import annotations

import argparse

from aisp_repro.config import load_config
from aisp_repro.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TinyLlama HH-RLHF AISP evaluation pipeline.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--tuning-results",
        default=None,
        help="Optional JSON file produced by scripts/tune_hparams.py.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    run_dir = run_experiment(config, tuning_results_path=args.tuning_results)
    print(run_dir)


if __name__ == "__main__":
    main()

