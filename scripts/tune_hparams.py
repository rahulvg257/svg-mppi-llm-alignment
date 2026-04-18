from __future__ import annotations

import argparse

from aisp_repro.config import load_config
from aisp_repro.experiment import run_tuning


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune Best-of-N and AISP hyperparameters.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    output_path = run_tuning(config)
    print(output_path)


if __name__ == "__main__":
    main()
