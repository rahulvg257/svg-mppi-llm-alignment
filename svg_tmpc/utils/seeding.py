# svg_tmpc/utils/seeding.py
"""Global RNG seeding for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
