"""Deterministic seeding for every stochastic component of a run.

Ref: Sec. 5.8 (the analysis is deterministic under the recorded initial settings).
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

DEFAULT_SEED = 20_260_219


def set_seed(seed: int = DEFAULT_SEED) -> int:
    """Seed python, numpy and torch from a single value and return it."""

    os.environ["PYTHONHASHSEED"] = str(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return int(seed)


def seed_rng(seed: int) -> np.random.Generator:
    """A numpy generator whose stream is fixed by the same seed vocabulary."""

    return np.random.default_rng(int(seed))
