"""Time encodings for the two clinical timelines.

Each structural token carries its acquisition time and each functional token its visit time,
so the alignment operator can decide what a query time is allowed to see.

Ref: Sec. 5.1 (``t^s_{i,k}`` and ``t^f_{i,l}``), Algorithm 1 steps 4-6.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def sinusoidal_time(years: Tensor, features: int) -> Tensor:
    """Sinusoidal expansion of a time vector, in years, into ``features`` channels."""

    if features % 2 != 0:
        raise ValueError("time feature count must be even")
    half = features // 2
    scale = torch.exp(
        torch.arange(half, dtype=torch.float32, device=years.device)
        * (-math.log(10_000.0) / max(half - 1, 1))
    )
    angles = years.unsqueeze(-1) * scale.unsqueeze(0)
    expanded: Tensor = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    return expanded


def relative_times(stamps: Tensor, reference: Tensor) -> Tensor:
    """Years between each stamp and a per-example reference, sign preserved."""

    return stamps - reference.unsqueeze(-1)


def time_ordering_penalty(stamps: Tensor) -> Tensor:
    """Non-negative count of inversions in each row's time order.

    A causal alignment has nothing to explain when this is zero.
    """

    if stamps.dim() != 2:
        raise ValueError("expected a two-dimensional stamp matrix")
    increments = stamps[:, 1:] - stamps[:, :-1]
    inversions = (increments < 0).to(torch.float32)
    total: Tensor = inversions.sum(dim=1)
    return total
