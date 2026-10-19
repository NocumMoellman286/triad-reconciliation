"""Attribution-consistency term.

A prediction whose arbitration direction contradicts the stream weight it reports pays a
penalty proportional to how far the weight sits from the margin that direction implies.

Ref: Sec. 5.5, ``L_cons = 1[alpha-hat contradicts a-hat] * |alpha - alpha_margin|``.
"""

from __future__ import annotations

from torch import Tensor

STRUCTURAL_INDEX = 0
FUNCTIONAL_INDEX = 1
CONCORDANT_INDEX = 2


def contradiction_mask(direction: Tensor, weight: Tensor, margin: float) -> Tensor:
    """Which entries report a direction their stream weight does not support."""

    structural = direction == STRUCTURAL_INDEX
    functional = direction == FUNCTIONAL_INDEX
    contradicts_structural = structural & (weight < margin)
    contradicts_functional = functional & (weight > 1.0 - margin)
    return contradicts_structural | contradicts_functional


def consistency_loss(
    direction: Tensor, weight: Tensor, margin: float, *, reduction: str = "mean"
) -> Tensor:
    """``1[alpha-hat contradicts a-hat] * |alpha - alpha_margin|``."""

    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(f"unsupported reduction {reduction!r}")
    distance = (weight - margin).abs()
    penalised = contradiction_mask(direction, weight, margin).to(distance.dtype) * distance
    if reduction == "sum":
        total: Tensor = penalised.sum()
        return total
    if reduction == "none":
        return penalised
    mean: Tensor = penalised.mean()
    return mean
