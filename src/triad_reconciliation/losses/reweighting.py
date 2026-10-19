"""Agreement-state re-weighting.

A pair the model reads as discordant carries more weight than its frequency in the cohort,
because the clinical significance of the minority is not its share of the records.

Ref: Sec. 5.5, ``omega_i = 1 + lambda * 1[c-hat_i = discordant]``.
"""

from __future__ import annotations

import torch
from torch import Tensor

DISCORDANT_INDEX = 1


def agreement_weights(predicted_agreement: Tensor, strength: float) -> Tensor:
    """The per-pair multiplier read off the inferred agreement state."""

    if strength < 0:
        raise ValueError("the re-weighting strength must be non-negative")
    indicator = (predicted_agreement == DISCORDANT_INDEX).to(torch.float32)
    weights: Tensor = 1.0 + float(strength) * indicator
    return weights


def discordant_weight_mass(weights: Tensor, predicted_agreement: Tensor) -> float:
    """Share of the batch weight carried by the pairs read as discordant.

    The discordant share of a batch and the share of its weight are different numbers, and the
    gap between them is what the re-weighting moves.

    Ref: Sec. 5.5, the weighted objective; Sec. 2.2, the ablation of that weighting.
    """

    total = float(weights.detach().sum())
    if total <= 0.0:
        return 0.0
    selected = weights.detach() * (predicted_agreement == DISCORDANT_INDEX).to(weights.dtype)
    return float(selected.sum()) / total
