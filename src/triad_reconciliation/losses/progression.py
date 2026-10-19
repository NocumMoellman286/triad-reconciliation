"""Progression term.

Ref: Sec. 5.5 (``L_prog`` inside the weighted objective), Sec. 5.1 (the progression marker is
a binary label at the index visit).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

Reduction = Literal["mean", "sum", "none"]


def progression_loss(logits: Tensor, targets: Tensor, *, reduction: Reduction = "mean") -> Tensor:
    """Binary cross-entropy with logits."""

    if logits.shape != targets.shape:
        raise ValueError("progression logits and targets must share a shape")
    loss: Tensor = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets.float(), reduction=reduction
    )
    return loss
