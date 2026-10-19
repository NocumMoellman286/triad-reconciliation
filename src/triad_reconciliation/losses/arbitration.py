"""Arbitration term.

Ref: Sec. 5.5 (``L_arb``), Sec. 5.1 (the arbitration label takes the three values
structural, functional and concordant).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

Reduction = Literal["mean", "sum", "none"]


def arbitration_loss(logits: Tensor, targets: Tensor, *, reduction: Reduction = "mean") -> Tensor:
    """Cross-entropy over the three arbitration classes."""

    if logits.dim() != 2 or logits.shape[1] != 3:
        raise ValueError("arbitration logits must be (batch, 3)")
    loss: Tensor = torch.nn.functional.cross_entropy(logits, targets.long(), reduction=reduction)
    return loss


def agreement_loss(logits: Tensor, targets: Tensor, *, reduction: Reduction = "mean") -> Tensor:
    """Cross-entropy over the hidden agreement state held in the same latent."""

    if logits.dim() != 2 or logits.shape[1] != 2:
        raise ValueError("agreement logits must be (batch, 2)")
    loss: Tensor = torch.nn.functional.cross_entropy(logits, targets.long(), reduction=reduction)
    return loss
