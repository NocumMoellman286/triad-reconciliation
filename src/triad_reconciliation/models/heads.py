"""Arbitration and agreement readouts.

Both read from the same reconciling state: the three-class arbitration direction and the
two-class agreement state are inferred from one latent, which is what keeps the weight of a
discordant pair attached to the state the direction came from.

Ref: Sec. 5.5 (a softmax classifier over the three arbitration classes, and the agreement
state inferred from the same latent).
"""

from __future__ import annotations

from torch import Tensor, nn

from triad_reconciliation.core.config import ModelConfig


class ArbitrationHead(nn.Module):
    """Three-class direction: structural-dominant, functional-dominant, concordant."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.vision_token_dim)
        self.projection = nn.Linear(config.vision_token_dim, 3)

    def forward(self, state: Tensor) -> Tensor:
        logits: Tensor = self.projection(self.norm(state))
        return logits


class AgreementHead(nn.Module):
    """Two-class hidden agreement state, read from the same latent as the direction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.vision_token_dim)
        self.projection = nn.Linear(config.vision_token_dim, 2)

    def forward(self, state: Tensor) -> Tensor:
        logits: Tensor = self.projection(self.norm(state))
        return logits


class ProgressionHead(nn.Module):
    """Progression probability at the pair's index visit."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(config.vision_token_dim)
        self.projection = nn.Linear(config.vision_token_dim, 1)

    def forward(self, state: Tensor) -> Tensor:
        logit: Tensor = self.projection(self.norm(state)).squeeze(-1)
        return logit
