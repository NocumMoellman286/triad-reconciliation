"""Cross-stream reconciliation.

The reconciliation state is read out of a single attention pass whose keys carry the
alignment weights of the two timelines, so an anticipative or blocked source contributes
nothing to it.

Ref: Sec. 5.4 (the reconciling state obtained through cross-stream attention), Sec. 5.5
(the stream-evidence weight read from that state).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from triad_reconciliation.core.config import ModelConfig

NEGATIVE_INFINITY = float("-inf")


class CrossStreamLayer(nn.Module):
    """One attention block over the query token and both token sets."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.norm_feed_forward = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim)
        )

    def forward(self, values: Tensor, valid: Tensor, bias: Tensor) -> Tensor:
        batch, length, _ = values.shape
        head_dim = self.dim // self.heads
        normalised = self.norm(values)
        queries = (
            self.query(normalised).reshape(batch, length, self.heads, head_dim).transpose(1, 2)
        )
        keys = self.key(normalised).reshape(batch, length, self.heads, head_dim).transpose(1, 2)
        contents = (
            self.value(normalised).reshape(batch, length, self.heads, head_dim).transpose(1, 2)
        )
        logits = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(head_dim)
        logits = logits + bias.reshape(batch, 1, 1, length)
        logits = logits.masked_fill(~valid.reshape(batch, 1, 1, length), NEGATIVE_INFINITY)
        attention = torch.softmax(logits, dim=-1)
        attended = (
            torch.matmul(attention, contents).transpose(1, 2).reshape(batch, length, self.dim)
        )
        updated = values + self.dropout(self.output(attended))
        expanded: Tensor = self.feed_forward(self.norm_feed_forward(updated))
        projected: Tensor = updated + expanded
        return projected


class CrossStreamReconciliation(nn.Module):
    """Builds the reconciling state from the two aligned token sets."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.vision_token_dim
        if config.language_dim != dim:
            raise ValueError("the two encoders must share the token width")
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.layers = nn.ModuleList(
            CrossStreamLayer(dim, config.reconciliation_heads, config.dropout)
            for _ in range(config.reconciliation_layers)
        )
        self.norm = nn.LayerNorm(dim)
        self.weight_head = nn.Linear(dim, 1)

    def forward(
        self,
        structural_tokens: Tensor,
        structural_valid: Tensor,
        structural_bias: Tensor,
        functional_tokens: Tensor,
        functional_valid: Tensor,
        functional_bias: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch = structural_tokens.shape[0]
        if structural_valid.shape[1] != structural_tokens.shape[1]:
            raise ValueError("structural validity must be expressed per token, not per visit")
        if functional_valid.shape[1] != functional_tokens.shape[1]:
            raise ValueError("functional validity must be expressed per token, not per visit")
        query = self.query.expand(batch, 1, -1)
        query_valid = torch.ones((batch, 1), dtype=torch.bool, device=structural_tokens.device)
        query_bias = torch.zeros(
            (batch, 1), dtype=structural_tokens.dtype, device=structural_tokens.device
        )
        values = torch.cat((query, structural_tokens, functional_tokens), dim=1)
        valid = torch.cat((query_valid, structural_valid, functional_valid), dim=1)
        bias = torch.cat((query_bias, structural_bias, functional_bias), dim=1)
        for layer in self.layers:
            values = layer(values, valid, bias)
        state: Tensor = self.norm(values)[:, 0]
        weight: Tensor = torch.sigmoid(self.weight_head(state)).squeeze(-1)
        return state, weight
