"""Fusion comparators.

One comparator fuses the two streams visit by visit and then pools; the other encodes the
last visit of each stream separately and concatenates the two embeddings without any
temporal modelling. Both are static in the sense that neither sees the order of the visits.

Ref: Sec. 2.1 (late fusion of per-visit embeddings and the multi-encoder static fusion
network), Table 1 rows 12-13, Table 7 rows 11-12.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from triad_reconciliation.comparators.single_stream import FIELD_WIDTH, WIDTH, masked_mean
from triad_reconciliation.core.config import ModelConfig
from triad_reconciliation.models.triad import StreamBatch

VOLUME_FLAT = 4 * 4 * 4


class LateFusionPerVisit(nn.Module):
    """Fuses each visit's two streams, then pools over the visits."""

    def __init__(self, config: ModelConfig) -> None:
        del config
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(8 + FIELD_WIDTH, WIDTH),
            nn.GELU(),
            nn.Linear(WIDTH, WIDTH),
        )
        self.head = nn.Sequential(nn.Linear(WIDTH, WIDTH // 2), nn.GELU(), nn.Linear(WIDTH // 2, 1))

    def forward(self, batch: StreamBatch) -> Tensor:
        visits = min(batch.morphometry.shape[1], batch.field_indices.shape[1])
        joined = torch.cat((batch.morphometry[:, :visits], batch.field_indices[:, :visits]), dim=-1)
        fused = self.fuse(joined)
        valid = batch.structural_valid[:, :visits] | batch.functional_valid[:, :visits]
        pooled = masked_mean(fused, valid)
        logit: Tensor = self.head(pooled).squeeze(-1)
        return logit


class MultiEncoderStaticFusion(nn.Module):
    """Two encoders on the last visit of each stream, concatenated without a time axis."""

    def __init__(self, config: ModelConfig) -> None:
        del config
        super().__init__()
        self.structural = nn.Sequential(
            nn.Linear(8, WIDTH),
            nn.GELU(),
            nn.Linear(WIDTH, WIDTH),
        )
        self.functional = nn.Sequential(
            nn.Linear(FIELD_WIDTH, WIDTH),
            nn.GELU(),
            nn.Linear(WIDTH, WIDTH),
        )
        self.head = nn.Sequential(nn.Linear(2 * WIDTH, WIDTH), nn.GELU(), nn.Linear(WIDTH, 1))

    def _last(self, values: Tensor, valid: Tensor) -> Tensor:
        weights = valid.to(values.dtype)
        positions = torch.arange(values.shape[1], device=values.device).expand_as(valid)
        latest = torch.where(valid, positions, torch.full_like(positions, -1)).amax(dim=1)
        rows = torch.arange(values.shape[0], device=values.device)
        return values[rows, latest.clamp(min=0)] * weights.sum(dim=1, keepdim=True).clamp(max=1.0)

    def forward(self, batch: StreamBatch) -> Tensor:
        structural = self.structural(self._last(batch.morphometry, batch.structural_valid))
        functional = self.functional(self._last(batch.field_indices, batch.functional_valid))
        logit: Tensor = self.head(torch.cat((structural, functional), dim=-1)).squeeze(-1)
        return logit
