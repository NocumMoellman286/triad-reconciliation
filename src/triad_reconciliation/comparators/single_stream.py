"""Single-stream comparators.

Five comparators read one stream each: a temporal convolution over the optic-disc
morphometry series, a volumetric network and a patch transformer over the tomography grid,
and two perimetry sequence models. Each has its own input surface and its own temporal
treatment, and none shares an encoder with the reconciliation stack.

Ref: Table 1 rows 6-10, Table 7 rows 6-10.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from triad_reconciliation.core.config import ModelConfig
from triad_reconciliation.models.encoders import open_one_slot
from triad_reconciliation.models.triad import StreamBatch
from triad_reconciliation.streams.optic_disc import patchify_volume
from triad_reconciliation.streams.timeenc import sinusoidal_time

WIDTH = 72
FIELD_WIDTH = 6


def masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    """Mean over the visit axis, skipping the padded positions."""

    weights = valid.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)


class FundusTemporalNetwork(nn.Module):
    """Dilated temporal convolution over the optic-disc morphometry series."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed = nn.Linear(8, WIDTH)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(WIDTH, WIDTH, kernel_size=3, padding=dilation, dilation=dilation)
            for dilation in (1, 2, 4)
        )
        self.norms = nn.ModuleList(nn.GroupNorm(8, WIDTH) for _ in range(3))
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Sequential(nn.Linear(WIDTH, WIDTH // 2), nn.GELU(), nn.Linear(WIDTH // 2, 1))

    def forward(self, batch: StreamBatch) -> Tensor:
        sequence = self.embed(batch.morphometry).transpose(1, 2)
        for convolution, norm in zip(self.convolutions, self.norms, strict=True):
            sequence = sequence + torch.nn.functional.gelu(norm(convolution(sequence)))
        pooled = self.dropout(masked_mean(sequence.transpose(1, 2), batch.structural_valid))
        logit: Tensor = self.head(pooled).squeeze(-1)
        return logit


class OpticNerveHeadVolumeNetwork(nn.Module):
    """Three-dimensional convolution over the tomography grid, pooled across visits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv3d(1, 12, kernel_size=3, padding=1),
            nn.GroupNorm(4, 12),
            nn.GELU(),
            nn.Conv3d(12, 20, kernel_size=3, padding=1),
            nn.GroupNorm(5, 20),
            nn.GELU(),
        )
        self.projection = nn.Linear(20, WIDTH)
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Sequential(nn.Linear(WIDTH, WIDTH // 2), nn.GELU(), nn.Linear(WIDTH // 2, 1))

    def forward(self, batch: StreamBatch) -> Tensor:
        batch_size, visits = batch.volume.shape[0], batch.volume.shape[1]
        volumes = batch.volume.reshape(batch_size * visits, 1, *batch.volume.shape[2:])
        features = self.blocks(volumes).mean(dim=(2, 3, 4))
        per_visit = self.projection(features).reshape(batch_size, visits, WIDTH)
        pooled = self.dropout(masked_mean(per_visit, batch.structural_valid))
        logit: Tensor = self.head(pooled).squeeze(-1)
        return logit


class OpticNerveHeadVisionTransformer(nn.Module):
    """Patch transformer over the tomography grid, read out at a class token."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        patch_width = config.volume_patch_edge**3
        patches = (4 // config.volume_patch_edge) ** 3
        self.patch = nn.Linear(patch_width, WIDTH)
        self.class_token = nn.Parameter(torch.zeros(1, 1, WIDTH))
        self.role = nn.Parameter(torch.zeros(1, patches + 1, WIDTH))
        layer = nn.TransformerEncoderLayer(
            WIDTH,
            4,
            dim_feedforward=2 * WIDTH,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(WIDTH, WIDTH // 2), nn.GELU(), nn.Linear(WIDTH // 2, 1))
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.role, std=0.02)

    def forward(self, batch: StreamBatch) -> Tensor:
        batch_size, visits = batch.volume.shape[0], batch.volume.shape[1]
        volumes = batch.volume.reshape(batch_size * visits, *batch.volume.shape[2:])
        patches = self.patch(patchify_volume(volumes, 2))
        prefix = self.class_token.expand(patches.shape[0], -1, -1) + self.role[:, :1]
        sequence = torch.cat((prefix, patches + self.role[:, 1:]), dim=1)
        read_out = self.encoder(sequence)[:, 0].reshape(batch_size, visits, WIDTH)
        pooled = masked_mean(read_out, batch.structural_valid)
        logit: Tensor = self.head(pooled).squeeze(-1)
        return logit


class PerimetrySequenceNetwork(nn.Module):
    """Recurrent model over the perimetry visit sequence."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed = nn.Linear(FIELD_WIDTH, WIDTH)
        self.recurrent = nn.LSTM(
            WIDTH, WIDTH, num_layers=2, batch_first=True, dropout=config.dropout
        )
        self.head = nn.Sequential(nn.Linear(WIDTH, WIDTH // 2), nn.GELU(), nn.Linear(WIDTH // 2, 1))

    def forward(self, batch: StreamBatch) -> Tensor:
        embedded = torch.nn.functional.gelu(self.embed(batch.field_indices))
        outputs, _ = self.recurrent(embedded)
        weights = batch.functional_valid.to(outputs.dtype).unsqueeze(-1)
        last = (outputs * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        logit: Tensor = self.head(last).squeeze(-1)
        return logit


class PerimetryFieldIndexTransformer(nn.Module):
    """Transformer over the field-index series, masked at padded visits."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.time_features = config.time_features
        self.embed = nn.Linear(FIELD_WIDTH, WIDTH)
        self.time_projection = nn.Linear(config.time_features, WIDTH)
        layer = nn.TransformerEncoderLayer(
            WIDTH,
            4,
            dim_feedforward=2 * WIDTH,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(WIDTH, WIDTH // 2), nn.GELU(), nn.Linear(WIDTH // 2, 1))

    def forward(self, batch: StreamBatch) -> Tensor:
        batch_size, visits = batch.functional_times.shape
        embedded = self.embed(batch.field_indices)
        stamps = sinusoidal_time(batch.functional_times.reshape(-1), self.time_features)
        embedded = embedded + self.time_projection(stamps).reshape(batch_size, visits, WIDTH)
        padding = open_one_slot(~batch.functional_valid)
        encoded = self.encoder(embedded, src_key_padding_mask=padding)
        pooled = masked_mean(encoded, batch.functional_valid)
        logit: Tensor = self.head(pooled).squeeze(-1)
        return logit
