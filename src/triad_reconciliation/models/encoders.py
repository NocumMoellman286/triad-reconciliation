"""The two stream encoders.

The structural stream is an optic-nerve-head tomography grid plus an optic-disc morphometry
block; the functional stream is a serialised perimetry report. Both produce token sets in a
shared width so the reconciliation module can attend across them.

Ref: Sec. 5.1 (the two streams), Sec. 5.4 (a backbone adapted to optic-disc imaging and an
adapter-enhanced language encoder), Algorithm 1 steps 4-5.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from triad_reconciliation.core.config import ModelConfig
from triad_reconciliation.models.adapters import BottleneckAdapter
from triad_reconciliation.streams.optic_disc import patchify_volume
from triad_reconciliation.streams.timeenc import sinusoidal_time


def open_one_slot(padding_mask: Tensor) -> Tensor:
    """Leave at least one position unmasked, so attention has a key to read.

    A row whose every position is padded makes softmax read an all-masked score row and produces
    a non-finite output that then propagates through the pooling. The released models keep one
    slot open and drop the row later through the visit mask.
    """

    empty = padding_mask.all(dim=1, keepdim=True)
    opened = padding_mask.clone()
    opened[:, :1] = torch.where(empty, torch.zeros_like(opened[:, :1]), opened[:, :1])
    return opened


class _AlignedBlock(nn.Module):
    """Pre-norm block with an adapter after the attention sublayer."""

    def __init__(self, dim: int, heads: int, bottleneck: int, dropout: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.adapter = BottleneckAdapter(dim, bottleneck)
        self.norm_feed_forward = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, 2 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * dim, dim)
        )

    def forward(self, values: Tensor, valid: Tensor) -> Tensor:
        normalised = self.norm_attention(values)
        attended: Tensor = self.attention(
            normalised, normalised, normalised, key_padding_mask=~valid, need_weights=False
        )[0]
        adapted = self.adapter(values + attended)
        expanded: Tensor = self.feed_forward(self.norm_feed_forward(adapted))
        projected: Tensor = adapted + expanded
        return projected


class VisionEncoder(nn.Module):
    """Encodes one optic-disc imaging series into a token set.

    Each visit contributes one morphometry token plus one token per tomography patch, and the
    acquisition time is added to both.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        patch_width = config.volume_patch_edge**3
        patches = (4 // config.volume_patch_edge) ** 3
        self.patch_projection = nn.Linear(patch_width, config.vision_token_dim)
        self.morphometry_projection = nn.Linear(8, config.vision_token_dim)
        self.time_projection = nn.Linear(config.time_features, config.vision_token_dim)
        self.role = nn.Parameter(torch.zeros(patches + 1, config.vision_token_dim))
        self.norm = nn.LayerNorm(config.vision_token_dim)

    @property
    def tokens_per_visit(self) -> int:
        return self.config.volume_patch_edge**3 + 1

    def forward(self, morphometry: Tensor, volume: Tensor, times: Tensor) -> tuple[Tensor, Tensor]:
        if morphometry.dim() != 3 or volume.dim() != 5:
            raise ValueError(
                "morphometry must be (batch, visits, 8) and volume (batch, visits, e, e, e)"
            )
        batch, visits = morphometry.shape[0], morphometry.shape[1]
        flat_volume = volume.reshape(batch * visits, *volume.shape[2:])
        patches = patchify_volume(flat_volume, self.config.volume_patch_edge)
        projected = self.patch_projection(patches)
        summary = self.morphometry_projection(morphometry.reshape(batch * visits, -1)).unsqueeze(1)
        stacked = torch.cat((summary, projected), dim=1)
        stamps = sinusoidal_time(times.reshape(-1), self.config.time_features)
        encoded = stacked + self.role.unsqueeze(0) + self.time_projection(stamps).unsqueeze(1)
        tokens = self.norm(encoded).reshape(batch, visits * self.tokens_per_visit, -1)
        valid = torch.ones((batch, visits), dtype=torch.bool, device=times.device)
        return tokens, valid


class LanguageEncoder(nn.Module):
    """Encodes serialised perimetry reports into a token set in the shared width."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.language_vocab, config.language_dim, padding_idx=0)
        self.visit_type = nn.Parameter(
            torch.zeros(config.max_perimetry_tokens + 1, config.language_dim)
        )
        self.blocks = nn.ModuleList(
            _AlignedBlock(
                config.language_dim,
                config.language_heads,
                config.adapter_bottleneck,
                config.dropout,
            )
            for _ in range(config.language_layers)
        )
        self.norm = nn.LayerNorm(config.language_dim)

    def forward(self, token_ids: Tensor, times: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        if token_ids.dim() != 3:
            raise ValueError("token ids must be (batch, visits, tokens)")
        batch, visits, length = token_ids.shape
        embedded = self.embedding(token_ids)
        summary = embedded.mean(dim=2, keepdim=True)
        stamped = sinusoidal_time(times.reshape(-1), self.config.language_dim).reshape(
            batch, visits, 1, self.config.language_dim
        )
        summary = summary + stamped
        sequence = torch.cat((summary, embedded), dim=2) + self.visit_type.unsqueeze(0)
        filled = (token_ids != 0) & masks.unsqueeze(-1)
        summary_slot = torch.ones_like(masks.unsqueeze(-1))
        combined = torch.cat((summary_slot, filled), dim=2)
        valid: Tensor = ~open_one_slot(~combined)
        flat = sequence.reshape(batch * visits, length + 1, self.config.language_dim)
        flat_valid = valid.reshape(batch * visits, length + 1)
        for block in self.blocks:
            flat = block(flat, flat_valid)
        tokens = self.norm(flat).reshape(batch, visits * (length + 1), self.config.language_dim)
        return tokens, valid.reshape(batch, -1)
