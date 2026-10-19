"""The unconstrained concatenated-timeline comparator.

Both streams land on one timeline and a single transformer reads them, so a token may attend
to any other token whatever its time: nothing in the comparator enforces monotonicity or
causality. Its post-hoc attention attribution is the strongest attribution proxy the study
compares the model's own direction against.

Ref: Sec. 2.1 (the unconstrained concatenated-timeline transformer and the parity statement),
Sec. 5.4 (the comparator is neither causal nor monotone), Table 1 row 14, Table 3 row 4.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from triad_reconciliation.core.config import ModelConfig
from triad_reconciliation.models.encoders import open_one_slot
from triad_reconciliation.models.triad import StreamBatch
from triad_reconciliation.streams.timeenc import sinusoidal_time

WIDTH = 96


class ConcatenatedTimelineTransformer(nn.Module):
    """One transformer over both streams on a single concatenated timeline."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.time_features = config.time_features
        self.structural = nn.Linear(8 + 4 * 4 * 4, WIDTH)
        self.functional = nn.Linear(6, WIDTH)
        self.time_projection = nn.Linear(config.time_features, WIDTH)
        self.class_token = nn.Parameter(torch.zeros(1, 1, WIDTH))
        layer = nn.TransformerEncoderLayer(
            WIDTH,
            4,
            dim_feedforward=2 * WIDTH,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=3, enable_nested_tensor=False)
        self.head = nn.Linear(WIDTH, 1)
        nn.init.normal_(self.class_token, std=0.02)

    def _tokens(self, batch: StreamBatch) -> tuple[Tensor, Tensor]:
        batch_size, visits = batch.morphometry.shape[0], batch.morphometry.shape[1]
        structural = self.structural(
            torch.cat((batch.morphometry, batch.volume.reshape(batch_size, visits, -1)), dim=-1)
        )
        structural = structural + self._stamp(batch.structural_times)
        functional = self.functional(batch.field_indices) + self._stamp(batch.functional_times)
        tokens = torch.cat((structural, functional), dim=1)
        valid = torch.cat((batch.structural_valid, batch.functional_valid), dim=1)
        prefix = self.class_token.expand(batch_size, -1, -1)
        return torch.cat((prefix, tokens), dim=1), torch.cat(
            (torch.ones_like(batch.structural_valid[:, :1]), valid), dim=1
        )

    def _stamp(self, times: Tensor) -> Tensor:
        shape = times.shape
        stamps = sinusoidal_time(times.reshape(-1), self.time_features)
        stamped: Tensor = self.time_projection(stamps).reshape(*shape, WIDTH)
        return stamped

    def forward(self, batch: StreamBatch) -> Tensor:
        tokens, valid = self._tokens(batch)
        encoded = self.encoder(tokens, src_key_padding_mask=open_one_slot(~valid))
        logit: Tensor = self.head(encoded[:, 0]).squeeze(-1)
        return logit

    def attention_attribution(self, batch: StreamBatch) -> tuple[Tensor, Tensor]:
        """Post-hoc attention of the class token, offered as the attribution proxy.

        The comparator produces no direction of its own, so the weights are read after the
        fact off the class token rather than emitted by the model.
        """

        tokens, valid = self._tokens(batch)
        attended = self.encoder.layers[0].self_attn(
            tokens,
            tokens,
            tokens,
            key_padding_mask=open_one_slot(~valid),
            need_weights=True,
            average_attn_weights=True,
        )
        weights: Tensor = attended[1][:, 0, 1:]
        return weights, valid[:, 1:]
