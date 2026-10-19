"""Non-anticipative monotone alignment.

The operator returns a weight per source token: increasing along the clinical timeline,
normalised within a stream, and exactly zero for any source whose time is at or beyond the
query time. The zero-weight requirement is what makes the operator causal, and monotonicity
is structural rather than learned, so no training step can reintroduce an anticipative path.

Ref: Sec. 5.4 (the alignment operator A(tau)), Algorithm 1 step 6.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

NEGATIVE_INFINITY = float("-inf")


class MonotoneCausalAlignment(nn.Module):
    """A monotone causal kernel over the two clinical timelines."""

    def __init__(self) -> None:
        super().__init__()
        self.slope = nn.Parameter(torch.zeros(()))

    @property
    def rate(self) -> Tensor:
        return torch.nn.functional.softplus(self.slope) + 1e-4

    def _weights(self, source: Tensor, allowed: Tensor) -> Tensor:
        logits = self.rate * source
        blocked = torch.where(allowed, logits, torch.full_like(logits, -1e30))
        shifted = blocked - blocked.amax(dim=1, keepdim=True)
        exponentials = torch.where(allowed, torch.exp(shifted), torch.zeros_like(shifted))
        total = exponentials.sum(dim=1, keepdim=True)
        return torch.where(
            total > 0, exponentials / total.clamp(min=1e-30), torch.zeros_like(exponentials)
        )

    def forward(
        self,
        query_years: Tensor,
        structural_times: Tensor,
        functional_times: Tensor,
        structural_valid: Tensor,
        functional_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return per-token alignment weights for the two streams.

        ``query_years`` is the pair's query time, so a source token whose time is at or beyond
        it receives a zero weight.
        """

        cutoff = query_years.reshape(-1, 1)
        structural_allowed = (structural_times < cutoff) & structural_valid
        functional_allowed = (functional_times < cutoff) & functional_valid
        return (
            self._weights(structural_times, structural_allowed),
            self._weights(functional_times, functional_allowed),
        )

    def attention_bias(self, weights: Tensor, valid: Tensor) -> Tensor:
        """Additive log-weight bias; a zero weight blocks the key exactly."""

        positive = (weights > 0) & valid
        logged = weights.clamp(min=1e-30).log()
        return logged.masked_fill(~positive, NEGATIVE_INFINITY)
