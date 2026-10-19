"""Bottleneck adapters for the language encoder.

Ref: Sec. 5.4 (the perimetry stream passes through an adapter-augmented language encoder that
tokenises both streams into a common token space).
"""

from __future__ import annotations

from torch import Tensor, nn


class BottleneckAdapter(nn.Module):
    """A residual bottleneck inserted after the attention sublayer of a language block."""

    def __init__(self, dim: int, bottleneck: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.scale = nn.Parameter(Tensor([1.0]))

    def forward(self, values: Tensor) -> Tensor:
        projected: Tensor = self.up(self.activation(self.down(self.norm(values))))
        return values + self.scale * projected
