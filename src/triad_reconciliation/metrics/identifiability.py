"""Identifiability of the stream-evidence weight.

The weight is pinned down only when the two streams carry different evidence: with the
evidence gap below the inverse Lipschitz constant of the fusion map, moving the weight leaves
the fused determination where it was, so no value of the weight can be recovered from the
record. The gap vanishes on concordant pairs, which is where the weight stops being
identified.

Ref: Proposition 1, Sec. 5.4-5.5; Table 3 reports the two subsets and the separation of the
weight the design predicts.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np


class FusionMap(Protocol):
    """Maps a stream-evidence pair to the representation the arbitration reads."""

    def __call__(self, evidence: np.ndarray) -> np.ndarray: ...


def evidence_gap(structural: np.ndarray, functional: np.ndarray) -> np.ndarray:
    return np.abs(
        np.asarray(structural, dtype=np.float64) - np.asarray(functional, dtype=np.float64)
    )


def estimate_lipschitz(
    fusion: Callable[[np.ndarray], np.ndarray],
    centres: np.ndarray,
    *,
    radius: float = 1e-3,
) -> float:
    """Maximum finite-difference operator norm of the fusion map across the centres."""

    points = np.asarray(centres, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("centres must be a two-column evidence matrix")
    baseline = np.atleast_2d(fusion(points))
    worst = 0.0
    for axis in range(2):
        shifted = points.copy()
        shifted[:, axis] += radius
        moved = np.atleast_2d(fusion(shifted))
        local = np.linalg.norm(moved - baseline, axis=1) / radius
        worst = max(worst, float(np.max(local)))
    return worst


def identifiable(gap: np.ndarray, lipschitz: float) -> np.ndarray:
    """Which pairs separate their streams by more than the inverse Lipschitz constant."""

    if lipschitz <= 0:
        return np.zeros(np.shape(gap), dtype=bool)
    threshold = 1.0 / float(lipschitz)
    return np.asarray(gap, dtype=np.float64) > threshold


def fusion_threshold(lipschitz: float) -> float:
    if lipschitz <= 0:
        return float("inf")
    return 1.0 / float(lipschitz)


def weight_sensitivity(
    fused: Callable[..., np.ndarray],
    weights: np.ndarray,
    structural: np.ndarray,
    functional: np.ndarray,
    *,
    step: float = 0.25,
    gain: float = 1.0,
) -> np.ndarray:
    """How far the fused determination moves when the weight is nudged, per pair."""

    weight = np.asarray(weights, dtype=np.float64)
    base = np.atleast_2d(fused(weight, structural, functional, gain=gain))
    shifted = np.clip(weight + float(step), 0.0, 1.0)
    moved = np.atleast_2d(fused(shifted, structural, functional, gain=gain))
    distances: np.ndarray = np.linalg.norm(moved - base, axis=1)
    return distances


FUSION_GAIN = 4.0


def linear_fusion(
    weight: np.ndarray, structural: np.ndarray, functional: np.ndarray, gain: float = 1.0
) -> np.ndarray:
    """The evidence-level fusion the weight operates on.

    A weight of one carries the structural evidence alone and a weight of zero the functional
    evidence alone, so the fused determination moves with the gap between them. The gain is
    the scale at which the fusion map turns evidence into the representation the arbitration
    reads; the manuscript prints no gain, so the value in ``FUSION_GAIN`` is an engineering
    default used consistently by the analysis and its checks.
    """

    left = _as_columns(structural)
    right = _as_columns(functional)
    values = np.asarray(weight, dtype=np.float64).reshape(-1, 1)
    mixed = values * left + (1.0 - values) * right
    return float(gain) * np.concatenate((mixed, np.abs(left - right)[:, :1]), axis=1)


def _as_columns(values: np.ndarray) -> np.ndarray:
    """A one-dimensional vector becomes a single column so the membership vector stays a row."""

    array = np.asarray(values, dtype=np.float64)
    return array[:, None] if array.ndim == 1 else array


def separation_of_weight(
    weights: np.ndarray, structurally_resolved: np.ndarray
) -> tuple[float, float]:
    """Median weight on each side of the recorded direction."""

    values = np.asarray(weights, dtype=np.float64)
    mask = np.asarray(structurally_resolved, dtype=bool)
    if not bool(mask.any()) or bool(mask.all()):
        return float("nan"), float("nan")
    return float(np.median(values[mask])), float(np.median(values[~mask]))
