"""Paired comparisons on shared test units.

Every comparison runs on the same visit-pairs, so the difference is estimated pair by pair
and the resampling keeps each pair's two predictions together.

Ref: Sec. 5.6 (comparisons in matched pairs, no cohort blended, the same test batch for every
method), Sec. 2.1 (the differences and their intervals).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad_reconciliation.inference.cluster_bootstrap import Interval, percentile_interval


@dataclass(frozen=True)
class PairedDifference:
    """A difference of two metrics computed on identical units."""

    estimate: float
    interval: Interval
    n_units: int
    n_patients: int

    @property
    def excludes_zero(self) -> bool:
        if not np.isfinite(self.interval.lower) or not np.isfinite(self.interval.upper):
            return False
        return bool(self.interval.lower > 0.0 or self.interval.upper < 0.0)

    def as_mapping(self) -> dict[str, float]:
        return {
            "estimate": self.estimate,
            "lower": self.interval.lower,
            "upper": self.interval.upper,
            "n_units": float(self.n_units),
            "n_patients": float(self.n_patients),
        }


def paired_difference(
    first: np.ndarray,
    second: np.ndarray,
    patients: np.ndarray,
    *,
    replicates: int,
    seed: int,
    level: float = 0.95,
) -> PairedDifference:
    """Difference of two unit-level metric values where both are defined."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    keys = np.asarray(patients)
    if not (left.shape == right.shape == keys.shape):
        raise ValueError("the two metric vectors and the patient labels must share a shape")
    usable = np.isfinite(left) & np.isfinite(right)
    difference = (left - right)[usable]
    clusters = keys[usable]
    estimate = float(difference.mean()) if difference.size else float("nan")
    generator = np.random.default_rng(seed)
    draws = np.empty(int(replicates), dtype=np.float64)
    groups: dict[object, list[int]] = {}
    for position, key in enumerate(clusters.tolist()):
        groups.setdefault(key, []).append(position)
    members = [np.asarray(rows, dtype=np.int64) for rows in groups.values()]
    for replicate in range(int(replicates)):
        if not members:
            draws[replicate] = float("nan")
            continue
        chosen = generator.integers(0, len(members), size=len(members))
        index = np.concatenate([members[position] for position in chosen])
        draws[replicate] = float(difference[index].mean())
    interval = percentile_interval(estimate, draws, level=level)
    return PairedDifference(
        estimate=estimate,
        interval=interval,
        n_units=int(difference.size),
        n_patients=int(np.unique(clusters).size),
    )


def exceeds_margin(difference: PairedDifference, margin: float) -> dict[str, bool]:
    """Whether the point estimate and the interval clear a pre-specified margin."""

    return {
        "point_estimate": bool(difference.estimate >= float(margin)),
        "interval_lower": bool(difference.interval.lower >= float(margin)),
    }


def shared_units_masks(*metric_columns: np.ndarray) -> np.ndarray:
    """Rows on which every supplied metric column is defined."""

    if not metric_columns:
        raise ValueError("at least one metric column is required")
    mask = np.ones(np.asarray(metric_columns[0]).shape, dtype=bool)
    for column in metric_columns:
        mask &= np.isfinite(np.asarray(column, dtype=np.float64))
    return mask
