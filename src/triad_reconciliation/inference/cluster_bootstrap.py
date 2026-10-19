"""Patient-clustered bootstrap.

Repeated measures from one patient enter and leave together, so the resampling unit is the
patient and not the eye, the visit or the visit-pair.

Ref: Sec. 5.6 (cluster-bootstrapped intervals with the patient as the resampling unit, 2,000
replicates, percentile intervals).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

Statistic = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class Interval:
    """A point estimate beside a percentile interval and the replicates behind it."""

    estimate: float
    lower: float
    upper: float
    replicates: int

    @property
    def half_width(self) -> float:
        return 0.5 * (self.upper - self.lower)

    def as_mapping(self) -> dict[str, float]:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "replicates": float(self.replicates),
        }


def percentile_interval(estimate: float, draws: np.ndarray, *, level: float = 0.95) -> Interval:
    """Percentile interval over bootstrap draws, ignoring non-finite replicates."""

    values = np.asarray(draws, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Interval(estimate=estimate, lower=float("nan"), upper=float("nan"), replicates=0)
    tail = 0.5 * (1.0 - float(level))
    lower, upper = np.quantile(finite, [tail, 1.0 - tail])
    return Interval(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        replicates=int(finite.size),
    )


def cluster_indices(labels: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Draw clusters with replacement and return the row indices they cover."""

    keys = np.asarray(labels)
    groups: dict[object, list[int]] = {}
    for position, key in enumerate(keys.tolist()):
        groups.setdefault(key, []).append(position)
    members = [np.asarray(rows, dtype=np.int64) for rows in groups.values()]
    chosen = generator.integers(0, len(members), size=len(members))
    return np.concatenate([members[index] for index in chosen])


def bootstrap_interval(
    statistic: Statistic,
    labels: np.ndarray,
    *,
    replicates: int,
    seed: int,
    level: float = 0.95,
    rows: np.ndarray | None = None,
) -> Interval:
    """Bootstrapped percentile interval of ``statistic`` over the supplied rows."""

    matrix = np.arange(np.asarray(labels).size) if rows is None else np.asarray(rows)
    estimate = float(statistic(matrix))
    generator = np.random.default_rng(seed)
    draws = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        index = cluster_indices(np.asarray(labels)[matrix], generator)
        draws[replicate] = statistics_guard(statistic, matrix[index])
    return percentile_interval(estimate, draws, level=level)


def statistics_guard(statistic: Statistic, rows: np.ndarray) -> float:
    """Evaluate a statistic, mapping an empty or degenerate draw to a non-finite value."""

    if rows.size == 0:
        return float("nan")
    try:
        return float(statistic(rows))
    except (ValueError, ZeroDivisionError, IndexError):
        return float("nan")


def cluster_standard_error(values: np.ndarray, labels: np.ndarray) -> float:
    """Cluster-robust standard error of a mean, from the between-cluster spread."""

    sample = np.asarray(values, dtype=np.float64)
    keys = np.asarray(labels)
    unique = np.unique(keys)
    if unique.size < 2:
        return float("nan")
    totals = np.array([sample[keys == key].sum() for key in unique], dtype=np.float64)
    counts = np.array([float((keys == key).sum()) for key in unique], dtype=np.float64)
    grand = sample.sum()
    total_count = counts.sum()
    residual = totals - counts * (grand / total_count)
    variance = float((residual**2).sum()) * unique.size / (unique.size - 1)
    error: float = float(np.sqrt(variance)) / total_count
    return error


def cluster_size_inflation(labels: np.ndarray) -> float:
    """How much uneven cluster sizes inflate the variance over the mean cluster size.

    This is the size component of a design effect. A full design effect also carries the
    within-cluster correlation of the statistic being estimated, which the caller would have
    to supply alongside the cluster memberships.
    """

    keys = np.asarray(labels)
    unique = np.unique(keys)
    if unique.size == 0:
        return float("nan")
    counts = np.array([float((keys == key).sum()) for key in unique], dtype=np.float64)
    mean_size = counts.mean()
    if counts.sum() <= 0 or mean_size <= 0:
        return float("nan")
    return float((counts**2).sum() / (counts.sum() * mean_size))
