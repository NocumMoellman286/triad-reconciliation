"""Structure-function regression comparators.

Both variants estimate the functional change from the structural change, so the ranking
signal is the size of the functional drop the optic disc implies. One fits a linear map, the
other a Nadaraya-Watson kernel smoother.

Ref: Sec. 2.1 (linear structure-function mapping and the kernel-regression pair),
Table 1 rows 4-5, Table 7 rows 4-5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad_reconciliation.analysis.batching import FlatFeatures

STRUCTURAL_SUMMARY_WIDTH = 8


@dataclass(frozen=True)
class RegressionResult:
    """Per-pair predicted functional change and the score derived from it."""

    name: str
    predicted_change: np.ndarray

    @property
    def score(self) -> np.ndarray:
        return -self.predicted_change


def structural_change(features: FlatFeatures) -> np.ndarray:
    """Difference between the latest and the earliest structural summary."""

    times = np.where(features.structural_mask, features.structural_times, np.inf)
    order = np.argsort(times, axis=1)
    ordered = np.take_along_axis(features.morphometry, order[:, :, None], axis=1)
    mask = np.take_along_axis(features.structural_mask, order, axis=1)
    counts = mask.sum(axis=1)
    rows = np.arange(features.n_pairs)
    earliest = ordered[:, 0]
    latest = ordered[rows, np.clip(counts - 1, 0, ordered.shape[1] - 1)]
    change = latest - earliest
    return np.where((counts >= 2)[:, None], change, 0.0)


def functional_change(features: FlatFeatures, column: int = 2) -> np.ndarray:
    """Difference between the latest and the earliest value of a perimetry index."""

    times = np.where(features.field_mask, features.field_times, np.inf)
    order = np.argsort(times, axis=1)
    ordered = np.take_along_axis(features.field_indices[:, :, column], order, axis=1)
    mask = np.take_along_axis(features.field_mask, order, axis=1)
    counts = mask.sum(axis=1)
    rows = np.arange(features.n_pairs)
    earliest = ordered[:, 0]
    latest = ordered[rows, np.clip(counts - 1, 0, ordered.shape[1] - 1)]
    change = latest - earliest
    return np.where(counts >= 2, change, 0.0)


def fit_linear(design: np.ndarray, target: np.ndarray, ridge: float = 1e-3) -> np.ndarray:
    """Ridge-regularised least squares with an intercept column."""

    augmented = np.concatenate((np.ones((design.shape[0], 1)), design), axis=1)
    gram = augmented.T @ augmented + float(ridge) * np.eye(augmented.shape[1])
    return np.linalg.solve(gram, augmented.T @ target)


def linear_structure_function(
    development: FlatFeatures, held_out: FlatFeatures
) -> RegressionResult:
    """Linear map from the structural change to the change in the field index."""

    weights = fit_linear(structural_change(development), functional_change(development))
    design = np.concatenate((np.ones((held_out.n_pairs, 1)), structural_change(held_out)), axis=1)
    return RegressionResult(name="linear_structure_function", predicted_change=design @ weights)


def median_bandwidth(points: np.ndarray) -> float:
    """Silverman-style bandwidth from the median pairwise distance."""

    if points.shape[0] < 2:
        return 1.0
    paired: np.ndarray = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    upper = paired[np.triu_indices(points.shape[0], k=1)]
    scale = float(np.median(upper)) if upper.size else 1.0
    bandwidth: float = max(scale * points.shape[0] ** (-1.0 / (points.shape[1] + 4)), 1e-3)
    return bandwidth


def kernel_structure_function(
    development: FlatFeatures, held_out: FlatFeatures, bandwidth: float | None = None
) -> RegressionResult:
    """Nadaraya-Watson smoother from the structural change to the functional change."""

    train_inputs = structural_change(development)
    train_targets = functional_change(development)
    test_inputs = structural_change(held_out)
    scale = median_bandwidth(train_inputs) if bandwidth is None else float(bandwidth)
    distances = np.linalg.norm(test_inputs[:, None, :] - train_inputs[None, :, :], axis=-1)
    weights = np.exp(-0.5 * (distances / scale) ** 2)
    totals = weights.sum(axis=1, keepdims=True)
    normalised = np.divide(weights, np.clip(totals, 1e-12, None))
    predicted = normalised @ train_targets
    fallback = float(np.mean(train_targets)) if train_targets.size else 0.0
    return RegressionResult(
        name="kernel_structure_function",
        predicted_change=np.where(totals[:, 0] > 1e-12, predicted, fallback),
    )
