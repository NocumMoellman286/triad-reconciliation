"""Calibration metrics.

Ref: Sec. 5.6 (expected calibration error, Brier score, calibration slope and intercept).
"""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, bins: int = 15
) -> float:
    """Binned gap between mean predicted probability and observed frequency."""

    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(values, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    for bucket in range(bins):
        selected = index == bucket
        count = int(selected.sum())
        if count == 0:
            continue
        gap = abs(float(values[selected].mean()) - float(targets[selected].mean()))
        total += gap * count
    return total / values.size


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    return float(np.mean((values - targets) ** 2))


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def calibration_slope_intercept(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    """Slope and intercept from a logistic recalibration of the reported probabilities."""

    design = np.stack(
        (np.ones_like(np.asarray(probabilities, dtype=np.float64)), _logit(probabilities)), axis=1
    )
    targets = np.asarray(labels, dtype=np.float64)
    coefficients = np.zeros(2, dtype=np.float64)
    for _ in range(120):
        linear = design @ coefficients
        expected = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(expected * (1.0 - expected), 1e-9, None)
        gradient = design.T @ (targets - expected)
        hessian = design.T @ (design * weights[:, None])
        step = np.linalg.solve(hessian + 1e-8 * np.eye(2), gradient)
        coefficients = coefficients + step
        if float(np.abs(step).max()) < 1e-10:
            break
    return float(coefficients[1]), float(coefficients[0])


def reliability_curve(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> np.ndarray:
    """Per-bin (lower edge, mean probability, observed frequency, count)."""

    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(values, edges[1:-1], right=False), 0, bins - 1)
    rows: list[list[float]] = []
    for bucket in range(bins):
        selected = index == bucket
        count = int(selected.sum())
        if count == 0:
            rows.append([float(edges[bucket]), float("nan"), float("nan"), 0.0])
            continue
        rows.append(
            [
                float(edges[bucket]),
                float(values[selected].mean()),
                float(targets[selected].mean()),
                float(count),
            ]
        )
    return np.asarray(rows, dtype=np.float64)
