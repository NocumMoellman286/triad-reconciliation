"""Discrimination metrics.

Rank statistics are computed directly so an interval resamples the same numbers the tables
print, without a second library entering the reported path.

Ref: Sec. 5.6 (area under the receiver-operating-characteristic curve, area under the
precision-recall curve, sensitivity at a fixed specificity).
"""

from __future__ import annotations

import numpy as np


def _average_ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    position = 0
    while position < order.size:
        end = position
        while end + 1 < order.size and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        mean_rank = 0.5 * (position + end) + 1.0
        ranks[order[position : end + 1]] = mean_rank
        position = end + 1
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the receiver-operating-characteristic curve via the rank identity."""

    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    positive = targets == 1
    n_positive = int(positive.sum())
    n_negative = int((~positive).sum())
    if n_positive == 0 or n_negative == 0:
        return float("nan")
    ranks = _average_ranks(values)
    statistic = ranks[positive].sum() - n_positive * (n_positive + 1) / 2.0
    return float(statistic / (n_positive * n_negative))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve as a step-weighted mean precision."""

    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if targets.sum() == 0:
        return float("nan")
    order = np.argsort(-values, kind="mergesort")
    sorted_targets = targets[order]
    cumulative = np.cumsum(sorted_targets)
    positions = np.arange(1, sorted_targets.size + 1, dtype=np.float64)
    precision = cumulative / positions
    increments = np.diff(np.concatenate(([0.0], cumulative))) > 0
    return float(precision[increments].sum() / cumulative[-1])


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cut points with their sensitivity and specificity, in descending score order."""

    values = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    if values.size == 0:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    order = np.argsort(-values, kind="mergesort")
    sorted_values = values[order]
    sorted_targets = targets[order]
    positive_total = max(int(sorted_targets.sum()), 1)
    negative_total = max(int((sorted_targets == 0).sum()), 1)
    true_positive = np.cumsum(sorted_targets)
    false_positive = np.cumsum(1 - sorted_targets)
    changes = np.flatnonzero(sorted_values[:-1] != sorted_values[1:])
    last_of_run = np.concatenate((changes, np.asarray([sorted_values.size - 1], dtype=np.int64)))
    thresholds = sorted_values[last_of_run]
    sensitivity = true_positive[last_of_run] / positive_total
    specificity = 1.0 - false_positive[last_of_run] / negative_total
    return thresholds, sensitivity, specificity


def sensitivity_at_specificity(
    scores: np.ndarray, labels: np.ndarray, target_specificity: float
) -> float:
    """The highest sensitivity whose specificity is at least the target."""

    thresholds, sensitivity, specificity = roc_curve(scores, labels)
    admissible = specificity >= float(target_specificity) - 1e-12
    if not bool(admissible.any()):
        return float("nan")
    return float(sensitivity[admissible].max())


def partial_auc(scores: np.ndarray, labels: np.ndarray, lower: float, upper: float) -> float:
    """Trapezoidal area under the receiver-operating-characteristic curve over a band."""

    _, sensitivity, specificity = roc_curve(scores, labels)
    false_positive = 1.0 - specificity
    order = np.argsort(false_positive)
    x_values = false_positive[order]
    y_values = sensitivity[order]
    keep = (x_values >= lower) & (x_values <= upper)
    if int(keep.sum()) < 2:
        return float("nan")
    return float(np.trapezoid(y_values[keep], x_values[keep]) / (upper - lower))
