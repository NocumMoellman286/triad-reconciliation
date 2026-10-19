"""Agreement metrics.

Ref: Sec. 5.1 (the recorded arbitration direction), Sec. 5.6 (Cohen's kappa with a
cluster-bootstrap interval), Table 3 (agreement against the recorded reference arbitration).
"""

from __future__ import annotations

import numpy as np

DIRECTION_CATEGORIES: tuple[str, ...] = ("structural", "functional")
ARBITRATION_CATEGORIES: tuple[str, ...] = ("structural", "functional", "concordant")


def confusion(
    labels_a: np.ndarray, labels_b: np.ndarray, categories: tuple[str, ...]
) -> np.ndarray:
    """Contingency table of two nominal labellings, in the declared category order."""

    position = {category: index for index, category in enumerate(categories)}
    table = np.zeros((len(categories), len(categories)), dtype=np.float64)
    for first, second in zip(labels_a, labels_b, strict=True):
        table[position[str(first)], position[str(second)]] += 1.0
    return table


def cohen_kappa(labels_a: np.ndarray, labels_b: np.ndarray, categories: tuple[str, ...]) -> float:
    """Chance-corrected agreement between two nominal labellings."""

    table = confusion(labels_a, labels_b, categories)
    total = table.sum()
    if total == 0:
        return float("nan")
    observed = float(np.trace(table)) / total
    rows = table.sum(axis=1) / total
    columns = table.sum(axis=0) / total
    expected = float(np.dot(rows, columns))
    if abs(1.0 - expected) < 1e-12:
        return 0.0
    kappa: float = (observed - expected) / (1.0 - expected)
    return kappa


def agreement_share(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    first = np.asarray(labels_a, dtype=object)
    second = np.asarray(labels_b, dtype=object)
    if first.size == 0:
        return float("nan")
    return float(np.mean([str(a) == str(b) for a, b in zip(first, second, strict=True)]))


def mean_pairwise_kappa(ratings: np.ndarray, categories: tuple[str, ...]) -> float:
    """Average Cohen's kappa over every reader pair in a reader-by-vignette matrix."""

    n_readers, n_items = ratings.shape
    if n_readers < 2:
        return float("nan")
    values: list[float] = []
    for first in range(n_readers):
        for second in range(first + 1, n_readers):
            left = ratings[first]
            right = ratings[second]
            keep = np.asarray(
                [
                    index
                    for index in range(n_items)
                    if left[index] is not None and right[index] is not None
                ],
                dtype=np.int64,
            )
            if keep.size < 2:
                continue
            values.append(cohen_kappa(left[keep], right[keep], categories))
    if not values:
        return float("nan")
    return float(np.mean(values))


def weighted_kappa(
    labels_a: np.ndarray, labels_b: np.ndarray, categories: tuple[str, ...], weights: np.ndarray
) -> float:
    """Weighted chance-corrected agreement, used where category order carries meaning."""

    table = confusion(labels_a, labels_b, categories)
    total = table.sum()
    if total == 0:
        return float("nan")
    observed = float((table * weights).sum()) / total
    rows = table.sum(axis=1) / total
    columns = table.sum(axis=0) / total
    expected = float(rows @ weights @ columns)
    if abs(1.0 - expected) < 1e-12:
        return 0.0
    kappa: float = (observed - expected) / (1.0 - expected)
    return kappa


def linear_weights(categories: tuple[str, ...]) -> np.ndarray:
    size = len(categories)
    distance = np.abs(np.subtract.outer(np.arange(size), np.arange(size))).astype(np.float64)
    return 1.0 - distance / max(size - 1, 1)
