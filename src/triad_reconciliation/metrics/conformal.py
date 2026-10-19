"""Conformal calibration and the deferral rule built on it.

A split-conformal quantile turns the progressor score into a set of directions, and a pair
whose set still holds more than one direction is deferred rather than decided.

Ref: Sec. 5.5 (the calibrated probability and the arbitral direction the reconciliation
emits), Sec. 5.6 (calibration is reported beside discrimination), Table 2 (the calibrated
deferral row and its coverage guarantee), Table 6 (deferred decisions).
"""

from __future__ import annotations

import numpy as np


def conformity_scores(probabilities: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Nonconformity of the reported class on a calibration split."""

    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.int64)
    return 1.0 - values[np.arange(values.shape[0]), targets]


def conformal_quantile(scores: np.ndarray, nominal_coverage: float) -> float:
    """The finite-sample quantile at the ceil((n+1) * coverage) / n order statistic."""

    values = np.sort(np.asarray(scores, dtype=np.float64))
    if values.size == 0:
        return 1.0
    rank = int(np.ceil((values.size + 1) * float(nominal_coverage)))
    rank = min(max(rank, 1), values.size)
    return float(values[rank - 1])


def direction_sets(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean membership of each direction in the conformal set."""

    membership: np.ndarray = (1.0 - np.asarray(probabilities, dtype=np.float64)) <= threshold
    return membership


def deferral_mask(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Pairs whose conformal set is not a singleton."""

    membership = direction_sets(probabilities, threshold)
    deferred: np.ndarray = membership.sum(axis=1) != 1
    return deferred


def decided_accuracy(
    probabilities: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float]:
    """Accuracy on the scored subset beside the share that was scored."""

    membership = direction_sets(probabilities, threshold)
    deferred = membership.sum(axis=1) != 1
    scored = ~deferred
    if not bool(scored.any()):
        return float("nan"), 0.0
    chosen = np.argmax(probabilities, axis=1)
    correct = (chosen == np.asarray(labels, dtype=np.int64))[scored]
    return float(correct.mean()), float(scored.mean())


def empirical_coverage(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Share of pairs whose true direction stayed inside the emitted set."""

    membership = direction_sets(probabilities, threshold)
    targets = np.asarray(labels, dtype=np.int64)
    if targets.size == 0:
        return float("nan")
    return float(np.mean(membership[np.arange(targets.size), targets]))


def decide_threshold(
    probabilities: np.ndarray, labels: np.ndarray, nominal_coverage: float
) -> float:
    """Conformal threshold fitted on the supplied split.

    Callers pass the calibration split here and the evaluation split to the coverage and
    deferral helpers, so the guarantee is stated on units the threshold never saw.
    """

    return conformal_quantile(conformity_scores(probabilities, labels), nominal_coverage)


def deferral_analysis(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    evaluation_probabilities: np.ndarray,
    evaluation_labels: np.ndarray,
    nominal_coverage: float,
) -> dict[str, float]:
    """Coverage, deferral share and decided accuracy for the calibrated deferral rule."""

    threshold = decide_threshold(calibration_probabilities, calibration_labels, nominal_coverage)
    accuracy, scored = decided_accuracy(evaluation_probabilities, evaluation_labels, threshold)
    return {
        "threshold": threshold,
        "coverage": empirical_coverage(evaluation_probabilities, evaluation_labels, threshold),
        "deferral_share": float(1.0 - scored),
        "decided_accuracy": accuracy,
        "nominal_coverage": float(nominal_coverage),
    }


def unguarded_analysis(
    evaluation_probabilities: np.ndarray, evaluation_labels: np.ndarray
) -> dict[str, float]:
    """The variant that decides every pair and keeps neither the set nor the guarantee."""

    values = np.asarray(evaluation_probabilities, dtype=np.float64)
    targets = np.asarray(evaluation_labels, dtype=np.int64)
    if targets.size == 0:
        return {"deferral_share": 0.0, "decided_accuracy": float("nan")}
    chosen = np.argmax(values, axis=1)
    return {
        "deferral_share": 0.0,
        "decided_accuracy": float(np.mean(chosen == targets)),
    }
