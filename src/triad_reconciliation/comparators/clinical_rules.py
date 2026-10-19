"""Clinical rules.

Three comparators decide progression the way the clinical standard does: an event-based
cluster rule over the sensitivity map, and two trend rules fitted per eye.

Ref: Sec. 2.1 (event- and trend-guided progression analysis, the visual field index trend and
the mean deviation trend), Table 1 rows 1-3, Table 7 rows 1-3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad_reconciliation.analysis.batching import FlatFeatures

CLUSTER_DROP_DB = 5.0
CLUSTER_POINT_COUNT = 4
VFI_SLOPE_LIMIT = -1.0
MD_SLOPE_LIMIT = -0.5
MD_INDEX = 0
VFI_INDEX = 2


@dataclass(frozen=True)
class RuleResult:
    """A clinical rule's graded score per pair, higher meaning more likely progression."""

    name: str
    score: np.ndarray


def _ordered(features: FlatFeatures, column: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(np.where(features.field_mask, features.field_times, np.inf), axis=1)
    values = np.take_along_axis(features.field_indices[:, :, column], order, axis=1)
    stamps = np.take_along_axis(features.field_times, order, axis=1)
    mask = np.take_along_axis(features.field_mask, order, axis=1)
    return values, stamps, mask


def per_eye_slope(values: np.ndarray, stamps: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Least-squares slope of one index against time in years, per pair."""

    n_pairs, length = values.shape
    counts = mask.sum(axis=1)
    mean_time = np.where(
        counts > 0, np.where(mask, stamps, 0.0).sum(axis=1) / np.clip(counts, 1, None), 0.0
    )
    mean_value = np.where(
        counts > 0, np.where(mask, values, 0.0).sum(axis=1) / np.clip(counts, 1, None), 0.0
    )
    deviation_time = np.where(mask, stamps - mean_time[:, None], 0.0)
    deviation_value = np.where(mask, values - mean_value[:, None], 0.0)
    denominator = (deviation_time**2).sum(axis=1)
    numerator = (deviation_time * deviation_value).sum(axis=1)
    slope = np.divide(numerator, denominator, out=np.zeros(n_pairs), where=denominator > 1e-9)
    return np.where(counts >= 2, slope, 0.0)


def guided_progression_analysis(features: FlatFeatures) -> RuleResult:
    """Event-based cluster rule: test points whose sensitivity fell by the cluster drop."""

    order = np.argsort(np.where(features.field_mask, features.field_times, np.inf), axis=1)
    sensitivity = np.take_along_axis(features.sensitivity, order[:, :, None], axis=1)
    mask = np.take_along_axis(features.field_mask, order, axis=1)
    counts = mask.sum(axis=1)
    rows = np.arange(features.n_pairs)
    baseline = sensitivity[:, 0]
    latest = sensitivity[rows, np.clip(counts - 1, 0, sensitivity.shape[1] - 1)]
    dropped = baseline - latest
    score = np.where(
        counts[:, None] >= 2, (dropped >= CLUSTER_DROP_DB).astype(np.float64), 0.0
    ).sum(axis=1)
    return RuleResult(name="guided_progression_analysis", score=score)


def visual_field_index_trend(features: FlatFeatures) -> RuleResult:
    """Trend of the visual field index, scored as minus its fitted slope."""

    values, stamps, mask = _ordered(features, VFI_INDEX)
    return RuleResult(name="visual_field_index_trend", score=-per_eye_slope(values, stamps, mask))


def mean_deviation_trend(features: FlatFeatures) -> RuleResult:
    """Trend of the mean deviation, scored as minus its fitted slope."""

    values, stamps, mask = _ordered(features, MD_INDEX)
    return RuleResult(name="mean_deviation_trend", score=-per_eye_slope(values, stamps, mask))


def rule_thresholds() -> dict[str, float]:
    """The pre-specified cut each rule applies to its own score."""

    return {
        "guided_progression_analysis": float(CLUSTER_POINT_COUNT),
        "visual_field_index_trend": float(-VFI_SLOPE_LIMIT),
        "mean_deviation_trend": float(-MD_SLOPE_LIMIT),
    }


def binary_decisions(result: RuleResult) -> np.ndarray:
    return (result.score >= rule_thresholds()[result.name]).astype(np.int64)


def all_rules(features: FlatFeatures) -> tuple[RuleResult, ...]:
    return (
        guided_progression_analysis(features),
        visual_field_index_trend(features),
        mean_deviation_trend(features),
    )
