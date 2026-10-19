"""Workflow outcomes.

The progression probability is turned into the two decisions a review actually makes - to
escalate treatment and to set the next interval - and each is compared with what the record
says was done. Lead time, reclassification and net benefit follow from the same probabilities.

Ref: Table 5 (sensitivity at matched specificity, treatment-escalation agreement,
review-interval agreement, median lead time, net reclassification improvement, net benefit at
a threshold probability), Sec. 5.7 (deployment-population comparison).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad_reconciliation.metrics.discrimination import sensitivity_at_specificity

ESCALATE = "escalate"
CONTINUE = "continue"


@dataclass(frozen=True)
class WorkflowOutcome:
    """The workflow-level numbers over one pair set."""

    sensitivity_matched: float
    escalation_agreement: float
    interval_agreement: float
    median_lead_time_months: float
    net_reclassification_improvement: float
    net_benefit: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "sensitivity_matched": self.sensitivity_matched,
            "escalation_agreement": self.escalation_agreement,
            "interval_agreement": self.interval_agreement,
            "median_lead_time_months": self.median_lead_time_months,
            "net_reclassification_improvement": self.net_reclassification_improvement,
            "net_benefit": self.net_benefit,
        }


def implied_escalation(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Escalate when the progression probability clears the decision threshold."""

    return np.asarray(probabilities, dtype=np.float64) >= float(threshold)


def implied_interval_months(probabilities: np.ndarray, bands: tuple[float, float]) -> np.ndarray:
    """Review interval in months: the shortest band above the probability."""

    values = np.asarray(probabilities, dtype=np.float64)
    low, high = float(bands[0]), float(bands[1])
    base = np.full(values.shape, 12.0)
    shortened = np.where(values >= low, 6.0, base)
    return np.where(values >= high, 3.0, shortened)


def agreement_share(predicted: np.ndarray, recorded: np.ndarray) -> float:
    left = np.asarray(predicted)
    right = np.asarray(recorded)
    if left.shape != right.shape:
        raise ValueError("the two decision vectors must share a shape")
    if left.size == 0:
        return float("nan")
    return float(np.mean(left == right))


def median_lead_time_months(detection_months: np.ndarray, reference_months: np.ndarray) -> float:
    """Median months by which a decision precedes the reference detection."""

    detected = np.asarray(detection_months, dtype=np.float64)
    reference = np.asarray(reference_months, dtype=np.float64)
    usable = np.isfinite(detected) & np.isfinite(reference)
    if not bool(usable.any()):
        return float("nan")
    return float(np.median(reference[usable] - detected[usable]))


def net_benefit(probabilities: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Decision-curve net benefit at a threshold probability."""

    values = np.asarray(probabilities, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    total = values.size
    if total == 0:
        return float("nan")
    positive = values >= float(threshold)
    true_positive = float(((positive) & (targets == 1)).sum()) / total
    false_positive = float(((positive) & (targets == 0)).sum()) / total
    odds = float(threshold) / max(1.0 - float(threshold), 1e-9)
    return true_positive - false_positive * odds


def reclassification_counts(
    baseline: np.ndarray, model: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Movement between decision bands, split by the true label."""

    base = np.asarray(baseline)
    updated = np.asarray(model)
    targets = np.asarray(labels, dtype=np.float64)
    events = targets == 1
    non_events = ~events
    if not bool(events.any()) or not bool(non_events.any()):
        return {
            "up_events": 0.0,
            "down_events": 0.0,
            "up_non_events": 0.0,
            "down_non_events": 0.0,
            "nri": float("nan"),
        }
    up_events = float(((updated > base) & events).sum()) / float(events.sum())
    down_events = float(((updated < base) & events).sum()) / float(events.sum())
    up_non_events = float(((updated > base) & non_events).sum()) / float(non_events.sum())
    down_non_events = float(((updated < base) & non_events).sum()) / float(non_events.sum())
    return {
        "up_events": up_events,
        "down_events": down_events,
        "up_non_events": up_non_events,
        "down_non_events": down_non_events,
        "nri": (up_events - down_events) - (up_non_events - down_non_events),
    }


def workflow_outcome(
    probabilities: np.ndarray,
    labels: np.ndarray,
    recorded_escalation: np.ndarray,
    recorded_interval: np.ndarray,
    detection_months: np.ndarray,
    reference_months: np.ndarray,
    baseline_probabilities: np.ndarray,
    *,
    matched_specificity: float,
    decision_threshold: float,
    interval_bands: tuple[float, float],
) -> WorkflowOutcome:
    """Assemble the workflow row of the clinical-outcomes table."""

    escalation = agreement_share(
        implied_escalation(probabilities, decision_threshold), recorded_escalation
    )
    interval = agreement_share(
        implied_interval_months(probabilities, interval_bands), recorded_interval
    )
    counts = reclassification_counts(
        implied_escalation(baseline_probabilities, decision_threshold),
        implied_escalation(probabilities, decision_threshold),
        labels,
    )
    return WorkflowOutcome(
        sensitivity_matched=sensitivity_at_specificity(probabilities, labels, matched_specificity),
        escalation_agreement=escalation,
        interval_agreement=interval,
        median_lead_time_months=median_lead_time_months(detection_months, reference_months),
        net_reclassification_improvement=counts["nri"],
        net_benefit=net_benefit(probabilities, labels, decision_threshold),
    )
