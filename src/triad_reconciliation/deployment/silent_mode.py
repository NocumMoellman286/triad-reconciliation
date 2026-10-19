"""Silent-mode deployment accounting.

The silent run scores every eligible encounter, emits an attribution with every scored
decision, and hands the deferred ones back to a reviewer. That run is a log; the release
computes its rates from whatever log the caller supplies rather than carrying one.

Ref: Sec. 5.7 (consecutive accrual and the coverage reported beside discrimination),
Table 6 (the silent-mode metrics).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MISSING_STREAM = "missing_stream"
PROCESSING_FAILURE = "processing_failure"


@dataclass(frozen=True)
class EncounterRecord:
    """One encounter as the silent run saw it."""

    encounter_id: str
    scored: bool
    failure_reason: str | None
    latency_seconds: float
    deferred: bool
    attribution_emitted: bool
    attribution_consistent: bool
    opened_by_reviewer: bool
    overrode_direction: bool
    deferred_returned: bool
    deferred_accepted: bool


@dataclass(frozen=True)
class SilentModeMetrics:
    """The rates Table 6 prints, over the log they were taken from."""

    eligible: int
    scored: int
    coverage: float
    unscored_missing_share: float
    unscored_processing_share: float
    median_latency_seconds: float
    latency_iqr: tuple[float, float]
    deferred_share: float
    attribution_share: float
    engagement_share: float
    override_share: float
    deferred_acceptance_share: float
    interpretability_share: float

    def as_mapping(self) -> dict[str, float]:
        return {
            "eligible": float(self.eligible),
            "scored": float(self.scored),
            "coverage": self.coverage,
            "unscored_missing_share": self.unscored_missing_share,
            "unscored_processing_share": self.unscored_processing_share,
            "median_latency_seconds": self.median_latency_seconds,
            "latency_iqr_lower": self.latency_iqr[0],
            "latency_iqr_upper": self.latency_iqr[1],
            "deferred_share": self.deferred_share,
            "attribution_share": self.attribution_share,
            "engagement_share": self.engagement_share,
            "override_share": self.override_share,
            "deferred_acceptance_share": self.deferred_acceptance_share,
            "interpretability_share": self.interpretability_share,
        }


def _share(count: int, total: int) -> float:
    return float(count) / float(total) if total else float("nan")


def silent_mode_metrics(records: tuple[EncounterRecord, ...]) -> SilentModeMetrics:
    """Fold a silent-run log into the reported rates."""

    eligible = len(records)
    scored_records = [record for record in records if record.scored]
    scored = len(scored_records)
    missing = sum(
        1 for record in records if not record.scored and record.failure_reason == MISSING_STREAM
    )
    processing = sum(
        1 for record in records if not record.scored and record.failure_reason == PROCESSING_FAILURE
    )
    latencies = np.asarray([record.latency_seconds for record in scored_records], dtype=np.float64)
    if latencies.size:
        lower, upper = np.quantile(latencies, [0.25, 0.75])
        median = float(np.median(latencies))
        iqr = (float(lower), float(upper))
    else:
        median, iqr = float("nan"), (float("nan"), float("nan"))
    deferred = sum(1 for record in scored_records if record.deferred)
    attributed = sum(1 for record in scored_records if record.attribution_emitted)
    consistent = sum(
        1
        for record in scored_records
        if record.attribution_emitted and record.attribution_consistent
    )
    opened = sum(1 for record in scored_records if record.opened_by_reviewer)
    overrode = sum(1 for record in scored_records if record.overrode_direction)
    returned = sum(1 for record in scored_records if record.deferred and record.deferred_returned)
    accepted = sum(
        1
        for record in scored_records
        if record.deferred and record.deferred_returned and record.deferred_accepted
    )
    return SilentModeMetrics(
        eligible=eligible,
        scored=scored,
        coverage=_share(scored, eligible),
        unscored_missing_share=_share(missing, eligible),
        unscored_processing_share=_share(processing, eligible),
        median_latency_seconds=median,
        latency_iqr=iqr,
        deferred_share=_share(deferred, scored),
        attribution_share=_share(attributed, scored),
        engagement_share=_share(opened, scored),
        override_share=_share(overrode, scored),
        deferred_acceptance_share=_share(accepted, returned),
        interpretability_share=_share(consistent, attributed),
    )


def deployment_log_summary(records: tuple[EncounterRecord, ...]) -> dict[str, float]:
    """Coverage beside the counts a reader needs to interpret it."""

    metrics = silent_mode_metrics(records)
    return {
        "eligible": float(metrics.eligible),
        "scored": float(metrics.scored),
        "coverage": metrics.coverage,
        "deferred": metrics.deferred_share,
    }
