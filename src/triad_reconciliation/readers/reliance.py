"""Reliance accounting over the reader study.

Over-reliance is the share of cases with wrong advice the reader followed, and under-reliance
the share of cases with correct advice the reader turned down. The two are counted on
different denominators, which is why the asymmetry they expose is readable.

Ref: Sec. 2.4 (the reliance decomposition and the pre-established limit on over-reliance),
Sec. 3.1 (rejecting correct advice more often than accepting wrong advice is the finding that
conditions the benefit), Table 4 (the reliance rows).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad_reconciliation.core.records import ReadershipResponse

ADVICE_CORRECT = "correct"
ADVICE_WRONG = "wrong"


@dataclass(frozen=True)
class RelianceReport:
    """The two reliance shares beside the counts they were taken over."""

    over_reliance: float
    under_reliance: float
    wrong_advice_cases: int
    correct_advice_cases: int
    adopted_wrong: int
    rejected_correct: int

    @property
    def asymmetry(self) -> float:
        return self.under_reliance - self.over_reliance

    def as_mapping(self) -> dict[str, float]:
        return {
            "over_reliance": self.over_reliance,
            "under_reliance": self.under_reliance,
            "wrong_advice_cases": float(self.wrong_advice_cases),
            "correct_advice_cases": float(self.correct_advice_cases),
            "adopted_wrong": float(self.adopted_wrong),
            "rejected_correct": float(self.rejected_correct),
            "asymmetry": self.asymmetry,
        }


def reliance_report(
    responses: tuple[ReadershipResponse, ...],
    reference: dict[str, str],
    advice: dict[str, str],
    *,
    arm: str = "model_assisted",
) -> RelianceReport:
    """Count adoption of wrong advice and rejection of correct advice."""

    adopted_wrong = 0
    wrong_cases = 0
    rejected_correct = 0
    correct_cases = 0
    for answer in responses:
        if answer.arm != arm:
            continue
        suggested = advice.get(answer.vignette_id)
        recorded = reference.get(answer.vignette_id)
        if suggested is None or recorded is None:
            continue
        if suggested == recorded:
            correct_cases += 1
            if answer.answered_direction != suggested:
                rejected_correct += 1
        else:
            wrong_cases += 1
            if answer.answered_direction == suggested:
                adopted_wrong += 1
    return RelianceReport(
        over_reliance=adopted_wrong / wrong_cases if wrong_cases else float("nan"),
        under_reliance=rejected_correct / correct_cases if correct_cases else float("nan"),
        wrong_advice_cases=wrong_cases,
        correct_advice_cases=correct_cases,
        adopted_wrong=adopted_wrong,
        rejected_correct=rejected_correct,
    )


def within_limit(report: RelianceReport, limit: float) -> dict[str, bool]:
    """Whether the point estimate and the interval stay under the pre-established limit."""

    return {
        "point_estimate": bool(report.over_reliance <= float(limit)),
        "under_limit_by": bool(report.over_reliance < float(limit)),
    }


def reader_strata(
    responses: tuple[ReadershipResponse, ...],
    reference: dict[str, str],
    *,
    arm: str,
    cut: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Split readers at a score index into the lower and upper halves of the study."""

    scored: list[tuple[str, float]] = []
    for reader in sorted({answer.reader_id for answer in responses if answer.arm == arm}):
        answers = [
            answer for answer in responses if answer.arm == arm and answer.reader_id == reader
        ]
        if not answers:
            continue
        correct = sum(
            1 for answer in answers if answer.answered_direction == reference[answer.vignette_id]
        )
        scored.append((reader, correct / len(answers)))
    scored.sort(key=lambda item: item[1])
    lower = np.asarray([name for name, _ in scored[:cut]], dtype=object)
    upper = np.asarray([name for name, _ in scored[cut:]], dtype=object)
    return lower, upper


def arm_gain(
    responses: tuple[ReadershipResponse, ...],
    reference: dict[str, str],
    readers: np.ndarray,
    *,
    baseline_arm: str = "unassisted",
    assisted_arm: str = "model_assisted",
) -> float:
    """Change in accuracy for a named set of readers between two arms."""

    selection = set(readers.tolist())
    by_arm: dict[str, list[int]] = {baseline_arm: [], assisted_arm: []}
    for answer in responses:
        if answer.reader_id not in selection:
            continue
        if answer.arm not in by_arm:
            continue
        by_arm[answer.arm].append(int(answer.answered_direction == reference[answer.vignette_id]))
    if not by_arm[baseline_arm] or not by_arm[assisted_arm]:
        return float("nan")
    return float(np.mean(by_arm[assisted_arm]) - np.mean(by_arm[baseline_arm]))
