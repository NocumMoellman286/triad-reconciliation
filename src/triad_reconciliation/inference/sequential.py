"""Fixed-sequence testing over the declared hypotheses.

The five hypotheses are tested in the order the text states them, and a later one is only
tested when the one before it met its criterion, so no adjustment is applied to the p-values
themselves.

Ref: Sec. 5.6 (multiplicity controlled by a fixed-sequence method, five hypotheses, four
pre-specified minimum detectable differences), Sec. 2.1-2.4 (the criteria met in order).
"""

from __future__ import annotations

from dataclasses import dataclass

HYPOTHESIS_ORDER: tuple[str, ...] = (
    "discrimination_vs_best_single_stream",
    "discrimination_vs_static_fusion",
    "discrimination_vs_unconstrained_timeline",
    "arbitration_agreement_vs_strongest_proxy",
    "reader_accuracy_with_advice",
)


@dataclass(frozen=True)
class HypothesisOutcome:
    """One hypothesis beside the criterion it was tested against."""

    identifier: str
    effect: float
    threshold: float
    interval_lower: float
    interval_upper: float
    tested: bool
    met: bool
    reason: str


@dataclass(frozen=True)
class SequenceReport:
    """The fixed-sequence outcome for the whole family."""

    outcomes: tuple[HypothesisOutcome, ...]

    @property
    def tested(self) -> tuple[HypothesisOutcome, ...]:
        return tuple(item for item in self.outcomes if item.tested)

    @property
    def stopped_at(self) -> str | None:
        for item in self.outcomes:
            if item.tested and not item.met:
                return item.identifier
        return None

    def as_mapping(self) -> dict[str, object]:
        return {
            "order": [item.identifier for item in self.outcomes],
            "tested": [item.identifier for item in self.tested],
            "met": [item.identifier for item in self.outcomes if item.met],
            "stopped_at": self.stopped_at,
        }


def fixed_sequence(
    effects: dict[str, float],
    thresholds: dict[str, float],
    intervals: dict[str, tuple[float, float]],
) -> SequenceReport:
    """Walk the declared order, stopping at the first criterion that is not met."""

    outcomes: list[HypothesisOutcome] = []
    blocked = False
    for identifier in HYPOTHESIS_ORDER:
        if identifier not in effects:
            outcomes.append(
                HypothesisOutcome(
                    identifier=identifier,
                    effect=float("nan"),
                    threshold=float("nan"),
                    interval_lower=float("nan"),
                    interval_upper=float("nan"),
                    tested=False,
                    met=False,
                    reason="not supplied by this run",
                )
            )
            continue
        effect = float(effects[identifier])
        threshold = float(thresholds.get(identifier, float("nan")))
        lower, upper = intervals.get(identifier, (float("nan"), float("nan")))
        if blocked:
            outcomes.append(
                HypothesisOutcome(
                    identifier=identifier,
                    effect=effect,
                    threshold=threshold,
                    interval_lower=float(lower),
                    interval_upper=float(upper),
                    tested=False,
                    met=False,
                    reason="the preceding hypothesis did not meet its criterion",
                )
            )
            continue
        met = effect >= threshold
        outcomes.append(
            HypothesisOutcome(
                identifier=identifier,
                effect=effect,
                threshold=threshold,
                interval_lower=float(lower),
                interval_upper=float(upper),
                tested=True,
                met=bool(met),
                reason="criterion met" if met else "effect below the pre-specified margin",
            )
        )
        blocked = not met
    return SequenceReport(outcomes=tuple(outcomes))


def intervals_from_pairs(
    differences: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float]]:
    """Extract the interval edges from an estimate/lower/upper triple per hypothesis."""

    return {key: (value[1], value[2]) for key, value in differences.items()}


def effects_from_pairs(differences: dict[str, tuple[float, float, float]]) -> dict[str, float]:
    return {key: value[0] for key, value in differences.items()}
