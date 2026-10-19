"""Stratum construction.

Ref: Sec. 2.2 (the discordant stratum carries the advantage), Table 5 (the reported subgroup
rows: disease stage, perimetry reliability, high myopia, age, sex and the recorded ancestry
groups), Sec. 5.6 (subgroup estimates use the same resampling procedure).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad_reconciliation.core.records import VisitPair


@dataclass(frozen=True)
class Stratum:
    """A named, declared partition of the held-out pairs."""

    name: str
    level: str
    members: tuple[VisitPair, ...]

    @property
    def n_pairs(self) -> int:
        return len(self.members)

    @property
    def n_patients(self) -> int:
        return len({pair.patient_id for pair in self.members})


def discordance_strata(pairs: tuple[VisitPair, ...]) -> tuple[Stratum, Stratum]:
    """The concordant and discordant strata the decomposition isolates."""

    discordant = tuple(pair for pair in pairs if pair.discordant)
    concordant = tuple(pair for pair in pairs if not pair.discordant)
    return (
        Stratum(name="agreement_state", level="discordant", members=discordant),
        Stratum(name="agreement_state", level="concordant", members=concordant),
    )


def stage_strata(pairs: tuple[VisitPair, ...]) -> tuple[Stratum, ...]:
    return tuple(
        Stratum(
            name="stage", level=level, members=tuple(pair for pair in pairs if pair.stage == level)
        )
        for level in ("early", "moderate", "advanced")
    )


def reliability_strata(pairs: tuple[VisitPair, ...]) -> tuple[Stratum, Stratum]:
    reliable = tuple(pair for pair in pairs if pair.functional_reliable)
    unreliable = tuple(pair for pair in pairs if not pair.functional_reliable)
    return (
        Stratum(name="perimetry_reliability", level="reliable", members=reliable),
        Stratum(name="perimetry_reliability", level="unreliable", members=unreliable),
    )


def myopia_strata(pairs: tuple[VisitPair, ...]) -> tuple[Stratum, Stratum]:
    myopic = tuple(pair for pair in pairs if pair.high_myopia)
    non_myopic = tuple(pair for pair in pairs if not pair.high_myopia)
    return (
        Stratum(name="myopia", level="high", members=myopic),
        Stratum(name="myopia", level="non_myopic", members=non_myopic),
    )


def age_strata(pairs: tuple[VisitPair, ...], cut: float = 65.0) -> tuple[Stratum, Stratum]:
    below = tuple(pair for pair in pairs if pair.age_years < cut)
    at_or_above = tuple(pair for pair in pairs if pair.age_years >= cut)
    return (
        Stratum(name="age", level=f"below_{int(cut)}", members=below),
        Stratum(name="age", level=f"at_or_above_{int(cut)}", members=at_or_above),
    )


def sex_strata(pairs: tuple[VisitPair, ...]) -> tuple[Stratum, Stratum]:
    female = tuple(pair for pair in pairs if pair.sex == "female")
    male = tuple(pair for pair in pairs if pair.sex == "male")
    return (
        Stratum(name="sex", level="female", members=female),
        Stratum(name="sex", level="male", members=male),
    )


def ancestry_strata(pairs: tuple[VisitPair, ...], groups: int = 3) -> tuple[Stratum, ...]:
    """The recorded ancestry groups, banded into the number of levels the tables print."""

    if groups < 2:
        raise ValueError("a banding needs at least two levels")
    bands = np.minimum((np.asarray([pair.ancestry_group for pair in pairs]) - 1), groups - 1)
    return tuple(
        Stratum(
            name="ancestry",
            level=f"group_{level + 1}",
            members=tuple(
                pair for pair, band in zip(pairs, bands.tolist(), strict=True) if band == level
            ),
        )
        for level in range(groups)
    )


def all_strata(pairs: tuple[VisitPair, ...]) -> tuple[Stratum, ...]:
    """Every declared stratum family in the order the tables print them."""

    return (
        *discordance_strata(pairs),
        *stage_strata(pairs),
        *reliability_strata(pairs),
        *myopia_strata(pairs),
        *age_strata(pairs),
        *sex_strata(pairs),
        *ancestry_strata(pairs),
    )


def stratum_sizes(pairs: tuple[VisitPair, ...]) -> dict[str, int]:
    return {f"{item.name}:{item.level}": item.n_pairs for item in all_strata(pairs)}
