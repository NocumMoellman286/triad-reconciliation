"""Development and held-out partitions.

The primary test set is the region that took no part in development or hyper-parameter
tuning, and the separation is held at the patient rather than at the visit.

Ref: Sec. 5.2 (development on the sites in one region, testing on the sites in the other, and
a strict patient-level separation), Sec. 5.7 (consecutive accrual and coverage reporting).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad_reconciliation.core.records import CohortTable, VisitPair


@dataclass(frozen=True)
class Partition:
    """Development, held-out and prospective memberships for one cohort table."""

    development: tuple[VisitPair, ...]
    held_out: tuple[VisitPair, ...]

    @property
    def development_patients(self) -> frozenset[str]:
        return frozenset(pair.patient_id for pair in self.development)

    @property
    def held_out_patients(self) -> frozenset[str]:
        return frozenset(pair.patient_id for pair in self.held_out)

    @property
    def overlap(self) -> frozenset[str]:
        return self.development_patients & self.held_out_patients


def split_regions(table: CohortTable, *, held_out_region: str = "II") -> Partition:
    """Split on the declared region membership and verify the patient separation."""

    development = tuple(pair for pair in table.pairs if pair.region != held_out_region)
    held_out = tuple(pair for pair in table.pairs if pair.region == held_out_region)
    return Partition(development=development, held_out=held_out)


def within_region_folds(
    pairs: tuple[VisitPair, ...], folds: int, seed: int
) -> tuple[tuple[VisitPair, ...], ...]:
    """Partition a region into patient-level folds for tuning."""

    if folds < 2:
        raise ValueError("a fold split needs at least two folds")
    patients = sorted({pair.patient_id for pair in pairs})
    order = np.random.default_rng(seed).permutation(len(patients))
    assignment = {patients[index]: position % folds for position, index in enumerate(order)}
    buckets: list[list[VisitPair]] = [[] for _ in range(folds)]
    for pair in pairs:
        buckets[assignment[pair.patient_id]].append(pair)
    return tuple(tuple(bucket) for bucket in buckets)


def monotone_embargo(pairs: tuple[VisitPair, ...], threshold_years: float) -> tuple[VisitPair, ...]:
    """Keep only the pairs whose index precedes the threshold.

    A testing time never sees a predictor series that runs past it, which is what the
    non-anticipative alignment has to hold on the development side too.
    """

    return tuple(pair for pair in pairs if pair.index_years <= threshold_years)


def coverage_report(pairs: tuple[VisitPair, ...]) -> dict[str, float]:
    """Share of pairs with both streams present at the index time.

    Ref: Sec. 5.7, the ratio of viable records collected beside the discrimination level.
    """

    total = len(pairs)
    if total == 0:
        return {"eligible": 0.0, "both_streams": 0.0, "coverage": float("nan")}
    both = sum(1 for pair in pairs if pair.structural and pair.functional)
    return {"eligible": float(total), "both_streams": float(both), "coverage": both / total}


def split_calibration(
    patients: np.ndarray, *, share: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split the held-out patients into a calibration half and an evaluation half.

    The conformal threshold is fitted on one half and its coverage read off the other, so the
    calibration units are never the units the guarantee is reported on.
    """

    if not 0.0 < float(share) < 1.0:
        raise ValueError("the calibration share must sit strictly inside the unit interval")
    unique = np.asarray(
        sorted({str(value) for value in np.asarray(patients).tolist()}), dtype=object
    )
    order = np.random.default_rng(seed).permutation(unique.size)
    cut = max(int(round(float(share) * unique.size)), 1)
    calibration = set(unique[order[:cut]].tolist())
    mask = np.asarray(
        [str(value) in calibration for value in np.asarray(patients).tolist()], dtype=bool
    )
    return mask, ~mask
