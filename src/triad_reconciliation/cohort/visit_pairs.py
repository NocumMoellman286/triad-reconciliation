"""Visit-pair construction across the two clinical timelines.

Ref: Sec. 5.1 (the visit-pair is an index visit and its closest neighbour on the opposing
timeline), Sec. 5.4 (the nearest perimetry visit to an imaging visit).
"""

from __future__ import annotations

from dataclasses import dataclass

from triad_reconciliation.core.records import EyeRecord, PerimetryVisit, StructuralVisit

MONTHS_PER_YEAR = 12.0


@dataclass(frozen=True)
class PairTemplate:
    """The timestamps a visit-pair is built around, before any label is attached."""

    pair_id: str
    patient_id: str
    eye_id: str
    index_years: float
    nearest_years: float
    index_stream: str
    structural: tuple[StructuralVisit, ...]
    functional: tuple[PerimetryVisit, ...]


def neighbour_window(imaging_interval_months: int, perimetry_interval_months: int) -> float:
    """How far apart an index visit and its nearest opposing visit may sit."""

    widest = max(imaging_interval_months, perimetry_interval_months)
    return 0.75 * widest / MONTHS_PER_YEAR


def nearest_functional(visits: tuple[PerimetryVisit, ...], stamp: float) -> PerimetryVisit | None:
    if not visits:
        return None
    return min(visits, key=lambda visit: (abs(visit.tested_years - stamp), visit.tested_years))


def nearest_structural(visits: tuple[StructuralVisit, ...], stamp: float) -> StructuralVisit | None:
    if not visits:
        return None
    return min(visits, key=lambda visit: (abs(visit.acquired_years - stamp), visit.acquired_years))


def _truncate_structural(
    visits: tuple[StructuralVisit, ...], cutoff: float
) -> tuple[StructuralVisit, ...]:
    return tuple(visit for visit in visits if visit.acquired_years <= cutoff)


def _truncate_functional(
    visits: tuple[PerimetryVisit, ...], cutoff: float
) -> tuple[PerimetryVisit, ...]:
    return tuple(visit for visit in visits if visit.tested_years <= cutoff)


def build_visit_pairs(
    record: EyeRecord,
    *,
    imaging_interval_months: int,
    perimetry_interval_months: int,
    index_stream: str = "union",
    max_pairs: int | None = None,
) -> tuple[PairTemplate, ...]:
    """Enumerate the pairs an eye contributes.

    An imaging visit is the index of a pair when a perimetry report sits inside the pairing
    window and vice versa; either way the predictor series are cut at the later of the two
    timestamps, so nothing recorded after the query time reaches the model.
    """

    if index_stream not in {"union", "structural", "functional"}:
        raise ValueError(f"unknown index stream {index_stream!r}")
    window = neighbour_window(imaging_interval_months, perimetry_interval_months)
    if index_stream == "structural":
        stamps = sorted({visit.acquired_years for visit in record.structural})
    elif index_stream == "functional":
        stamps = sorted({visit.tested_years for visit in record.functional})
    else:
        stamps = sorted(
            {visit.acquired_years for visit in record.structural}
            | {visit.tested_years for visit in record.functional}
        )
    templates: list[PairTemplate] = []
    for order, stamp in enumerate(stamps):
        structural_near = nearest_structural(record.structural, stamp)
        functional_near = nearest_functional(record.functional, stamp)
        if structural_near is None or functional_near is None:
            continue
        structural_distance = abs(structural_near.acquired_years - stamp)
        functional_distance = abs(functional_near.tested_years - stamp)
        if (
            index_stream != "functional"
            and functional_distance <= structural_distance
            and functional_distance <= window
        ):
            index_stream_label = "structural"
            nearest_years = functional_near.tested_years
        elif structural_distance <= window:
            index_stream_label = "functional"
            nearest_years = structural_near.acquired_years
        elif functional_distance <= window:
            index_stream_label = "structural"
            nearest_years = functional_near.tested_years
        else:
            continue
        cutoff = max(stamp, nearest_years)
        templates.append(
            PairTemplate(
                pair_id=f"{record.eye_id}-p{order:03d}",
                patient_id=record.patient_id,
                eye_id=record.eye_id,
                index_years=stamp,
                nearest_years=nearest_years,
                index_stream=index_stream_label,
                structural=_truncate_structural(record.structural, cutoff),
                functional=_truncate_functional(record.functional, cutoff),
            )
        )
    templates.sort(key=lambda item: (item.index_years, item.pair_id))
    if max_pairs is not None:
        templates = templates[:max_pairs]
    return tuple(templates)


def pair_gap_years(templates: tuple[PairTemplate, ...]) -> tuple[float, ...]:
    return tuple(abs(item.index_years - item.nearest_years) for item in templates)


def discordant_share(labels: tuple[str, ...]) -> float:
    if not labels:
        return float("nan")
    return sum(1 for label in labels if label != "concordant") / len(labels)
