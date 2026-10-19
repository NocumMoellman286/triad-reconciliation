"""Reader for the cohort table a caller supplies.

The analysed records are not distributed: the Data availability statement withholds both
institution and site identities. This module is the whole of the release's cohort surface — it
reads a table, validates it against the released schema and builds the visit-pairs it declares.

Ref: Sec. 5.2 (the private anonymised multicentre set and the units of analysis).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from triad_reconciliation.cohort.schema import CohortSchemaError, normalise_row
from triad_reconciliation.core.jsonio import read_jsonl
from triad_reconciliation.core.records import (
    CohortTable,
    EyeRecord,
    PerimetryVisit,
    StructuralVisit,
    VisitPair,
)

Stage = str


class CohortNotStaged(FileNotFoundError):
    """Raised when a command needs a cohort table and none was supplied."""

    def __init__(self, hint: str) -> None:
        super().__init__(hint)
        self.hint = hint


def _index_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {
        "eye": [],
        "structural": [],
        "functional": [],
        "pair": [],
    }
    for line, row in enumerate(rows, start=1):
        normalised = normalise_row(row, line)
        grouped[str(normalised["record"])].append(normalised)
    return grouped


def _eye_index(eye_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in eye_rows:
        key = str(row["eye_id"])
        if key in index:
            raise CohortSchemaError(f"eye {key} appears twice in the eye records")
        index[key] = row
    return index


def _structural_by_eye(
    rows: list[dict[str, Any]], eyes: dict[str, dict[str, Any]]
) -> dict[str, list[StructuralVisit]]:
    grouped: dict[str, list[StructuralVisit]] = {}
    for row in rows:
        eye_id = str(row["eye_id"])
        covariates = eyes.get(eye_id)
        if covariates is None:
            raise CohortSchemaError(f"structural visit references unknown eye {eye_id}")
        grouped.setdefault(eye_id, []).append(
            StructuralVisit(
                patient_id=str(row["patient_id"]),
                eye_id=eye_id,
                site=str(covariates["site"]),
                acquired_years=float(row["time_years"]),
                morphometry=tuple(row["morphometry"]),
                volume=tuple(row["volume"]),
            )
        )
    for visits in grouped.values():
        visits.sort(key=lambda visit: visit.acquired_years)
    return grouped


def _functional_by_eye(
    rows: list[dict[str, Any]], eyes: dict[str, dict[str, Any]]
) -> dict[str, list[PerimetryVisit]]:
    grouped: dict[str, list[PerimetryVisit]] = {}
    for row in rows:
        eye_id = str(row["eye_id"])
        covariates = eyes.get(eye_id)
        if covariates is None:
            raise CohortSchemaError(f"perimetry visit references unknown eye {eye_id}")
        grouped.setdefault(eye_id, []).append(
            PerimetryVisit(
                patient_id=str(row["patient_id"]),
                eye_id=eye_id,
                site=str(covariates["site"]),
                tested_years=float(row["time_years"]),
                sensitivity_db=tuple(row["sensitivity_db"]),
                mean_deviation_db=float(row["md"]),
                pattern_deviation_db=float(row["pd"]),
                visual_field_index=float(row["vfi"]),
                false_positive_rate=float(row["fp"]),
                false_negative_rate=float(row["fn"]),
                fixation_losses=int(row["fix"]),
                duration_seconds=float(row["duration_s"]),
                gaze_tracking_ok=bool(row["gaze_ok"]),
            )
        )
    for visits in grouped.values():
        visits.sort(key=lambda visit: visit.tested_years)
    return grouped


def _truncate_at(visits: list[StructuralVisit], cutoff: float) -> tuple[StructuralVisit, ...]:
    kept = [visit for visit in visits if visit.acquired_years <= cutoff]
    return tuple(kept)


def _truncate_perimetry(visits: list[PerimetryVisit], cutoff: float) -> tuple[PerimetryVisit, ...]:
    kept = [visit for visit in visits if visit.tested_years <= cutoff]
    return tuple(kept)


def read_cohort(path: Path) -> CohortTable:
    """Read a cohort table and build the visit-pairs it declares.

    Every predictor series is truncated at the pair's index time, which is the embargo the
    study imposes: a testing time never sees a predictor recorded after it.
    """

    rows = read_jsonl(Path(path))
    grouped = _index_rows(rows)
    eyes = _eye_index(grouped["eye"])
    structural = _structural_by_eye(grouped["structural"], eyes)
    functional = _functional_by_eye(grouped["functional"], eyes)

    pairs: list[VisitPair] = []
    for row in grouped["pair"]:
        eye_id = str(row["eye_id"])
        covariates = eyes.get(eye_id)
        if covariates is None:
            raise CohortSchemaError(f"visit-pair references unknown eye {eye_id}")
        index_years = float(row["index_years"])
        nearest_years = float(row["nearest_years"])
        cutoff = max(index_years, nearest_years)
        eye_functional = _truncate_perimetry(functional.get(eye_id, []), cutoff)
        pairs.append(
            VisitPair(
                pair_id=str(row["pair_id"]),
                patient_id=str(row["patient_id"]),
                eye_id=eye_id,
                site=str(covariates["site"]),
                region=str(covariates["region"]),  # type: ignore[arg-type]
                index_years=index_years,
                nearest_years=nearest_years,
                structural=_truncate_at(structural.get(eye_id, []), cutoff),
                functional=eye_functional,
                progression=int(row["progression"]),
                arbitration=str(row["arbitration"]),  # type: ignore[arg-type]
                stage=str(covariates["stage"]),  # type: ignore[arg-type]
                high_myopia=bool(covariates["high_myopia"]),
                age_years=float(covariates["age_years"]),
                sex=str(covariates["sex"]),  # type: ignore[arg-type]
                ancestry_group=int(covariates["ancestry_group"]),
                functional_reliable=bool(eye_functional and eye_functional[-1].reliability_ok),
            )
        )
    pairs.sort(key=lambda pair: pair.pair_id)
    return CohortTable(pairs=tuple(pairs), source=Path(path).name)


def read_eyes(path: Path) -> tuple[EyeRecord, ...]:
    """Assemble complete eye records from the same table, stream by stream."""

    rows = read_jsonl(Path(path))
    grouped = _index_rows(rows)
    eyes = _eye_index(grouped["eye"])
    structural = _structural_by_eye(grouped["structural"], eyes)
    functional = _functional_by_eye(grouped["functional"], eyes)
    records: list[EyeRecord] = []
    for eye_id, covariates in eyes.items():
        records.append(
            EyeRecord(
                patient_id=str(covariates["patient_id"]),
                eye_id=eye_id,
                site=str(covariates["site"]),
                region=str(covariates["region"]),  # type: ignore[arg-type]
                disc_stage=str(covariates["stage"]),  # type: ignore[arg-type]
                high_myopia=bool(covariates["high_myopia"]),
                age_years=float(covariates["age_years"]),
                sex=str(covariates["sex"]),  # type: ignore[arg-type]
                ancestry_group=int(covariates["ancestry_group"]),
                structural=tuple(structural.get(eye_id, [])),
                functional=tuple(functional.get(eye_id, [])),
            )
        )
    records.sort(key=lambda record: record.eye_id)
    return tuple(records)
