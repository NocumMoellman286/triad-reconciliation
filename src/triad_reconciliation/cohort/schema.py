"""Cohort table schema.

The released table is a single JSONL stream discriminated by a ``record`` field, so a caller
can supply visits, eye-level covariates and labelled visit-pairs without a second file.

Ref: Sec. 5.1 (streams and visit-pairs), Sec. 5.2 (units of analysis and the patient-level
separation between development and testing).
"""

from __future__ import annotations

from typing import Any

from triad_reconciliation.core.records import (
    AGREEMENT_STATES,
    ARBITRATION_CLASSES,
    N_MORPHOMETRY,
    N_PERIMETRY_POINTS,
    VOLUME_EDGE,
)

RECORD_KINDS: tuple[str, ...] = ("eye", "structural", "functional", "pair")

EYE_KEYS: tuple[str, ...] = (
    "patient_id",
    "eye_id",
    "site",
    "region",
    "stage",
    "high_myopia",
    "age_years",
    "sex",
    "ancestry_group",
)

STRUCTURAL_KEYS: tuple[str, ...] = ("patient_id", "eye_id", "time_years", "morphometry", "volume")

FUNCTIONAL_KEYS: tuple[str, ...] = (
    "patient_id",
    "eye_id",
    "time_years",
    "sensitivity_db",
    "md",
    "pd",
    "vfi",
    "fp",
    "fn",
    "fix",
    "duration_s",
    "gaze_ok",
)

PAIR_KEYS: tuple[str, ...] = (
    "pair_id",
    "patient_id",
    "eye_id",
    "index_years",
    "nearest_years",
    "index_stream",
    "progression",
    "arbitration",
)

STAGES: tuple[str, ...] = ("early", "moderate", "advanced")


class CohortSchemaError(ValueError):
    """Raised when a supplied row cannot be placed in the released schema."""


def _require(row: dict[str, Any], keys: tuple[str, ...], line: int) -> None:
    missing = [key for key in keys if key not in row]
    if missing:
        raise CohortSchemaError(f"line {line} ({row.get('record')!r}) is missing {missing}")


def _numeric_vector(value: Any, width: int, label: str, line: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise CohortSchemaError(f"line {line}: {label} must hold exactly {width} values")
    try:
        return tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise CohortSchemaError(f"line {line}: {label} holds a non-numeric entry") from error


def normalise_row(row: dict[str, Any], line: int) -> dict[str, Any]:
    """Validate one row and return it with numeric fields coerced."""

    kind = row.get("record")
    if kind not in RECORD_KINDS:
        raise CohortSchemaError(f"line {line}: unknown record kind {kind!r}")
    normalised: dict[str, Any] = dict(row)
    if kind == "eye":
        _require(row, EYE_KEYS, line)
        normalised["age_years"] = float(row["age_years"])
        normalised["high_myopia"] = bool(row["high_myopia"])
        normalised["ancestry_group"] = int(row["ancestry_group"])
        if row["stage"] not in STAGES:
            raise CohortSchemaError(f"line {line}: unknown stage {row['stage']!r}")
        if row["region"] not in {"I", "II"}:
            raise CohortSchemaError(f"line {line}: unknown region {row['region']!r}")
    elif kind == "structural":
        _require(row, STRUCTURAL_KEYS, line)
        normalised["time_years"] = float(row["time_years"])
        normalised["morphometry"] = _numeric_vector(
            row["morphometry"], N_MORPHOMETRY, "morphometry", line
        )
        normalised["volume"] = _numeric_vector(row["volume"], VOLUME_EDGE**3, "volume", line)
    elif kind == "functional":
        _require(row, FUNCTIONAL_KEYS, line)
        normalised["time_years"] = float(row["time_years"])
        normalised["sensitivity_db"] = _numeric_vector(
            row["sensitivity_db"], N_PERIMETRY_POINTS, "sensitivity_db", line
        )
        for key in ("md", "pd", "vfi", "fp", "fn", "duration_s"):
            normalised[key] = float(row[key])
        normalised["fix"] = int(row["fix"])
        normalised["gaze_ok"] = bool(row["gaze_ok"])
    else:
        _require(row, PAIR_KEYS, line)
        normalised["index_years"] = float(row["index_years"])
        normalised["nearest_years"] = float(row["nearest_years"])
        normalised["progression"] = int(row["progression"])
        if row["arbitration"] not in ARBITRATION_CLASSES:
            raise CohortSchemaError(f"line {line}: unknown arbitration {row['arbitration']!r}")
        if row["index_stream"] not in {"structural", "functional"}:
            raise CohortSchemaError(f"line {line}: unknown index stream {row['index_stream']!r}")
        if normalised["progression"] not in (0, 1):
            raise CohortSchemaError(f"line {line}: progression must be binary")
    return normalised


def agreement_state_of(arbitration: str) -> str:
    """Read the agreement state off an arbitration label."""

    state = "concordant" if arbitration == "concordant" else "discordant"
    if state not in AGREEMENT_STATES:
        raise CohortSchemaError(f"unreachable agreement state {state!r}")
    return state


def schema_card() -> dict[str, Any]:
    """A machine-readable description of the released table."""

    return {
        "format": "jsonl",
        "record_kinds": list(RECORD_KINDS),
        "eye_keys": list(EYE_KEYS),
        "structural_keys": list(STRUCTURAL_KEYS),
        "functional_keys": list(FUNCTIONAL_KEYS),
        "pair_keys": list(PAIR_KEYS),
        "morphometry_width": N_MORPHOMETRY,
        "volume_edge": VOLUME_EDGE,
        "perimetry_points": N_PERIMETRY_POINTS,
        "arbitration_classes": list(ARBITRATION_CLASSES),
    }
