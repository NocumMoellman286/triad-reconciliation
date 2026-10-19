"""Record types shared by the cohort, stream and model layers.

The units of analysis are the patient, the eye, the visit and the visit-pair, in the order
the study declares them.

Ref: Sec. 5.1 (streams, visit-pair, arbitration label), Sec. 5.2 (units of analysis).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ArbitrationLabel = Literal["structural", "functional", "concordant"]
AgreementState = Literal["concordant", "discordant"]
Region = Literal["I", "II"]
StreamKind = Literal["structural", "functional"]

ARBITRATION_CLASSES: tuple[ArbitrationLabel, ...] = ("structural", "functional", "concordant")
AGREEMENT_STATES: tuple[AgreementState, ...] = ("concordant", "discordant")

N_MORPHOMETRY = 8
N_PERIMETRY_POINTS = 24
VOLUME_EDGE = 4


@dataclass(frozen=True)
class StructuralVisit:
    """One optic-disc imaging encounter.

    ``morphometry`` holds the optic-disc summary the imaging layer emits and ``volume`` the
    optic-nerve-head tomography grid flattened in (depth, row, column) order.

    Ref: Sec. 5.1, the structural stream.
    """

    patient_id: str
    eye_id: str
    site: str
    acquired_years: float
    morphometry: tuple[float, ...]
    volume: tuple[float, ...]

    @property
    def volume_shape(self) -> tuple[int, int, int]:
        return (VOLUME_EDGE, VOLUME_EDGE, VOLUME_EDGE)


@dataclass(frozen=True)
class PerimetryVisit:
    """One automated perimetry report.

    Ref: Sec. 5.1, the functional stream.
    """

    patient_id: str
    eye_id: str
    site: str
    tested_years: float
    sensitivity_db: tuple[float, ...]
    mean_deviation_db: float
    pattern_deviation_db: float
    visual_field_index: float
    false_positive_rate: float
    false_negative_rate: float
    fixation_losses: int
    duration_seconds: float
    gaze_tracking_ok: bool

    @property
    def reliability_ok(self) -> bool:
        return (
            self.false_positive_rate <= 0.15
            and self.false_negative_rate <= 0.20
            and self.fixation_losses <= 2
            and self.gaze_tracking_ok
        )


@dataclass(frozen=True)
class EyeRecord:
    """An eye with both streams in clinical-time order."""

    patient_id: str
    eye_id: str
    site: str
    region: Region
    disc_stage: Literal["early", "moderate", "advanced"]
    high_myopia: bool
    age_years: float
    sex: Literal["female", "male"]
    ancestry_group: int
    structural: tuple[StructuralVisit, ...]
    functional: tuple[PerimetryVisit, ...]

    @property
    def first_years(self) -> float:
        stamps = [v.acquired_years for v in self.structural]
        stamps += [v.tested_years for v in self.functional]
        return min(stamps)


@dataclass(frozen=True)
class VisitPair:
    """An index visit and its nearest return visit on the opposing timeline.

    Ref: Sec. 5.1 (the visit-pair is where progression and discordance are examined),
    Sec. 5.4 (the comparison of interest runs between the imaging visit and the nearest
    perimetry visit).
    """

    pair_id: str
    patient_id: str
    eye_id: str
    site: str
    region: Region
    index_years: float
    nearest_years: float
    structural: tuple[StructuralVisit, ...]
    functional: tuple[PerimetryVisit, ...]
    progression: int
    arbitration: ArbitrationLabel
    stage: Literal["early", "moderate", "advanced"]
    high_myopia: bool
    age_years: float
    sex: Literal["female", "male"]
    ancestry_group: int
    functional_reliable: bool

    @property
    def discordant(self) -> bool:
        return self.arbitration != "concordant"

    @property
    def gap_years(self) -> float:
        return abs(self.index_years - self.nearest_years)


@dataclass(frozen=True)
class CohortTable:
    """A cohort table as the reader presents it.

    Every table this release reads is one a caller supplied; the field records which file it
    came from so a report can name its own input without carrying a path from the host.
    """

    pairs: tuple[VisitPair, ...]
    source: str

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def eyes(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for pair in self.pairs:
            seen.setdefault(pair.eye_id, None)
        return tuple(seen)

    @property
    def patients(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for pair in self.pairs:
            seen.setdefault(pair.patient_id, None)
        return tuple(seen)

    @property
    def discordant_share(self) -> float:
        if not self.pairs:
            return float("nan")
        return sum(1 for pair in self.pairs if pair.discordant) / len(self.pairs)


@dataclass(frozen=True)
class ProgressionScores:
    """Model output for a batch of visit-pairs.

    Ref: Sec. 5.1, ``f_theta`` maps the two streams to the progression probability, the
    arbitration direction and the stream-evidence weight.
    """

    progression_logit: float
    arbitration_logits: tuple[float, float, float]
    agreement_logits: tuple[float, float]
    stream_weight: float


@dataclass(frozen=True)
class ReadershipResponse:
    """One reader's answer on one vignette in one study arm."""

    reader_id: str
    vignette_id: str
    arm: Literal["unassisted", "model_only", "model_assisted"]
    answered_direction: ArbitrationLabel
    seconds: float


@dataclass
class TrainingHistory:
    """Per-epoch objectives in the order Algorithm 1 visits the development pairs."""

    total: list[float] = field(default_factory=list)
    progression: list[float] = field(default_factory=list)
    arbitration: list[float] = field(default_factory=list)
    consistency: list[float] = field(default_factory=list)
    weight_sum: list[float] = field(default_factory=list)
