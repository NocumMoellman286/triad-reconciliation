"""The comparator roster.

Each comparator carries its family and its input, so a reported row can be reconstructed
without consulting another source.

Ref: Table 1 (fifteen rows), Table 7 (the roster with family and input).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn

from triad_reconciliation.comparators.clinical_rules import RuleResult, all_rules
from triad_reconciliation.comparators.fusion import LateFusionPerVisit, MultiEncoderStaticFusion
from triad_reconciliation.comparators.single_stream import (
    FundusTemporalNetwork,
    OpticNerveHeadVisionTransformer,
    OpticNerveHeadVolumeNetwork,
    PerimetryFieldIndexTransformer,
    PerimetrySequenceNetwork,
)
from triad_reconciliation.comparators.timeline import ConcatenatedTimelineTransformer
from triad_reconciliation.core.config import ModelConfig

PROPOSED = "TRIAD (this work)"


@dataclass(frozen=True)
class ComparatorSpec:
    """One row of the roster."""

    name: str
    family: str
    inputs: str
    learned: bool
    table_row: int
    module_factory: str | None = field(default=None)


ROSTER: tuple[ComparatorSpec, ...] = (
    ComparatorSpec("guided progression analysis", "clinical rule", "serial perimetry", False, 1),
    ComparatorSpec("visual field index trend", "clinical rule", "serial perimetry", False, 2),
    ComparatorSpec("mean deviation trend", "clinical rule", "serial perimetry", False, 3),
    ComparatorSpec(
        "linear structure-function mapping",
        "statistical",
        "optic-disc summary and perimetry indices",
        False,
        4,
    ),
    ComparatorSpec(
        "kernel regression, radiomic + perimetry",
        "statistical",
        "optic-disc summary and perimetry indices",
        False,
        5,
    ),
    ComparatorSpec(
        "temporal CNN, fundus only",
        "single-stream visual",
        "serial optic-disc morphometry",
        True,
        6,
        "FundusTemporalNetwork",
    ),
    ComparatorSpec(
        "3D CNN, tomography only",
        "single-stream visual",
        "serial optic-nerve-head grids",
        True,
        7,
        "OpticNerveHeadVolumeNetwork",
    ),
    ComparatorSpec(
        "vision transformer, serial tomography",
        "single-stream visual",
        "serial optic-nerve-head grids",
        True,
        8,
        "OpticNerveHeadVisionTransformer",
    ),
    ComparatorSpec(
        "LSTM, perimetry sequences",
        "single-stream functional",
        "serial perimetry as text",
        True,
        9,
        "PerimetrySequenceNetwork",
    ),
    ComparatorSpec(
        "temporal transformer, field indices",
        "single-stream functional",
        "serial perimetry indices",
        True,
        10,
        "PerimetryFieldIndexTransformer",
    ),
    ComparatorSpec(
        "multimodal language model, zero-shot",
        "language",
        "serialised report text with an optic-disc summary",
        False,
        11,
    ),
    ComparatorSpec(
        "late fusion of per-visit embeddings",
        "static multimodal",
        "both streams, visit by visit",
        True,
        12,
        "LateFusionPerVisit",
    ),
    ComparatorSpec(
        "multi-encoder static fusion",
        "static multimodal",
        "both streams, last visit",
        True,
        13,
        "MultiEncoderStaticFusion",
    ),
    ComparatorSpec(
        "concatenated-timeline transformer",
        "temporal multimodal",
        "both streams on a concatenated timeline",
        True,
        14,
        "ConcatenatedTimelineTransformer",
    ),
    ComparatorSpec(PROPOSED, "temporal multimodal", "both streams, aligned and monotone", True, 15),
)


def learned_modules(config: ModelConfig) -> dict[str, nn.Module]:
    """Instantiate every learned comparator under its roster name."""

    return {
        "temporal CNN, fundus only": FundusTemporalNetwork(config),
        "3D CNN, tomography only": OpticNerveHeadVolumeNetwork(config),
        "vision transformer, serial tomography": OpticNerveHeadVisionTransformer(config),
        "LSTM, perimetry sequences": PerimetrySequenceNetwork(config),
        "temporal transformer, field indices": PerimetryFieldIndexTransformer(config),
        "late fusion of per-visit embeddings": LateFusionPerVisit(config),
        "multi-encoder static fusion": MultiEncoderStaticFusion(config),
        "concatenated-timeline transformer": ConcatenatedTimelineTransformer(config),
    }


def clinical_rule_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in ROSTER if spec.family == "clinical rule")


RULE_NAME_MAP = {
    "guided_progression_analysis": "guided progression analysis",
    "visual_field_index_trend": "visual field index trend",
    "mean_deviation_trend": "mean deviation trend",
}


def rule_results(features: object) -> dict[str, RuleResult]:
    """Score every clinical rule on a pair set, under its roster name."""

    results = all_rules(features)  # type: ignore[arg-type]
    return {RULE_NAME_MAP[result.name]: result for result in results}


def roster_by_row() -> dict[int, ComparatorSpec]:
    return {spec.table_row: spec for spec in ROSTER}


def roster_names() -> tuple[str, ...]:
    return tuple(spec.name for spec in ROSTER)


def families() -> tuple[str, ...]:
    seen: dict[str, None] = {}
    for spec in ROSTER:
        seen.setdefault(spec.family, None)
    return tuple(seen)
