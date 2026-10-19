"""End-to-end study assembly.

Builds the cohort view, fits the reconciliation stack and the roster, and evaluates every
reported family of numbers on the held-out region.

Ref: Sec. 2 (the reported analyses), Sec. 5.1-5.8 (the estimands, the cohort, the statistics),
Algorithm 1 (the training procedure).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from triad_reconciliation.analysis.batching import FlatFeatures, build_flat_features
from triad_reconciliation.analysis.trainer import (
    fit_binary_module,
    predict_probabilities,
    progression_only_loss_function,
    reconciliation_loss_function,
    reconciliation_outputs,
)
from triad_reconciliation.cohort.reader import CohortNotStaged, read_cohort
from triad_reconciliation.cohort.splits import Partition, split_calibration, split_regions
from triad_reconciliation.comparators import roster
from triad_reconciliation.comparators.clinical_rules import all_rules
from triad_reconciliation.comparators.structure_function import (
    kernel_structure_function,
    linear_structure_function,
)
from triad_reconciliation.comparators.zero_shot import zero_shot_scores
from triad_reconciliation.core.config import ExperimentConfig
from triad_reconciliation.core.records import ARBITRATION_CLASSES, CohortTable
from triad_reconciliation.core.seeding import set_seed
from triad_reconciliation.metrics.agreement import ARBITRATION_CATEGORIES, cohen_kappa
from triad_reconciliation.metrics.calibration import brier_score, expected_calibration_error
from triad_reconciliation.metrics.conformal import deferral_analysis, unguarded_analysis
from triad_reconciliation.metrics.discrimination import (
    auroc,
    average_precision,
    sensitivity_at_specificity,
)
from triad_reconciliation.metrics.excess_risk import (
    Decomposition,
    decompose,
    effective_pairs,
)
from triad_reconciliation.metrics.identifiability import (
    FUSION_GAIN,
    estimate_lipschitz,
    evidence_gap,
    identifiable,
    linear_fusion,
    separation_of_weight,
)
from triad_reconciliation.models.triad import TriadModel, VariantFlags
from triad_reconciliation.strata.partitions import stratum_sizes

METHOD_ORDER: tuple[str, ...] = roster.roster_names()
RULE_ROSTER_NAMES = {
    "guided_progression_analysis": "guided progression analysis",
    "visual_field_index_trend": "visual field index trend",
    "mean_deviation_trend": "mean deviation trend",
}
STAGING_HINT = (
    "no cohort table is staged. This release does not distribute or generate the analysed "
    "records; pass one with --cohort, as the Data availability statement requires."
)


def _direction_kappa(predicted: np.ndarray, recorded: np.ndarray) -> float:
    """Chance-corrected agreement between the predicted and recorded arbitration direction.

    Both sides are read over the three arbitration classes: a model that answers concordant on
    a discordant pair has missed the direction, and dropping that answer would score it as if
    it had never been made.

    Ref: Table 1 caption (kappa on the discordant subset).
    """

    if predicted.size == 0:
        return float("nan")
    left = np.asarray(
        [ARBITRATION_CLASSES[int(value)] for value in predicted.tolist()], dtype=object
    )
    right = np.asarray(
        [ARBITRATION_CLASSES[int(value)] for value in recorded.tolist()], dtype=object
    )
    return cohen_kappa(left, right, ARBITRATION_CATEGORIES)


@dataclass
class MethodScore:
    """One method's discrimination, calibration and agreement on the held-out region."""

    name: str
    probabilities: np.ndarray
    direction: np.ndarray | None = None
    stream_weight: np.ndarray | None = None

    def metrics(
        self, labels: np.ndarray, discordant: np.ndarray, arbitration: np.ndarray
    ) -> dict[str, float]:
        kappa = float("nan")
        if self.direction is not None and bool(discordant.any()):
            kappa = _direction_kappa(
                np.asarray(self.direction)[discordant], np.asarray(arbitration)[discordant]
            )
        return {
            "auroc": auroc(self.probabilities, labels),
            "auprc": average_precision(self.probabilities, labels),
            "sensitivity": sensitivity_at_specificity(self.probabilities, labels, 0.95),
            "ece": expected_calibration_error(self.probabilities, labels, 15),
            "brier": brier_score(self.probabilities, labels),
            "kappa": kappa,
        }


@dataclass
class VariantScore:
    """One component-analysis row."""

    variant: str
    overall_auroc: float
    discordant_auroc: float
    kappa: float
    discordant_delta: float
    epochs: int


@dataclass
class StudyResult:
    """Everything the tables are rendered from."""

    source: str
    partition: Partition
    stratum_sizes: dict[str, int]
    methods: list[MethodScore]
    variants: list[VariantScore]
    discordance: dict[str, float]
    arbitration: dict[str, float]
    deferral: dict[str, float]
    unguarded: dict[str, float]
    identifiability: dict[str, float]
    excess_risk: dict[str, float]
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    discordant_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    arbitration_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    patients: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=object))
    epochs: int = 0

    def table_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for score in self.methods:
            rows.append(
                {
                    "method": score.name,
                    **score.metrics(self.labels, self.discordant_mask, self.arbitration_index),
                }
            )
        return rows

    def variant_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "variant": item.variant,
                "auroc_all": item.overall_auroc,
                "auroc_discordant": item.discordant_auroc,
                "kappa": item.kappa,
                "delta_discordant": item.discordant_delta,
                "epochs": float(item.epochs),
            }
            for item in self.variants
        ]


def build_table(config: ExperimentConfig, source: Path | None) -> CohortTable:
    """Read the cohort table the caller supplied.

    The release holds no records of its own, so a command that needs one cannot proceed
    without it.
    """

    del config
    if source is None:
        raise CohortNotStaged(STAGING_HINT)
    return read_cohort(Path(source))


def prepare(
    config: ExperimentConfig, source: Path | None
) -> tuple[FlatFeatures, FlatFeatures, CohortTable]:
    """Read the table, split the two regions and build both feature views."""

    set_seed(config.cohort.seed)
    table = build_table(config, source)
    partition = split_regions(table, held_out_region=config.cohort.held_out_region)
    development = build_flat_features(partition.development, config.model)
    held_out = build_flat_features(partition.held_out, config.model)
    return development, held_out, table


def fit_reconciliation(
    development: FlatFeatures, config: ExperimentConfig, flags: VariantFlags
) -> TriadModel:
    """Fit the reconciliation stack under one variant."""

    model = TriadModel(config.model, flags)
    loss = reconciliation_loss_function(
        config.train.consistency_weight if flags.use_attribution_consistency else 0.0,
        config.train.reweighting_strength if flags.use_agreement_reweighting else 0.0,
        config.model.stream_weight_margin,
        supervise_arbitration=flags.use_arbitration_head,
    )
    fit_binary_module(model, development, config.train, loss_function=loss, seed=config.cohort.seed)
    return model


def comparator_scores(
    development: FlatFeatures, held_out: FlatFeatures, config: ExperimentConfig
) -> list[MethodScore]:
    """Fit and score every roster entry on the shared held-out units."""

    collected: dict[str, MethodScore] = {}
    for result in all_rules(held_out):
        collected[RULE_ROSTER_NAMES[result.name]] = MethodScore(
            name=RULE_ROSTER_NAMES[result.name],
            probabilities=np.asarray(result.score, dtype=np.float64),
        )
    linear = linear_structure_function(development, held_out)
    collected["linear structure-function mapping"] = MethodScore(
        name="linear structure-function mapping", probabilities=linear.score
    )
    kernel = kernel_structure_function(development, held_out)
    collected["kernel regression, radiomic + perimetry"] = MethodScore(
        name="kernel regression, radiomic + perimetry", probabilities=kernel.score
    )
    collected["multimodal language model, zero-shot"] = MethodScore(
        name="multimodal language model, zero-shot", probabilities=zero_shot_scores(held_out)
    )
    loss = progression_only_loss_function()
    for name, module in roster.learned_modules(config.model).items():
        fit_binary_module(
            module, development, config.train, loss_function=loss, seed=config.cohort.seed
        )
        collected[name] = MethodScore(
            name=name, probabilities=predict_probabilities(module, held_out, config.train)
        )
    return [
        collected[name] for name in METHOD_ORDER if name in collected and name != roster.PROPOSED
    ]


def variant_scores(
    development: FlatFeatures, held_out: FlatFeatures, config: ExperimentConfig
) -> list[VariantScore]:
    """Fit every reported variant and score it on the held-out region."""

    labels = held_out.progression
    discordant = held_out.discordant
    variants: tuple[VariantFlags, ...] = (
        VariantFlags(),
        VariantFlags(use_reconciliation_state=False),
        VariantFlags(use_arbitration_head=False),
        VariantFlags(use_attribution_consistency=False),
        VariantFlags(use_agreement_reweighting=False),
        VariantFlags(use_monotone_alignment=False),
        VariantFlags(use_language_encoder=False),
        VariantFlags(use_calibrated_deferral=False),
        VariantFlags(use_reconciliation_state=False, use_arbitration_head=False),
    )
    scores: list[VariantScore] = []
    reference = float("nan")
    for flags in variants:
        model = fit_reconciliation(development, config, flags)
        outputs = reconciliation_outputs(model, held_out, config.train)
        overall = auroc(outputs["progression"], labels)
        stratum = auroc(outputs["progression"][discordant], labels[discordant])
        kappa = _direction_kappa(outputs["direction"][discordant], held_out.arbitration[discordant])
        if np.isnan(reference):
            reference = stratum
        scores.append(
            VariantScore(
                variant=flags.name,
                overall_auroc=overall,
                discordant_auroc=stratum,
                kappa=kappa,
                discordant_delta=stratum - reference,
                epochs=config.train.epochs,
            )
        )
    return scores


def discordance_analysis(
    development: FlatFeatures, held_out: FlatFeatures, config: ExperimentConfig
) -> dict[str, float]:
    """Concordant and discordant stratum areas for the three headline methods."""

    labels = held_out.progression
    discordant = held_out.discordant
    concordant = ~discordant
    model = fit_reconciliation(development, config, VariantFlags())
    outputs = reconciliation_outputs(model, held_out, config.train)
    modules = roster.learned_modules(config.model)
    loss = progression_only_loss_function()
    fitted: dict[str, np.ndarray] = {}
    for name in ("concatenated-timeline transformer", "multi-encoder static fusion"):
        module = modules[name]
        fit_binary_module(
            module, development, config.train, loss_function=loss, seed=config.cohort.seed
        )
        fitted[name] = predict_probabilities(module, held_out, config.train)
    triad = outputs["progression"]
    return {
        "discordant_share": float(discordant.mean()),
        "concordant_share": float(concordant.mean()),
        "triad_concordant": auroc(triad[concordant], labels[concordant]),
        "unconstrained_concordant": auroc(
            fitted["concatenated-timeline transformer"][concordant], labels[concordant]
        ),
        "static_concordant": auroc(
            fitted["multi-encoder static fusion"][concordant], labels[concordant]
        ),
        "triad_discordant": auroc(triad[discordant], labels[discordant]),
        "unconstrained_discordant": auroc(
            fitted["concatenated-timeline transformer"][discordant], labels[discordant]
        ),
        "static_discordant": auroc(
            fitted["multi-encoder static fusion"][discordant], labels[discordant]
        ),
    }


def arbitration_analysis(
    outputs: dict[str, np.ndarray], held_out: FlatFeatures
) -> dict[str, float]:
    """Agreement with the recorded direction on the discordant pairs."""

    keep = held_out.discordant
    recorded = np.asarray(
        [ARBITRATION_CLASSES[int(value)] for value in held_out.arbitration[keep].tolist()],
        dtype=object,
    )
    weights = outputs["stream_weight"][keep]
    mask = recorded == "structural"
    structural, functional = separation_of_weight(weights, mask)
    return {
        "kappa": _direction_kappa(outputs["direction"][keep], held_out.arbitration[keep]),
        "separation_structural": structural,
        "separation_functional": functional,
        "resolved_share": float((held_out.arbitration[keep] != 2).mean()),
    }


def identifiability_analysis(held_out: FlatFeatures) -> dict[str, float]:
    """Where the stream-evidence weight is pinned down and where it is not.

    Ref: Proposition 1.
    """

    structural = held_out.structural_summary()
    field = held_out.masked_field_mean()
    structural_evidence = np.clip((structural[:, 0] - 0.35) / 0.55, 0.0, 1.0)
    functional_evidence = np.clip((100.0 - field[:, 2]) / 82.0, 0.0, 1.0)
    points = np.stack((structural_evidence, functional_evidence), axis=1)
    sampled = points[
        np.linspace(0, points.shape[0] - 1, min(points.shape[0], 256)).astype(np.int64)
    ]

    def fusion(evidence: np.ndarray) -> np.ndarray:
        weight = 0.5 * np.ones(evidence.shape[0])
        return linear_fusion(weight, evidence, evidence * 0.0, gain=FUSION_GAIN)

    lipschitz = estimate_lipschitz(fusion, sampled, radius=1e-4)
    gap = evidence_gap(structural_evidence, functional_evidence)
    flags = identifiable(gap, lipschitz)
    discordant = held_out.discordant
    return {
        "lipschitz": lipschitz,
        "threshold": float("inf") if lipschitz <= 0 else 1.0 / lipschitz,
        "identifiable_share": float(flags.mean()),
        "identifiable_among_discordant": (
            float(flags[discordant].mean()) if bool(discordant.any()) else float("nan")
        ),
        "identifiable_among_concordant": (
            float(flags[~discordant].mean()) if bool((~discordant).any()) else float("nan")
        ),
        "median_gap_discordant": (
            float(np.median(gap[discordant])) if bool(discordant.any()) else float("nan")
        ),
        "median_gap_concordant": (
            float(np.median(gap[~discordant])) if bool((~discordant).any()) else float("nan")
        ),
    }


def excess_risk_analysis(
    development: FlatFeatures, held_out: FlatFeatures, config: ExperimentConfig
) -> dict[str, float]:
    """The two-term excess-risk decomposition, fitted over re-weighting settings.

    Ref: Theorem 1 and its corollary.
    """

    discordant = held_out.discordant
    settings = (0.0, 1.0, 2.0, 4.0)
    errors: list[float] = []
    counts: list[float] = []
    alignments: list[float] = []
    for strength in settings:
        flags = VariantFlags()
        model = TriadModel(config.model, flags)
        loss = reconciliation_loss_function(
            config.train.consistency_weight, float(strength), config.model.stream_weight_margin
        )
        fit_binary_module(
            model, development, config.train, loss_function=loss, seed=config.cohort.seed
        )
        outputs = reconciliation_outputs(model, held_out, config.train)
        predicted = outputs["direction"]
        error = float(np.mean(predicted[discordant] != held_out.arbitration[discordant]))
        weights = 1.0 + float(strength) * outputs["agreement"].astype(np.float64)
        errors.append(error)
        counts.append(effective_pairs(weights[discordant]))
        alignments.append(float(np.mean(np.abs(outputs["stream_weight"][discordant] - 0.5))))
    decomposition: Decomposition = decompose(
        np.asarray(alignments), np.asarray(counts), np.asarray(errors), errors[-1]
    )
    return {
        "settings": float(len(settings)),
        **decomposition.as_mapping(),
        "effective_first": counts[0],
        "effective_last": counts[-1],
    }


def deferral_analysis_for(
    outputs: dict[str, np.ndarray],
    labels: np.ndarray,
    patients: np.ndarray,
    config: ExperimentConfig,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """Calibrated deferral beside the variant that keeps no guarantee."""

    probabilities = np.stack((1.0 - outputs["progression"], outputs["progression"]), axis=1)
    binary_labels = np.asarray(labels, dtype=np.int64)
    calibration, evaluation = split_calibration(
        patients, share=config.cohort.calibration_share, seed=seed
    )
    guarded = deferral_analysis(
        probabilities[calibration],
        binary_labels[calibration],
        probabilities[evaluation],
        binary_labels[evaluation],
        config.inference.nominal_coverage,
    )
    unguarded = unguarded_analysis(probabilities[evaluation], binary_labels[evaluation])
    return guarded, unguarded


def run_study(config: ExperimentConfig, source: Path | None) -> StudyResult:
    """Assemble every reported family of numbers for one experiment file.

    The cohort table is an input, not a default: without one the call raises rather than
    producing numbers from records the release does not hold.
    """

    development, held_out, table = prepare(config, source)
    model = fit_reconciliation(development, config, VariantFlags())
    outputs = reconciliation_outputs(model, held_out, config.train)
    proposed = MethodScore(
        name=roster.PROPOSED,
        probabilities=outputs["progression"],
        direction=outputs["direction"],
        stream_weight=outputs["stream_weight"],
    )
    methods = [*comparator_scores(development, held_out, config), proposed]
    guarded, unguarded = deferral_analysis_for(
        outputs, held_out.progression, held_out.patients, config, config.cohort.seed
    )
    partition = split_regions(table, held_out_region=config.cohort.held_out_region)
    result = StudyResult(
        source=table.source,
        partition=partition,
        stratum_sizes=stratum_sizes(partition.held_out),
        methods=methods,
        variants=[],
        discordance={},
        arbitration={},
        deferral=guarded,
        unguarded=unguarded,
        identifiability={},
        excess_risk={},
        labels=held_out.progression,
        discordant_mask=held_out.discordant,
        arbitration_index=held_out.arbitration,
        patients=held_out.patients,
        epochs=config.train.epochs,
    )
    result.variants = variant_scores(development, held_out, config)
    result.discordance = discordance_analysis(development, held_out, config)
    result.arbitration = arbitration_analysis(outputs, held_out)
    result.identifiability = identifiability_analysis(held_out)
    result.excess_risk = excess_risk_analysis(development, held_out, config)
    return result
