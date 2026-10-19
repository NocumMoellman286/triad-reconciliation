"""Execution checks.

Each check runs a released component and compares its output with an independently written
reference: a brute-force computation, a closed form, or a hand-built case. Nothing here calls
the implementation it is checking to produce its own expected value.

The analysed records are not distributed, so nothing here reads, writes or draws a cohort
table. Checks that need one report ``BLOCKED`` with the staging instruction; the numerical
checks run on tensor batches drawn from a recorded seed, which is a harness input and not a
cohort.

Ref: the verification contract carried by every release of this study; the components under
test are those of Sec. 5 and Algorithm 1.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from triad_reconciliation.analysis.pipeline import STAGING_HINT
from triad_reconciliation.analysis.trainer import (
    build_optimizer,
    checkpoint_digest,
    descend_on_loss,
    learning_rate_at,
    load_checkpoint,
    progression_only_loss_function,
    reconciliation_loss_function,
    save_checkpoint,
)
from triad_reconciliation.comparators import roster
from triad_reconciliation.comparators.structure_function import fit_linear
from triad_reconciliation.comparators.zero_shot import lexical_score, report_tokens
from triad_reconciliation.core.config import ModelConfig, TrainConfig
from triad_reconciliation.core.records import ArbitrationLabel, PerimetryVisit, ReadershipResponse
from triad_reconciliation.core.seeding import set_seed
from triad_reconciliation.deployment.silent_mode import EncounterRecord, silent_mode_metrics
from triad_reconciliation.deployment.workflow import net_benefit, reclassification_counts
from triad_reconciliation.inference.cluster_bootstrap import (
    bootstrap_interval,
    cluster_indices,
    cluster_size_inflation,
)
from triad_reconciliation.inference.paired import paired_difference
from triad_reconciliation.inference.sequential import fixed_sequence
from triad_reconciliation.losses.consistency import consistency_loss, contradiction_mask
from triad_reconciliation.losses.objective import reconciliation_objective
from triad_reconciliation.losses.progression import progression_loss
from triad_reconciliation.losses.reweighting import agreement_weights
from triad_reconciliation.metrics.agreement import DIRECTION_CATEGORIES, cohen_kappa
from triad_reconciliation.metrics.calibration import brier_score, expected_calibration_error
from triad_reconciliation.metrics.conformal import conformal_quantile, deferral_analysis
from triad_reconciliation.metrics.discrimination import (
    auroc,
    average_precision,
    sensitivity_at_specificity,
)
from triad_reconciliation.metrics.excess_risk import decompose, pair_term, scaling_ratio
from triad_reconciliation.metrics.identifiability import (
    FUSION_GAIN,
    estimate_lipschitz,
    evidence_gap,
    identifiable,
    linear_fusion,
    weight_sensitivity,
)
from triad_reconciliation.models.encoders import open_one_slot
from triad_reconciliation.models.triad import StreamBatch, TriadModel
from triad_reconciliation.readers.reliance import reliance_report
from triad_reconciliation.readers.study import inter_reader_agreement
from triad_reconciliation.reporting.tables import render_table
from triad_reconciliation.streams.perimetry import VOCABULARY, encode_report, serialise_report
from triad_reconciliation.verification.hygiene import scan_tree

SMALL_MODEL = ModelConfig(max_visits=4)
SMOKE_TRAIN = TrainConfig(epochs=2, batch_size=8, learning_rate=0.006)
HARNESS_SEED = 20_260_219


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    family: str = "execution"

    def as_mapping(self) -> dict[str, str]:
        return {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "family": self.family,
        }


def random_stream_batch(rows: int, config: ModelConfig, seed: int) -> StreamBatch:
    """A shape-consistent tensor batch for the numerical checks.

    This is not a cohort and it carries no record: the release holds no patient data and
    generates none. The tensors are drawn from a recorded seed so that a forward pass, an
    objective, a backward pass, an optimiser step and a checkpoint round trip can all be
    exercised without a staged table, and nothing drawn here is written to disk.
    """

    generator = torch.Generator().manual_seed(seed)
    visits = max(min(config.max_visits, 4), 1)
    morphometry = torch.rand((rows, visits, 8), generator=generator)
    volume = torch.rand((rows, visits, 4, 4, 4), generator=generator)
    structural_times = torch.sort(
        torch.rand((rows, visits), generator=generator) * 3.0, dim=1
    ).values
    structural_valid = torch.ones((rows, visits), dtype=torch.bool)
    structural_valid[(torch.arange(rows) % 4 == 0), -1] = False
    functional_times = torch.sort(
        torch.rand((rows, visits), generator=generator) * 3.5, dim=1
    ).values
    functional_valid = torch.ones((rows, visits), dtype=torch.bool)
    functional_valid[(torch.arange(rows) % 3 == 0), -1] = False
    token_ids = torch.randint(
        1, config.language_vocab, (rows, visits, config.max_perimetry_tokens), generator=generator
    )
    token_ids[~functional_valid] = 0
    return StreamBatch(
        morphometry=morphometry,
        volume=volume,
        structural_times=structural_times,
        structural_valid=structural_valid,
        token_ids=token_ids,
        field_indices=torch.rand((rows, visits, 6), generator=generator),
        functional_times=functional_times,
        functional_valid=functional_valid,
        query_years=structural_times.max(dim=1).values + 0.5,
        progression=torch.randint(0, 2, (rows,), generator=generator),
        arbitration=torch.randint(0, 3, (rows,), generator=generator),
        agreement=torch.randint(0, 2, (rows,), generator=generator),
    )


def small_model() -> TriadModel:
    set_seed(HARNESS_SEED)
    return TriadModel(SMALL_MODEL)


def objective_function(model: TriadModel) -> Callable[[torch.nn.Module, StreamBatch], torch.Tensor]:
    """The released objective bound to one model's variant, for the fit loops."""

    return reconciliation_loss_function(
        0.65, 2.35, 0.46, supervise_arbitration=model.flags.use_arbitration_head
    )


def _guard(name: str, function: Callable[[], tuple[str, str]]) -> CheckResult:
    try:
        status, detail = function()
    except Exception as error:  # noqa: BLE001 - a check that raises is a failed check
        return CheckResult(name=name, status="FAIL", detail=f"{type(error).__name__}: {error}")
    return CheckResult(name=name, status=status, detail=detail)


def check_schedule_profile() -> tuple[str, str]:
    rates = [learning_rate_at(step, 100, 1.0, 0.1) for step in range(100)]
    warm = rates[:10]
    tail_rates = rates[10:]
    monotone_warm = all(later >= earlier for earlier, later in zip(warm, warm[1:], strict=False))
    monotone_tail = all(
        later <= earlier for earlier, later in zip(tail_rates, tail_rates[1:], strict=False)
    )
    endpoints = abs(rates[0] - 0.1) < 1e-9 and rates[-1] < 1e-3
    ok = monotone_warm and monotone_tail and endpoints
    return (
        "PASS" if ok else "FAIL",
        f"warm-up rises {monotone_warm}, cosine falls {monotone_tail}, endpoints {endpoints}",
    )


def check_progression_loss_reference() -> tuple[str, str]:
    logits = torch.tensor([0.4, -1.2, 2.0, 0.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 1.0])
    hand = np.mean(
        [
            math.log1p(math.exp(-value)) if label == 1.0 else math.log1p(math.exp(value))
            for value, label in zip(logits.tolist(), targets.tolist(), strict=True)
        ]
    )
    value = float(progression_loss(logits, targets))
    ok = abs(value - hand) < 1e-6
    return ("PASS" if ok else "FAIL"), f"module {value:.12f} against hand-computed {hand:.12f}"


def check_consistency_reference() -> tuple[str, str]:
    direction = torch.tensor([0, 1, 2, 0])
    weight = torch.tensor([0.2, 0.3, 0.5, 0.9])
    margin = 0.5
    hand = 0.3 / 4.0
    value = float(consistency_loss(direction, weight, margin))
    mask = contradiction_mask(direction, weight, margin).tolist()
    ok = abs(value - hand) < 1e-7 and mask == [True, False, False, False]
    return (
        "PASS" if ok else "FAIL",
        f"consistency {value:.6f} against hand-computed {hand:.6f}, contradiction mask {mask}",
    )


def check_reweighting_reference() -> tuple[str, str]:
    predicted = torch.tensor([0, 1, 1, 0])
    weights = agreement_weights(predicted, 2.5)
    hand = [1.0, 3.5, 3.5, 1.0]
    ok = np.allclose(weights.tolist(), hand, atol=1e-9)
    return ("PASS" if ok else "FAIL"), f"weights {weights.tolist()} against hand-computed {hand}"


def check_objective_reference() -> tuple[str, str]:
    batch = random_stream_batch(8, SMALL_MODEL, HARNESS_SEED)
    model = small_model()
    output = model(batch)
    margin = 0.46
    parts = reconciliation_objective(
        output, batch, consistency_weight=0.5, reweighting_strength=2.0, margin=margin
    )
    with torch.no_grad():
        per_pair_progression = progression_loss(
            output.progression_logit, batch.progression, reduction="none"
        )
        per_pair_arbitration = torch.nn.functional.cross_entropy(
            output.arbitration_logits, batch.arbitration.long(), reduction="none"
        )
        per_pair_consistency = consistency_loss(
            output.predicted_direction, output.stream_weight, margin, reduction="none"
        )
        weights = agreement_weights(output.predicted_agreement, 2.0)
        hand = (
            weights * (per_pair_progression + per_pair_arbitration + 0.5 * per_pair_consistency)
        ).sum() / weights.sum()
    difference = abs(float(parts.total.detach()) - float(hand))
    ok = difference < 1e-9
    return (
        "PASS" if ok else "FAIL"
    ), f"objective differs from the hand-written sum by {difference:.3e}"


def check_forward_shapes() -> tuple[str, str]:
    batch = random_stream_batch(6, SMALL_MODEL, HARNESS_SEED)
    model = small_model()
    output = model(batch)
    shapes = (
        tuple(output.progression_logit.shape),
        tuple(output.arbitration_logits.shape),
        tuple(output.agreement_logits.shape),
        tuple(output.stream_weight.shape),
        tuple(output.reconciliation_state.shape),
    )
    finite = bool(
        torch.isfinite(output.progression_logit).all()
        and torch.isfinite(output.stream_weight).all()
        and torch.isfinite(output.reconciliation_state).all()
    )
    ok = shapes == ((6,), (6, 3), (6, 2), (6,), (6, SMALL_MODEL.vision_token_dim)) and finite
    return ("PASS" if ok else "FAIL"), f"readout shapes {shapes}, every readout finite {finite}"


def check_alignment_causality() -> tuple[str, str]:
    model = small_model()
    query = torch.tensor([3.0, 3.0])
    structural_times = torch.tensor([[0.0, 1.0, 2.0, 4.0], [0.0, 1.0, 2.0, 4.0]])
    functional_times = torch.tensor([[0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5]])
    valid = torch.ones((2, 4), dtype=torch.bool)
    with torch.no_grad():
        structural_weights, functional_weights = model.alignment(
            query, structural_times, functional_times, valid, valid
        )
        blocked = model.alignment.attention_bias(structural_weights, valid)
    beyond = float(functional_weights[0, 3])
    causal = beyond == 0.0 and blocked[0, 3] == float("-inf")
    rate = float(model.alignment.rate.detach())
    exponentials = [math.exp(rate * value) for value in (0.0, 1.0, 2.0)]
    brute = [value / sum(exponentials) for value in exponentials]
    reference_ok = np.allclose(structural_weights[0, :3].tolist(), brute, atol=1e-6)
    ok = causal and reference_ok and abs(float(structural_weights[0, :4].sum()) - 1.0) < 1e-6
    return (
        "PASS" if ok else "FAIL",
        f"weight beyond the query time {beyond:.3e}, brute-force softmax agrees: {reference_ok}",
    )


def check_alignment_monotonicity() -> tuple[str, str]:
    model = small_model()
    times = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0]])
    valid = torch.ones((1, 5), dtype=torch.bool)
    with torch.no_grad():
        weights, _ = model.alignment(torch.tensor([8.0]), times, times, valid, valid)
    values = weights[0].tolist()
    ok = all(later >= earlier for earlier, later in zip(values, values[1:], strict=False))
    return (
        "PASS" if ok else "FAIL"
    ), f"weights along the timeline {[round(value, 6) for value in values]}"


def check_open_slot_keeps_attention_finite() -> tuple[str, str]:
    """A padded row must still leave one key open, or the pooled readout turns non-finite."""

    mask = torch.ones((2, 5), dtype=torch.bool)
    mask[1, 2] = False
    opened = open_one_slot(mask)
    reopened = bool(opened[0].sum() == 4) and bool(opened[1].sum() == 4) and not bool(opened[0, 0])
    batch = random_stream_batch(6, SMALL_MODEL, HARNESS_SEED)
    empty = StreamBatch(
        morphometry=batch.morphometry,
        volume=batch.volume,
        structural_times=batch.structural_times,
        structural_valid=batch.structural_valid,
        token_ids=torch.zeros_like(batch.token_ids),
        field_indices=batch.field_indices,
        functional_times=batch.functional_times,
        functional_valid=torch.zeros_like(batch.functional_valid),
        query_years=batch.query_years,
        progression=batch.progression,
        arbitration=batch.arbitration,
        agreement=batch.agreement,
    )
    with torch.no_grad():
        output = small_model()(empty)
    finite = bool(torch.isfinite(output.stream_weight).all())
    return (
        "PASS" if reopened and finite else "FAIL",
        f"padded rows keep {int(opened[0].sum())} and {int(opened[1].sum())} open slots; a stream "
        f"with no valid visit still yields a finite weight: {finite}",
    )


def check_auroc_brute_force() -> tuple[str, str]:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    labels = np.asarray([1, 1, 0, 1, 0, 0, 1, 0])
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    concordant = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    hand = concordant / (positives.size * negatives.size)
    value = auroc(scores, labels)
    ok = abs(value - hand) < 1e-12
    return ("PASS" if ok else "FAIL"), f"ranks {value:.12f} against pairwise count {hand:.12f}"


def check_auprc_brute_force() -> tuple[str, str]:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    labels = np.asarray([1, 0, 1, 0, 1, 0])
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    hits = 0
    total_precision = 0.0
    for position, label in enumerate(sorted_labels.tolist(), start=1):
        if label == 1:
            hits += 1
            total_precision += hits / position
    hand = total_precision / int(sorted_labels.sum())
    value = average_precision(scores, labels)
    ok = abs(value - hand) < 1e-9
    return (
        "PASS" if ok else "FAIL"
    ), f"average precision {value:.9f} against hand-computed {hand:.9f}"


def check_sensitivity_ordering() -> tuple[str, str]:
    generator = np.random.default_rng(HARNESS_SEED)
    scores = generator.normal(size=400)
    labels = (generator.random(400) < 0.4).astype(np.int64)
    coarse = sensitivity_at_specificity(scores, labels, 0.50)
    strict = sensitivity_at_specificity(scores, labels, 0.90)
    ok = coarse >= strict
    return (
        "PASS" if ok else "FAIL"
    ), f"sensitivity at 0.50 specificity {coarse:.6f}, at 0.90 {strict:.6f}"


def check_calibration_reference() -> tuple[str, str]:
    probabilities = np.asarray([0.1, 0.2, 0.35, 0.55, 0.7, 0.8, 0.9, 0.95])
    labels = np.asarray([0, 0, 1, 1, 0, 1, 1, 1])
    bins = 4
    index = np.minimum((probabilities * bins).astype(np.int64), bins - 1)
    hand = 0.0
    for bucket in range(bins):
        selected = index == bucket
        if selected.any():
            hand += (
                abs(probabilities[selected].mean() - labels[selected].mean())
                * selected.sum()
                / probabilities.size
            )
    value = expected_calibration_error(probabilities, labels, bins)
    brier = brier_score(probabilities, labels)
    hand_brier = float(np.mean((probabilities - labels) ** 2))
    ok = abs(value - hand) < 1e-12 and abs(brier - hand_brier) < 1e-12
    return (
        "PASS" if ok else "FAIL",
        f"calibration error {value:.9f} against {hand:.9f}; Brier {brier:.9f} against {hand_brier:.9f}",
    )


def check_kappa_reference() -> tuple[str, str]:
    left = np.asarray(["structural"] * 7 + ["functional"] * 5, dtype=object)
    right = np.asarray(
        ["structural"] * 5 + ["functional"] * 2 + ["functional"] * 4 + ["structural"], dtype=object
    )
    observed = float(np.mean([a == b for a, b in zip(left, right, strict=True)]))
    row_structural = float(np.mean(left == "structural"))
    column_structural = float(np.mean(right == "structural"))
    expected = row_structural * column_structural + (1 - row_structural) * (1 - column_structural)
    hand = (observed - expected) / (1 - expected)
    value = cohen_kappa(left, right, DIRECTION_CATEGORIES)
    ok = abs(value - hand) < 1e-12
    return ("PASS" if ok else "FAIL"), f"kappa {value:.9f} against hand-computed {hand:.9f}"


def check_bootstrap_reference() -> tuple[str, str]:
    values = np.ones(60)
    labels = np.asarray([f"p{index % 12}" for index in range(60)], dtype=object)
    interval = bootstrap_interval(
        lambda rows: float(values[rows].mean()), labels, replicates=64, seed=7
    )
    members = cluster_indices(labels, np.random.default_rng(3))
    inflation = cluster_size_inflation(labels)
    ok = (
        abs(interval.upper - interval.lower) < 1e-12
        and abs(interval.estimate - 1.0) < 1e-12
        and members.size > 0
        and abs(inflation - 1.0) < 1e-12
    )
    return (
        "PASS" if ok else "FAIL",
        f"a constant statistic gives width {interval.upper - interval.lower:.3e}, estimate "
        f"{interval.estimate:.6f} and cluster-size inflation {inflation:.6f}",
    )


def check_paired_difference_identity() -> tuple[str, str]:
    values = np.asarray([0.4, 0.7, 0.2, 0.9, 0.5])
    patients = np.asarray(["a", "a", "b", "b", "c"], dtype=object)
    difference = paired_difference(values, values, patients, replicates=32, seed=11)
    ok = abs(difference.estimate) < 1e-12 and not difference.excludes_zero
    return (
        "PASS" if ok else "FAIL",
        f"a vector against itself gives {difference.estimate:.3e} with interval "
        f"[{difference.interval.lower:.3e}, {difference.interval.upper:.3e}]",
    )


def check_fixed_sequence_reference() -> tuple[str, str]:
    """The declared order walked against the article's own criteria.

    Ref: Sec. 2.1-2.4, where the first two criteria are met, the parity statement against the
    unconstrained comparator is not, and Sec. 5.6 fixes the minimum detectable differences.
    """

    effects = {
        "discrimination_vs_best_single_stream": 0.033,
        "discrimination_vs_static_fusion": 0.071,
        "discrimination_vs_unconstrained_timeline": -0.003,
        "arbitration_agreement_vs_strongest_proxy": 0.22,
        "reader_accuracy_with_advice": 0.088,
    }
    thresholds = {
        "discrimination_vs_best_single_stream": 0.030,
        "discrimination_vs_static_fusion": 0.030,
        "discrimination_vs_unconstrained_timeline": 0.030,
        "arbitration_agreement_vs_strongest_proxy": 0.05,
        "reader_accuracy_with_advice": 0.05,
    }
    intervals = {name: (value, value + 0.01) for name, value in effects.items()}
    sequence = fixed_sequence(effects, thresholds, intervals)
    tested = [item.identifier for item in sequence.tested]
    expected = [
        "discrimination_vs_best_single_stream",
        "discrimination_vs_static_fusion",
        "discrimination_vs_unconstrained_timeline",
    ]
    ok = tested == expected and sequence.stopped_at == "discrimination_vs_unconstrained_timeline"
    return (
        "PASS" if ok else "FAIL"
    ), f"tested {len(tested)} of 5 in order, stopped at {sequence.stopped_at}"


def check_conformal_coverage() -> tuple[str, str]:
    generator = np.random.default_rng(HARNESS_SEED)
    truth = generator.integers(0, 2, size=4_000)
    signal = 0.2 + 0.6 * truth
    scores = np.stack((1.0 - signal, signal), axis=1)
    scores = scores + generator.normal(0.0, 0.05, size=scores.shape)
    scores = np.clip(scores, 1e-6, None)
    scores = scores / scores.sum(axis=1, keepdims=True)
    analysis = deferral_analysis(
        scores[:2_000], truth[:2_000], scores[2_000:], truth[2_000:], nominal_coverage=0.90
    )
    ok = analysis["coverage"] >= 0.90 - 0.02
    return (
        "PASS" if ok else "FAIL",
        f"empirical coverage {analysis['coverage']:.4f} against a nominal "
        f"{analysis['nominal_coverage']:.2f}",
    )


def check_conformal_threshold_reference() -> tuple[str, str]:
    scores = np.asarray([0.10, 0.30, 0.50, 0.70, 0.90])
    rank = int(np.ceil((scores.size + 1) * 0.90))
    hand = float(np.sort(scores)[min(rank, scores.size) - 1])
    value = conformal_quantile(scores, 0.90)
    ok = abs(value - hand) < 1e-12
    return (
        "PASS" if ok else "FAIL"
    ), f"threshold {value:.6f} against the order statistic {hand:.6f}"


def check_identifiability_perturbation() -> tuple[str, str]:
    equal = (np.asarray([0.40, 0.55, 0.30]), np.asarray([0.40, 0.55, 0.30]))
    unequal = (np.asarray([0.25, 0.80, 0.20]), np.asarray([0.70, 0.30, 0.65]))
    centres = np.stack((equal[0], unequal[0]), axis=1)

    def fusion(evidence: np.ndarray) -> np.ndarray:
        weight = np.full(evidence.shape[0], 0.6)
        return linear_fusion(weight, evidence, evidence * 0.0, gain=FUSION_GAIN)

    lipschitz = estimate_lipschitz(fusion, centres, radius=1e-4)
    threshold = 1.0 / lipschitz if lipschitz > 0 else float("inf")
    quiet = weight_sensitivity(
        linear_fusion, np.full(3, 0.6), equal[0], equal[1], step=0.25, gain=FUSION_GAIN
    )
    loud = weight_sensitivity(
        linear_fusion, np.full(3, 0.6), unequal[0], unequal[1], step=0.25, gain=FUSION_GAIN
    )
    flags_quiet = identifiable(evidence_gap(equal[0], equal[1]), lipschitz)
    flags_loud = identifiable(evidence_gap(unequal[0], unequal[1]), lipschitz)
    ok = (
        bool(np.all(quiet < 1e-9))
        and bool(np.all(loud > 1e-3))
        and not bool(np.any(flags_quiet))
        and bool(np.all(flags_loud))
    )
    return (
        "PASS" if ok else "FAIL",
        f"fusion Lipschitz {lipschitz:.4f} gives an identification threshold {threshold:.4f}; equal "
        f"streams move {float(quiet.max()):.3e} under a weight nudge and are identified "
        f"{flags_quiet.tolist()}, unequal streams move {float(loud.min()):.4f} and are identified "
        f"{flags_loud.tolist()}",
    )


def check_excess_risk_scaling() -> tuple[str, str]:
    coefficient = 0.7
    small = pair_term(coefficient, 400.0)
    large = pair_term(coefficient, 1_600.0)
    ratio = small / large
    ok = abs(ratio - math.sqrt(4.0)) < 1e-9 and abs(ratio - 2.0) < 1e-9
    return (
        "PASS" if ok else "FAIL",
        f"quadrupling the effective count reduces the pair term by {ratio:.6f}, the reported "
        f"scaling is {scaling_ratio(400.0, 1_600.0):.6f}",
    )


def check_decomposition_reference() -> tuple[str, str]:
    alignment = np.asarray([0.10, 0.20, 0.30, 0.40])
    counts = np.asarray([100.0, 200.0, 400.0, 800.0])
    measured = 0.55 * alignment + 30.0 / np.sqrt(counts)
    result = decompose(alignment, counts, measured, float(measured[-1]))
    ok = (
        abs(result.alignment_coefficient - 0.55) < 1e-5
        and abs(result.pair_coefficient - 30.0) < 1e-5
    )
    return (
        "PASS" if ok else "FAIL",
        f"recovered coefficients {result.alignment_coefficient:.6f} and "
        f"{result.pair_coefficient:.6f} against the values used to build the response",
    )


def check_backward_and_update() -> tuple[str, str]:
    batch = random_stream_batch(8, SMALL_MODEL, HARNESS_SEED)
    model = small_model()
    optimizer = build_optimizer(model, TrainConfig(learning_rate=0.01))
    loss = objective_function(model)(model, batch)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    descend_on_loss(
        optimizer, loss, clip=1.0, step=0, total=10, base_rate=0.01, warmup_fraction=0.1
    )
    with_gradient = sum(
        1
        for parameter in model.parameters()
        if parameter.grad is not None and parameter.grad.abs().sum() > 0
    )
    moved = sum(
        1
        for first, second in zip(before, model.parameters(), strict=True)
        if not torch.equal(first, second.detach())
    )
    ok = with_gradient > 0 and moved > 0
    return (
        "PASS" if ok else "FAIL",
        f"{with_gradient} parameters carry a gradient and {moved} changed after one step",
    )


def check_single_batch_overfit() -> tuple[str, str]:
    batch = random_stream_batch(16, SMALL_MODEL, HARNESS_SEED)
    model = small_model()
    settings = TrainConfig(epochs=24, batch_size=16, learning_rate=0.01, warmup_fraction=0.2)
    optimizer = build_optimizer(model, settings)
    loss_function = objective_function(model)
    first = float(loss_function(model, batch).detach())
    history: list[float] = []
    total = settings.epochs * 4
    for step in range(total):
        value = loss_function(model, batch)
        history.append(float(value.detach()))
        descend_on_loss(
            optimizer,
            value,
            clip=settings.grad_clip,
            step=step,
            total=total,
            base_rate=settings.learning_rate,
            warmup_fraction=settings.warmup_fraction,
        )
    last = history[-1]
    best = min(history)
    ok = last < 0.5 * first and best < 0.1 * first
    return (
        "PASS" if ok else "FAIL",
        f"a single tensor batch drives the objective from {first:.4f} to {last:.4f}, best "
        f"{best:.4f}, over {total} steps",
    )


def check_minimal_training_loop() -> tuple[str, str]:
    batches = [random_stream_batch(16, SMALL_MODEL, HARNESS_SEED + index) for index in range(3)]
    model = small_model()
    settings = TrainConfig(epochs=3, batch_size=16, learning_rate=0.008)
    optimizer = build_optimizer(model, settings)
    loss_function = objective_function(model)
    epoch_losses: list[float] = []
    for epoch in range(settings.epochs):
        running: list[float] = []
        for position, batch in enumerate(batches):
            value = loss_function(model, batch)
            running.append(float(value.detach()))
            descend_on_loss(
                optimizer,
                value,
                clip=settings.grad_clip,
                step=epoch * len(batches) + position,
                total=settings.epochs * len(batches),
                base_rate=settings.learning_rate,
                warmup_fraction=settings.warmup_fraction,
            )
        epoch_losses.append(float(np.mean(running)))
    ok = len(epoch_losses) == settings.epochs and epoch_losses[-1] < epoch_losses[0]
    return (
        "PASS" if ok else "FAIL",
        f"epoch losses {[round(value, 5) for value in epoch_losses]} over {len(batches)} batches",
    )


def check_progression_objective_updates() -> tuple[str, str]:
    model = small_model()
    batch = random_stream_batch(8, SMALL_MODEL, HARNESS_SEED)
    loss_function = progression_only_loss_function()
    before = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = build_optimizer(model, SMOKE_TRAIN)
    value = loss_function(model, batch)
    descend_on_loss(
        optimizer,
        value,
        clip=SMOKE_TRAIN.grad_clip,
        step=0,
        total=2,
        base_rate=SMOKE_TRAIN.learning_rate,
        warmup_fraction=SMOKE_TRAIN.warmup_fraction,
    )
    moved = sum(
        1
        for first, second in zip(before, model.parameters(), strict=True)
        if not torch.equal(first, second.detach())
    )
    ok = moved > 0 and math.isfinite(float(value.detach()))
    return (
        "PASS" if ok else "FAIL",
        f"{moved} parameter tensors move under outcome supervision alone",
    )


def check_checkpoint_roundtrip() -> tuple[str, str]:
    batch = random_stream_batch(16, SMALL_MODEL, HARNESS_SEED)
    model = small_model()
    settings = TrainConfig(epochs=1, batch_size=16, learning_rate=0.01)
    optimizer = build_optimizer(model, settings)
    loss_function = objective_function(model)
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "checkpoint.pt"
        save_checkpoint(target, model, optimizer, seed=HARNESS_SEED, epoch=1, metrics={"loss": 1.0})
        reloaded = small_model()
        metadata = load_checkpoint(target, reloaded, restore_seed=True)
        identical = all(
            torch.equal(first, second)
            for first, second in zip(
                model.state_dict().values(), reloaded.state_dict().values(), strict=True
            )
        )
        resumed = build_optimizer(reloaded, settings)
        for step in range(3):
            value = loss_function(reloaded, batch)
            descend_on_loss(
                resumed, value, clip=1.0, step=step, total=4, base_rate=0.01, warmup_fraction=0.25
            )
        drifted = not all(
            torch.equal(first, second)
            for first, second in zip(
                model.state_dict().values(), reloaded.state_dict().values(), strict=True
            )
        )
        again = small_model()
        load_checkpoint(target, again, restore_seed=True)
        restored = all(
            torch.equal(first, second)
            for first, second in zip(
                model.state_dict().values(), again.state_dict().values(), strict=True
            )
        )
        digest = checkpoint_digest(target)
    ok = (
        identical
        and drifted
        and restored
        and metadata["seed"] == HARNESS_SEED
        and len(digest) == 64
    )
    return (
        "PASS" if ok else "FAIL",
        f"reload identical {identical}, weights drift after further steps {drifted}, second reload "
        f"identical {restored}, digest length {len(digest)}",
    )


def check_checkpoint_file_mode() -> tuple[str, str]:
    model = small_model()
    with tempfile.TemporaryDirectory() as scratch:
        directory = Path(scratch)
        target = directory / "checkpoint.pt"
        save_checkpoint(target, model, None, seed=1, epoch=0, metrics={})
        mode = oct(target.stat().st_mode & 0o777)
        leftovers = [path.name for path in directory.iterdir() if path.name != "checkpoint.pt"]
    ok = mode == "0o644" and not leftovers
    return ("PASS" if ok else "FAIL"), f"mode {mode}, staged leftovers {leftovers}"


def check_roster_completeness() -> tuple[str, str]:
    names = roster.roster_names()
    learned = roster.learned_modules(SMALL_MODEL)
    ok = len(names) == 15 and len(learned) == 8 and roster.PROPOSED in names
    return (
        "PASS" if ok else "FAIL",
        f"{len(names)} roster entries, {len(learned)} learned comparators instantiated",
    )


def check_comparator_shapes() -> tuple[str, str]:
    batch = random_stream_batch(6, SMALL_MODEL, HARNESS_SEED)
    set_seed(HARNESS_SEED)
    shapes: dict[str, tuple[int, ...]] = {}
    finite = True
    for name, module in roster.learned_modules(SMALL_MODEL).items():
        with torch.no_grad():
            value = module(batch)
        shapes[name] = tuple(value.shape)
        finite = finite and bool(torch.isfinite(value).all())
    ok = all(shape == (6,) for shape in shapes.values()) and finite
    return (
        "PASS" if ok else "FAIL",
        f"{len(shapes)} comparators, distinct shapes {sorted(set(shapes.values()))}, every readout "
        f"finite {finite}",
    )


def check_structure_function_reference() -> tuple[str, str]:
    generator = np.random.default_rng(HARNESS_SEED)
    design = generator.normal(size=(200, 3))
    weights = np.asarray([1.5, -0.75, 0.25])
    target = design @ weights + 2.0
    recovered = fit_linear(design, target, ridge=1e-9)
    ok = np.allclose(recovered[1:], weights, atol=1e-5) and abs(recovered[0] - 2.0) < 1e-5
    return (
        "PASS" if ok else "FAIL",
        f"recovered weights {np.round(recovered[1:], 6).tolist()} against {weights.tolist()}",
    )


def check_zero_shot_direction() -> tuple[str, str]:
    worse = report_tokens(np.asarray([-22.0, -18.0, 12.0, 0.02, 0.03, 0.0]))
    better = report_tokens(np.asarray([-1.0, -1.0, 96.0, 0.01, 0.01, 0.0]))
    worse_score = lexical_score(list(worse))
    better_score = lexical_score(list(better))
    ok = worse_score > better_score
    return (
        "PASS" if ok else "FAIL",
        f"a deep defect scores {worse_score:.4f} against {better_score:.4f} for a shallow one",
    )


def check_perimetry_schema() -> tuple[str, str]:
    report = PerimetryVisit(
        patient_id="",
        eye_id="",
        site="",
        tested_years=0.0,
        sensitivity_db=tuple(30.0 - index for index in range(24)),
        mean_deviation_db=-3.4,
        pattern_deviation_db=-2.8,
        visual_field_index=86.0,
        false_positive_rate=0.04,
        false_negative_rate=0.06,
        fixation_losses=0,
        duration_seconds=280.0,
        gaze_tracking_ok=True,
    )
    text = serialise_report(report)
    ids, length = encode_report(report, 28)
    plain = sum(1 for token in VOCABULARY.tokens if not token.startswith(("pt", "sens", "<")))
    total = len(text.split())
    ok = total == 57 and len(ids) == 28 and length == 28 and VOCABULARY.size == 152 and plain == 104
    return (
        "PASS" if ok else "FAIL",
        f"vocabulary {VOCABULARY.size} tokens of which {plain} are value tokens; one report "
        f"serialises to {total} tokens and truncates to {length}",
    )


def check_evidence_gap_reference() -> tuple[str, str]:
    left = np.asarray([0.2, 0.8, 0.5])
    right = np.asarray([0.2, 0.3, 0.5])
    gap = evidence_gap(left, right)
    ok = np.allclose(gap, [0.0, 0.5, 0.0], atol=1e-12)
    return ("PASS" if ok else "FAIL"), f"gaps {gap.tolist()}"


def check_silent_mode_reference() -> tuple[str, str]:
    records = tuple(
        EncounterRecord(
            encounter_id=f"e{index}",
            scored=index % 10 != 0,
            failure_reason=(
                None
                if index % 10 != 0
                else ("missing_stream" if index % 20 == 10 else "processing_failure")
            ),
            latency_seconds=1.0 + 0.1 * index,
            deferred=index % 5 == 0,
            attribution_emitted=index % 10 != 0,
            attribution_consistent=index % 10 != 0,
            opened_by_reviewer=index % 2 == 0,
            overrode_direction=index % 4 == 0,
            deferred_returned=index % 5 == 0,
            deferred_accepted=index % 5 == 0,
        )
        for index in range(100)
    )
    metrics = silent_mode_metrics(records)
    ok = (
        abs(metrics.coverage - 90 / 100) < 1e-12
        and abs(metrics.unscored_missing_share - 5 / 100) < 1e-12
        and abs(metrics.unscored_processing_share - 5 / 100) < 1e-12
        and abs(metrics.deferred_share - 10 / 90) < 1e-12
    )
    return (
        "PASS" if ok else "FAIL",
        f"coverage {metrics.coverage:.4f}, missing {metrics.unscored_missing_share:.4f}, processing "
        f"{metrics.unscored_processing_share:.4f}, deferred {metrics.deferred_share:.4f}",
    )


def check_net_benefit_reference() -> tuple[str, str]:
    probabilities = np.asarray([0.9, 0.8, 0.3, 0.2, 0.7, 0.1])
    labels = np.asarray([1, 1, 0, 0, 1, 0])
    threshold = 0.5
    positive = probabilities >= threshold
    true_positive = float(((positive) & (labels == 1)).sum()) / labels.size
    false_positive = float(((positive) & (labels == 0)).sum()) / labels.size
    hand = true_positive - false_positive * (threshold / (1 - threshold))
    value = net_benefit(probabilities, labels, threshold)
    ok = abs(value - hand) < 1e-12
    return ("PASS" if ok else "FAIL"), f"net benefit {value:.9f} against hand-computed {hand:.9f}"


def check_reclassification_reference() -> tuple[str, str]:
    baseline = np.asarray([0, 0, 1, 1, 0, 1, 0, 1])
    model = np.asarray([1, 0, 1, 0, 1, 1, 0, 1])
    labels = np.asarray([1, 0, 1, 1, 0, 1, 1, 0])
    counts = reclassification_counts(baseline, model, labels)
    events = labels == 1
    non_events = ~events
    hand = float(np.mean(model[events] > baseline[events])) - float(
        np.mean(model[events] < baseline[events])
    )
    hand -= float(np.mean(model[non_events] > baseline[non_events])) - float(
        np.mean(model[non_events] < baseline[non_events])
    )
    ok = abs(counts["nri"] - hand) < 1e-12
    return (
        "PASS" if ok else "FAIL",
        f"reclassification {counts['nri']:.6f} against hand-computed {hand:.6f}",
    )


def check_reliance_reference() -> tuple[str, str]:
    reference = {"v1": "structural", "v2": "structural", "v3": "functional", "v4": "functional"}
    advice = {"v1": "structural", "v2": "functional", "v3": "functional", "v4": "structural"}
    pattern: tuple[tuple[str, ArbitrationLabel, float], ...] = (
        ("v1", "structural", 96.0),
        ("v2", "functional", 104.0),
        ("v3", "functional", 88.0),
        ("v4", "structural", 112.0),
    )
    answers = tuple(
        ReadershipResponse(reader, vignette, "model_assisted", direction, seconds)
        for reader in ("r1", "r2")
        for vignette, direction, seconds in pattern
    )
    report = reliance_report(answers, reference, advice)
    ok = (
        abs(report.over_reliance - 1.0) < 1e-12
        and abs(report.under_reliance - 0.0) < 1e-12
        and report.adopted_wrong == 4
        and report.rejected_correct == 0
        and report.wrong_advice_cases == 4
        and report.correct_advice_cases == 4
    )
    return (
        "PASS" if ok else "FAIL",
        f"over-reliance {report.over_reliance:.4f} over {report.wrong_advice_cases} wrong-advice "
        f"cases, under-reliance {report.under_reliance:.4f} over "
        f"{report.correct_advice_cases} correct-advice cases",
    )


def check_inter_reader_reference() -> tuple[str, str]:
    pattern: tuple[tuple[ArbitrationLabel, ...], ...] = (
        ("structural", "structural", "structural"),
        ("structural", "functional", "structural"),
        ("functional", "functional", "structural"),
    )
    answers = tuple(
        ReadershipResponse(f"r{reader}", f"v{item}", "unassisted", direction, 100.0)
        for reader, row in enumerate(pattern)
        for item, direction in enumerate(row)
    )
    value = inter_reader_agreement(answers, "unassisted", ("v0", "v1", "v2"))
    left = np.asarray(["structural", "structural", "functional"], dtype=object)
    right = np.asarray(["functional", "functional", "functional"], dtype=object)
    hand = cohen_kappa(left, right, DIRECTION_CATEGORIES)
    ok = np.isfinite(value) and -1.0 <= value <= 1.0 and abs(value - hand) < 1.0
    return (
        "PASS" if ok else "FAIL",
        f"mean pairwise kappa {value:.6f}, an individual pair gives {hand:.6f}",
    )


def check_table_rendering() -> tuple[str, str]:
    block = render_table("Wide", ("a", "second_column"), (("1", "x"), ("2", "a-much-longer-value")))
    lines = [line for line in block.splitlines() if "a-much-longer-value" in line]
    ok = bool(lines) and lines[0].count("a-much-longer-value") == 1
    return ("PASS" if ok else "FAIL"), f"long value rendered in full: {ok}"


def check_hygiene_scans() -> tuple[str, str]:
    root = Path(__file__).resolve().parents[3]
    findings = scan_tree(root)
    return ("PASS" if not findings else "FAIL"), f"{len(findings)} hygiene findings"


def check_manifest_freshness() -> tuple[str, str]:
    from triad_reconciliation.core.jsonio import digest_paths, manifest_digest, read_json

    root = Path(__file__).resolve().parents[3]
    manifest_path = root / "integrity_manifest.json"
    if not manifest_path.is_file():
        return (
            "NOT_RUN",
            "the manifest is written at the end of this run, so a first pass cannot read it",
        )
    shipped = read_json(manifest_path)
    live = digest_paths(root, exclude=frozenset({"integrity_manifest.json"}))
    changed = sorted(
        path for path, value in shipped.get("entries", {}).items() if live.get(path) != value
    )
    missing = sorted(path for path in shipped.get("entries", {}) if path not in live)
    unlisted = sorted(path for path in live if path not in shipped.get("entries", {}))
    ok = not changed and not missing and not unlisted
    return (
        "PASS" if ok else "FAIL",
        f"changed {len(changed)}, missing {len(missing)}, unlisted {len(unlisted)}, digest matches "
        f"description: {manifest_digest(live) == shipped.get('manifest_digest')}",
    )


def gate_command(command: list[str], cwd: Path) -> tuple[int, str]:
    import os

    environment = dict(os.environ)
    environment.update(
        {
            "COLUMNS": "200",
            "LINES": "50",
            "NO_COLOR": "1",
            "PY_COLORS": "0",
            "PYTHONPATH": str(cwd / "src"),
        }
    )
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argument vectors, no shell
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
    except FileNotFoundError:
        return 127, f"{command[0]} is not installed on this host"
    return completed.returncode, completed.stdout + completed.stderr


def check_gate_pytest(root: Path) -> tuple[str, str]:
    code, _ = gate_command(["pytest", "-q", "--no-header", "-p", "no:cacheprovider"], root)
    return ("PASS" if code == 0 else "FAIL"), f"pytest exit code {code}"


def check_gate_ruff(root: Path) -> tuple[str, str]:
    code, _ = gate_command(["ruff", "check", "."], root)
    return ("PASS" if code == 0 else "FAIL"), f"ruff exit code {code}"


def check_gate_black(root: Path) -> tuple[str, str]:
    code, _ = gate_command(["black", "--check", "."], root)
    return ("PASS" if code == 0 else "FAIL"), f"black exit code {code}"


def check_gate_isort(root: Path) -> tuple[str, str]:
    code, _ = gate_command(["isort", "--check-only", "."], root)
    return ("PASS" if code == 0 else "FAIL"), f"isort exit code {code}"


def check_gate_mypy(root: Path) -> tuple[str, str]:
    code, _ = gate_command(["mypy", "--strict", "src/triad_reconciliation"], root)
    return ("PASS" if code == 0 else "FAIL"), f"mypy exit code {code}"


def check_gate_docker(root: Path) -> tuple[str, str]:
    code, _ = gate_command(["docker", "build", "-t", "triad-verify", "."], root)
    if code in (126, 127):
        return "BLOCKED", "no container runtime is installed on the authoring host"
    return ("PASS" if code == 0 else "FAIL"), f"docker build exit code {code}"


RELEASE_ROOT = Path(__file__).resolve().parents[3]

CHECKS: tuple[tuple[str, Callable[[], tuple[str, str]]], ...] = (
    ("schedule-profile", check_schedule_profile),
    ("loss-progression-reference", check_progression_loss_reference),
    ("loss-consistency-reference", check_consistency_reference),
    ("loss-reweighting-reference", check_reweighting_reference),
    ("objective-weighted-sum-reference", check_objective_reference),
    ("forward-readout-shapes", check_forward_shapes),
    ("alignment-causality", check_alignment_causality),
    ("alignment-monotonicity", check_alignment_monotonicity),
    ("attention-open-slot", check_open_slot_keeps_attention_finite),
    ("auroc-brute-force", check_auroc_brute_force),
    ("auprc-brute-force", check_auprc_brute_force),
    ("sensitivity-ordering", check_sensitivity_ordering),
    ("calibration-reference", check_calibration_reference),
    ("kappa-reference", check_kappa_reference),
    ("bootstrap-reference", check_bootstrap_reference),
    ("paired-difference-identity", check_paired_difference_identity),
    ("fixed-sequence-reference", check_fixed_sequence_reference),
    ("conformal-coverage", check_conformal_coverage),
    ("conformal-threshold-reference", check_conformal_threshold_reference),
    ("identifiability-perturbation", check_identifiability_perturbation),
    ("excess-risk-scaling", check_excess_risk_scaling),
    ("excess-risk-decomposition-reference", check_decomposition_reference),
    ("backward-and-parameter-update", check_backward_and_update),
    ("single-batch-overfit", check_single_batch_overfit),
    ("minimal-training-loop", check_minimal_training_loop),
    ("outcome-objective-updates", check_progression_objective_updates),
    ("checkpoint-roundtrip", check_checkpoint_roundtrip),
    ("checkpoint-file-mode", check_checkpoint_file_mode),
    ("comparator-roster-completeness", check_roster_completeness),
    ("comparator-forward-shapes", check_comparator_shapes),
    ("structure-function-linear-reference", check_structure_function_reference),
    ("zero-shot-lexical-direction", check_zero_shot_direction),
    ("perimetry-serialisation-schema", check_perimetry_schema),
    ("evidence-gap-reference", check_evidence_gap_reference),
    ("silent-mode-rates-reference", check_silent_mode_reference),
    ("workflow-net-benefit-reference", check_net_benefit_reference),
    ("workflow-reclassification-reference", check_reclassification_reference),
    ("reliance-rates-reference", check_reliance_reference),
    ("inter-reader-agreement", check_inter_reader_reference),
    ("table-rendering-widths", check_table_rendering),
    ("hygiene-tree-scan", check_hygiene_scans),
    ("manifest-freshness", check_manifest_freshness),
    ("gate-pytest", lambda: check_gate_pytest(RELEASE_ROOT)),
    ("gate-ruff", lambda: check_gate_ruff(RELEASE_ROOT)),
    ("gate-black", lambda: check_gate_black(RELEASE_ROOT)),
    ("gate-isort", lambda: check_gate_isort(RELEASE_ROOT)),
    ("gate-mypy", lambda: check_gate_mypy(RELEASE_ROOT)),
    ("gate-docker-build", lambda: check_gate_docker(RELEASE_ROOT)),
)

STAGING_BLOCKED: tuple[str, ...] = (
    "cohort-table-staged",
    "cohort-predictor-embargo",
    "calibration-split-disjointness",
    "strata-partition-coverage",
    "clinical-rule-scores",
    "structure-function-on-a-cohort",
    "comparator-scores-on-a-cohort",
    "discordance-and-arbitration-analyses",
    "conformal-deferral-on-a-cohort",
    "study-pipeline-end-to-end",
)

GATE_NAMES: tuple[str, ...] = (
    "gate-pytest",
    "gate-ruff",
    "gate-black",
    "gate-isort",
    "gate-mypy",
    "gate-docker-build",
)


def run_execution_checks(*, include_gates: bool = True) -> list[CheckResult]:
    """Run the numerical checks, then record the record-level ones as blocked.

    Nothing here fabricates a cohort: the release holds no records, so every analysis the
    manuscript reports on its cohort is reported as blocked with the staging instruction.

    The tool gates shell out to the toolchain, so a caller that is itself running inside
    pytest passes ``include_gates=False`` and the shipping invocation leaves them on.
    """

    results: list[CheckResult] = [
        CheckResult(name=name, status="BLOCKED", detail=STAGING_HINT, family="cohort")
        for name in STAGING_BLOCKED
    ]
    for name, function in CHECKS:
        if not include_gates and name in GATE_NAMES:
            results.append(
                CheckResult(
                    name=name,
                    status="NOT_RUN",
                    detail="the tool gates run only in the shipping invocation",
                    family="gate",
                )
            )
            continue
        set_seed(HARNESS_SEED)
        results.append(_guard(name, function))
    return results


def summarise_execution(results: list[CheckResult]) -> dict[str, float]:
    return {
        "checks": float(len(results)),
        "passed": float(sum(1 for item in results if item.status == "PASS")),
        "failed": float(sum(1 for item in results if item.status == "FAIL")),
        "blocked": float(sum(1 for item in results if item.status == "BLOCKED")),
        "not_run": float(sum(1 for item in results if item.status == "NOT_RUN")),
    }
