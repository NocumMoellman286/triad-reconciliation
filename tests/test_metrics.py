"""Discrimination, calibration, agreement, conformal, excess-risk and identifiability."""

from __future__ import annotations

import math

import numpy as np
import pytest

from triad_reconciliation.metrics.agreement import (
    DIRECTION_CATEGORIES,
    agreement_share,
    cohen_kappa,
    confusion,
    linear_weights,
    mean_pairwise_kappa,
    weighted_kappa,
)
from triad_reconciliation.metrics.calibration import (
    brier_score,
    calibration_slope_intercept,
    expected_calibration_error,
    reliability_curve,
)
from triad_reconciliation.metrics.conformal import (
    conformal_quantile,
    decide_threshold,
    decided_accuracy,
    deferral_analysis,
    deferral_mask,
    direction_sets,
    empirical_coverage,
    unguarded_analysis,
)
from triad_reconciliation.metrics.discrimination import (
    auroc,
    average_precision,
    partial_auc,
    roc_curve,
    sensitivity_at_specificity,
)
from triad_reconciliation.metrics.excess_risk import (
    alignment_error,
    decompose,
    effective_pairs,
    excess_risk,
    fit_decomposition,
    pair_term,
    scaling_ratio,
)
from triad_reconciliation.metrics.identifiability import (
    FUSION_GAIN,
    estimate_lipschitz,
    evidence_gap,
    fusion_threshold,
    identifiable,
    linear_fusion,
    separation_of_weight,
    weight_sensitivity,
)


def test_auroc_matches_a_pairwise_count() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    labels = np.asarray([1, 1, 0, 1, 0, 0])
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    hand = sum(
        1.0 if first > second else 0.5 if first == second else 0.0
        for first in positives
        for second in negatives
    ) / (positives.size * negatives.size)
    assert auroc(scores, labels) == pytest.approx(hand, abs=1e-12)


def test_auroc_handles_ties_and_degenerate_labels() -> None:
    assert auroc(np.asarray([0.5, 0.5, 0.5, 0.5]), np.asarray([1, 0, 1, 0])) == pytest.approx(0.5)
    assert math.isnan(auroc(np.asarray([0.5, 0.5]), np.asarray([1, 1])))


def test_average_precision_matches_a_step_sum() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.6])
    labels = np.asarray([1, 0, 1, 1])
    hand = (1 / 1 + 2 / 3 + 3 / 4) / 3
    assert average_precision(scores, labels) == pytest.approx(hand, abs=1e-12)


def test_roc_curve_is_monotone_in_the_cut_point() -> None:
    scores = np.asarray([0.9, 0.7, 0.4, 0.2, 0.1])
    labels = np.asarray([1, 0, 1, 0, 1])
    thresholds, sensitivity, specificity = roc_curve(scores, labels)
    assert len(thresholds) == len(sensitivity) == len(specificity)
    assert all(
        later >= earlier for earlier, later in zip(sensitivity, sensitivity[1:], strict=False)
    )
    assert sensitivity[-1] == pytest.approx(1.0)


def test_sensitivity_at_specificity_moves_the_right_way() -> None:
    generator = np.random.default_rng(4)
    scores = generator.normal(size=300)
    labels = (generator.random(300) < 0.4).astype(np.int64)
    assert sensitivity_at_specificity(scores, labels, 0.5) >= sensitivity_at_specificity(
        scores, labels, 0.9
    )
    assert math.isnan(sensitivity_at_specificity(scores, labels, 1.5))


def test_partial_auc_is_bounded() -> None:
    generator = np.random.default_rng(9)
    scores = generator.normal(size=200)
    labels = (generator.random(200) < 0.5).astype(np.int64)
    value = partial_auc(scores, labels, 0.0, 0.3)
    assert 0.0 <= value <= 1.0


def test_calibration_error_matches_a_hand_binning() -> None:
    probabilities = np.asarray([0.05, 0.2, 0.45, 0.65, 0.85, 0.95])
    labels = np.asarray([0, 0, 1, 1, 1, 1])
    bins = 3
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
    assert expected_calibration_error(probabilities, labels, bins) == pytest.approx(hand, abs=1e-12)


def test_brier_and_reliability_curve() -> None:
    probabilities = np.asarray([0.1, 0.6, 0.9])
    labels = np.asarray([0, 1, 1])
    assert brier_score(probabilities, labels) == pytest.approx(
        float(np.mean((probabilities - labels) ** 2))
    )
    curve = reliability_curve(probabilities, labels, 4)
    assert curve.shape == (4, 4)
    assert int(curve[:, 3].sum()) == probabilities.size


def test_calibration_slope_recovers_a_well_specified_forecast() -> None:
    generator = np.random.default_rng(11)
    truth = generator.random(3000)
    labels = (generator.random(3000) < truth).astype(np.float64)
    slope, intercept = calibration_slope_intercept(truth, labels)
    assert abs(slope - 1.0) < 0.35
    assert abs(intercept) < 0.25


def test_confusion_and_kappa_match_a_hand_computation() -> None:
    left = np.asarray(["structural", "structural", "functional", "functional"], dtype=object)
    right = np.asarray(["structural", "functional", "functional", "functional"], dtype=object)
    table = confusion(left, right, DIRECTION_CATEGORIES)
    assert float(table.sum()) == 4.0
    observed = float(np.trace(table)) / 4.0
    rows = table.sum(axis=1) / 4.0
    columns = table.sum(axis=0) / 4.0
    hand = (observed - float(rows @ columns)) / (1.0 - float(rows @ columns))
    assert cohen_kappa(left, right, DIRECTION_CATEGORIES) == pytest.approx(hand, abs=1e-12)


def test_kappa_on_a_chance_labelling_is_near_zero() -> None:
    generator = np.random.default_rng(13)
    left = generator.choice(["structural", "functional"], size=4000)
    right = generator.choice(["structural", "functional"], size=4000)
    assert abs(cohen_kappa(left, right, DIRECTION_CATEGORIES)) < 0.06
    assert agreement_share(left, right) == pytest.approx(float(np.mean(left == right)))


def test_weighted_kappa_and_mean_pairwise() -> None:
    weights = linear_weights(DIRECTION_CATEGORIES)
    left = np.asarray(["structural", "functional"], dtype=object)
    right = np.asarray(["structural", "functional"], dtype=object)
    assert weighted_kappa(left, right, DIRECTION_CATEGORIES, weights) == pytest.approx(1.0)
    ratings = np.asarray(
        [["structural", "functional", "structural"], ["structural", "functional", "functional"]],
        dtype=object,
    )
    value = mean_pairwise_kappa(ratings, DIRECTION_CATEGORIES)
    assert -1.0 <= value <= 1.0


def test_conformal_threshold_is_the_finite_sample_order_statistic() -> None:
    scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
    rank = min(int(np.ceil((scores.size + 1) * 0.8)), scores.size)
    assert conformal_quantile(scores, 0.8) == pytest.approx(float(np.sort(scores)[rank - 1]))


def test_conformal_flagging_and_coverage() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.3, 0.7], [0.5, 0.5]])
    membership = direction_sets(probabilities, 0.61)
    assert membership.shape == (3, 2)
    flags = deferral_mask(probabilities, 0.61)
    assert not bool(flags[0])
    assert bool(flags[2])
    coverage = empirical_coverage(probabilities, np.asarray([0, 1, 0]), 0.61)
    assert 0.0 <= coverage <= 1.0


def test_deferral_analysis_reports_the_guarantee_from_the_evaluation_split() -> None:
    generator = np.random.default_rng(17)
    truth = generator.integers(0, 2, size=4_000)
    signal = 0.25 + 0.5 * truth
    scores = np.stack((1.0 - signal, signal), axis=1) + generator.normal(0.0, 0.07, size=(4_000, 2))
    scores = np.clip(scores, 1e-6, None)
    scores = scores / scores.sum(axis=1, keepdims=True)
    analysis = deferral_analysis(
        scores[:2_000], truth[:2_000], scores[2_000:], truth[2_000:], nominal_coverage=0.90
    )
    assert analysis["coverage"] >= 0.88
    assert 0.0 < analysis["deferral_share"] < 1.0
    assert analysis["decided_accuracy"] > 0.5
    threshold = decide_threshold(scores[:2_000], truth[:2_000], 0.90)
    assert threshold == pytest.approx(analysis["threshold"])


def test_unguarded_variant_defers_nothing() -> None:
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8]])
    report = unguarded_analysis(probabilities, np.asarray([0, 1]))
    assert report["deferral_share"] == 0.0
    assert report["decided_accuracy"] == pytest.approx(1.0)
    accuracy, share = decided_accuracy(probabilities, np.asarray([0, 1]), 0.5)
    assert accuracy <= 1.0 and 0.0 <= share <= 1.0


def test_alignment_error_and_effective_counts() -> None:
    assert alignment_error(
        np.asarray([1.0, 2.0, 3.0]), np.asarray([1.5, 2.0, 4.0])
    ) == pytest.approx(0.5)
    assert effective_pairs(np.ones(50)) == pytest.approx(50.0)
    assert effective_pairs(np.asarray([10.0] + [1.0] * 9)) < 10.0
    assert effective_pairs(np.zeros(0)) == 0.0
    with pytest.raises(ValueError):
        alignment_error(np.zeros(2), np.zeros(3))


def test_pair_term_scales_with_the_inverse_square_root() -> None:
    assert pair_term(1.0, 100.0) == pytest.approx(0.1)
    assert scaling_ratio(100.0, 400.0) == pytest.approx(2.0)
    assert math.isinf(pair_term(1.0, 0.0))
    with pytest.raises(ValueError):
        scaling_ratio(0.0, 10.0)


def test_decomposition_recovers_the_coefficients_that_built_the_response() -> None:
    alignment = np.asarray([0.1, 0.2, 0.3, 0.45])
    counts = np.asarray([144.0, 289.0, 576.0, 900.0])
    measured = 0.5 * alignment + 12.0 / np.sqrt(counts)
    left, right = fit_decomposition(alignment, counts, measured)
    assert left == pytest.approx(0.5, abs=1e-5)
    assert right == pytest.approx(12.0, abs=1e-5)
    result = decompose(alignment, counts, measured, float(measured[-1]))
    assert result.residual == pytest.approx(0.0, abs=1e-7)
    assert result.predicted == pytest.approx(result.measured, abs=1e-7)


def test_decomposition_needs_two_settings() -> None:
    with pytest.raises(ValueError):
        fit_decomposition(np.zeros(1), np.ones(1), np.zeros(1))
    with pytest.raises(ValueError):
        fit_decomposition(np.zeros(3), np.ones(2), np.zeros(3))
    assert excess_risk(np.asarray([0.3, 0.5]), 0.2) == pytest.approx(0.2)


def test_evidence_gap_and_identifiability_threshold() -> None:
    assert evidence_gap(np.asarray([0.2, 0.9]), np.asarray([0.2, 0.4])).tolist() == pytest.approx(
        [0.0, 0.5]
    )
    assert identifiable(np.asarray([0.5]), lipschitz=4.0).tolist() == [True]
    assert identifiable(np.asarray([0.1]), lipschitz=4.0).tolist() == [False]
    assert fusion_threshold(0.0) == float("inf")


def test_lipschitz_estimate_matches_the_analytic_slope() -> None:
    def fusion(evidence: np.ndarray) -> np.ndarray:
        return np.asarray(evidence[:, :1] * 3.0)

    centres = np.asarray([[0.2, 0.4], [0.6, 0.8]])
    assert estimate_lipschitz(fusion, centres, radius=1e-6) == pytest.approx(3.0, rel=1e-4)
    with pytest.raises(ValueError):
        estimate_lipschitz(fusion, np.zeros((2, 3)))


def test_weight_sensitivity_is_zero_when_the_streams_agree() -> None:
    equal = np.asarray([0.4, 0.4])
    weights = np.asarray([0.5, 0.5])
    quiet = weight_sensitivity(linear_fusion, weights, equal, equal, gain=FUSION_GAIN)
    assert float(quiet.max()) == pytest.approx(0.0)
    loud = weight_sensitivity(
        linear_fusion, weights, np.asarray([0.1, 0.9]), np.asarray([0.8, 0.2]), gain=FUSION_GAIN
    )
    assert float(loud.min()) > 0.0


def test_separation_of_weight_reads_the_two_medians() -> None:
    structural, functional = separation_of_weight(
        np.asarray([0.9, 0.8, 0.2, 0.1]), np.asarray([True, True, False, False])
    )
    assert structural == pytest.approx(0.85)
    assert functional == pytest.approx(0.15)
    degenerate = separation_of_weight(np.asarray([0.5]), np.asarray([True]))
    assert math.isnan(degenerate[0])
