"""Resampling, paired comparison and the fixed sequence."""

from __future__ import annotations

import math

import numpy as np
import pytest

from triad_reconciliation.inference.cluster_bootstrap import (
    Interval,
    bootstrap_interval,
    cluster_indices,
    cluster_size_inflation,
    cluster_standard_error,
    percentile_interval,
    statistics_guard,
)
from triad_reconciliation.inference.paired import (
    exceeds_margin,
    paired_difference,
    shared_units_masks,
)
from triad_reconciliation.inference.sequential import (
    HYPOTHESIS_ORDER,
    fixed_sequence,
)


def _labels(size: int, clusters: int) -> np.ndarray:
    return np.asarray([f"p{index % clusters}" for index in range(size)], dtype=object)


def test_cluster_indices_keep_every_member_of_a_drawn_cluster() -> None:
    labels = np.asarray(["a", "a", "a", "b", "b", "c"], dtype=object)
    sizes = {"a": 3, "b": 2, "c": 1}
    index = cluster_indices(labels, np.random.default_rng(3))
    counts: dict[str, int] = {}
    for position in index.tolist():
        counts[labels[position]] = counts.get(labels[position], 0) + 1
    assert sum(counts.get(name, 0) / size for name, size in sizes.items()) == pytest.approx(
        3.0, abs=0.0
    )
    assert all(counts.get(name, 0) % size == 0 for name, size in sizes.items() if name in counts)


def test_bootstrap_interval_brackets_a_known_mean() -> None:
    generator = np.random.default_rng(5)
    values = generator.normal(0.4, 1.0, size=400)
    labels = _labels(400, 60)
    interval = bootstrap_interval(
        lambda rows: float(values[rows].mean()), labels, replicates=200, seed=11
    )
    assert interval.lower <= interval.estimate <= interval.upper
    assert interval.replicates <= 200
    assert interval.half_width >= 0.0


def test_bootstrap_interval_is_degenerate_for_a_constant() -> None:
    values = np.ones(40)
    labels = _labels(40, 8)
    interval = bootstrap_interval(
        lambda rows: float(values[rows].mean()), labels, replicates=32, seed=13
    )
    assert interval.upper - interval.lower == pytest.approx(0.0)


def test_percentile_interval_ignores_non_finite_draws() -> None:
    interval = percentile_interval(0.5, np.asarray([0.1, float("nan"), 0.9]), level=0.5)
    assert interval.replicates == 2
    assert math.isfinite(interval.lower) and math.isfinite(interval.upper)
    empty = percentile_interval(0.5, np.asarray([float("nan")]))
    assert empty.replicates == 0
    assert isinstance(empty, Interval)


def test_statistics_guard_maps_a_degenerate_draw_to_a_non_finite_value() -> None:
    assert math.isnan(statistics_guard(lambda rows: float(rows.mean()), np.zeros(0, dtype=int)))

    def broken(rows: np.ndarray) -> float:
        raise ValueError("no draws")

    assert math.isnan(statistics_guard(broken, np.asarray([1])))


def test_cluster_standard_error_grows_with_the_between_cluster_spread() -> None:
    tight = np.asarray([1.0, 1.0, 1.0, 1.0] + [2.0, 2.0, 2.0, 2.0])
    wide = np.asarray([0.0, 0.0, 0.0, 0.0] + [4.0, 4.0, 4.0, 4.0])
    labels = np.asarray(["a"] * 4 + ["b"] * 4, dtype=object)
    assert cluster_standard_error(wide, labels) > cluster_standard_error(tight, labels)
    assert math.isnan(cluster_standard_error(tight, np.asarray(["a"] * 8, dtype=object)))


def test_cluster_size_inflation_reports_uneven_cluster_sizes() -> None:
    even = np.asarray(["a"] * 3 + ["b"] * 3, dtype=object)
    uneven = np.asarray(["a"] * 5 + ["b"], dtype=object)
    assert cluster_size_inflation(even) == pytest.approx(1.0)
    assert cluster_size_inflation(uneven) > 1.0
    assert math.isnan(cluster_size_inflation(np.zeros(0, dtype=object)))


def test_paired_difference_on_identical_vectors_is_exactly_zero() -> None:
    values = np.asarray([0.3, 0.6, 0.2, 0.9])
    patients = np.asarray(["a", "a", "b", "b"], dtype=object)
    difference = paired_difference(values, values, patients, replicates=32, seed=17)
    assert difference.estimate == pytest.approx(0.0)
    assert difference.interval.upper - difference.interval.lower == pytest.approx(0.0)
    assert not difference.excludes_zero
    assert difference.n_units == 4


def test_paired_difference_detects_a_shift() -> None:
    first = np.asarray([0.8] * 60)
    second = np.asarray([0.2] * 60)
    patients = _labels(60, 20)
    difference = paired_difference(first, second, patients, replicates=64, seed=19)
    assert difference.estimate == pytest.approx(0.6)
    assert difference.excludes_zero
    margin = exceeds_margin(difference, 0.3)
    assert margin["point_estimate"] and margin["interval_lower"]


def test_paired_difference_ignores_rows_where_a_metric_is_undefined() -> None:
    first = np.asarray([1.0, float("nan"), 1.0])
    second = np.asarray([0.0, 0.0, 0.0])
    patients = np.asarray(["a", "b", "c"], dtype=object)
    difference = paired_difference(first, second, patients, replicates=16, seed=23)
    assert difference.n_units == 2
    with pytest.raises(ValueError):
        paired_difference(first, second[:2], patients, replicates=4, seed=1)


def test_shared_units_masks_intersects_the_columns() -> None:
    mask = shared_units_masks(
        np.asarray([1.0, float("nan"), 2.0]), np.asarray([1.0, 1.0, float("nan")])
    )
    assert mask.tolist() == [True, False, False]
    with pytest.raises(ValueError):
        shared_units_masks()


def test_fixed_sequence_stops_at_the_first_criterion_not_met() -> None:
    effects = {
        name: value
        for name, value in zip(HYPOTHESIS_ORDER, [0.04, 0.05, -0.01, 0.3, 0.1], strict=True)
    }
    thresholds = dict.fromkeys(HYPOTHESIS_ORDER, 0.03)
    intervals = {name: (effect - 0.01, effect + 0.01) for name, effect in effects.items()}
    report = fixed_sequence(effects, thresholds, intervals)
    assert [item.identifier for item in report.tested] == list(HYPOTHESIS_ORDER[:3])
    assert report.stopped_at == HYPOTHESIS_ORDER[2]
    assert HYPOTHESIS_ORDER[3] not in [item.identifier for item in report.outcomes if item.tested]
    assert report.as_mapping()["stopped_at"] == HYPOTHESIS_ORDER[2]


def test_fixed_sequence_runs_to_the_end_when_every_criterion_is_met() -> None:
    effects = dict.fromkeys(HYPOTHESIS_ORDER, 0.5)
    thresholds = dict.fromkeys(HYPOTHESIS_ORDER, 0.1)
    intervals = dict.fromkeys(HYPOTHESIS_ORDER, (0.4, 0.6))
    report = fixed_sequence(effects, thresholds, intervals)
    assert report.stopped_at is None
    assert len(report.tested) == len(HYPOTHESIS_ORDER)


def test_fixed_sequence_records_a_hypothesis_this_run_did_not_supply() -> None:
    report = fixed_sequence(
        {HYPOTHESIS_ORDER[0]: 0.5}, {HYPOTHESIS_ORDER[0]: 0.1}, {HYPOTHESIS_ORDER[0]: (0.4, 0.6)}
    )
    unsupplied = [item for item in report.outcomes if item.reason == "not supplied by this run"]
    assert unsupplied
