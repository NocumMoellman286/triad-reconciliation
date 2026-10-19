"""Comparators, strata, reader study, deployment accounting and reporting."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from triad_reconciliation.comparators import roster
from triad_reconciliation.comparators.clinical_rules import per_eye_slope
from triad_reconciliation.comparators.structure_function import fit_linear
from triad_reconciliation.comparators.timeline import ConcatenatedTimelineTransformer
from triad_reconciliation.comparators.zero_shot import lexical_score, report_tokens
from triad_reconciliation.core.config import ModelConfig
from triad_reconciliation.core.records import ReadershipResponse
from triad_reconciliation.core.seeding import set_seed
from triad_reconciliation.deployment.silent_mode import (
    MISSING_STREAM,
    PROCESSING_FAILURE,
    EncounterRecord,
    deployment_log_summary,
    silent_mode_metrics,
)
from triad_reconciliation.deployment.workflow import (
    agreement_share,
    implied_escalation,
    implied_interval_months,
    median_lead_time_months,
    net_benefit,
    reclassification_counts,
    workflow_outcome,
)
from triad_reconciliation.readers.reliance import (
    arm_gain,
    reader_strata,
    reliance_report,
    within_limit,
)
from triad_reconciliation.readers.study import (
    ARMS,
    Vignette,
    accuracy,
    arm_answers,
    inter_reader_agreement,
    reader_matrix,
    stream_switch_share,
    summarise_arm,
)
from triad_reconciliation.reporting.tables import (
    all_tables,
    render_roster,
    render_table,
    render_workflow,
)

CONFIG = ModelConfig(max_visits=5)


def test_roster_declares_fifteen_rows_with_families_and_inputs() -> None:
    assert len(roster.ROSTER) == 15
    assert roster.roster_by_row()[15].name == roster.PROPOSED
    assert all(spec.family and spec.inputs for spec in roster.ROSTER)
    assert len(roster.families()) >= 6
    assert set(roster.clinical_rule_names()) == {
        "guided progression analysis",
        "visual field index trend",
        "mean deviation trend",
    }


def test_learned_modules_instantiate_under_their_roster_names() -> None:
    modules = roster.learned_modules(CONFIG)
    assert len(modules) == 8
    for name in modules:
        assert name in roster.roster_names()


def test_trend_rules_recover_a_known_slope() -> None:
    stamps = np.asarray([[0.0, 1.0, 2.0]])
    values = np.asarray([[10.0, 8.0, 6.0]])
    mask = np.ones((1, 3), dtype=bool)
    assert per_eye_slope(values, stamps, mask)[0] == pytest.approx(-2.0)


def test_linear_regression_recovers_an_exact_map() -> None:
    generator = np.random.default_rng(3)
    design = generator.normal(size=(120, 4))
    weights = np.asarray([0.5, -1.5, 2.0, 0.25])
    recovered = fit_linear(design, design @ weights + 3.0, ridge=1e-10)
    assert recovered[1:] == pytest.approx(weights, abs=1e-6)
    assert recovered[0] == pytest.approx(3.0, abs=1e-6)


def test_lexical_score_rises_as_the_field_worsens() -> None:
    deep = list(report_tokens(np.asarray([-24.0, -20.0, 8.0, 0.02, 0.02, 0.0])))
    shallow = list(report_tokens(np.asarray([-1.0, -1.0, 98.0, 0.01, 0.01, 0.0])))
    assert lexical_score(deep) > lexical_score(shallow)


def test_unconstrained_comparator_emits_a_post_hoc_attribution(tensor_batch) -> None:
    set_seed(5)
    model = ConcatenatedTimelineTransformer(CONFIG)
    with torch.no_grad():
        logit = model(tensor_batch)
        weights, valid = model.attention_attribution(tensor_batch)
    assert logit.shape == (tensor_batch.progression.shape[0],)
    assert weights.shape == valid.shape
    assert bool((weights >= 0.0).all())


def _responses() -> tuple[tuple[ReadershipResponse, ...], dict[str, str], dict[str, str]]:
    """Two readers over four vignettes, hand-written for the accounting tests."""

    reference = {"v1": "structural", "v2": "structural", "v3": "functional", "v4": "functional"}
    advice = {"v1": "structural", "v2": "functional", "v3": "functional", "v4": "structural"}
    answers = tuple(
        answer
        for reader in ("r1", "r2")
        for answer in (
            ReadershipResponse(reader, "v1", "unassisted", "structural", 130.0),
            ReadershipResponse(reader, "v2", "unassisted", "structural", 128.0),
            ReadershipResponse(reader, "v3", "model_assisted", "functional", 100.0),
            ReadershipResponse(reader, "v4", "model_assisted", "structural", 104.0),
        )
    )
    return answers, reference, advice


def test_reader_study_arms_and_accuracy() -> None:
    answers, reference, _ = _responses()
    discordant = {"v1": False, "v2": True, "v3": True, "v4": False}
    assert ARMS == ("unassisted", "model_only", "model_assisted")
    unassisted = arm_answers(answers, "unassisted", discordant, only_discordant=False)
    assert len(unassisted) == 4
    assert accuracy(unassisted, reference) == pytest.approx(1.0)
    summary = summarise_arm(answers, reference, discordant, "unassisted", only_discordant=False)
    assert summary.n_answers == 4
    assert summary.median_seconds == pytest.approx(129.0)
    assert math.isfinite(summary.arbitration_kappa)
    empty = summarise_arm((), reference, discordant, "model_only", only_discordant=False)
    assert empty.n_answers == 0 and math.isnan(empty.accuracy)


def test_reader_matrix_and_inter_reader_agreement() -> None:
    answers, _, _ = _responses()
    matrix = reader_matrix(answers, "unassisted", ("v1", "v2", "v3", "v4"))
    assert matrix.shape == (2, 4)
    assert inter_reader_agreement(answers, "model_assisted", ("v3", "v4")) == pytest.approx(1.0)
    assert math.isnan(inter_reader_agreement(answers, "unassisted", ("v3", "v4")))
    assert math.isnan(inter_reader_agreement((), "unassisted", ("v1",)))


def test_stream_switch_share_counts_readers_who_moved() -> None:
    answers, _, _ = _responses()
    baseline = {"r1/v1": "structural", "r1/v3": "functional", "r2/v3": "structural"}
    share = stream_switch_share(answers, baseline, arm="model_assisted")
    assert share == pytest.approx(0.5)


def test_reliance_report_counts_both_denominators() -> None:
    answers, reference, advice = _responses()
    report = reliance_report(answers, reference, advice)
    assert report.wrong_advice_cases == 2
    assert report.correct_advice_cases == 2
    assert report.over_reliance == pytest.approx(1.0)
    assert report.under_reliance == pytest.approx(0.0)
    assert report.adopted_wrong == 2 and report.rejected_correct == 0
    limit = within_limit(report, 0.2)
    assert limit["point_estimate"] is False


def test_reader_strata_and_arm_gain() -> None:
    answers, reference, _ = _responses()
    lower, upper = reader_strata(answers, reference, arm="model_assisted", cut=1)
    assert lower.size + upper.size == 2
    gain = arm_gain(answers, reference, lower)
    assert math.isfinite(gain)


def test_vignette_counts_and_consensus_endorsement() -> None:
    vignette = Vignette("v1", "structural", "functional", True)
    assert vignette.discordant
    assert vignette.reference != vignette.model_direction


def test_silent_mode_rates_match_a_hand_count() -> None:
    records = tuple(
        EncounterRecord(
            encounter_id=f"e{index}",
            scored=index >= 4,
            failure_reason=(
                None if index >= 4 else (MISSING_STREAM if index % 2 == 0 else PROCESSING_FAILURE)
            ),
            latency_seconds=2.0 + 0.5 * index,
            deferred=index == 5,
            attribution_emitted=index >= 4,
            attribution_consistent=index >= 5,
            opened_by_reviewer=index >= 6,
            overrode_direction=index == 4,
            deferred_returned=index == 5,
            deferred_accepted=index == 5,
        )
        for index in range(8)
    )
    metrics = silent_mode_metrics(records)
    assert metrics.eligible == 8
    assert metrics.scored == 4
    assert metrics.coverage == pytest.approx(0.5)
    assert metrics.unscored_missing_share == pytest.approx(0.25)
    assert metrics.unscored_processing_share == pytest.approx(0.25)
    assert metrics.deferred_share == pytest.approx(0.25)
    assert metrics.attribution_share == pytest.approx(1.0)
    assert metrics.deferred_acceptance_share == pytest.approx(1.0)
    assert metrics.interpretability_share == pytest.approx(0.75)
    summary = deployment_log_summary(records)
    assert summary["scored"] == 4.0


def test_silent_mode_handles_an_empty_log() -> None:
    metrics = silent_mode_metrics(())
    assert metrics.eligible == 0
    assert math.isnan(metrics.coverage)


def test_workflow_decision_rules() -> None:
    probabilities = np.asarray([0.05, 0.25, 0.6, 0.95])
    escalation = implied_escalation(probabilities, 0.5)
    assert escalation.tolist() == [False, False, True, True]
    intervals = implied_interval_months(probabilities, (0.10, 0.50))
    assert intervals.tolist() == [12.0, 6.0, 3.0, 3.0]
    assert agreement_share(escalation, escalation) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        agreement_share(escalation, escalation[:2])


def test_net_benefit_and_reclassification() -> None:
    probabilities = np.asarray([0.8, 0.9, 0.2, 0.1])
    labels = np.asarray([1, 1, 0, 0])
    assert net_benefit(probabilities, labels, 0.5) == pytest.approx(0.5)
    assert math.isnan(net_benefit(np.zeros(0), np.zeros(0), 0.5))
    baseline = np.asarray([0, 0, 1, 1])
    updated = np.asarray([1, 0, 1, 0])
    counts = reclassification_counts(baseline, updated, labels)
    assert counts["up_events"] == pytest.approx(0.5)
    assert counts["down_non_events"] == pytest.approx(0.5)
    assert counts["nri"] == pytest.approx(1.0)
    assert math.isnan(reclassification_counts(baseline, updated, np.ones(4))["nri"])


def test_lead_time_uses_the_usable_rows() -> None:
    assert median_lead_time_months(
        np.asarray([2.0, 4.0]), np.asarray([6.0, 10.0])
    ) == pytest.approx(5.0)
    assert math.isnan(median_lead_time_months(np.asarray([float("nan")]), np.asarray([1.0])))


def test_workflow_outcome_assembles_every_row() -> None:
    generator = np.random.default_rng(23)
    probabilities = generator.random(400)
    baseline = generator.random(400)
    labels = (generator.random(400) < 0.4).astype(np.int64)
    outcome = workflow_outcome(
        probabilities,
        labels,
        implied_escalation(probabilities, 0.2),
        implied_interval_months(probabilities, (0.1, 0.5)),
        detection_months=generator.random(400),
        reference_months=generator.random(400) + 1.0,
        baseline_probabilities=baseline,
        matched_specificity=0.90,
        decision_threshold=0.20,
        interval_bands=(0.1, 0.5),
    )
    mapping = outcome.as_mapping()
    assert mapping["escalation_agreement"] == pytest.approx(1.0)
    assert 0.0 <= mapping["sensitivity_matched"] <= 1.0
    assert math.isfinite(mapping["net_benefit"])


def test_table_rendering_is_data_derived_and_complete() -> None:
    block = render_table("T", ("a", "b"), [["1", "x"], ["2", "y-long-value"]])
    lines = block.splitlines()
    assert lines[0] == "T"
    body = [line for line in lines if line.startswith(("1", "2"))]
    assert len({len(line) for line in body}) == 1
    assert "y-long-value" in body[1]


def test_roster_and_missing_table_renderers() -> None:
    roster_block = render_roster()
    assert "concatenated-timeline transformer" in roster_block
    missing = render_workflow(None)
    assert missing.count("---") >= 6
    assert callable(all_tables)
