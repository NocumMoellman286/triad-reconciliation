"""The cohort table contract: schema, reader, pairing and partitions.

The analysed records are not distributed and this release generates none. These tests cover
the surface the release does own: the schema it validates against, the reader that reads a
table a caller supplies, the pairing rule the manuscript describes, and the partition
arithmetic. The table they read is a literal of the smallest acceptable shape, declared in
``conftest``; nothing here draws a cohort.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from triad_reconciliation.analysis.pipeline import STAGING_HINT, build_table, prepare
from triad_reconciliation.cohort.reader import CohortNotStaged, read_cohort, read_eyes
from triad_reconciliation.cohort.schema import (
    CohortSchemaError,
    agreement_state_of,
    normalise_row,
    schema_card,
)
from triad_reconciliation.cohort.splits import (
    coverage_report,
    monotone_embargo,
    split_calibration,
    split_regions,
    within_region_folds,
)
from triad_reconciliation.cohort.visit_pairs import (
    build_visit_pairs,
    discordant_share,
    neighbour_window,
    pair_gap_years,
)
from triad_reconciliation.core.config import CohortConfig, ExperimentConfig, ModelConfig
from triad_reconciliation.core.jsonio import write_jsonl
from triad_reconciliation.core.records import (
    VOLUME_EDGE,
    CohortTable,
    EyeRecord,
    PerimetryVisit,
    StructuralVisit,
    VisitPair,
)
from triad_reconciliation.strata.partitions import (
    age_strata,
    all_strata,
    ancestry_strata,
    discordance_strata,
    myopia_strata,
    reliability_strata,
    sex_strata,
    stage_strata,
    stratum_sizes,
)


def _structural(eye: str, years: float) -> StructuralVisit:
    return StructuralVisit(
        patient_id="P1",
        eye_id=eye,
        site="A",
        acquired_years=years,
        morphometry=(0.45, 0.43, 1.42, 2.5, 0.12, 84.0, 78.0, 0.91),
        volume=tuple([0.5] * (VOLUME_EDGE**3)),
    )


def _report(eye: str, years: float) -> PerimetryVisit:
    return PerimetryVisit(
        patient_id="P1",
        eye_id=eye,
        site="A",
        tested_years=years,
        sensitivity_db=tuple([30.0] * 24),
        mean_deviation_db=-2.0,
        pattern_deviation_db=-1.8,
        visual_field_index=92.0,
        false_positive_rate=0.02,
        false_negative_rate=0.03,
        fixation_losses=0,
        duration_seconds=280.0,
        gaze_tracking_ok=True,
    )


def _pair(
    pair_id: str, patient: str, region: str, arbitration: str, index_years: float
) -> VisitPair:
    return VisitPair(
        pair_id=pair_id,
        patient_id=patient,
        eye_id=f"{patient}-L",
        site="A",
        region=region,
        index_years=index_years,
        nearest_years=index_years,
        structural=(_structural(f"{patient}-L", index_years),),
        functional=(_report(f"{patient}-L", index_years),),
        progression=0,
        arbitration=arbitration,
        stage="early",
        high_myopia=False,
        age_years=61.0,
        sex="female",
        ancestry_group=1,
        functional_reliable=True,
    )


def _two_region_table() -> CohortTable:
    """Four hand-written pairs, two per region, for the partition arithmetic only."""

    return CohortTable(
        pairs=(
            _pair("a", "P1", "I", "structural", 0.0),
            _pair("b", "P2", "I", "concordant", 0.5),
            _pair("c", "P3", "II", "functional", 1.0),
            _pair("d", "P4", "II", "concordant", 1.5),
        ),
        source="hand-written",
    )


def test_schema_card_declares_the_released_table() -> None:
    card = schema_card()
    assert card["format"] == "jsonl"
    assert card["morphometry_width"] == 8
    assert card["perimetry_points"] == 24
    assert card["volume_edge"] == VOLUME_EDGE
    assert list(card["arbitration_classes"]) == ["structural", "functional", "concordant"]


def test_normalise_row_coerces_and_rejects() -> None:
    eye = {
        "record": "eye",
        "patient_id": "p",
        "eye_id": "e",
        "site": "A",
        "region": "I",
        "stage": "early",
        "high_myopia": 1,
        "age_years": "61.5",
        "sex": "female",
        "ancestry_group": 2,
    }
    normalised = normalise_row(eye, 1)
    assert normalised["age_years"] == 61.5
    assert normalised["high_myopia"] is True
    with pytest.raises(CohortSchemaError):
        normalise_row({**eye, "stage": "unknown"}, 2)
    with pytest.raises(CohortSchemaError):
        normalise_row({"record": "pair"}, 3)
    with pytest.raises(CohortSchemaError):
        normalise_row({**eye, "high_myopia": False, "region": "III"}, 4)


def test_agreement_state_reads_off_the_label() -> None:
    assert agreement_state_of("concordant") == "concordant"
    assert agreement_state_of("structural") == "discordant"
    assert agreement_state_of("functional") == "discordant"


def test_reader_reads_the_table_a_caller_supplies(caller_table: Path) -> None:
    table = read_cohort(caller_table)
    assert table.n_pairs == 1
    assert table.source == "cohort.jsonl"
    pair = table.pairs[0]
    assert pair.region == "II"
    assert pair.arbitration == "concordant"
    assert not pair.discordant
    assert len(pair.structural) == 1 and len(pair.functional) == 1
    assert len(pair.structural[0].morphometry) == 8
    assert len(pair.functional[0].sensitivity_db) == 24


def test_reader_truncates_every_series_at_the_query_time(
    caller_table: Path, tmp_path: Path
) -> None:
    import json

    lines = caller_table.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    later = next(row for row in records if row["record"] == "functional")
    later["time_years"] = 9.0
    lines.append(json.dumps(later, sort_keys=True))
    target = tmp_path / "late.jsonl"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pair = read_cohort(target).pairs[0]
    assert all(report.tested_years <= 9.0 for report in pair.functional)
    assert all(visit.acquired_years <= 9.0 for visit in pair.structural)


def test_reader_rejects_a_visit_for_an_unknown_eye(tmp_path: Path) -> None:
    target = tmp_path / "bad.jsonl"
    write_jsonl(
        target,
        [
            {
                "record": "structural",
                "patient_id": "p",
                "eye_id": "missing",
                "time_years": 0.0,
                "morphometry": [0.0] * 8,
                "volume": [0.0] * (VOLUME_EDGE**3),
            }
        ],
    )
    with pytest.raises(CohortSchemaError):
        read_cohort(target)


def test_read_eyes_assembles_complete_records(caller_table: Path) -> None:
    eyes = read_eyes(caller_table)
    assert len(eyes) == 1
    record = eyes[0]
    assert record.structural and record.functional
    assert record.region == "II"
    assert record.disc_stage == "early"


def test_build_table_stops_without_a_staged_table(smoke_config: ExperimentConfig) -> None:
    with pytest.raises(CohortNotStaged) as raised:
        build_table(smoke_config, None)
    assert raised.value.hint == STAGING_HINT
    assert "--cohort" in STAGING_HINT


def test_build_table_reads_the_source_it_was_given(
    smoke_config: ExperimentConfig, caller_table: Path
) -> None:
    table = build_table(smoke_config, caller_table)
    assert table.n_pairs == 1
    with pytest.raises(CohortNotStaged):
        prepare(smoke_config, None)


def test_pairing_rule_and_window() -> None:
    structural = tuple(_structural("E1", 1.0 * index) for index in range(4))
    functional = tuple(_report("E1", 0.5 * index) for index in range(8))
    record = EyeRecord(
        patient_id="P1",
        eye_id="E1",
        site="A",
        region="II",
        disc_stage="early",
        high_myopia=False,
        age_years=60.0,
        sex="female",
        ancestry_group=1,
        structural=structural,
        functional=functional,
    )
    pairs = build_visit_pairs(
        record,
        imaging_interval_months=12,
        perimetry_interval_months=6,
        index_stream="functional",
    )
    assert len(pairs) == len(functional)
    assert all(pair.index_stream == "functional" for pair in pairs)
    assert neighbour_window(12, 6) == pytest.approx(0.75)
    assert max(pair_gap_years(pairs)) <= 0.5 + 1e-9
    assert discordant_share(("concordant", "structural")) == 0.5
    assert discordant_share(()) != discordant_share(("concordant",))


def test_patient_level_separation_and_folds() -> None:
    table = _two_region_table()
    partition = split_regions(table, held_out_region="II")
    assert not partition.overlap
    assert all(pair.region == "II" for pair in partition.held_out)
    assert len(partition.held_out) == 2
    folds = within_region_folds(partition.development, 2, seed=3)
    owners: dict[str, int] = {}
    for index, fold in enumerate(folds):
        for pair in fold:
            assert owners.setdefault(pair.patient_id, index) == index
    with pytest.raises(ValueError):
        within_region_folds(partition.development, 1, seed=3)


def test_monotone_embargo_and_coverage_report() -> None:
    table = _two_region_table()
    kept = monotone_embargo(table.pairs, threshold_years=1.0)
    assert all(pair.index_years <= 1.0 for pair in kept)
    report = coverage_report(table.pairs)
    assert report["coverage"] == pytest.approx(1.0)
    assert coverage_report(())["eligible"] == 0.0


def test_calibration_split_is_disjoint() -> None:
    patients = np.asarray(["p1", "p1", "p2", "p3", "p4"], dtype=object)
    first, second = split_calibration(patients, share=0.5, seed=1)
    assert first.any() and second.any()
    assert not set(patients[first].tolist()) & set(patients[second].tolist())
    assert first.sum() + second.sum() == patients.size
    with pytest.raises(ValueError):
        split_calibration(patients, share=1.5, seed=1)


def test_configuration_holds_no_record_level_constant() -> None:
    """The reading convention is fixed; the records are not."""

    defaults = CohortConfig()
    assert defaults.held_out_region == "II"
    assert defaults.imaging_interval_months == 12
    assert defaults.perimetry_interval_months == 6
    assert not hasattr(defaults, "region_two_pairs")
    assert not hasattr(defaults, "stage_mix")
    assert ModelConfig().max_visits > 0


def test_strata_cover_the_partition() -> None:
    """The declared strata families, on the four hand-written pairs."""

    pairs = _two_region_table().pairs
    discordant, concordant = discordance_strata(pairs)
    assert discordant.n_pairs + concordant.n_pairs == len(pairs)
    assert discordant.n_pairs == 2 and concordant.n_pairs == 2
    assert sum(item.n_pairs for item in stage_strata(pairs)) == len(pairs)
    for left, right in (reliability_strata(pairs), myopia_strata(pairs), sex_strata(pairs)):
        assert left.n_pairs + right.n_pairs == len(pairs)
    below, above = age_strata(pairs)
    assert below.n_pairs + above.n_pairs == len(pairs)
    assert sum(item.n_pairs for item in ancestry_strata(pairs, 3)) == len(pairs)
    sizes = stratum_sizes(pairs)
    assert sizes["agreement_state:discordant"] == 2
    assert all(0 <= value <= len(pairs) for value in sizes.values())
    assert all(item.n_patients <= item.n_pairs for item in all_strata(pairs))
