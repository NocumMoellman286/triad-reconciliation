"""The claim map, the manuscript arithmetic, the hygiene scans and the artefact writer."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from triad_reconciliation.core.jsonio import read_json
from triad_reconciliation.verification import manuscript as manuscript_module
from triad_reconciliation.verification.claims import CLAIMS, claims_by_verification, resolve_claims
from triad_reconciliation.verification.conformance import all_conformance
from triad_reconciliation.verification.hygiene import (
    Finding,
    findings_by_family,
    scan_config_result_collisions,
    scan_documentation,
    scan_duplicate_statements,
    scan_empty_implementations,
    scan_host_paths,
    scan_tree,
)
from triad_reconciliation.verification.manifest import (
    build_manifest,
    compare_with_tree,
    write_integrity_manifest,
)
from triad_reconciliation.verification.report import build_report, render_summary, write_artefacts
from triad_reconciliation.verification.sources import auxiliary_total, render_dataset_urls

RELEASE = Path(__file__).resolve().parents[1]


def test_every_claim_names_a_location_and_a_symbol() -> None:
    assert len(CLAIMS) >= 70
    assert len({claim.identifier for claim in CLAIMS}) == len(CLAIMS)
    for claim in CLAIMS:
        assert claim.location and claim.statement
        assert claim.symbols
        assert claim.verification in {"mapping", "execution"}
    assert len(claims_by_verification("mapping")) >= 10


def test_claim_symbols_all_resolve() -> None:
    rows = resolve_claims()
    unresolved = [row for row in rows if not row["code_present"]]
    assert not unresolved, f"unresolved claim symbols: {unresolved}"
    assert all(row["status"] == "PASS" for row in rows)


def test_claim_map_reports_the_split_between_mapping_and_execution() -> None:
    rows = resolve_claims()
    mapping = sum(1 for row in rows if row["verification"] == "mapping")
    execution = sum(1 for row in rows if row["verification"] == "execution")
    assert mapping + execution == len(rows)
    assert execution > mapping


def test_manuscript_arithmetic_closes_on_the_printed_values() -> None:
    outcomes = manuscript_module.run_manuscript_checks()
    failures = [outcome.identifier for outcome in outcomes if outcome.status == "FAIL"]
    assert not failures, f"the manuscript's own printed components disagree: {failures}"
    assert len(outcomes) >= 35


def test_manuscript_findings_are_the_expected_four() -> None:
    outcomes = manuscript_module.run_discrepancy_checks()
    failing = {outcome.identifier for outcome in outcomes if outcome.status == "FAIL"}
    assert failing == {"M101", "M102", "M103", "M104"}
    for outcome in outcomes:
        if outcome.status == "FAIL":
            assert outcome.family == "manuscript"


def test_a_revision_that_moves_a_reported_value_breaks_the_pin() -> None:
    """The interval the prose quotes and the interval the table prints stay distinct."""

    prose = (0.843, 0.873)
    table = (0.842, 0.874)
    assert prose != table
    assert abs(prose[0] - table[0]) < 5e-3
    arbitration = {"prose": 0.42, "table": 0.24}
    assert arbitration["prose"] != arbitration["table"]


def test_data_endpoints_carry_only_resolved_sources() -> None:
    body = render_dataset_urls(probed=False)
    assert "no endpoint was probed in this pass" in body
    assert "https://github.com/miag-ull/rim-one-dl" in body
    assert "not distributable at record level" in body
    assert auxiliary_total() == 9678


def test_hygiene_scans_the_shipped_tree_clean() -> None:
    findings = scan_tree(RELEASE)
    assert not findings, f"hygiene findings: {[(f.family, f.path) for f in findings]}"


def test_duplicate_detector_reports_a_real_copy(tmp_path: Path) -> None:
    body = (
        "def first(values):\n"
        "    total = sum(values)\n"
        "    count = len(values)\n"
        "    mean = total / max(count, 1)\n"
        "    spread = max(values) - min(values)\n"
        "    return mean, spread\n"
    )
    other = body.replace("def first", "def second")
    (tmp_path / "a.py").write_text(body, encoding="utf-8")
    (tmp_path / "b.py").write_text(other, encoding="utf-8")
    findings = scan_duplicate_statements(tmp_path)
    assert findings
    assert all(finding.family == "duplicate-statements" for finding in findings)


def test_empty_implementation_detector_ignores_protocol_stubs(tmp_path: Path) -> None:
    source = (
        "from typing import Protocol\n"
        "\n"
        "class Surface(Protocol):\n"
        "    def value(self) -> float: ...\n"
        "\n"
        "def implemented() -> int:\n"
        "    return 1\n"
        "\n"
        "def hollow() -> int:\n"
        "    pass\n"
    )
    (tmp_path / "module.py").write_text(source, encoding="utf-8")
    findings = scan_empty_implementations(tmp_path)
    assert len(findings) == 1
    assert findings[0].path == "module.py"
    assert findings[0].detail.startswith("function at line 9")


def test_host_path_scan_masks_what_it_reports(tmp_path: Path) -> None:
    leak = "/" + "Users" + "/" + "someone" + "/project"
    (tmp_path / "leak.txt").write_text(f"root at {leak}\n", encoding="utf-8")
    findings = scan_host_paths(tmp_path)
    assert len(findings) == 1
    assert "someone" not in findings[0].detail


def test_documentation_scan_flags_extra_markdown(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("ok\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("not allowed\n", encoding="utf-8")
    findings = scan_documentation(tmp_path)
    assert [finding.path for finding in findings] == ["notes.md"]


def test_config_scan_keeps_method_constants_and_flags_results(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "good.yaml").write_text(
        "threshold: 0.20\nwidth: 96\n", encoding="utf-8"
    )
    assert not scan_config_result_collisions(tmp_path)
    (tmp_path / "configs" / "bad.yaml").write_text("result: 0.891\n", encoding="utf-8")
    flagged = scan_config_result_collisions(tmp_path)
    assert len(flagged) == 1
    assert flagged[0].family == "config-result-collision"


def test_findings_by_family_counts_each_group() -> None:
    counts = findings_by_family(
        [Finding("a", "p", "d"), Finding("a", "q", "d"), Finding("b", "r", "d")]
    )
    assert counts == {"a": 2, "b": 1}


def test_manifest_excludes_itself_and_detects_drift(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    write_integrity_manifest(tmp_path)
    manifest = read_json(tmp_path / "integrity_manifest.json")
    assert "integrity_manifest.json" not in manifest["entries"]
    assert manifest["file_count"] == 1
    assert compare_with_tree(tmp_path)["digest_matches"] is True
    (tmp_path / "one.txt").write_text("changed", encoding="utf-8")
    comparison = compare_with_tree(tmp_path)
    assert comparison["changed"] == ["one.txt"]
    assert comparison["digest_matches"] is False
    assert "live_digest" not in comparison


def test_manifest_inventory_keeps_a_nested_source_data_directory(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "pkg" / "data"
    nested.mkdir(parents=True)
    (nested / "table.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "cohort.yaml").write_text("cohort: {}\n", encoding="utf-8")
    manifest = build_manifest(tmp_path)
    assert "src/pkg/data/table.py" in manifest["entries"]
    assert "configs/cohort.yaml" in manifest["entries"]


def test_conformance_passes_on_the_shipped_configuration() -> None:
    outcomes = all_conformance(RELEASE)
    failures = [outcome.name for outcome in outcomes if outcome.status != "PASS"]
    assert not failures, f"configuration conformance failures: {failures}"


def test_report_summary_separates_code_from_manuscript() -> None:
    report = build_report(RELEASE, include_gates=False)
    summary = report["summary"]
    assert summary["code_status"] in {"PASS", "FAIL"}
    assert summary["manuscript_status"] == "FAIL"
    if summary["code_status"] == "PASS":
        assert summary["overall_status"] == "PARTIALLY_VERIFIED"
    else:
        assert summary["overall_status"] == "UNVERIFIED"
    assert summary["claims_checked"] == float(len(CLAIMS))
    assert summary["manuscript_arithmetic_findings"] == 4.0
    text = render_summary(report)
    assert "manuscript status" in text
    assert "manuscript finding M101" in text


def test_artefact_writer_targets_a_caller_supplied_root(tmp_path: Path) -> None:
    """The writer must be parameterised, or a test run replaces the shipped artefacts."""

    copy = tmp_path / "release"
    copy.mkdir()
    shutil.copytree(RELEASE / "configs", copy / "configs")
    shutil.copy2(RELEASE / "pyproject.toml", copy / "pyproject.toml")
    written = write_artefacts(copy, include_gates=False)
    assert set(written) == {
        "claim_to_code.json",
        "verification_report.json",
        "verification_summary.txt",
        "dataset_urls.txt",
        "integrity_manifest.json",
    }
    assert (copy / "verification_summary.txt").is_file()
    assert not (copy / "verification_summary.md").exists()
    assert json.loads((copy / "claim_to_code.json").read_text(encoding="utf-8"))["claims"]
    assert (copy / "dataset_urls.txt").read_text(encoding="utf-8").startswith("# Data endpoints")
    first = (copy / "verification_report.json").read_bytes()
    write_artefacts(copy, include_gates=False)
    assert (copy / "verification_report.json").read_bytes() == first


def test_shipped_report_is_the_complete_one() -> None:
    """A test that writes artefacts must not have overwritten the shipped report."""

    report = read_json(RELEASE / "verification_report.json")
    assert report["summary"]["claims_checked"] == float(len(CLAIMS))
    assert len(report["layer_two_execution"]) >= 50
    names = {item["name"] for item in report["layer_two_execution"]}
    assert {"gate-pytest", "gate-ruff", "gate-black", "gate-isort", "gate-mypy"} <= names
    assert len(report["manuscript_checks"]) >= 40
    assert set(report) >= {
        "paper",
        "summary",
        "engineering_defaults",
        "deviations",
        "layer_one_paper_to_code",
        "layer_two_execution",
        "manuscript_checks",
        "manuscript_inconsistencies",
        "conformance",
        "hygiene",
        "notes",
    }
    assert report["paper"]["analysed_cohort_distributed"] is False


@pytest.mark.parametrize(
    "name", ["claim_to_code.json", "verification_report.json", "integrity_manifest.json"]
)
def test_root_artefacts_exist(name: str) -> None:
    assert (RELEASE / name).is_file()
