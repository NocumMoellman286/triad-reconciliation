"""Assembly of the released artefacts.

One call writes everything: the claim map, the verification report, the plain-text summary,
the endpoint list and, last of all, the integrity manifest. The order matters because the
manifest digests the tree that contains the report, and because a check may read the endpoint
list.

Ref: the release contract; the claims are drawn from every section and table of the manuscript.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from triad_reconciliation.core.jsonio import atomic_write_text, write_json
from triad_reconciliation.verification import (
    conformance as conformance_module,
    execution as execution_module,
    hygiene as hygiene_module,
    manuscript as manuscript_module,
)
from triad_reconciliation.verification.claims import CLAIMS, resolve_claims
from triad_reconciliation.verification.manifest import write_integrity_manifest
from triad_reconciliation.verification.sources import auxiliary_total, render_dataset_urls

PAPER_TITLE = "Temporal Vision-Language Reconciliation of Optic Disc Images and Visual Fields"
PAPER_VENUE = "npj Digital Medicine"
PACKAGE = "triad_reconciliation"

ENGINEERING_DEFAULTS: dict[str, str] = {
    "agreement.supervision": "the agreement state is inferred from the latent and the manuscript prints "
    "no signal that supervises it directly; the released readout is trained against the arbitration "
    "label collapsed to concordant or discordant, and the switch that turns that off exists only for "
    "the component analysis",
    "arbitration.outcome_derived_rule": "in the variant that removes the arbitration head the direction "
    "is composed from the progression readout and the stream weight; the manuscript prints no rule for "
    "this composition",
    "cohort.pairing_window": "the window a perimetry visit is paired with the nearest imaging "
    "visit is a released constant; Sec. 5.4 names the pairing without printing the window",
    "cohort.calibration_share": "the fraction of held-out patients whose scores the conformal "
    "threshold is fitted on; the manuscript prints the nominal coverage, not the split",
    "cohort.held_out_region": "the region label the release treats as held out; Sec. 5.2 prints "
    "the two regions without printing which label a supplied table will carry",
    "cohort.row_grouping": "the reader groups rows by the eye_id a caller supplies; the "
    "manuscript names the units of analysis without printing a column convention",
    "identifiability.fusion_gain": "the fusion map's gain is used by the identifiability analysis and by "
    "its check; the manuscript prints no gain",
    "loss.consistency_margin": "the margin the stream weight is compared against in the consistency term "
    "is a released constant; the manuscript names the term without printing the margin",
    "loss.reweighting_strength": "lambda is fixed on the development group per Sec. 5.5; no value is printed",
    "model.widths": "token widths, head counts, layer counts and dropout are released constants; the "
    "manuscript names the encoders and the reconciliation operator without printing any shape",
    "model.volume_edge": "the optic-nerve-head grid is carried at the resolution the released table uses",
    "perimetry.vocabulary": "the perimetry serialisation schema and its bucket edges are release designs; "
    "the manuscript names the schema without printing it",
    "stages.bandwidth": "the kernel structure-function comparator takes its bandwidth from the median "
    "pairwise distance; no bandwidth is printed",
    "table.renderer": "reported values are rendered from what ran; no value in the tables is transcribed "
    "from the manuscript",
    "train.optimizer": "AdamW under a cosine schedule is printed in Algorithm 1; the learning rate, "
    "weight decay, batch size, warm-up, epoch count, precision and clipping come from the release",
    "workflow.decision_rule": "the progression probability is mapped to treatment escalation and to a "
    "review interval by released bands; the manuscript prints the threshold probability only",
}

DEVIATIONS: list[dict[str, str]] = [
    {
        "location": "Data availability, p. 17; Sec. 5.2, p. 13-14",
        "departs": "The analysed cohort is not distributed and this release produces nothing in its "
        "place. Every command that reads records takes the table from the caller with --cohort and "
        "stops without one.",
        "why": "The manuscript withholds institution and site identities and states that the records are "
        "held under the sites' agreements. Every record-level quantity the article reports is therefore "
        "reported BLOCKED with that reason, and no value is produced from records the release does not "
        "hold.",
    },
    {
        "location": "Sec. 5.5, p. 15",
        "departs": "The agreement state carries supervision through the collapsed arbitration label.",
        "why": "The manuscript describes the state as inferred from the latent and prints no loss term "
        "for it; without any signal the re-weighting strength has no effect on the objective and its "
        "reported ablation would be untestable.",
    },
    {
        "location": "Sec. 2.1, p. 5; Table 7 row 13",
        "departs": "The zero-shot comparator is reproduced as a prompt-template lexical scorer over the "
        "serialisation vocabulary. No language-model weights ship.",
        "why": "The manuscript names a multimodal language model prompted zero-shot and releases no "
        "checkpoint, so the comparator is rebuilt from its prompt rather than from weights.",
    },
    {
        "location": "Sec. 2.4, p. 11; Table 4, p. 12",
        "departs": "The reader study is computed from a response table the caller supplies; no reader "
        "responses ship.",
        "why": "Reader responses are site-held records. The release carries the accounting, the reliance "
        "decomposition and the inter-reader agreement, and reports the table values as BLOCKED.",
    },
    {
        "location": "Table 6, p. 23; Sec. 5.7, p. 17",
        "departs": "The silent-mode and workflow tables are computed from a deployment log the caller "
        "supplies; no log ships.",
        "why": "The silent run was performed at the participating sites. The release carries every rate "
        "the table prints and reports the values as BLOCKED.",
    },
    {
        "location": "Sec. 5.6, p. 15-16",
        "departs": "Minimum detectable differences and bootstrap replicate counts are read as design "
        "constants; the manuscript prints four of them for five hypotheses.",
        "why": "The mismatch is recorded as a manuscript finding rather than resolved by inventing a "
        "threshold for the unmapped hypothesis.",
    },
]


def _verdict(code_failed: bool, unfinished: bool) -> str:
    """Three verdicts.

    A failure in a code family makes the release unverified. A disagreement inside the
    article's own printed values, or a check that could not run, leaves it partially verified:
    the article's arithmetic is not a defect of this release, and neither is a cohort that is
    not distributed, but neither can the release be called fully verified while they stand.
    """

    if code_failed:
        return "UNVERIFIED"
    if unfinished:
        return "PARTIALLY_VERIFIED"
    return "VERIFIED"


def build_report(root: Path, *, include_gates: bool = True) -> dict[str, Any]:
    claims = resolve_claims()
    executions = execution_module.run_execution_checks(include_gates=include_gates)
    manuscript_outcomes = manuscript_module.all_manuscript_outcomes()
    conformance = conformance_module.all_conformance(Path(root))
    findings = hygiene_module.scan_tree(Path(root))

    code_failures = [item for item in executions if item.status == "FAIL"]
    manuscript_failures = [item for item in manuscript_outcomes if item.status == "FAIL"]
    conformance_failures = [item for item in conformance if item.status == "FAIL"]
    claim_failures = [item for item in claims if item["status"] == "FAIL"]
    hygiene_failures = [item.as_mapping() for item in findings]

    code_failed = bool(code_failures or conformance_failures or claim_failures or hygiene_failures)
    unfinished = any(item.status in {"BLOCKED", "NOT_RUN"} for item in executions) or any(
        item.status in {"BLOCKED", "NOT_RUN"} for item in conformance
    )
    code_status = "FAIL" if code_failed else "PASS"
    manuscript_status = "PASS" if not manuscript_failures else "FAIL"
    overall = _verdict(code_failed, unfinished or bool(manuscript_failures))

    summary = {
        "overall_status": overall,
        "code_status": code_status,
        "manuscript_status": manuscript_status,
        "claims_checked": float(len(claims)),
        "claims_with_code_failures": float(len(claim_failures)),
        "execution_checks": float(len(executions)),
        "execution_failures": float(len(code_failures)),
        "execution_blocked": float(sum(1 for item in executions if item.status == "BLOCKED")),
        "execution_not_run": float(sum(1 for item in executions if item.status == "NOT_RUN")),
        "manuscript_checks": float(len(manuscript_outcomes)),
        "manuscript_arithmetic_findings": float(len(manuscript_failures)),
        "conformance_checks": float(len(conformance)),
        "conformance_failures": float(len(conformance_failures)),
        "hygiene_findings": float(len(hygiene_failures)),
    }

    return {
        "paper": {
            "title": PAPER_TITLE,
            "venue": PAPER_VENUE,
            "package": PACKAGE,
            "analysed_cohort_distributed": False,
            "cohort_source": "supplied by the caller with --cohort",
            "auxiliary_image_total": auxiliary_total(),
        },
        "summary": summary,
        "engineering_defaults": ENGINEERING_DEFAULTS,
        "deviations": DEVIATIONS,
        "layer_one_paper_to_code": {"claims": claims, "summary": _claim_summary(claims)},
        "layer_two_execution": [item.as_mapping() for item in executions],
        "manuscript_checks": [asdict(item) for item in manuscript_outcomes],
        "manuscript_inconsistencies": [asdict(item) for item in manuscript_failures],
        "conformance": [asdict(item) for item in conformance],
        "hygiene": {
            "families": hygiene_module.findings_by_family(findings),
            "findings": hygiene_failures,
        },
        "notes": _notes(summary),
    }


def _claim_summary(claims: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "total": len(claims),
        "resolved": sum(1 for item in claims if item["code_present"]),
        "mapping": sum(1 for item in claims if item["verification"] == "mapping"),
        "execution": sum(1 for item in claims if item["verification"] == "execution"),
        "unresolved": [item["id"] for item in claims if not item["code_present"]],
    }


def _notes(summary: dict[str, Any]) -> str:
    held = (
        " The analysed records are not distributed and this release generates none: every "
        "record-level analysis is reported BLOCKED with the staging instruction, the numerical "
        "checks run on tensor batches drawn from a recorded seed, and the reader is exercised on "
        "a hand-written table in the unit suite."
    )
    if summary["manuscript_arithmetic_findings"]:
        return (
            "A manuscript finding is a disagreement between two statements inside the article's own "
            "printed values, or between the number of hypotheses and the number of pre-specified "
            "thresholds it prints. It is not a defect of this release and it does not enter the code "
            "status." + held
        )
    return held.strip()


def render_summary(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "Verification summary",
        "===================",
        f"overall status            : {summary['overall_status']}",
        f"code status               : {summary['code_status']}",
        f"manuscript status         : {summary['manuscript_status']}",
        f"claims checked            : {summary['claims_checked']:.0f}",
        f"claims with code failures : {summary['claims_with_code_failures']:.0f}",
        f"manuscript findings       : {summary['manuscript_arithmetic_findings']:.0f}",
        f"execution checks          : {summary['execution_checks']:.0f}",
        f"execution failures        : {summary['execution_failures']:.0f}",
        f"execution blocked         : {summary['execution_blocked']:.0f}",
        f"execution not run         : {summary['execution_not_run']:.0f}",
        f"conformance checks        : {summary['conformance_checks']:.0f}",
        f"conformance failures      : {summary['conformance_failures']:.0f}",
        f"hygiene findings          : {summary['hygiene_findings']:.0f}",
        "",
        report["notes"],
        "",
    ]
    for item in report["manuscript_inconsistencies"]:
        lines.append(f"manuscript finding {item['identifier']} at {item['location']}")
    if report["manuscript_inconsistencies"]:
        lines.append("")
    for item in report["hygiene"]["findings"]:
        lines.append(f"hygiene finding {item['family']} at {item['path']}: {item['detail']}")
    return "\n".join(lines) + "\n"


def write_artefacts(
    root: Path, probed: bool = False, *, include_gates: bool = True
) -> dict[str, str]:
    """Write every root artefact, with the manifest last."""

    base = Path(root).resolve()
    report = build_report(base, include_gates=include_gates)
    claim_map = {
        "paper": report["paper"],
        "summary": report["summary"],
        "claims": report["layer_one_paper_to_code"]["claims"],
        "claim_summary": report["layer_one_paper_to_code"]["summary"],
        "engineering_defaults": report["engineering_defaults"],
        "deviations": report["deviations"],
    }
    write_json(base / "claim_to_code.json", claim_map)
    write_json(base / "verification_report.json", report)
    atomic_write_text(base / "verification_summary.txt", render_summary(report))
    atomic_write_text(base / "dataset_urls.txt", render_dataset_urls(probed))
    write_integrity_manifest(base)
    return {
        "claim_to_code.json": "written",
        "verification_report.json": "written",
        "verification_summary.txt": "written",
        "dataset_urls.txt": "written",
        "integrity_manifest.json": "written",
    }


def claim_count() -> int:
    return len(CLAIMS)
