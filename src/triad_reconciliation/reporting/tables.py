"""Reported tables.

Every renderer produces the same rows as a plain-text block and as a JSON-ready mapping, so a
reader can reconstruct a row without consulting another source.

Ref: Table 1 (discrimination and arbitration agreement), Table 2 (component analysis),
Table 3 (arbitration agreement and the design analyses), Table 5 (clinical outcomes),
Table 6 (silent-mode metrics), Table 7 (the comparator roster).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from triad_reconciliation.analysis.pipeline import StudyResult
from triad_reconciliation.comparators import roster
from triad_reconciliation.deployment.silent_mode import SilentModeMetrics
from triad_reconciliation.deployment.workflow import WorkflowOutcome

MISSING = "---"


def format_value(value: Any) -> str:
    """One cell, rendered the way the tables print it."""

    if value is None:
        return MISSING
    if isinstance(value, float):
        if value != value:
            return MISSING
        if abs(value) >= 100.0:
            return f"{value:,.1f}"
        return f"{value:.3f}"
    return str(value)


def render_table(title: str, headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Aligned plain-text block whose column widths come from the data."""

    rendered = [[format_value(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for position, cell in enumerate(row):
            widths[position] = max(widths[position], len(cell))
    lines = [title, "=" * len(title), ""]
    lines.append(
        "  ".join(header.ljust(widths[position]) for position, header in enumerate(headers))
    )
    lines.append("  ".join("-" * widths[position] for position in range(len(headers))))
    for row in rendered:
        lines.append("  ".join(cell.ljust(widths[position]) for position, cell in enumerate(row)))
    return "\n".join(lines)


def discrimination_rows(result: StudyResult) -> list[list[Any]]:
    """Table 1, one row per roster entry in the printed order."""

    metrics = result.table_rows()
    by_name = {row["method"]: row for row in metrics}
    rows: list[list[Any]] = []
    for position, name in enumerate(roster.roster_names(), start=1):
        row = by_name.get(name)
        if row is None:
            continue
        rows.append(
            [
                position,
                name,
                row["auroc"],
                row["auprc"],
                row["sensitivity"],
                row["ece"],
                row["brier"],
                row["kappa"],
            ]
        )
    return rows


def render_discrimination(result: StudyResult) -> str:
    return render_table(
        "Progression discrimination and arbitration agreement on the held-out region",
        ("#", "method", "AUROC", "AUPRC", "Sens@95Spec", "ECE", "Brier", "kappa"),
        discrimination_rows(result),
    )


def variant_rows(result: StudyResult) -> list[list[Any]]:
    """Table 2, the component analysis in the order the flags are declared."""

    return [
        [
            row["variant"],
            row["auroc_all"],
            row["auroc_discordant"],
            row["kappa"],
            row["delta_discordant"],
        ]
        for row in result.variant_rows()
    ]


def render_variants(result: StudyResult) -> str:
    return render_table(
        "Component analysis on the held-out region",
        ("variant", "AUROC (all)", "AUROC (discordant)", "kappa", "delta discordant"),
        variant_rows(result),
    )


def design_rows(result: StudyResult) -> list[list[Any]]:
    """Table 3's model rows beside the design analyses of the same table."""

    rows: list[list[Any]] = [
        [
            "recorded arbitration, reader vs reader (ceiling)",
            float("nan"),
            MISSING,
            MISSING,
        ],
        [
            "reconciliation model",
            result.arbitration.get("kappa", float("nan")),
            result.arbitration.get("separation_structural", float("nan")),
            result.arbitration.get("separation_functional", float("nan")),
        ],
    ]
    for name in (
        "triad_concordant",
        "unconstrained_concordant",
        "static_concordant",
        "triad_discordant",
        "unconstrained_discordant",
        "static_discordant",
    ):
        if name in result.discordance:
            rows.append([name, result.discordance[name], MISSING, MISSING])
    rows.append(
        [
            "identifiable share of the stream weight",
            result.identifiability.get("identifiable_share", float("nan")),
            result.identifiability.get("median_gap_discordant", float("nan")),
            result.identifiability.get("median_gap_concordant", float("nan")),
        ]
    )
    return rows


def render_design(result: StudyResult) -> str:
    return render_table(
        "Arbitration agreement and the design analyses",
        ("analysis", "value", "structural side", "functional side"),
        design_rows(result),
    )


def workflow_rows(outcome: WorkflowOutcome | None) -> list[list[Any]]:
    """Table 5's workflow rows; a missing outcome is reported as undefined, not as a zero."""

    if outcome is None:
        labels = (
            "sensitivity at matched specificity",
            "treatment-escalation agreement",
            "review-interval agreement",
            "median lead time (months)",
            "net reclassification improvement",
            "net benefit at threshold",
        )
        return [[label, MISSING, MISSING, MISSING] for label in labels]
    mapping = outcome.as_mapping()
    return [
        ["sensitivity at matched specificity", mapping["sensitivity_matched"], MISSING, MISSING],
        ["treatment-escalation agreement", mapping["escalation_agreement"], MISSING, MISSING],
        ["review-interval agreement", mapping["interval_agreement"], MISSING, MISSING],
        ["median lead time (months)", mapping["median_lead_time_months"], MISSING, MISSING],
        [
            "net reclassification improvement",
            mapping["net_reclassification_improvement"],
            MISSING,
            MISSING,
        ],
        ["net benefit at threshold", mapping["net_benefit"], MISSING, MISSING],
    ]


def render_workflow(outcome: WorkflowOutcome | None) -> str:
    return render_table(
        "Clinical outcomes on the held-out region",
        ("outcome", "value", "reference", "delta"),
        workflow_rows(outcome),
    )


def silent_mode_rows(metrics: SilentModeMetrics | None) -> list[list[Any]]:
    if metrics is None:
        labels = (
            "eligible encounters",
            "scored encounters (coverage)",
            "unscored, missing imaging or perimetry",
            "unscored, processing or interface failure",
            "median end-to-end latency per visit-pair (s)",
            "deferred decisions",
            "attribution emitted with every scored decision",
            "reviewer engagement rate",
            "override rate on the arbitration direction",
            "acceptance rate on deferred decisions",
            "interpretability audit",
        )
        return [[label, MISSING] for label in labels]
    mapping = metrics.as_mapping()
    return [
        ["eligible encounters", mapping["eligible"]],
        ["scored encounters (coverage)", mapping["scored"]],
        ["coverage share", mapping["coverage"]],
        ["unscored, missing imaging or perimetry", mapping["unscored_missing_share"]],
        ["unscored, processing or interface failure", mapping["unscored_processing_share"]],
        ["median end-to-end latency per visit-pair (s)", mapping["median_latency_seconds"]],
        ["interquartile range, lower (s)", mapping["latency_iqr_lower"]],
        ["interquartile range, upper (s)", mapping["latency_iqr_upper"]],
        ["deferred decisions", mapping["deferred_share"]],
        ["attribution emitted with every scored decision", mapping["attribution_share"]],
        ["reviewer engagement rate", mapping["engagement_share"]],
        ["override rate on the arbitration direction", mapping["override_share"]],
        ["acceptance rate on deferred decisions", mapping["deferred_acceptance_share"]],
        ["interpretability audit", mapping["interpretability_share"]],
    ]


def render_silent_mode(metrics: SilentModeMetrics | None) -> str:
    return render_table(
        "Silent-mode implementation metrics", ("metric", "value"), silent_mode_rows(metrics)
    )


def roster_rows() -> list[list[Any]]:
    return [[spec.table_row, spec.name, spec.family, spec.inputs] for spec in roster.ROSTER]


def render_roster() -> str:
    return render_table("Comparator roster", ("#", "comparator", "family", "input"), roster_rows())


def stratum_rows(result: StudyResult) -> list[list[Any]]:
    return [[name, count] for name, count in sorted(result.stratum_sizes.items())]


def render_strata(result: StudyResult) -> str:
    return render_table(
        "Declared strata on the held-out region", ("stratum", "pairs"), stratum_rows(result)
    )


def all_tables(result: StudyResult) -> str:
    blocks = [
        render_discrimination(result),
        render_variants(result),
        render_design(result),
        render_workflow(None),
        render_silent_mode(None),
        render_roster(),
        render_strata(result),
    ]
    return "\n\n\n".join(blocks) + "\n"
