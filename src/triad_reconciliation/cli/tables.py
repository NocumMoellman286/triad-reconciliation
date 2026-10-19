"""Render every reported table from a study run.

Ref: Tables 1-7 of the manuscript, rendered from what the released pipeline produced.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from triad_reconciliation.analysis.pipeline import run_study
from triad_reconciliation.cli.common import add_common, announce, load_context
from triad_reconciliation.core.io_text import render_block
from triad_reconciliation.core.jsonio import atomic_write_text, write_json
from triad_reconciliation.reporting.tables import (
    discrimination_rows,
    render_design,
    render_discrimination,
    render_roster,
    render_silent_mode,
    render_strata,
    render_variants,
    render_workflow,
    roster_rows,
    silent_mode_rows,
    stratum_rows,
    variant_rows,
    workflow_rows,
)

LOGGER = logging.getLogger("triad_reconciliation.cli.tables")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render the reported tables from a study run.")
    add_common(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = load_context(args, LOGGER, "runs/tables")
    result = run_study(context.config, Path(args.cohort) if args.cohort else None)
    directory = context.output
    blocks = [
        render_discrimination(result),
        render_variants(result),
        render_design(result),
        render_workflow(None),
        render_silent_mode(None),
        render_roster(),
        render_strata(result),
    ]
    note = (
        "Rows produced by the released pipeline on the table the caller supplied are marked by their "
        "own "
        "values; the workflow and silent-mode tables carry no value because no deployment log ships "
        "with this release and the manuscript's own values are held by the participating sites."
    )
    atomic_write_text(directory / "reported_tables.txt", "\n\n".join(blocks) + "\n" + note + "\n")
    write_json(
        directory / "table_rows.json",
        {
            "discrimination": [list(row) for row in discrimination_rows(result)],
            "components": [list(row) for row in variant_rows(result)],
            "roster": [list(row) for row in roster_rows()],
            "strata": [list(row) for row in stratum_rows(result)],
            "workflow": [list(row) for row in workflow_rows(None)],
            "silent_mode": [list(row) for row in silent_mode_rows(None)],
            "deferral": dict(result.deferral),
            "unguarded": dict(result.unguarded),
            "identifiability": dict(result.identifiability),
            "excess_risk": dict(result.excess_risk),
        },
    )
    atomic_write_text(
        directory / "table_summary.txt",
        render_block(
            "Study run summary",
            ("quantity", "value"),
            [
                ["cohort table", result.source],
                ["held-out pairs", len(result.partition.held_out)],
                ["development pairs", len(result.partition.development)],
                ["methods scored", len(result.methods)],
                ["variants fitted", len(result.variants)],
                ["deferral share", result.deferral.get("deferral_share", float("nan"))],
                ["conformal coverage", result.deferral.get("coverage", float("nan"))],
            ],
            note=note,
        ),
    )
    announce(context, directory / "reported_tables.txt")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
