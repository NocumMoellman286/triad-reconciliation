"""Evaluate the fitted model on the held-out region and write a plain-text report.

Ref: Sec. 2.1-2.4 (the reported analyses), Table 1 (the discrimination table), Table 3 (the
agreement table).
"""

from __future__ import annotations

import argparse
import logging

from triad_reconciliation.analysis.pipeline import (
    arbitration_analysis,
    discordance_analysis,
    fit_reconciliation,
)
from triad_reconciliation.analysis.trainer import reconciliation_outputs
from triad_reconciliation.cli.common import add_common, announce, load_context, variant_flags
from triad_reconciliation.core.io_text import render_block
from triad_reconciliation.core.jsonio import atomic_write_text, write_json
from triad_reconciliation.metrics.calibration import brier_score, expected_calibration_error
from triad_reconciliation.metrics.discrimination import (
    auroc,
    average_precision,
    sensitivity_at_specificity,
)

LOGGER = logging.getLogger("triad_reconciliation.cli.evaluate")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the reconciliation model on the held-out region."
    )
    add_common(parser)
    parser.add_argument("--variant", default="full", help="component-analysis variant name")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = load_context(args, LOGGER, "runs/evaluate")
    model = fit_reconciliation(context.development, context.config, variant_flags(args.variant))
    outputs = reconciliation_outputs(model, context.held_out, context.config.train)
    labels = context.held_out.progression
    metrics = {
        "auroc": auroc(outputs["progression"], labels),
        "auprc": average_precision(outputs["progression"], labels),
        "sensitivity_at_95_specificity": sensitivity_at_specificity(
            outputs["progression"], labels, 0.95
        ),
        "expected_calibration_error": expected_calibration_error(
            outputs["progression"], labels, 15
        ),
        "brier_score": brier_score(outputs["progression"], labels),
    }
    metrics.update(discordance_analysis(context.development, context.held_out, context.config))
    metrics.update(arbitration_analysis(outputs, context.held_out))
    write_json(context.output / "held_out_metrics.json", metrics)
    rows = [[name, value] for name, value in metrics.items()]
    body = render_block(
        "Held-out evaluation",
        ("quantity", "value"),
        rows,
        note=(
            "These numbers are produced by the released pipeline on the table the caller supplied. They "
            "are not the manuscript's held-out values: the analysed cohort is private and is not "
            "distributed, so every cohort-level value the article prints is reported NOT_RUN."
        ),
    )
    atomic_write_text(context.output / "held_out_report.txt", body)
    announce(context, context.output / "held_out_report.txt")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
