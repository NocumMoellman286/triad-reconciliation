"""Fit the reconciliation stack and write a checkpoint beside a run report.

Ref: Algorithm 1 (the training procedure), Sec. 5.8 (a deterministic analysis under the
recorded initial settings).
"""

from __future__ import annotations

import argparse
import logging

from triad_reconciliation.analysis.pipeline import fit_reconciliation
from triad_reconciliation.analysis.trainer import (
    build_optimizer,
    checkpoint_digest,
    save_checkpoint,
)
from triad_reconciliation.cli.common import add_common, load_context, variant_flags
from triad_reconciliation.core.jsonio import atomic_write_text, write_json

LOGGER = logging.getLogger("triad_reconciliation.cli.train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the reconciliation model on the development region."
    )
    add_common(parser)
    parser.add_argument("--variant", default="full", help="component-analysis variant name")
    return parser


def render_report(
    config_name: str, epochs: int, parameters: int, digest: str, checkpoint_name: str
) -> str:
    lines = [
        "Run report",
        "==========",
        f"experiment        : {config_name}",
        f"epochs            : {epochs}",
        f"parameters        : {parameters}",
        f"checkpoint digest : {digest}",
        f"checkpoint path   : {checkpoint_name}",
        "",
        "The checkpoint carries the seed it was trained under, so a resumed run restores the",
        "same stream of pseudo-random numbers.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    context = load_context(args, LOGGER, "runs/train")
    flags = variant_flags(args.variant)
    LOGGER.info("fitting %s over %d development pairs", flags.name, context.development.n_pairs)
    model = fit_reconciliation(context.development, context.config, flags)
    checkpoint = context.output / "reconciliation.pt"
    optimizer = build_optimizer(model, context.config.train)
    save_checkpoint(
        checkpoint,
        model,
        optimizer,
        seed=context.config.cohort.seed,
        epoch=context.config.train.epochs,
        metrics={"development_pairs": float(context.development.n_pairs)},
    )
    write_json(context.output / "run_metrics.json", {"parameters": model.parameter_count()})
    atomic_write_text(
        context.output / "run_report.txt",
        render_report(
            context.config.name,
            context.config.train.epochs,
            model.parameter_count(),
            checkpoint_digest(checkpoint),
            checkpoint.name,
        ),
    )
    from triad_reconciliation.cli.common import announce

    announce(context, context.output / "run_report.txt")
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
