"""Path, argument and context helpers shared by the command-line entry points."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triad_reconciliation.analysis.batching import FlatFeatures, build_flat_features
from triad_reconciliation.analysis.pipeline import STAGING_HINT, build_table
from triad_reconciliation.cohort.splits import Partition, split_regions
from triad_reconciliation.core.config import ExperimentConfig, apply_overrides, load_experiment
from triad_reconciliation.core.logging_setup import configure_logging
from triad_reconciliation.core.paths import repository_root
from triad_reconciliation.core.records import CohortTable
from triad_reconciliation.core.seeding import set_seed
from triad_reconciliation.models.triad import VariantFlags

VARIANTS: dict[str, VariantFlags] = {
    "full": VariantFlags(),
    "without_reconciliation_state": VariantFlags(use_reconciliation_state=False),
    "without_arbitration_head": VariantFlags(use_arbitration_head=False),
    "without_attribution_consistency": VariantFlags(use_attribution_consistency=False),
    "without_agreement_state_reweighting": VariantFlags(use_agreement_reweighting=False),
    "without_monotone_causal_alignment": VariantFlags(use_monotone_alignment=False),
    "with_numeric_perimetry_input": VariantFlags(use_language_encoder=False),
    "without_calibrated_deferral": VariantFlags(use_calibrated_deferral=False),
    "interaction_state_and_head": VariantFlags(
        use_reconciliation_state=False, use_arbitration_head=False
    ),
}


@dataclass(frozen=True)
class RunContext:
    """Everything a command needs once its arguments have been read."""

    config: ExperimentConfig
    root: Path
    table: CohortTable
    partition: Partition
    development: FlatFeatures
    held_out: FlatFeatures
    output: Path
    logger: logging.Logger


def add_common(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--experiment",
        default="configs/experiment/main.yaml",
        help="experiment file, relative to the release root",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="section.key=value override applied on top of the experiment file",
    )
    parser.add_argument(
        "--cohort",
        default=None,
        help="path to the cohort table the caller supplies; required by every command "
        "that reads records",
    )
    parser.add_argument("--out", default=None, help="directory for reports written by this command")
    parser.add_argument("--log-level", default="INFO")
    return parser


def variant_flags(name: str) -> VariantFlags:
    if name not in VARIANTS:
        raise ValueError(f"unknown variant {name!r}; known variants are {sorted(VARIANTS)}")
    return VARIANTS[name]


def load_context(args: argparse.Namespace, logger: logging.Logger, default_out: str) -> RunContext:
    """Resolve the experiment, read the cohort table and prepare the two regions.

    A command that reads records cannot run without a staged table; it stops with the staging
    instruction rather than producing anything.
    """

    configure_logging(str(args.log_level))
    root = repository_root()
    path = Path(args.experiment)
    config = apply_overrides(
        load_experiment(path if path.is_absolute() else root / path, root=root),
        list(args.override),
    )
    if args.cohort is None:
        raise SystemExit(f"error: {STAGING_HINT}")
    table = build_table(config, Path(args.cohort))
    set_seed(config.cohort.seed)
    partition = split_regions(table, held_out_region=config.cohort.held_out_region)
    development = build_flat_features(partition.development, config.model)
    held_out = build_flat_features(partition.held_out, config.model)
    output = Path(args.out) if args.out else root / default_out
    output.mkdir(parents=True, exist_ok=True)
    return RunContext(
        config=config,
        root=root,
        table=table,
        partition=partition,
        development=development,
        held_out=held_out,
        output=output,
        logger=logger,
    )


def announce(context: RunContext, written: Any) -> None:
    """Record where a command's report landed, relative to the release root."""

    target = Path(written)
    try:
        rendered = target.resolve().relative_to(context.root.resolve()).as_posix()
    except ValueError:
        rendered = target.name
    context.logger.info("wrote %s", rendered)
