"""Write the released artefacts from a real verification pass.

The driver holds no results of its own: it runs the claim map, the execution checks, the
manuscript arithmetic and the hygiene scans, and writes what they returned. Run it twice with
no edits in between — the manifest-freshness check can only read a manifest a previous run
wrote.

Ref: the release contract.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from triad_reconciliation.core.logging_setup import configure_logging
from triad_reconciliation.core.paths import repository_root
from triad_reconciliation.verification.report import render_summary, write_artefacts

LOGGER = logging.getLogger("triad_reconciliation.cli.verify")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify the release and write its artefacts.")
    parser.add_argument(
        "--root", default=None, help="release root, defaulting to the resolved repository"
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level)
    root = Path(args.root) if args.root else repository_root()
    written = write_artefacts(root)
    for name, state in written.items():
        LOGGER.info("%s: %s", name, state)
    from triad_reconciliation.core.jsonio import read_json

    report = read_json(root / "verification_report.json")
    summary = render_summary(report)
    for line in summary.splitlines():
        LOGGER.info("%s", line)
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
