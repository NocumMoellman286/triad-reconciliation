"""Logging setup for the command-line entry points."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Install a single stream handler so library modules never print directly."""

    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_FORMAT)
    for name in ("matplotlib", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)
