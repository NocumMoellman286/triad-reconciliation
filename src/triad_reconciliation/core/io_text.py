"""Plain-text block rendering for command-line reports.

Reports leave through this module so that every one of them is a ``.txt`` file with a stable
layout, whatever the length of the values it carries.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        if value != value:
            return "---"
        return f"{value:.6f}" if abs(value) < 10.0 else f"{value:,.3f}"
    return str(value)


def render_block(
    title: str, headers: Sequence[str], rows: Iterable[Sequence[Any]], *, note: str = ""
) -> str:
    """An aligned block whose column widths come from the data it carries."""

    rendered = [[format_cell(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in rendered:
        for position, cell in enumerate(row):
            if position < len(widths):
                widths[position] = max(widths[position], len(cell))
    lines = [title, "=" * len(title), ""]
    lines.append("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    lines.append("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rendered:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    if note:
        lines.extend(["", note])
    return "\n".join(lines) + "\n"
