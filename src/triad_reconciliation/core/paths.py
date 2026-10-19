"""Repository-root resolution and path rendering.

Every path that reaches a report is rendered relative to the repository root, so a shipped
artefact never carries the directory layout of the machine that wrote it.
"""

from __future__ import annotations

from pathlib import Path


def repository_root(anchor: Path | None = None) -> Path:
    """Resolve the release root from an anchor inside the shipped tree.

    ``configs/experiment/<name>.yaml`` sits two levels below the root; joining that layout
    with a module anchor keeps the resolution independent of the working directory.
    """

    base = anchor if anchor is not None else Path(__file__).resolve()
    for candidate in base.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    return Path(__file__).resolve().parents[3]


def relative_to_root(path: Path, root: Path | None = None) -> str:
    """Render ``path`` relative to the release root, falling back to its name."""

    base = root if root is not None else repository_root()
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return resolved.name


def experiment_config_root(config_path: Path) -> Path:
    """The manifest root implied by a ``configs/experiment`` file."""

    return Path(config_path).resolve().parents[2]
