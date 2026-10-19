"""Atomic artefact writing and JSON round-tripping."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

TextWriter = Callable[[Any], str]


def atomic_write_text(target: Path, payload: str, *, mode: int = 0o644) -> Path:
    """Write ``payload`` to ``target`` through a temporary file in the same directory."""

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, raw = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    staged = Path(raw)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.chmod(staged, mode)
        os.replace(staged, target)
    except BaseException:
        if staged.exists():
            staged.unlink()
        raise
    return target


def write_json(target: Path, payload: Any, *, indent: int = 2) -> Path:
    """Serialise ``payload`` with sorted keys so two identical runs match byte for byte."""

    body = json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False)
    return atomic_write_text(target, body + "\n")


def read_json(source: Path) -> Any:
    return json.loads(Path(source).read_text(encoding="utf-8"))


def write_jsonl(target: Path, rows: Iterable[dict[str, Any]]) -> Path:
    body = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    return atomic_write_text(target, body)


def read_jsonl(source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in Path(source).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def digest_paths(
    root: Path,
    *,
    skip_names: frozenset[str] = frozenset(
        {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git"}
    ),
    skip_top_level: frozenset[str] = frozenset({"data", "runs", "models"}),
    exclude: frozenset[str] = frozenset(),
) -> dict[str, str]:
    """Digest every shipped file below ``root``.

    Build caches are skipped wherever they appear; a data or artefact root is skipped only
    at the top level of the tree so that a source package directory named ``data`` still
    contributes its files.
    """

    import hashlib

    entries: dict[str, str] = {}
    base = Path(root).resolve()
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if any(part in skip_names for part in relative.parts):
            continue
        if relative.parts[0] in skip_top_level and len(relative.parts) > 1:
            continue
        key = relative.as_posix()
        if key in exclude:
            continue
        entries[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return entries


def manifest_digest(entries: dict[str, str]) -> str:
    """Fold a path-to-digest mapping into one digest in walk order."""

    import hashlib

    flat = "\n".join(f"{path}:{value}" for path, value in entries.items())
    return hashlib.sha256(flat.encode("utf-8")).hexdigest()
