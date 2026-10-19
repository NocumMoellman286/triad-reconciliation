"""Integrity manifest.

The manifest is the only artefact that carries digests, and it is the last write of a run, so
the report it names is already final when the digests are taken. It excludes itself.

Ref: the release contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from triad_reconciliation.core.jsonio import digest_paths, manifest_digest, read_json, write_json

MANIFEST_NAME = "integrity_manifest.json"


def build_manifest(root: Path) -> dict[str, Any]:
    base = Path(root).resolve()
    entries = digest_paths(base, exclude=frozenset({MANIFEST_NAME}))
    return {
        "algorithm": "sha256",
        "excludes": [MANIFEST_NAME],
        "file_count": len(entries),
        "manifest_digest": manifest_digest(entries),
        "entries": entries,
    }


def write_integrity_manifest(root: Path) -> Path:
    base = Path(root).resolve()
    payload = build_manifest(base)
    return write_json(base / MANIFEST_NAME, payload)


def load_manifest(root: Path) -> dict[str, Any]:
    payload: dict[str, Any] = read_json(Path(root).resolve() / MANIFEST_NAME)
    return payload


def compare_with_tree(root: Path) -> dict[str, Any]:
    """Compare the shipped manifest with the live tree.

    The comparison reports counts and outcomes only; the digests stay in the manifest.
    """

    base = Path(root).resolve()
    shipped = load_manifest(base)
    entries = shipped.get("entries", {})
    live = digest_paths(base, exclude=frozenset({MANIFEST_NAME}))
    changed = sorted(path for path, value in entries.items() if live.get(path) != value)
    missing = sorted(path for path in entries if path not in live)
    unlisted = sorted(path for path in live if path not in entries)
    return {
        "shipped_files": len(entries),
        "live_files": len(live),
        "changed": changed,
        "missing": missing,
        "unlisted": unlisted,
        "digest_matches": manifest_digest(live) == shipped.get("manifest_digest"),
    }
