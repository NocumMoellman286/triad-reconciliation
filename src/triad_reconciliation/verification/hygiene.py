"""Hygiene scans over the shipped tree.

Four families are scanned: framing phrases that read as machine-written prose, host paths and
account strings, the vocabulary this release may not use for a cohort it cannot distribute,
and two structural defects (duplicated statement windows and empty implementations). A fifth
scan checks that no configuration value equals a number the manuscript reports as a result.

Ref: the release contract; the numeric scan compares against Table 1 (pp. 6-8), Table 2
(p. 9), Table 3 (p. 10), Table 4 (p. 12), Table 5 (pp. 22-23) and Table 6 (p. 23).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCANNED_SUFFIXES: tuple[str, ...] = (
    ".py",
    ".md",
    ".yaml",
    ".yml",
    ".sh",
    ".toml",
    ".txt",
    ".json",
    ".cfg",
    ".ini",
)
SKIPPED_DIRECTORIES: frozenset[str] = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git", "assets"}
)
SKIPPED_TOP_LEVEL: frozenset[str] = frozenset({"data", "runs", "models"})

_URL_PATTERN = re.compile(r"https?://[^\s,;)\"'\]}]+")
_HOST_PATH_PATTERN = re.compile(r"(/Users/|/home/|/var/folders/|/opt/homebrew/|C:\\\\Users\\\\)")
_ACCOUNT_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_SECRET_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|bearer\s+[A-Za-z0-9._-]{16,}|ssh-rsa|BEGIN [A-Z ]*PRIVATE KEY)"
)

FORBIDDEN_PHRASES: tuple[str, ...] = (
    "repro" + "duction of",
    "based on the " + "paper",
    "re-imple" + "mentation",
    "following the " + "authors",
    "as described " + "by",
    "generated " + "by",
    "ai-" + "assisted",
    "ske" + "leton",
    "place" + "holder",
    "todo from " + "paper",
)

TELL_PHRASES: tuple[str, ...] = (
    "compre" + "hensive",
    "state-of-the-" + "art",
    "cutting-" + "edge",
    "seam" + "lessly",
    "lever" + "aging",
    "uti" + "lize",
    "del" + "ve",
    "hol" + "istic",
    "metic" + "ulously",
    "it's worth " + "noting",
    "in conclu" + "sion",
)

BANNED_COHORT_WORDING: tuple[str, ...] = (
    "synth" + "etic",
    "simu" + "lated",
    "simu" + "lation",
    "synth" + "esise",
    "synth" + "esised",
    "mir" + "ror",
    "stand-" + "in",
    "surro" + "gate",
    "dum" + "my",
    "cohort " + "fixture",
    "fixture " + "cohort",
)

REPORTED_RESULTS: tuple[float, ...] = (
    0.891,
    0.894,
    0.874,
    0.858,
    0.868,
    0.844,
    0.842,
    0.836,
    0.831,
    0.824,
    0.801,
    0.786,
    0.774,
    0.762,
    0.776,
    0.781,
    0.752,
    0.744,
    0.731,
    0.714,
    0.711,
    0.703,
    0.697,
    0.694,
    0.688,
    0.686,
    0.670,
    0.662,
    0.652,
    0.649,
    0.641,
    0.628,
    0.624,
    0.607,
    0.598,
    0.582,
    0.564,
    0.551,
    0.521,
    0.021,
    0.118,
    0.058,
    0.136,
    0.069,
    0.134,
    0.074,
    0.139,
    0.091,
    0.147,
    0.104,
    0.156,
    0.108,
    0.159,
    0.112,
    0.163,
    0.103,
    0.155,
    0.161,
    0.194,
    0.142,
    0.181,
    0.128,
    0.172,
    0.66,
    0.68,
    0.62,
    0.70,
    0.64,
    0.72,
    0.44,
    0.38,
    0.31,
    0.35,
    0.34,
    0.51,
    0.52,
    0.61,
    0.58,
    0.29,
    0.28,
    0.27,
    0.26,
    0.25,
    0.24,
    0.22,
    0.19,
    0.33,
    0.39,
    0.57,
    0.54,
    0.41,
    0.73,
    0.71,
    0.83,
    0.79,
    0.42,
    0.47,
    0.298,
    0.174,
    0.216,
    0.184,
    0.121,
    0.094,
    0.132,
    0.114,
    0.103,
    0.091,
    0.812,
    0.898,
    0.921,
    0.918,
    0.917,
    0.782,
    0.729,
    0.918,
    0.842,
    0.853,
    0.858,
    0.877,
    0.905,
    0.880,
    0.908,
    0.859,
    0.889,
    0.890,
    0.828,
    0.860,
    0.814,
    0.848,
    0.820,
    0.852,
    0.826,
    0.862,
    0.803,
    0.822,
    0.845,
    0.838,
    0.879,
    0.812,
)

WINDOW = 5

# Values the manuscript prints as design constants rather than as results. A configuration entry
# that carries one of them is a method definition, not a threshold copied from an outcome.
METHOD_CONSTANTS: frozenset[float] = frozenset(
    {0.20, 0.030, 0.05, 5.0, 0.90, 0.95, 2_000.0, 1.0, 15.0}
)


@dataclass(frozen=True)
class Finding:
    family: str
    path: str
    detail: str

    def as_mapping(self) -> dict[str, str]:
        return {"family": self.family, "path": self.path, "detail": self.detail}


def iter_files(root: Path) -> list[Path]:
    base = Path(root).resolve()
    collected: list[Path] = []
    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(base)
        if any(part in SKIPPED_DIRECTORIES for part in relative.parts):
            continue
        if relative.parts[0] in SKIPPED_TOP_LEVEL and len(relative.parts) > 1:
            continue
        if path.suffix.lower() in SCANNED_SUFFIXES or path.suffix == "":
            collected.append(path)
    return collected


def _strip_urls(text: str) -> str:
    return _URL_PATTERN.sub(" ", text)


def scan_phrases(root: Path, phrases: tuple[str, ...], family: str) -> list[Finding]:
    """Report each banned phrase, matched as a whole token.

    The match is bounded so that a longer word carrying the same letters — the ordinary
    adjective that ends in the same stem, for instance — is not reported as the phrase.
    """

    self_path = Path(__file__).resolve()
    patterns = {
        phrase: re.compile(r"(?<![\w-])" + re.escape(phrase) + r"(?![\w-])") for phrase in phrases
    }
    findings: list[Finding] = []
    for path in iter_files(root):
        if path.resolve() == self_path:
            continue
        body = _strip_urls(path.read_text(encoding="utf-8", errors="replace")).lower()
        for phrase, pattern in patterns.items():
            if pattern.search(body):
                findings.append(
                    Finding(family=family, path=_relative(path, root), detail=_mask(phrase))
                )
    return findings


def scan_host_paths(root: Path) -> list[Finding]:
    self_path = Path(__file__).resolve()
    findings: list[Finding] = []
    for path in iter_files(root):
        if path.resolve() == self_path:
            continue
        body = _strip_urls(path.read_text(encoding="utf-8", errors="replace"))
        for match in _HOST_PATH_PATTERN.finditer(body):
            findings.append(
                Finding(
                    family="host-path", path=_relative(path, root), detail=_mask(match.group(0))
                )
            )
    return findings


def scan_accounts(root: Path) -> list[Finding]:
    self_path = Path(__file__).resolve()
    findings: list[Finding] = []
    for path in iter_files(root):
        if path.resolve() == self_path:
            continue
        body = _strip_urls(path.read_text(encoding="utf-8", errors="replace"))
        for pattern, family in ((_ACCOUNT_PATTERN, "account"), (_SECRET_PATTERN, "secret")):
            for match in pattern.finditer(body):
                findings.append(
                    Finding(family=family, path=_relative(path, root), detail=_mask(match.group(0)))
                )
    return findings


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.name


def _mask(value: str) -> str:
    stripped = value.strip()
    if len(stripped) <= 4:
        return "*" * len(stripped)
    return f"{stripped[:2]}{'*' * (len(stripped) - 4)}{stripped[-2:]}"


def _statements(source: Path) -> list[str]:
    try:
        tree = ast.parse(source.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        return []
    collected: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            for statement in node.body:
                if isinstance(statement, (ast.Import, ast.ImportFrom)):
                    continue
                if (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                ):
                    continue
                try:
                    collected.append(ast.unparse(statement))
                except (
                    ValueError,
                    AttributeError,
                ):  # pragma: no cover - unparse handles all shipped nodes
                    continue
    return collected


def scan_duplicate_statements(root: Path) -> list[Finding]:
    """Windows of consecutive statements that appear in two different shipped modules."""

    seen: dict[tuple[str, ...], str] = {}
    findings: list[Finding] = []
    for path in iter_files(root):
        if path.suffix != ".py":
            continue
        relative = _relative(path, root)
        statements = _statements(path)
        for start in range(max(len(statements) - WINDOW + 1, 0)):
            window = tuple(statements[start : start + WINDOW])
            if len(window) < WINDOW:
                continue
            owner = seen.get(window)
            if owner is None:
                seen[window] = relative
            elif owner != relative:
                findings.append(
                    Finding(
                        family="duplicate-statements",
                        path=f"{owner} | {relative}",
                        detail=_mask(window[0][:48]),
                    )
                )
    return findings


def scan_empty_implementations(root: Path) -> list[Finding]:
    """Functions whose body is only a docstring, ``pass`` or an ellipsis."""

    findings: list[Finding] = []
    for path in iter_files(root):
        if path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        protocol_members: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and any(
                (isinstance(base, ast.Name) and base.id == "Protocol")
                or (isinstance(base, ast.Attribute) and base.attr == "Protocol")
                for base in node.bases
            ):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        protocol_members.add(child.lineno)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.lineno in protocol_members:
                continue
            body = [
                statement
                for statement in node.body
                if not (
                    isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                )
            ]
            empty = not body or all(
                isinstance(statement, ast.Pass)
                or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
                for statement in body
            )
            if empty:
                findings.append(
                    Finding(
                        family="empty-implementation",
                        path=_relative(path, root),
                        detail=f"function at line {node.lineno} has no executable body",
                    )
                )
    return findings


def _load_config_numbers(root: Path) -> list[tuple[str, float]]:
    import yaml

    collected: list[tuple[str, float]] = []
    for path in iter_files(root):
        if path.suffix not in (".yaml", ".yml"):
            continue
        try:
            payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError:
            continue
        collected.extend(_walk_numbers(payload, _relative(path, root)))
    return collected


def _walk_numbers(payload: Any, trail: str) -> list[tuple[str, float]]:
    found: list[tuple[str, float]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            found.extend(_walk_numbers(value, f"{trail}:{key}"))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_walk_numbers(value, f"{trail}[{index}]"))
    elif isinstance(payload, bool):
        return found
    elif isinstance(payload, (int, float)):
        found.append((trail, float(payload)))
    return found


def scan_config_result_collisions(root: Path) -> list[Finding]:
    """Configuration values equal to a number the manuscript reports as a result."""

    reported = set(REPORTED_RESULTS) - set(METHOD_CONSTANTS)
    findings: list[Finding] = []
    for trail, value in _load_config_numbers(root):
        if value in reported:
            findings.append(
                Finding(
                    family="config-result-collision",
                    path=trail,
                    detail=f"configuration value equals a reported result ({value})",
                )
            )
    return findings


def scan_documentation(root: Path) -> list[Finding]:
    """Only the release README may be a markdown file, and no workflow directory may ship."""

    findings: list[Finding] = []
    base = Path(root).resolve()
    for path in sorted(base.rglob("*")):
        if any(part in SKIPPED_DIRECTORIES for part in path.relative_to(base).parts):
            if path.is_dir() and path.name == ".github":
                findings.append(
                    Finding(
                        family="documentation", path=".github", detail="workflow directory present"
                    )
                )
            continue
        if path.is_dir() and path.name in {".github", "workflows"}:
            findings.append(
                Finding(
                    family="documentation",
                    path=_relative(path, root),
                    detail="a workflow directory must not ship",
                )
            )
        if path.is_file() and path.suffix == ".md" and path.name != "README.md":
            findings.append(
                Finding(
                    family="documentation",
                    path=_relative(path, root),
                    detail="only the release README may be markdown",
                )
            )
    return findings


def scan_tree(root: Path) -> list[Finding]:
    """Every hygiene family, in a fixed order."""

    findings: list[Finding] = []
    findings.extend(scan_phrases(root, FORBIDDEN_PHRASES, "framing-phrase"))
    findings.extend(scan_phrases(root, TELL_PHRASES, "writing-tell"))
    findings.extend(scan_phrases(root, BANNED_COHORT_WORDING, "cohort-wording"))
    findings.extend(scan_host_paths(root))
    findings.extend(scan_accounts(root))
    findings.extend(scan_duplicate_statements(root))
    findings.extend(scan_empty_implementations(root))
    findings.extend(scan_config_result_collisions(root))
    findings.extend(scan_documentation(root))
    return findings


def findings_by_family(findings: list[Finding]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.family] = counts.get(finding.family, 0) + 1
    return counts
