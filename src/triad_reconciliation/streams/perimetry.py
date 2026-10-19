"""Perimetry report serialisation.

The functional encoder consumes a report as a sequence of text tokens, so the report schema
is the interface between the cohort layer and the language encoder.

Ref: Sec. 5.1 (the functional stream is a series of automated perimetry reports),
Sec. 5.4 (the language encoder receives the serialised report), Code availability in the
manuscript names a schema for perimetry serialisation.
"""

from __future__ import annotations

from dataclasses import dataclass

from triad_reconciliation.core.records import N_PERIMETRY_POINTS, PerimetryVisit

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
FIELD_PREFIX = "pt"
SENSITIVITY_PREFIX = "sens"
RELIABLE_TOKEN = "rel_ok"
UNRELIABLE_TOKEN = "rel_poor"
INDEX_NAMES: tuple[str, ...] = ("md", "pd", "vfi", "fp", "fn", "fix")

MORPHOMETRY_NAMES: tuple[str, ...] = (
    "vcdr",
    "hcdr",
    "rim_area",
    "disc_area",
    "cup_volume",
    "rnfl_mean",
    "rnfl_inferior",
    "image_quality",
)


@dataclass(frozen=True)
class PerimetryVocabulary:
    """Fixed token vocabulary built from field names, value buckets and reliability marks."""

    tokens: tuple[str, ...]
    index: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.tokens)

    def encode(self, token: str) -> int:
        return self.index.get(token, self.index[PAD_TOKEN])


def bucket_token(prefix: str, value: float, low: float, high: float, buckets: int) -> str:
    """The vocabulary token a scalar falls in, using the serialisation's own buckets."""

    span = (high - low) / (buckets - 1)
    position = int(round((float(value) - low) / span))
    return f"{prefix}{min(max(position, 0), buckets - 1):02d}"


def _bucket(value: float, low: float, high: float, buckets: int) -> str:
    return bucket_token("b", value, low, high, buckets)


def build_vocabulary() -> PerimetryVocabulary:
    """Assemble the vocabulary in a fixed order so a rerun reproduces every token id."""

    tokens: list[str] = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, RELIABLE_TOKEN, UNRELIABLE_TOKEN]
    tokens += [f"{FIELD_PREFIX}{index:02d}" for index in range(N_PERIMETRY_POINTS)]
    tokens += [f"sens{index:02d}" for index in range(21)]
    for name in ("md", "pd", "vfi", "fp", "fn", "fix"):
        tokens += [f"{name}{index:02d}" for index in range(17)]
    return PerimetryVocabulary(
        tokens=tuple(tokens), index={token: i for i, token in enumerate(tokens)}
    )


VOCABULARY = build_vocabulary()


def serialise_report(report: PerimetryVisit) -> str:
    """Render one perimetry report as whitespace-separated tokens."""

    parts: list[str] = [BOS_TOKEN]
    for position, value in enumerate(report.sensitivity_db[:N_PERIMETRY_POINTS]):
        parts.append(f"{FIELD_PREFIX}{position:02d}")
        parts.append(f"{SENSITIVITY_PREFIX}{min(max(int(round(value / 2.0)), 0), 20):02d}")
    parts.append(f"md{_bucket(report.mean_deviation_db, -30.0, 2.0, 17)}")
    parts.append(f"pd{_bucket(report.pattern_deviation_db, -30.0, 2.0, 17)}")
    parts.append(f"vfi{_bucket(report.visual_field_index, 0.0, 100.0, 17)}")
    parts.append(f"fp{_bucket(report.false_positive_rate, 0.0, 0.6, 17)}")
    parts.append(f"fn{_bucket(report.false_negative_rate, 0.0, 0.6, 17)}")
    parts.append(f"fix{min(report.fixation_losses, 16):02d}")
    parts.append(RELIABLE_TOKEN if report.reliability_ok else UNRELIABLE_TOKEN)
    parts.append(EOS_TOKEN)
    return " ".join(parts)


def encode_report(report: PerimetryVisit, max_tokens: int) -> tuple[tuple[int, ...], int]:
    """Encode a report to padded token ids and report the unpadded length."""

    ids = [VOCABULARY.encode(piece) for piece in serialise_report(report).split()]
    trimmed = ids[:max_tokens]
    length = len(trimmed)
    if length < max_tokens:
        trimmed = trimmed + [VOCABULARY.index[PAD_TOKEN]] * (max_tokens - length)
    return tuple(trimmed), length


def serialise_series(reports: tuple[PerimetryVisit, ...], max_visits: int) -> list[str]:
    """Serialise a perimetry series, keeping the most recent visits in clinical order."""

    ordered = sorted(reports, key=lambda item: item.tested_years)[-max_visits:]
    return [serialise_report(report) for report in ordered]
