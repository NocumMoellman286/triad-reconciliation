"""The zero-shot multimodal language comparator.

The comparator is prompted with the serialised perimetry series beside a textual summary of
the optic disc and answers without any fitting. No language-model weights travel with this
release, so the scorer here is the prompt template's own lexical prior over the same
vocabulary the functional encoder reads.

Ref: Sec. 2.1 (the multimodal language model prompted zero-shot), Table 1 row 11, Table 7
row 13.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad_reconciliation.analysis.batching import FlatFeatures
from triad_reconciliation.streams.perimetry import (
    RELIABLE_TOKEN,
    UNRELIABLE_TOKEN,
    bucket_token,
)

FIELD_BUCKETS = 17
DEVIATION_RANGE = (-30.0, 2.0)
INDEX_RANGE = (0.0, 100.0)
RELIABILITY_SHARE = 0.03
STRUCTURAL_KEYS = ("vcdr", "rim_area", "rnfl_mean", "rnfl_inferior")
WORSENING_WEIGHT = {
    "vfi": 0.44,
    "md": 0.30,
    "pd": 0.16,
    "fp": 0.02,
    "fn": 0.02,
    "fix": 0.06,
}


@dataclass(frozen=True)
class PromptBundle:
    """The prompt the comparator reads and the pair it was built for."""

    pair_id: str
    prompt: str


def disc_summary(morphometry: np.ndarray) -> str:
    """A short textual reading of the most recent optic-disc summary."""

    summary = morphometry[: len(STRUCTURAL_KEYS)]
    return " ".join(
        f"{name}={value:.3f}" for name, value in zip(STRUCTURAL_KEYS, summary, strict=True)
    )


def report_tokens(field_row: np.ndarray) -> tuple[str, ...]:
    """The vocabulary tokens one perimetry visit contributes to the prompt."""

    mean_deviation, pattern, index, false_positive, false_negative, fixations = (
        float(field_row[0]),
        float(field_row[1]),
        float(field_row[2]),
        float(field_row[3]),
        float(field_row[4]),
        float(field_row[5]),
    )
    reliable = false_positive <= 0.15 and false_negative <= 0.20 and fixations <= 2.0
    return (
        bucket_token("md", mean_deviation, *DEVIATION_RANGE, FIELD_BUCKETS),
        bucket_token("pd", pattern, *DEVIATION_RANGE, FIELD_BUCKETS),
        bucket_token("vfi", index, *INDEX_RANGE, FIELD_BUCKETS),
        bucket_token("fp", false_positive, 0.0, 0.6, FIELD_BUCKETS),
        bucket_token("fn", false_negative, 0.0, 0.6, FIELD_BUCKETS),
        f"fix{min(int(fixations), 16):02d}",
        RELIABLE_TOKEN if reliable else UNRELIABLE_TOKEN,
    )


def build_prompt(features: FlatFeatures, row: int) -> PromptBundle:
    """One prompt per visit-pair: the disc summary beside the serialised perimetry series."""

    summary = disc_summary(features.structural_summary()[row])
    visits: list[str] = []
    for position in range(features.field_mask.shape[1]):
        if not features.field_mask[row, position]:
            continue
        stamps = " ".join(report_tokens(features.field_indices[row, position]))
        visits.append(f"visit{position:02d} {stamps}")
    return PromptBundle(
        pair_id=features.pair_ids[row], prompt=f"disc {summary} :: " + " ; ".join(visits)
    )


def _bucket_direction(token: str, prefix: str) -> float:
    """How far into the worse end of a bucket scale a token sits."""

    if not token.startswith(prefix):
        return 0.0
    index = int(token[len(prefix) :])
    if prefix == "vfi":
        return float(FIELD_BUCKETS - 1 - index) / float(FIELD_BUCKETS - 1)
    return float(index) / float(FIELD_BUCKETS - 1)


def lexical_score(tokens: list[str]) -> float:
    """Signed lexical prior over the prompt's own tokens."""

    score = 0.0
    for token in tokens:
        if token == UNRELIABLE_TOKEN:
            score += RELIABILITY_SHARE
            continue
        for name, weight in WORSENING_WEIGHT.items():
            score += weight * _bucket_direction(token, name)
    return score


def _morphometry_worsening(morphometry: np.ndarray) -> float:
    cup_disc = float(morphometry[0])
    nerve_fibre = float(morphometry[5])
    return 0.5 * float(np.clip((cup_disc - 0.35) / 0.55, 0.0, 1.0)) + 0.5 * float(
        np.clip((100.0 - nerve_fibre) / 60.0, 0.0, 1.0)
    )


def zero_shot_scores(features: FlatFeatures) -> np.ndarray:
    """A score per pair, higher meaning the prompt reads as more advanced disease."""

    summary = features.structural_summary()
    scores = np.zeros(features.n_pairs, dtype=np.float64)
    for row in range(features.n_pairs):
        bundle = build_prompt(features, row)
        tokens = bundle.prompt.replace("::", "").replace(";", " ").split()
        scores[row] = lexical_score(tokens) + 0.25 * _morphometry_worsening(summary[row])
    return scores


def template_signature() -> dict[str, object]:
    """What the prompt template emits, for the comparator roster and the report."""

    return {
        "block": "disc summary beside the serialised perimetry series",
        "field_tokens": list(WORSENING_WEIGHT),
        "buckets": FIELD_BUCKETS,
        "fit_on": "none, the comparator is prompted rather than fitted",
    }
