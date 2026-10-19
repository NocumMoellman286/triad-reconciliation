"""Reader study and reliance accounting.

Ref: Sec. 2.4 and Table 4 (a three-arm reader study over 320 vignettes of which 224 were
discordant, each reader in every arm with a washout between arms), Sec. 3.1 (the reliance
decomposition that conditions the benefit).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from triad_reconciliation.core.records import ARBITRATION_CLASSES, ReadershipResponse
from triad_reconciliation.metrics.agreement import (
    ARBITRATION_CATEGORIES,
    DIRECTION_CATEGORIES,
    cohen_kappa,
    mean_pairwise_kappa,
)

ARMS: tuple[str, ...] = ("unassisted", "model_only", "model_assisted")


@dataclass(frozen=True)
class Vignette:
    """One reader-facing case, with the reference decision and the model's advice."""

    vignette_id: str
    reference: str
    model_direction: str
    discordant: bool


@dataclass(frozen=True)
class ArmSummary:
    """Accuracy, arbitration concordance and reading time for one study arm."""

    arm: str
    accuracy: float
    n_answers: int
    arbitration_kappa: float
    median_seconds: float


def accuracy(answers: list[ReadershipResponse], references: dict[str, str]) -> float:
    if not answers:
        return float("nan")
    correct = sum(
        1 for answer in answers if answer.answered_direction == references[answer.vignette_id]
    )
    return correct / len(answers)


def arm_answers(
    responses: tuple[ReadershipResponse, ...],
    arm: str,
    discordant: dict[str, bool],
    *,
    only_discordant: bool,
) -> list[ReadershipResponse]:
    return [
        answer
        for answer in responses
        if answer.arm == arm and (discordant[answer.vignette_id] if only_discordant else True)
    ]


def summarise_arm(
    responses: tuple[ReadershipResponse, ...],
    references: dict[str, str],
    discordant: dict[str, bool],
    arm: str,
    *,
    only_discordant: bool,
) -> ArmSummary:
    answers = arm_answers(responses, arm, discordant, only_discordant=only_discordant)
    if not answers:
        return ArmSummary(
            arm=arm,
            accuracy=float("nan"),
            n_answers=0,
            arbitration_kappa=float("nan"),
            median_seconds=float("nan"),
        )
    predicted = np.asarray([answer.answered_direction for answer in answers], dtype=object)
    recorded = np.asarray([references[answer.vignette_id] for answer in answers], dtype=object)
    kappa = (
        cohen_kappa(predicted, recorded, DIRECTION_CATEGORIES)
        if set(predicted.tolist()) <= set(DIRECTION_CATEGORIES)
        else float("nan")
    )
    seconds = np.asarray([answer.seconds for answer in answers], dtype=np.float64)
    return ArmSummary(
        arm=arm,
        accuracy=accuracy(answers, references),
        n_answers=len(answers),
        arbitration_kappa=kappa,
        median_seconds=float(np.median(seconds)),
    )


def reader_matrix(
    responses: tuple[ReadershipResponse, ...], arm: str, vignette_ids: tuple[str, ...]
) -> np.ndarray:
    """A reader-by-vignette matrix of the recorded directions in one arm."""

    readers = sorted({answer.reader_id for answer in responses if answer.arm == arm})
    position = {vignette_id: index for index, vignette_id in enumerate(vignette_ids)}
    matrix = np.full((len(readers), len(vignette_ids)), None, dtype=object)
    row_of = {reader: index for index, reader in enumerate(readers)}
    for answer in responses:
        if answer.arm != arm or answer.vignette_id not in position:
            continue
        matrix[row_of[answer.reader_id], position[answer.vignette_id]] = answer.answered_direction
    return matrix


def inter_reader_agreement(
    responses: tuple[ReadershipResponse, ...], arm: str, vignette_ids: tuple[str, ...]
) -> float:
    matrix = reader_matrix(responses, arm, vignette_ids)
    if matrix.size == 0:
        return float("nan")
    return mean_pairwise_kappa(matrix, ARBITRATION_CATEGORIES)


def stream_switch_share(
    responses: tuple[ReadershipResponse, ...],
    unhinted: dict[str, str],
    *,
    arm: str,
) -> float:
    """How often the advice moved a reader off the direction they had reached alone."""

    joined: dict[str, dict[str, str]] = {}
    for answer in responses:
        if answer.arm == arm:
            joined.setdefault(answer.reader_id, {})[answer.vignette_id] = answer.answered_direction
    moved = 0
    total = 0
    for reader, answers in joined.items():
        for vignette_id, direction in answers.items():
            baseline = unhinted.get(f"{reader}/{vignette_id}")
            if baseline is None:
                continue
            total += 1
            if baseline != direction:
                moved += 1
    return float(moved / total) if total else float("nan")


def arbitration_categories_used(responses: tuple[ReadershipResponse, ...]) -> tuple[str, ...]:
    used = {answer.answered_direction for answer in responses}
    return tuple(category for category in ARBITRATION_CLASSES if category in used)
