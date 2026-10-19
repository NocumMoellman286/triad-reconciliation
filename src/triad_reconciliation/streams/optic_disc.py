"""Optic-disc stream construction and the per-stream evidence summaries.

The structural stream is an optic-disc morphometry block plus an optic-nerve-head tomography
grid; patch order is fixed so the vision encoder sees the same token layout on every visit.

Ref: Sec. 5.1 (the structural stream gathers optic discs, fundus colour photography and
optic nerve head tomography), Proposition 1 (the two streams' evidence gap decides whether the
stream-evidence weight is identifiable).
"""

from __future__ import annotations

import torch
from torch import Tensor

from triad_reconciliation.core.records import N_MORPHOMETRY, VOLUME_EDGE, StructuralVisit


def patchify_volume(volume: Tensor, patch_edge: int) -> Tensor:
    """Split ``(batch, edge, edge, edge)`` volumes into ``(batch, n_patch, patch_volume)``."""

    if volume.dim() != 4:
        raise ValueError("expected a four-dimensional volume batch")
    edge = volume.shape[1]
    if edge % patch_edge != 0:
        raise ValueError("patch edge must divide the volume edge")
    step = edge // patch_edge
    folded = volume.reshape(volume.shape[0], patch_edge, step, patch_edge, step, patch_edge, step)
    permuted = folded.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
    patches: Tensor = permuted.reshape(volume.shape[0], patch_edge**3, step**3)
    return patches


def volume_to_tensor(visits: tuple[StructuralVisit, ...]) -> Tensor:
    """Stack a series of visits into a ``(visits, edge, edge, edge)`` volume batch."""

    rows = [
        torch.tensor(visit.volume, dtype=torch.float32).reshape(
            VOLUME_EDGE, VOLUME_EDGE, VOLUME_EDGE
        )
        for visit in visits
    ]
    return torch.stack(rows, dim=0)


def morphometry_to_tensor(visits: tuple[StructuralVisit, ...]) -> Tensor:
    rows = [
        torch.tensor(visit.morphometry[:N_MORPHOMETRY], dtype=torch.float32) for visit in visits
    ]
    return torch.stack(rows, dim=0)


def acquisition_stamps(visits: tuple[StructuralVisit, ...]) -> Tensor:
    return torch.tensor([visit.acquired_years for visit in visits], dtype=torch.float32)


def structural_evidence(morphometry: Tensor) -> Tensor:
    """Structural evidence as a bounded damage score per visit.

    The manuscript defines the evidence gap between the two streams without printing the
    functional form, so the summary reads the vertical cup-disc ratio together with the mean
    retinal nerve-fibre layer thickness. Index order matches ``MORPHOMETRY_NAMES``.
    """

    cup_disc = morphometry[..., 0].clamp(0.30, 0.92)
    nerve_fibre = morphometry[..., 5].clamp(38.0, 102.0)
    ratio_term = (cup_disc - 0.30) / 0.62
    thickness_term = (102.0 - nerve_fibre) / 64.0
    score: Tensor = 0.5 * ratio_term.clamp(0.0, 1.0) + 0.5 * thickness_term.clamp(0.0, 1.0)
    return score.clamp(0.0, 1.0)


def functional_evidence(mean_deviation: Tensor, visual_field_index: Tensor) -> Tensor:
    """Functional evidence as a bounded damage score per visit.

    The two perimetry indices enter through the depth of the defect, so a deeper mean
    deviation and a lower field index both raise the score.
    """

    deviation_term = ((-mean_deviation).clamp(-4.0, 34.0) + 4.0) / 38.0
    index_term = (100.0 - visual_field_index.clamp(2.0, 100.0)) / 82.0
    score: Tensor = 0.5 * deviation_term.clamp(0.0, 1.0) + 0.5 * index_term.clamp(0.0, 1.0)
    return score.clamp(0.0, 1.0)


def evidence_gap(structural: Tensor, functional: Tensor) -> Tensor:
    """Absolute gap between the two streams' evidence at the same visit-pair."""

    return (structural - functional).abs()
