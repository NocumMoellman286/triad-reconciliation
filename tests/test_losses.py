"""Loss terms and the assembled objective."""

from __future__ import annotations

import math

import pytest
import torch

from triad_reconciliation.core.config import ModelConfig
from triad_reconciliation.core.seeding import set_seed
from triad_reconciliation.losses.arbitration import agreement_loss, arbitration_loss
from triad_reconciliation.losses.consistency import consistency_loss, contradiction_mask
from triad_reconciliation.losses.objective import reconciliation_objective
from triad_reconciliation.losses.progression import progression_loss
from triad_reconciliation.losses.reweighting import agreement_weights, discordant_weight_mass
from triad_reconciliation.models.triad import TriadModel, VariantFlags


def test_progression_loss_matches_the_logistic_form() -> None:
    logits = torch.tensor([0.5, -2.0, 1.5])
    targets = torch.tensor([1.0, 0.0, 1.0])
    hand = (
        sum(
            math.log1p(math.exp(-value)) if label == 1.0 else math.log1p(math.exp(value))
            for value, label in zip(logits.tolist(), targets.tolist(), strict=True)
        )
        / 3
    )
    assert float(progression_loss(logits, targets)) == pytest.approx(hand, abs=1e-6)


def test_progression_loss_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        progression_loss(torch.zeros(3), torch.zeros(4))


def test_arbitration_and_agreement_losses_use_the_declared_class_counts() -> None:
    arbitration = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    assert float(arbitration_loss(arbitration, torch.tensor([0, 1]))) < 0.25
    agreement = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    assert float(agreement_loss(agreement, torch.tensor([0, 1]))) < 0.25
    with pytest.raises(ValueError):
        arbitration_loss(torch.zeros(2, 2), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError):
        agreement_loss(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long))


def test_contradiction_mask_covers_both_directions() -> None:
    direction = torch.tensor([0, 1, 2, 0, 1])
    weight = torch.tensor([0.9, 0.1, 0.5, 0.2, 0.8])
    mask = contradiction_mask(direction, weight, 0.5)
    assert mask.tolist() == [False, False, False, True, True]


def test_consistency_loss_is_zero_when_nothing_contradicts() -> None:
    direction = torch.tensor([0, 1, 2])
    weight = torch.tensor([0.9, 0.1, 0.5])
    assert float(consistency_loss(direction, weight, 0.5)) == 0.0


def test_consistency_loss_scales_with_the_distance_to_the_margin() -> None:
    direction = torch.tensor([0, 0])
    near = torch.tensor([0.45, 0.05])
    far = torch.tensor([0.20, 0.05])
    near_loss = float(consistency_loss(direction, near, 0.5))
    far_loss = float(consistency_loss(direction, far, 0.5))
    assert far_loss > near_loss
    with pytest.raises(ValueError):
        consistency_loss(direction, near, 0.5, reduction="median")


def test_reweighting_applies_the_strength_only_to_discordant_pairs() -> None:
    weights = agreement_weights(torch.tensor([0, 1, 1, 0]), 3.0)
    assert weights.tolist() == [1.0, 4.0, 4.0, 1.0]
    with pytest.raises(ValueError):
        agreement_weights(torch.tensor([0]), -1.0)


def test_discordant_weight_mass_separates_frequency_from_weight() -> None:
    predicted = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0])
    weights = agreement_weights(predicted, 3.0)
    assert discordant_weight_mass(weights, predicted) == pytest.approx(8.0 / 14.0)
    assert discordant_weight_mass(torch.zeros(4), torch.zeros(4, dtype=torch.long)) == 0.0


def test_objective_equals_a_hand_written_weighted_sum(tensor_batch) -> None:
    batch = tensor_batch
    set_seed(20260219)
    model = TriadModel(ModelConfig(max_visits=5))
    output = model(batch)
    margin = 0.46
    parts = reconciliation_objective(
        output, batch, consistency_weight=0.5, reweighting_strength=2.0, margin=margin
    )
    with torch.no_grad():
        hand_weights = agreement_weights(output.predicted_agreement, 2.0)
        inner = (
            progression_loss(output.progression_logit, batch.progression, reduction="none")
            + arbitration_loss(output.arbitration_logits, batch.arbitration, reduction="none")
            + 0.5
            * consistency_loss(
                output.predicted_direction, output.stream_weight, margin, reduction="none"
            )
        )
        hand = (hand_weights * inner).sum() / hand_weights.sum()
    assert float(parts.total.detach()) == pytest.approx(float(hand), abs=1e-9)


def test_objective_reports_every_term(tensor_batch) -> None:
    batch = tensor_batch
    set_seed(20260219)
    model = TriadModel(ModelConfig(max_visits=5))
    parts = reconciliation_objective(
        model(batch), batch, consistency_weight=0.5, reweighting_strength=2.0, margin=0.46
    )
    scalars = parts.scalars()
    assert set(scalars) == {
        "total",
        "progression",
        "arbitration",
        "agreement",
        "consistency",
        "weight_sum",
    }
    assert all(math.isfinite(value) for value in scalars.values())


def test_objective_can_drop_arbitration_supervision(tensor_batch) -> None:
    batch = tensor_batch
    set_seed(7)
    model = TriadModel(ModelConfig(max_visits=5), VariantFlags(use_arbitration_head=False))
    supervised = reconciliation_objective(
        model(batch),
        batch,
        consistency_weight=0.5,
        reweighting_strength=2.0,
        margin=0.46,
        supervise_arbitration=True,
    )
    dropped = reconciliation_objective(
        model(batch),
        batch,
        consistency_weight=0.5,
        reweighting_strength=2.0,
        margin=0.46,
        supervise_arbitration=False,
    )
    assert float(dropped.arbitration.detach()) == 0.0
    assert not math.isclose(float(supervised.total.detach()), float(dropped.total.detach()))
