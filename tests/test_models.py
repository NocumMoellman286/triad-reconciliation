"""Encoders, adapters, alignment, reconciliation, heads and the assembled stack."""

from __future__ import annotations

import math

import pytest
import torch

from triad_reconciliation.core.config import ModelConfig
from triad_reconciliation.core.seeding import set_seed
from triad_reconciliation.models.adapters import BottleneckAdapter
from triad_reconciliation.models.alignment import MonotoneCausalAlignment
from triad_reconciliation.models.encoders import LanguageEncoder, VisionEncoder
from triad_reconciliation.models.heads import AgreementHead, ArbitrationHead, ProgressionHead
from triad_reconciliation.models.reconciliation import CrossStreamReconciliation
from triad_reconciliation.models.triad import TriadModel, VariantFlags
from triad_reconciliation.streams.optic_disc import patchify_volume

CONFIG = ModelConfig(max_visits=5)


def test_patchify_covers_the_grid_once() -> None:
    volume = torch.arange(4**3, dtype=torch.float32).reshape(1, 4, 4, 4)
    patches = patchify_volume(volume, 2)
    assert patches.shape == (1, 8, 8)
    assert float(patches.sum()) == float(volume.sum())


def test_patchify_rejects_an_uneven_grid() -> None:
    with pytest.raises(ValueError):
        patchify_volume(torch.zeros(1, 5, 5, 5), 2)


def test_vision_encoder_shapes(tensor_batch) -> None:
    batch = tensor_batch
    encoder = VisionEncoder(CONFIG)
    visits = batch.morphometry.shape[1]
    tokens, valid = encoder(batch.morphometry, batch.volume, batch.structural_times)
    assert tokens.shape == (6, visits * encoder.tokens_per_visit, CONFIG.vision_token_dim)
    assert valid.shape == (6, visits)
    assert encoder.tokens_per_visit == 9


def test_language_encoder_shapes_and_padding(tensor_batch) -> None:
    batch = tensor_batch
    encoder = LanguageEncoder(CONFIG)
    visits = batch.functional_times.shape[1]
    tokens, valid = encoder(batch.token_ids, batch.functional_times, batch.functional_valid)
    assert tokens.shape == (6, visits * (CONFIG.max_perimetry_tokens + 1), CONFIG.language_dim)
    assert valid.shape == (6, visits * (CONFIG.max_perimetry_tokens + 1))
    assert bool(valid.any()) and not bool(valid.all())


def test_adapter_is_residual_and_rescales() -> None:
    adapter = BottleneckAdapter(8, 3)
    values = torch.randn(2, 4, 8)
    with torch.no_grad():
        adapter.up.weight.zero_()
        adapter.up.bias.zero_()
    assert torch.allclose(adapter(values), values)
    with torch.no_grad():
        adapter.up.weight.fill_(1.0)
    assert not torch.allclose(adapter(values), values)


def test_alignment_blocks_everything_at_or_beyond_the_query() -> None:
    alignment = MonotoneCausalAlignment()
    times = torch.tensor([[0.0, 1.0, 2.0, 3.5]])
    valid = torch.ones((1, 4), dtype=torch.bool)
    with torch.no_grad():
        weights, _ = alignment(torch.tensor([3.0]), times, times, valid, valid)
    assert float(weights[0, 3]) == 0.0
    assert float(weights[0, :3].sum()) == pytest.approx(1.0)
    bias = alignment.attention_bias(weights, valid)
    assert bias[0, 3] == float("-inf")


def test_alignment_is_monotone_along_the_timeline() -> None:
    alignment = MonotoneCausalAlignment()
    times = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0]])
    valid = torch.ones((1, 5), dtype=torch.bool)
    with torch.no_grad():
        weights, _ = alignment(torch.tensor([9.0]), times, times, valid, valid)
    values = weights[0].tolist()
    assert all(later >= earlier for earlier, later in zip(values, values[1:], strict=False))
    assert float(alignment.rate.detach()) > 0.0


def test_alignment_matches_a_brute_force_softmax() -> None:
    alignment = MonotoneCausalAlignment()
    times = torch.tensor([[0.0, 1.0, 2.0]])
    valid = torch.ones((1, 3), dtype=torch.bool)
    with torch.no_grad():
        weights, _ = alignment(torch.tensor([5.0]), times, times, valid, valid)
    rate = float(alignment.rate.detach())
    exponentials = [math.exp(rate * value) for value in times[0].tolist()]
    reference = [value / sum(exponentials) for value in exponentials]
    assert weights[0].tolist() == pytest.approx(reference, abs=1e-7)


def test_alignment_returns_zero_when_nothing_is_admissible() -> None:
    alignment = MonotoneCausalAlignment()
    times = torch.tensor([[5.0, 6.0]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    with torch.no_grad():
        weights, _ = alignment(torch.tensor([1.0]), times, times, valid, valid)
    assert weights.sum().item() == 0.0


def test_reconciliation_rejects_visit_level_validity(tensor_batch) -> None:
    module = CrossStreamReconciliation(CONFIG)
    with torch.no_grad():
        tokens = torch.randn(6, 4 * 9, CONFIG.vision_token_dim)
        with pytest.raises(ValueError):
            module(
                tokens,
                torch.ones(6, 4, dtype=torch.bool),
                torch.zeros(6, 45),
                torch.randn(6, 20, CONFIG.vision_token_dim),
                torch.ones(6, 20, dtype=torch.bool),
                torch.zeros(6, 20),
            )


def test_reconciliation_shapes_and_weight_bounds(tensor_batch) -> None:
    batch = tensor_batch
    module = CrossStreamReconciliation(CONFIG)
    vision = VisionEncoder(CONFIG)
    with torch.no_grad():
        structural, visits_valid = vision(batch.morphometry, batch.volume, batch.structural_times)
        visits = visits_valid.shape[1]
        structural_valid = (
            visits_valid.unsqueeze(-1).expand(6, visits, vision.tokens_per_visit).reshape(6, -1)
        )
        functional = torch.randn(6, visits * 29, CONFIG.language_dim)
        functional_valid = torch.zeros(6, visits * 29, dtype=torch.bool)
        functional_valid[:, :10] = True
        state, weight = module(
            structural,
            structural_valid,
            torch.zeros(6, visits * vision.tokens_per_visit),
            functional,
            functional_valid,
            torch.zeros(6, visits * 29),
        )
    assert state.shape == (6, CONFIG.vision_token_dim)
    assert weight.shape == (6,)
    assert bool((weight >= 0.0).all() and (weight <= 1.0).all())


def test_heads_emit_the_declared_class_counts(tensor_batch) -> None:
    batch = tensor_batch
    model = TriadModel(CONFIG)
    output = model(batch)
    assert ArbitrationHead(CONFIG)(output.reconciliation_state).shape == (6, 3)
    assert AgreementHead(CONFIG)(output.reconciliation_state).shape == (6, 2)
    assert ProgressionHead(CONFIG)(output.reconciliation_state).shape == (6,)


def test_triad_readouts_are_consistent_with_the_logits(tensor_batch) -> None:
    batch = tensor_batch
    model = TriadModel(CONFIG)
    output = model(batch)
    assert torch.allclose(output.predicted_direction, output.arbitration_logits.argmax(dim=-1))
    assert torch.allclose(output.predicted_agreement, output.agreement_logits.argmax(dim=-1))
    probability = output.progression_probability
    assert bool((probability > 0.0).all() and (probability < 1.0).all())


def test_forward_is_deterministic_under_a_fixed_seed(tensor_batch) -> None:
    batch = tensor_batch
    set_seed(4242)
    first = TriadModel(CONFIG)(batch).progression_logit.detach().clone()
    set_seed(4242)
    second = TriadModel(CONFIG)(batch).progression_logit.detach().clone()
    assert torch.equal(first, second)


def test_variant_flags_name_the_removed_mechanism() -> None:
    assert VariantFlags().name == "full"
    assert VariantFlags(use_reconciliation_state=False).name == "without_reconciliation_state"
    interaction = VariantFlags(use_reconciliation_state=False, use_arbitration_head=False)
    assert interaction.name == "without_reconciliation_state+arbitration_head"


def test_late_fusion_variant_skips_the_reconciliation_module(tensor_batch) -> None:
    batch = tensor_batch
    set_seed(7)
    model = TriadModel(CONFIG, VariantFlags(use_reconciliation_state=False))
    assert not hasattr(model, "reconciliation")
    output = model(batch)
    assert output.reconciliation_state.shape == (6, CONFIG.vision_token_dim)


def test_numeric_perimetry_variant_replaces_the_language_encoder(tensor_batch) -> None:
    batch = tensor_batch
    set_seed(7)
    model = TriadModel(CONFIG, VariantFlags(use_language_encoder=False))
    assert not hasattr(model, "language")
    output = model(batch)
    assert output.arbitration_logits.shape == (6, 3)


def test_unrestricted_variant_lets_a_future_token_through(tensor_batch) -> None:
    batch = tensor_batch
    set_seed(7)
    model = TriadModel(CONFIG, VariantFlags(use_monotone_alignment=False))
    output = model(batch)
    assert bool(torch.isfinite(output.reconciliation_state).all())


def test_outcome_derived_direction_variant_has_no_three_class_head(tensor_batch) -> None:
    batch = tensor_batch
    set_seed(7)
    model = TriadModel(CONFIG, VariantFlags(use_arbitration_head=False))
    output = model(batch)
    assert output.arbitration_logits.shape == (6, 3)
    direct = model._outcome_derived_direction(output.progression_logit, output.stream_weight)
    assert torch.equal(direct, output.arbitration_logits)


def test_parameter_count_is_positive_and_stable() -> None:
    set_seed(3)
    model = TriadModel(CONFIG)
    assert model.parameter_count() > 0
    assert model.parameter_count() == model.parameter_count()
