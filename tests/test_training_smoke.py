"""Training, schedule, checkpointing and the minimal training loop.

The release holds no records, so these tests run on the tensor harness: a shape-consistent
batch drawn from a recorded seed. Nothing here reads or writes a cohort table.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from triad_reconciliation.analysis.pipeline import STAGING_HINT, build_table
from triad_reconciliation.analysis.trainer import (
    build_optimizer,
    checkpoint_digest,
    descend_on_loss,
    learning_rate_at,
    load_checkpoint,
    progression_only_loss_function,
    reconciliation_loss_function,
    save_checkpoint,
)
from triad_reconciliation.cohort.reader import CohortNotStaged
from triad_reconciliation.comparators import roster
from triad_reconciliation.core.config import ExperimentConfig, ModelConfig, TrainConfig
from triad_reconciliation.core.seeding import set_seed
from triad_reconciliation.models.triad import StreamBatch, TriadModel
from triad_reconciliation.verification.execution import HARNESS_SEED, random_stream_batch

SMOKE = TrainConfig(epochs=2, batch_size=8, learning_rate=0.006)
MODEL = ModelConfig(max_visits=4)


def _objective(model: TriadModel):
    return reconciliation_loss_function(
        0.65, 2.35, 0.46, supervise_arbitration=model.flags.use_arbitration_head
    )


def test_learning_rate_warms_up_then_decays() -> None:
    rates = [learning_rate_at(step, 40, 1.0, 0.25) for step in range(40)]
    assert rates[0] < rates[9]
    assert rates[-1] < 0.01
    assert all(later <= earlier for earlier, later in zip(rates[10:], rates[11:], strict=False))


def test_learning_rate_is_flat_when_there_are_no_steps() -> None:
    assert learning_rate_at(0, 0, 0.5, 0.1) == pytest.approx(0.5)


def test_optimizer_is_adamw_and_rejects_anything_else(model: TriadModel) -> None:
    optimizer = build_optimizer(model, SMOKE)
    assert isinstance(optimizer, torch.optim.AdamW)
    with pytest.raises(ValueError):
        build_optimizer(model, TrainConfig(optimizer="sgd"))


def test_descend_populates_gradients_and_moves_weights(tensor_batch: StreamBatch) -> None:
    set_seed(HARNESS_SEED)
    model = TriadModel(MODEL)
    optimizer = build_optimizer(model, SMOKE)
    loss = _objective(model)(model, tensor_batch)
    before = [parameter.detach().clone() for parameter in model.parameters()]
    norm = descend_on_loss(
        optimizer, loss, 1.0, step=0, total=4, base_rate=0.01, warmup_fraction=0.25
    )
    assert math.isfinite(norm) and norm > 0.0
    moved = sum(
        1
        for first, second in zip(before, model.parameters(), strict=True)
        if not torch.equal(first, second.detach())
    )
    assert moved > 0


def test_fit_is_reproducible_for_a_seed() -> None:
    set_seed(HARNESS_SEED)
    first = _fit_once(HARNESS_SEED)
    set_seed(HARNESS_SEED)
    second = _fit_once(HARNESS_SEED)
    assert first.step_losses == second.step_losses
    assert first.epoch_losses == second.epoch_losses


def _fit_once(seed: int):
    """One short fit over the harness batches, with the seed fixed by the caller."""

    from triad_reconciliation.analysis.trainer import FitHistory, build_optimizer as _opt

    model = TriadModel(MODEL)
    optimizer = _opt(model, SMOKE)
    loss_function = _objective(model)
    batches = [random_stream_batch(8, MODEL, seed + index) for index in range(3)]
    history = FitHistory()
    step = 0
    for _epoch in range(SMOKE.epochs):
        running: list[float] = []
        for batch in batches:
            value = loss_function(model, batch)
            history.step_losses.append(float(value.detach()))
            running.append(float(value.detach()))
            descend_on_loss(
                optimizer,
                value,
                SMOKE.grad_clip,
                step=step,
                total=SMOKE.epochs * len(batches),
                base_rate=SMOKE.learning_rate,
                warmup_fraction=SMOKE.warmup_fraction,
            )
            step += 1
        history.epoch_losses.append(float(np.mean(running)))
    return history


def test_two_epochs_decrease_the_epoch_loss() -> None:
    set_seed(HARNESS_SEED)
    history = _fit_once(HARNESS_SEED)
    assert len(history.epoch_losses) == SMOKE.epochs
    assert history.epoch_losses[-1] < history.epoch_losses[0]


def test_single_batch_overfit_drives_the_loss_down() -> None:
    batch = random_stream_batch(16, MODEL, HARNESS_SEED)
    set_seed(HARNESS_SEED)
    model = TriadModel(MODEL)
    settings = TrainConfig(epochs=24, batch_size=16, learning_rate=0.01, warmup_fraction=0.2)
    optimizer = build_optimizer(model, settings)
    loss_function = _objective(model)
    first = float(loss_function(model, batch).detach())
    history: list[float] = []
    total = settings.epochs * 4
    for step in range(total):
        value = loss_function(model, batch)
        history.append(float(value.detach()))
        descend_on_loss(
            optimizer,
            value,
            settings.grad_clip,
            step=step,
            total=total,
            base_rate=settings.learning_rate,
            warmup_fraction=settings.warmup_fraction,
        )
    assert history[-1] < 0.5 * first
    assert min(history) < 0.1 * first


def test_checkpoint_round_trip_restores_weights_and_seed(
    tensor_batch: StreamBatch, tmp_path: Path
) -> None:
    del tensor_batch
    set_seed(HARNESS_SEED)
    model = TriadModel(MODEL)
    optimizer = build_optimizer(model, SMOKE)
    target = tmp_path / "checkpoint.pt"
    save_checkpoint(target, model, optimizer, seed=HARNESS_SEED, epoch=2, metrics={"loss": 1.0})
    reloaded = TriadModel(MODEL)
    metadata = load_checkpoint(target, reloaded, restore_seed=True)
    assert metadata["seed"] == HARNESS_SEED
    assert metadata["epoch"] == 2
    for first, second in zip(
        model.state_dict().values(), reloaded.state_dict().values(), strict=True
    ):
        assert torch.equal(first, second)
    assert len(checkpoint_digest(target)) == 64


def test_checkpoint_survives_further_optimisation(
    tensor_batch: StreamBatch, tmp_path: Path
) -> None:
    set_seed(HARNESS_SEED)
    model = TriadModel(MODEL)
    optimizer = build_optimizer(model, SMOKE)
    target = tmp_path / "checkpoint.pt"
    save_checkpoint(target, model, optimizer, seed=HARNESS_SEED, epoch=1, metrics={})
    loss_function = _objective(model)
    for step in range(4):
        value = loss_function(model, tensor_batch)
        descend_on_loss(
            optimizer, value, 1.0, step=step, total=4, base_rate=0.05, warmup_fraction=0.25
        )
    drifted = TriadModel(MODEL)
    load_checkpoint(target, drifted, restore_seed=False)
    assert any(
        not torch.equal(first, second)
        for first, second in zip(
            model.state_dict().values(), drifted.state_dict().values(), strict=True
        )
    )


def test_checkpoint_file_is_world_readable_with_no_staging_left(tmp_path: Path) -> None:
    set_seed(HARNESS_SEED)
    model = TriadModel(MODEL)
    target = tmp_path / "checkpoint.pt"
    save_checkpoint(target, model, None, seed=1, epoch=0, metrics={})
    assert oct(target.stat().st_mode & 0o777) == "0o644"
    assert [path.name for path in tmp_path.iterdir()] == ["checkpoint.pt"]


def test_comparators_emit_a_probability_per_row(tensor_batch: StreamBatch) -> None:
    set_seed(HARNESS_SEED)
    module = roster.learned_modules(MODEL)["late fusion of per-visit embeddings"]
    optimizer = build_optimizer(module, SMOKE)
    loss_function = progression_only_loss_function()
    value = loss_function(module, tensor_batch)
    descend_on_loss(optimizer, value, 1.0, step=0, total=2, base_rate=0.01, warmup_fraction=0.5)
    with torch.no_grad():
        logits = module(tensor_batch)
    probabilities = 1.0 / (1.0 + torch.exp(-logits))
    assert probabilities.shape == (tensor_batch.progression.shape[0],)
    assert bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all())


def test_reconciliation_outputs_expose_every_readout(tensor_batch: StreamBatch) -> None:
    set_seed(HARNESS_SEED)
    model = TriadModel(MODEL)
    outputs = _reconciliation_arrays(model, tensor_batch)
    assert set(outputs) == {"progression", "direction", "stream_weight", "agreement"}
    for value in outputs.values():
        assert value.shape == (tensor_batch.progression.shape[0],)
    assert set(np.unique(outputs["direction"]).tolist()) <= {0, 1, 2}


def _reconciliation_arrays(model: TriadModel, batch: StreamBatch) -> dict[str, np.ndarray]:
    """The readouts of one batch, read out without going through the batching layer."""

    with torch.no_grad():
        output = model(batch)
    return {
        "progression": output.progression_probability.numpy(),
        "direction": output.predicted_direction.numpy(),
        "stream_weight": output.stream_weight.numpy(),
        "agreement": output.predicted_agreement.numpy(),
    }


def test_the_pipeline_stops_without_a_staged_table(smoke_config: ExperimentConfig) -> None:
    with pytest.raises(CohortNotStaged) as raised:
        build_table(smoke_config, None)
    assert raised.value.hint == STAGING_HINT


def test_the_suite_leaves_no_run_directory_behind(release_root: Path) -> None:
    assert not (release_root / "runs").exists()
    assert np.isfinite(float((release_root / "pyproject.toml").stat().st_size))
