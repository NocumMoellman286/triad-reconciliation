"""Training loop, schedule and checkpointing.

The optimiser is AdamW under a cosine schedule, as the training algorithm prints, with a
linear warm-up and gradient clipping around it and the release's own constants for
everything the manuscript does not report.

Ref: Algorithm 1 step 10 (AdamW with a cosine schedule), Sec. 5.8 (a deterministic analysis
under the recorded initial settings).
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from triad_reconciliation.analysis.batching import FlatFeatures, iterate_batches
from triad_reconciliation.core.config import TrainConfig
from triad_reconciliation.core.seeding import set_seed
from triad_reconciliation.models.triad import StreamBatch, TriadModel, TriadOutput

LossFunction = Callable[[nn.Module, StreamBatch], Tensor]


@dataclass
class FitHistory:
    """Objectives recorded per step and per epoch."""

    step_losses: list[float] = field(default_factory=list)
    epoch_losses: list[float] = field(default_factory=list)

    @property
    def final_step_loss(self) -> float:
        return self.step_losses[-1] if self.step_losses else float("nan")

    @property
    def first_step_loss(self) -> float:
        return self.step_losses[0] if self.step_losses else float("nan")


def learning_rate_at(
    step: int, total_steps: int, base_rate: float, warmup_fraction: float
) -> float:
    """Linear warm-up into a cosine decay, evaluated at one step."""

    if total_steps <= 0:
        return float(base_rate)
    warmup = max(int(round(float(warmup_fraction) * total_steps)), 1)
    if step < warmup:
        return float(base_rate) * float(step + 1) / float(warmup)
    progress = float(step - warmup) / float(max(total_steps - warmup, 1))
    progress = min(max(progress, 0.0), 1.0)
    return float(base_rate) * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(module: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with the release's betas and weight decay."""

    if config.optimizer != "adamw":
        raise ValueError(f"the training algorithm prints adamw, not {config.optimizer!r}")
    return torch.optim.AdamW(
        module.parameters(),
        lr=config.learning_rate,
        betas=tuple(config.betas),  # type: ignore[arg-type]
        weight_decay=config.weight_decay,
    )


def descend_on_loss(
    optimizer: torch.optim.Optimizer,
    loss: Tensor,
    clip: float,
    *,
    step: int,
    total: int,
    base_rate: float,
    warmup_fraction: float,
) -> float:
    """One optimisation step: the schedule, the gradient pass, the clip and the update."""

    rate = learning_rate_at(step, total, base_rate, warmup_fraction)
    for group in optimizer.param_groups:
        group["lr"] = rate
    optimizer.zero_grad(set_to_none=True)
    torch.autograd.backward(loss)
    norm = float(
        torch.nn.utils.clip_grad_norm_(list(optimizer.param_groups[0]["params"]), clip).detach()
    )
    optimizer.step()
    return norm


def fit_binary_module(
    module: nn.Module,
    features: FlatFeatures,
    config: TrainConfig,
    *,
    loss_function: LossFunction,
    seed: int,
) -> FitHistory:
    """Fit a module that emits a progression logit, over the development pairs."""

    set_seed(seed)
    module.train()
    optimizer = build_optimizer(module, config)
    batches_per_epoch = max(math.ceil(features.n_pairs / max(config.batch_size, 1)), 1)
    total_steps = max(config.epochs * batches_per_epoch, 1)
    history = FitHistory()
    cursor = 0
    for epoch in range(config.epochs):
        epoch_losses: list[float] = []
        for _, batch in iterate_batches(
            features, config.batch_size, shuffle=True, seed=seed + epoch
        ):
            loss = loss_function(module, batch)
            descend_on_loss(
                optimizer,
                loss,
                config.grad_clip,
                step=cursor,
                total=total_steps,
                base_rate=config.learning_rate,
                warmup_fraction=config.warmup_fraction,
            )
            value = float(loss.detach())
            history.step_losses.append(value)
            epoch_losses.append(value)
            cursor += 1
        history.epoch_losses.append(float(np.mean(epoch_losses)))
    module.eval()
    return history


def reconciliation_loss_function(
    consistency_weight: float,
    reweighting_strength: float,
    margin: float,
    *,
    supervise_arbitration: bool = True,
    supervise_agreement: bool = True,
) -> LossFunction:
    """The objective of the reconciliation stack, as a callable the fit loop can use.

    The two supervision switches carry the component analysis: the variant that keeps only
    outcome supervision drops the arbitration term, so its direction is never trained.

    Ref: Sec. 5.5 (the objective), Table 2 rows 3 and 9.
    """

    from triad_reconciliation.losses.objective import reconciliation_objective

    def loss_function(module: nn.Module, batch: StreamBatch) -> Tensor:
        if not isinstance(module, TriadModel):
            raise TypeError("the reconciliation objective needs the reconciliation module")
        output = module(batch)
        parts = reconciliation_objective(
            output,
            batch,
            consistency_weight=consistency_weight,
            reweighting_strength=reweighting_strength,
            margin=margin,
            supervise_arbitration=supervise_arbitration,
            supervise_agreement=supervise_agreement,
        )
        return parts.total

    return loss_function


def progression_only_loss_function() -> LossFunction:
    """Progression supervision alone, for the comparators and the component analysis.

    A comparator returns its logit directly; the reconciliation stack returns its full set of
    readouts, and this objective reads the progression one out of it.
    """

    def loss_function(module: nn.Module, batch: StreamBatch) -> Tensor:
        readout = module(batch)
        logits = readout.progression_logit if isinstance(readout, TriadOutput) else readout
        if not isinstance(logits, Tensor):
            raise TypeError("a comparator must return a progression logit")
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, batch.progression.float()
        )

    return loss_function


def predict_logits(module: nn.Module, features: FlatFeatures, config: TrainConfig) -> np.ndarray:
    """Progression logits for every pair, evaluated in batches without gradients."""

    module.eval()
    collected: list[np.ndarray] = []
    with torch.no_grad():
        for _, batch in iterate_batches(features, config.batch_size):
            output = module(batch)
            if isinstance(output, Tensor):
                values = output.detach().cpu().numpy()
            else:
                raise TypeError("prediction expects a comparator that returns a logit")
            collected.append(values)
    if not collected:
        return np.zeros(0, dtype=np.float64)
    stacked: np.ndarray = np.concatenate(collected).astype(np.float64)
    return stacked


def predict_probabilities(
    module: nn.Module, features: FlatFeatures, config: TrainConfig
) -> np.ndarray:
    logits = predict_logits(module, features, config)
    probabilities: np.ndarray = 1.0 / (1.0 + np.exp(-logits))
    return probabilities


def reconciliation_outputs(
    model: TriadModel, features: FlatFeatures, config: TrainConfig
) -> dict[str, np.ndarray]:
    """Every readout the reconciliation stack emits, for a pair set."""

    model.eval()
    probabilities: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    agreements: list[np.ndarray] = []
    with torch.no_grad():
        for _, batch in iterate_batches(features, config.batch_size):
            output = model(batch)
            probabilities.append(output.progression_probability.cpu().numpy())
            directions.append(output.predicted_direction.cpu().numpy())
            weights.append(output.stream_weight.cpu().numpy())
            agreements.append(output.predicted_agreement.cpu().numpy())
    return {
        "progression": np.concatenate(probabilities) if probabilities else np.zeros(0),
        "direction": np.concatenate(directions) if directions else np.zeros(0, dtype=np.int64),
        "stream_weight": np.concatenate(weights) if weights else np.zeros(0),
        "agreement": np.concatenate(agreements) if agreements else np.zeros(0, dtype=np.int64),
    }


def save_checkpoint(
    target: Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    seed: int,
    epoch: int,
    metrics: dict[str, float],
) -> Path:
    """Write a checkpoint through a temporary file in the target directory."""

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, raw = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    staged = Path(raw)
    try:
        with os.fdopen(handle, "wb") as stream:
            torch.save(
                {
                    "state_dict": module.state_dict(),
                    "optimizer": optimizer.state_dict() if optimizer is not None else None,
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "metrics": {key: float(value) for key, value in metrics.items()},
                },
                stream,
            )
        os.chmod(staged, 0o644)
        os.replace(staged, target)
    except BaseException:
        if staged.exists():
            staged.unlink()
        raise
    return target


def load_checkpoint(
    source: Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    restore_seed: bool = True,
) -> dict[str, Any]:
    """Restore a module and its seed, returning the recorded metadata."""

    payload = torch.load(Path(source), map_location="cpu", weights_only=False)
    module.load_state_dict(payload["state_dict"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if restore_seed:
        set_seed(int(payload["seed"]))
    return {
        "seed": int(payload["seed"]),
        "epoch": int(payload["epoch"]),
        "metrics": {key: float(value) for key, value in payload.get("metrics", {}).items()},
    }


def checkpoint_digest(source: Path) -> str:
    """Digest of the tensor payload, which is what a checkpoint's content is."""

    import hashlib

    payload = torch.load(Path(source), map_location="cpu", weights_only=False)
    digest = hashlib.sha256()
    for name in sorted(payload["state_dict"]):
        tensor = payload["state_dict"][name]
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    digest.update(str(int(payload["seed"])).encode("utf-8"))
    return digest.hexdigest()
