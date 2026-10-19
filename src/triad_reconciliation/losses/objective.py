"""The assembled reconciliation objective.

Ref: Sec. 5.5, ``L = sum_i omega_i (L_prog + L_arb + mu L_cons) / sum_i omega_i``.
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor

from triad_reconciliation.losses.arbitration import agreement_loss, arbitration_loss
from triad_reconciliation.losses.consistency import consistency_loss
from triad_reconciliation.losses.progression import progression_loss
from triad_reconciliation.losses.reweighting import agreement_weights
from triad_reconciliation.models.triad import StreamBatch, TriadOutput


@dataclass(frozen=True)
class ObjectiveParts:
    """The weighted total beside the terms it is built from."""

    total: Tensor
    progression: Tensor
    arbitration: Tensor
    agreement: Tensor
    consistency: Tensor
    weight_sum: Tensor

    def scalars(self) -> dict[str, float]:
        return {
            "total": float(self.total.detach()),
            "progression": float(self.progression.detach()),
            "arbitration": float(self.arbitration.detach()),
            "agreement": float(self.agreement.detach()),
            "consistency": float(self.consistency.detach()),
            "weight_sum": float(self.weight_sum.detach()),
        }


def reconciliation_objective(
    output: TriadOutput,
    batch: StreamBatch,
    *,
    consistency_weight: float,
    reweighting_strength: float,
    margin: float,
    supervise_arbitration: bool = True,
    supervise_agreement: bool = True,
) -> ObjectiveParts:
    """Evaluate the objective for one batch.

    The manuscript prints no supervision signal for the agreement state it infers from the
    latent, so the readout is trained against the coarsened arbitration label unless the
    caller turns that off. The two supervision switches carry the component analysis, not a
    reporting option.
    """

    per_pair_progression = progression_loss(
        output.progression_logit, batch.progression, reduction="none"
    )
    if supervise_arbitration:
        per_pair_arbitration = arbitration_loss(
            output.arbitration_logits, batch.arbitration, reduction="none"
        )
    else:
        per_pair_arbitration = per_pair_progression * 0.0
    if supervise_agreement:
        agreement_term = agreement_loss(output.agreement_logits, batch.agreement, reduction="mean")
    else:
        agreement_term = output.agreement_logits.sum() * 0.0
    per_pair_consistency = consistency_loss(
        output.predicted_direction, output.stream_weight, margin, reduction="none"
    )
    weights = agreement_weights(output.predicted_agreement, reweighting_strength)
    inner = (
        per_pair_progression
        + per_pair_arbitration
        + float(consistency_weight) * per_pair_consistency
    )
    total = (weights * inner).sum() / weights.sum().clamp(min=1e-9)
    return ObjectiveParts(
        total=total,
        progression=per_pair_progression.mean(),
        arbitration=per_pair_arbitration.mean(),
        agreement=agreement_term,
        consistency=per_pair_consistency.mean(),
        weight_sum=weights.sum(),
    )
