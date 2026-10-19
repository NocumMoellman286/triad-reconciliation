"""The assembled reconciliation model.

Ref: Sec. 5.1 (``f_theta`` maps the two streams to the progression probability, the
arbitration direction and the stream-evidence weight), Sec. 5.4-5.5, Algorithm 1. The
variant flags carry the component analysis of Table 2: each flag removes exactly the
mechanism the corresponding row removes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from triad_reconciliation.core.config import ModelConfig
from triad_reconciliation.models.alignment import MonotoneCausalAlignment
from triad_reconciliation.models.encoders import LanguageEncoder, VisionEncoder
from triad_reconciliation.models.heads import AgreementHead, ArbitrationHead, ProgressionHead
from triad_reconciliation.models.reconciliation import CrossStreamReconciliation


@dataclass(frozen=True)
class VariantFlags:
    """Which mechanisms a reported variant keeps.

    Ref: Table 2 rows, the full model and the seven single-mechanism removals beside the
    interaction row that removes two at once.
    """

    use_reconciliation_state: bool = True
    use_arbitration_head: bool = True
    use_attribution_consistency: bool = True
    use_agreement_reweighting: bool = True
    use_monotone_alignment: bool = True
    use_language_encoder: bool = True
    use_calibrated_deferral: bool = True

    @property
    def name(self) -> str:
        removed = [
            key
            for key, value in (
                ("reconciliation_state", self.use_reconciliation_state),
                ("arbitration_head", self.use_arbitration_head),
                ("attribution_consistency", self.use_attribution_consistency),
                ("agreement_reweighting", self.use_agreement_reweighting),
                ("monotone_alignment", self.use_monotone_alignment),
                ("language_encoder", self.use_language_encoder),
                ("calibrated_deferral", self.use_calibrated_deferral),
            )
            if not value
        ]
        return "full" if not removed else "without_" + "+".join(removed)


@dataclass(frozen=True)
class StreamBatch:
    """One batch of visit-pairs, padded to the widest series in the batch."""

    morphometry: Tensor
    volume: Tensor
    structural_times: Tensor
    structural_valid: Tensor
    token_ids: Tensor
    field_indices: Tensor
    functional_times: Tensor
    functional_valid: Tensor
    query_years: Tensor
    progression: Tensor
    arbitration: Tensor
    agreement: Tensor

    def to(self, device: torch.device) -> StreamBatch:
        return StreamBatch(
            morphometry=self.morphometry.to(device),
            volume=self.volume.to(device),
            structural_times=self.structural_times.to(device),
            structural_valid=self.structural_valid.to(device),
            token_ids=self.token_ids.to(device),
            field_indices=self.field_indices.to(device),
            functional_times=self.functional_times.to(device),
            functional_valid=self.functional_valid.to(device),
            query_years=self.query_years.to(device),
            progression=self.progression.to(device),
            arbitration=self.arbitration.to(device),
            agreement=self.agreement.to(device),
        )


@dataclass(frozen=True)
class TriadOutput:
    """Model readouts for a batch."""

    progression_logit: Tensor
    arbitration_logits: Tensor
    agreement_logits: Tensor
    stream_weight: Tensor
    reconciliation_state: Tensor

    @property
    def progression_probability(self) -> Tensor:
        return torch.sigmoid(self.progression_logit)

    @property
    def predicted_direction(self) -> Tensor:
        return self.arbitration_logits.argmax(dim=-1)

    @property
    def predicted_agreement(self) -> Tensor:
        return self.agreement_logits.argmax(dim=-1)


class TriadModel(nn.Module):
    """Two encoders, a monotone alignment operator, a reconciling state and three readouts."""

    def __init__(self, config: ModelConfig, flags: VariantFlags | None = None) -> None:
        super().__init__()
        self.config = config
        self.flags = flags if flags is not None else VariantFlags()
        self.vision = VisionEncoder(config)
        if self.flags.use_language_encoder:
            self.language = LanguageEncoder(config)
        else:
            self.numeric_functional = nn.Linear(6, config.language_dim)
        if self.flags.use_reconciliation_state:
            self.reconciliation = CrossStreamReconciliation(config)
        else:
            self.late_fusion = nn.Sequential(
                nn.Linear(2 * config.vision_token_dim, config.vision_token_dim),
                nn.GELU(),
                nn.Linear(config.vision_token_dim, config.vision_token_dim),
            )
            self.weight_head = nn.Linear(config.vision_token_dim, 1)
        self.alignment = MonotoneCausalAlignment()
        self.progression_head = ProgressionHead(config)
        self.arbitration_head = ArbitrationHead(config)
        self.agreement_head = AgreementHead(config)

    def _expand_visits(self, values: Tensor, width: int) -> Tensor:
        return values.unsqueeze(-1).expand(*values.shape, width).reshape(values.shape[0], -1)

    def _functional_stream(self, batch: StreamBatch) -> tuple[Tensor, Tensor, Tensor]:
        if self.flags.use_language_encoder:
            tokens, valid = self.language(
                batch.token_ids, batch.functional_times, batch.functional_valid
            )
            width = self.config.max_perimetry_tokens + 1
        else:
            tokens = self.numeric_functional(batch.field_indices)
            valid = batch.functional_valid
            width = 1
        times = self._expand_visits(batch.functional_times, width)
        expanded_valid = self._expand_visits(batch.functional_valid, width)
        if width > 1:
            expanded_valid = valid & expanded_valid
        return tokens, times, expanded_valid

    def _late_fused_state(
        self,
        structural_tokens: Tensor,
        structural_valid: Tensor,
        functional_tokens: Tensor,
        functional_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Per-visit late fusion, the representation the ablation falls back to."""

        visits = min(structural_valid.shape[1], functional_valid.shape[1])
        structural_width = structural_tokens.shape[1] // structural_valid.shape[1]
        functional_width = functional_tokens.shape[1] // functional_valid.shape[1]
        pooled_structural = structural_tokens.reshape(
            structural_tokens.shape[0], structural_valid.shape[1], structural_width, -1
        )[:, :visits].mean(dim=2)
        pooled_functional = functional_tokens.reshape(
            functional_tokens.shape[0], functional_valid.shape[1], functional_width, -1
        )[:, :visits].mean(dim=2)
        joined = self.late_fusion(torch.cat((pooled_structural, pooled_functional), dim=-1))
        valid = structural_valid[:, :visits] | functional_valid[:, :visits]
        weights = valid.to(joined.dtype).unsqueeze(-1)
        state = (joined * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        weight = torch.sigmoid(self.weight_head(state)).squeeze(-1)
        return state, weight

    def _outcome_derived_direction(
        self, progression_logit: Tensor, stream_weight: Tensor
    ) -> Tensor:
        """Read the direction off the outcome when no three-class head is trained.

        The variant that keeps only outcome supervision has no classifier of its own, so the
        direction is composed from the progression readout and the stream weight. The
        manuscript prints no rule for this composition.
        """

        magnitude = 6.0 * torch.tanh(progression_logit)
        towards_structural = stream_weight >= 0.5
        signed = torch.where(towards_structural, magnitude, -magnitude)
        return torch.stack((signed, -signed, -0.5 * magnitude.abs()), dim=-1)

    def forward(self, batch: StreamBatch) -> TriadOutput:
        structural_tokens, structural_visits_valid = self.vision(
            batch.morphometry, batch.volume, batch.structural_times
        )
        functional_tokens, functional_times, functional_valid = self._functional_stream(batch)
        structural_times = self._expand_visits(batch.structural_times, self.vision.tokens_per_visit)
        structural_valid = self._expand_visits(
            structural_visits_valid, self.vision.tokens_per_visit
        )
        if self.flags.use_monotone_alignment:
            structural_weights, functional_weights = self.alignment(
                batch.query_years,
                structural_times,
                functional_times,
                structural_valid,
                functional_valid,
            )
        else:
            structural_weights = structural_valid.to(structural_tokens.dtype)
            functional_weights = functional_valid.to(functional_tokens.dtype)
        if self.flags.use_reconciliation_state:
            state, weight = self.reconciliation(
                structural_tokens,
                structural_valid,
                self.alignment.attention_bias(structural_weights, structural_valid),
                functional_tokens,
                functional_valid,
                self.alignment.attention_bias(functional_weights, functional_valid),
            )
        else:
            state, weight = self._late_fused_state(
                structural_tokens, structural_valid, functional_tokens, functional_valid
            )
        progression_logit = self.progression_head(state)
        if self.flags.use_arbitration_head:
            arbitration_logits = self.arbitration_head(state)
        else:
            arbitration_logits = self._outcome_derived_direction(progression_logit, weight)
        return TriadOutput(
            progression_logit=progression_logit,
            arbitration_logits=arbitration_logits,
            agreement_logits=self.agreement_head(state),
            stream_weight=weight,
            reconciliation_state=state,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
