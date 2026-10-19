"""Turn visit-pairs into the tensors the model consumes and the flat features the clinical
and statistical comparators consume.

Both views are cut at the pair's query time, so a batch can never carry a predictor recorded
after the encounter it is asked about.

Ref: Sec. 5.1 (the streams and the elements of a pair), Sec. 5.4 (the predictor embargo at a
testing time), Algorithm 1 steps 4-5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from triad_reconciliation.core.config import ModelConfig
from triad_reconciliation.core.records import (
    ARBITRATION_CLASSES,
    N_MORPHOMETRY,
    N_PERIMETRY_POINTS,
    VOLUME_EDGE,
    VisitPair,
)
from triad_reconciliation.models.triad import StreamBatch
from triad_reconciliation.streams.perimetry import encode_report

N_FIELD_INDICES = 6


@dataclass(frozen=True)
class FlatFeatures:
    """Array view of a pair set, one row per visit-pair."""

    pair_ids: tuple[str, ...]
    patients: np.ndarray
    eyes: np.ndarray
    regions: np.ndarray
    sites: np.ndarray
    progression: np.ndarray
    arbitration: np.ndarray
    agreement: np.ndarray
    morphometry: np.ndarray
    volume: np.ndarray
    structural_times: np.ndarray
    structural_mask: np.ndarray
    field_indices: np.ndarray
    field_times: np.ndarray
    field_mask: np.ndarray
    sensitivity: np.ndarray
    token_ids: np.ndarray
    query_years: np.ndarray

    @property
    def n_pairs(self) -> int:
        return int(self.progression.shape[0])

    @property
    def discordant(self) -> np.ndarray:
        flagged: np.ndarray = self.agreement == 1
        return flagged

    def structural_summary(self) -> np.ndarray:
        """The most recent structural visit's morphometry, zero where none is recorded."""

        latest = np.argmax(np.where(self.structural_mask, self.structural_times, -np.inf), axis=1)
        rows = self.morphometry[np.arange(self.n_pairs), latest]
        present = self.structural_mask.any(axis=1).reshape(-1, 1)
        selected: np.ndarray = np.where(present, rows, 0.0)
        return selected

    def latest_field_indices(self) -> np.ndarray:
        latest = np.argmax(np.where(self.field_mask, self.field_times, -np.inf), axis=1)
        rows = self.field_indices[np.arange(self.n_pairs), latest]
        present = self.field_mask.sum(axis=1).reshape(-1, 1)
        selected: np.ndarray = np.where(present > 0, rows, 0.0)
        return selected

    def masked_structural_mean(self) -> np.ndarray:
        mask = self.structural_mask[:, :, None].astype(np.float64)
        total = mask.sum(axis=1)
        summed = (self.morphometry * mask).sum(axis=1)
        averaged: np.ndarray = np.divide(
            summed, np.clip(total, 1e-9, None), out=np.zeros_like(summed), where=total > 1e-9
        )
        return averaged

    def masked_field_mean(self) -> np.ndarray:
        mask = self.field_mask[:, :, None].astype(np.float64)
        total = mask.sum(axis=1)
        summed = (self.field_indices * mask).sum(axis=1)
        averaged: np.ndarray = np.divide(
            summed, np.clip(total, 1e-9, None), out=np.zeros_like(summed), where=total > 1e-9
        )
        return averaged


def _pad_structural(pairs: tuple[VisitPair, ...], max_visits: int) -> tuple[np.ndarray, ...]:
    n_pairs = len(pairs)
    morphometry = np.zeros((n_pairs, max_visits, N_MORPHOMETRY), dtype=np.float32)
    volume = np.zeros(
        (n_pairs, max_visits, VOLUME_EDGE, VOLUME_EDGE, VOLUME_EDGE), dtype=np.float32
    )
    times = np.zeros((n_pairs, max_visits), dtype=np.float32)
    mask = np.zeros((n_pairs, max_visits), dtype=bool)
    for row, pair in enumerate(pairs):
        visits = pair.structural[-max_visits:]
        for position, visit in enumerate(visits):
            morphometry[row, position] = np.asarray(visit.morphometry, dtype=np.float32)
            volume[row, position] = np.asarray(visit.volume, dtype=np.float32).reshape(
                VOLUME_EDGE, VOLUME_EDGE, VOLUME_EDGE
            )
            times[row, position] = visit.acquired_years
            mask[row, position] = True
    return morphometry, volume, times, mask


def _pad_functional(
    pairs: tuple[VisitPair, ...], max_visits: int, max_tokens: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_pairs = len(pairs)
    token_ids = np.zeros((n_pairs, max_visits, max_tokens), dtype=np.int64)
    field_indices = np.zeros((n_pairs, max_visits, N_FIELD_INDICES), dtype=np.float32)
    sensitivity = np.zeros((n_pairs, max_visits, N_PERIMETRY_POINTS), dtype=np.float32)
    times = np.zeros((n_pairs, max_visits), dtype=np.float32)
    mask = np.zeros((n_pairs, max_visits), dtype=bool)
    for row, pair in enumerate(pairs):
        reports = pair.functional[-max_visits:]
        for position, report in enumerate(reports):
            ids, _ = encode_report(report, max_tokens)
            token_ids[row, position] = np.asarray(ids, dtype=np.int64)
            sensitivity[row, position] = np.asarray(report.sensitivity_db, dtype=np.float32)
            field_indices[row, position] = np.asarray(
                (
                    report.mean_deviation_db,
                    report.pattern_deviation_db,
                    report.visual_field_index,
                    report.false_positive_rate,
                    report.false_negative_rate,
                    float(report.fixation_losses),
                ),
                dtype=np.float32,
            )
            times[row, position] = report.tested_years
            mask[row, position] = True
    return token_ids, field_indices, sensitivity, times, mask


def build_flat_features(pairs: tuple[VisitPair, ...], config: ModelConfig) -> FlatFeatures:
    morphometry, volume, structural_times, structural_mask = _pad_structural(
        pairs, config.max_visits
    )
    token_ids, field_indices, sensitivity, field_times, field_mask = _pad_functional(
        pairs, config.max_visits, config.max_perimetry_tokens
    )
    arbitration_index = {str(label): index for index, label in enumerate(ARBITRATION_CLASSES)}
    return FlatFeatures(
        pair_ids=tuple(pair.pair_id for pair in pairs),
        patients=np.asarray([pair.patient_id for pair in pairs], dtype=object),
        eyes=np.asarray([pair.eye_id for pair in pairs], dtype=object),
        regions=np.asarray([pair.region for pair in pairs], dtype=object),
        sites=np.asarray([pair.site for pair in pairs], dtype=object),
        progression=np.asarray([pair.progression for pair in pairs], dtype=np.int64),
        arbitration=np.asarray(
            [arbitration_index[str(pair.arbitration)] for pair in pairs], dtype=np.int64
        ),
        agreement=np.asarray([1 if pair.discordant else 0 for pair in pairs], dtype=np.int64),
        morphometry=morphometry,
        volume=volume,
        structural_times=structural_times,
        structural_mask=structural_mask,
        field_indices=field_indices,
        field_times=field_times,
        field_mask=field_mask,
        sensitivity=sensitivity,
        token_ids=token_ids,
        query_years=np.asarray([pair.index_years for pair in pairs], dtype=np.float32),
    )


def to_stream_batch(features: FlatFeatures) -> StreamBatch:
    """The torch view of the flat features, for the reconciliation model and its comparators."""

    return StreamBatch(
        morphometry=torch.from_numpy(features.morphometry),
        volume=torch.from_numpy(features.volume),
        structural_times=torch.from_numpy(features.structural_times),
        structural_valid=torch.from_numpy(features.structural_mask),
        token_ids=torch.from_numpy(features.token_ids),
        field_indices=torch.from_numpy(features.field_indices),
        functional_times=torch.from_numpy(features.field_times),
        functional_valid=torch.from_numpy(features.field_mask),
        query_years=torch.from_numpy(features.query_years),
        progression=torch.from_numpy(features.progression),
        arbitration=torch.from_numpy(features.arbitration),
        agreement=torch.from_numpy(features.agreement),
    )


def iterate_batches(
    features: FlatFeatures, batch_size: int, *, shuffle: bool = False, seed: int = 0
) -> list[tuple[np.ndarray, StreamBatch]]:
    """Slice the flat features into batches, keeping the row index beside each tensor batch."""

    order = np.arange(features.n_pairs)
    if shuffle:
        order = np.random.default_rng(seed).permutation(order)
    batches: list[tuple[np.ndarray, StreamBatch]] = []
    for start in range(0, order.size, batch_size):
        rows = order[start : start + batch_size]
        batches.append((rows, to_stream_batch(slice_features(features, rows))))
    return batches


def slice_features(features: FlatFeatures, rows: np.ndarray) -> FlatFeatures:
    selection = np.asarray(rows, dtype=np.int64)
    return FlatFeatures(
        pair_ids=tuple(features.pair_ids[index] for index in selection.tolist()),
        patients=features.patients[selection],
        eyes=features.eyes[selection],
        regions=features.regions[selection],
        sites=features.sites[selection],
        progression=features.progression[selection],
        arbitration=features.arbitration[selection],
        agreement=features.agreement[selection],
        morphometry=features.morphometry[selection],
        volume=features.volume[selection],
        structural_times=features.structural_times[selection],
        structural_mask=features.structural_mask[selection],
        field_indices=features.field_indices[selection],
        field_times=features.field_times[selection],
        field_mask=features.field_mask[selection],
        sensitivity=features.sensitivity[selection],
        token_ids=features.token_ids[selection],
        query_years=features.query_years[selection],
    )
