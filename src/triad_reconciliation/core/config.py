"""Configuration loading for the released model, cohort and training settings.

Ref: Sec. 5.5, Algorithm 1 (AdamW with a cosine schedule), Sec. 5.6 (2,000 cluster-bootstrap
replicates, four pre-specified minimum detectable differences).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Reconciliation stack shape.

    The manuscript names the two encoders and the reconciliation operator without printing
    any width, depth or head count; the values here are engineering defaults and are marked
    as such in the verification report.
    """

    vision_token_dim: int = 96
    volume_patch_edge: int = 2
    morphometry_tokens: int = 1
    language_dim: int = 96
    language_layers: int = 2
    language_heads: int = 4
    language_vocab: int = 512
    max_perimetry_tokens: int = 28
    adapter_bottleneck: int = 24
    reconciliation_heads: int = 4
    reconciliation_layers: int = 2
    dropout: float = 0.12
    max_visits: int = 8
    time_features: int = 16
    stream_weight_margin: float = 0.46


@dataclass(frozen=True)
class CohortConfig:
    """How the pipeline reads the cohort table a caller supplies.

    The analysed records are not distributed. What the release fixes here is the reading
    convention the manuscript describes: the two clinical grids a visit-pair is taken between,
    the region that is held out, and the split the conformal threshold is fitted on. No count,
    no follow-up distribution and no label mix is fixed, because none of those is the reader's
    to choose.

    Ref: Sec. 5.2 (the two regions), Sec. 5.4 (the imaging and perimetry grids), Sec. 5.6 (the
    calibration split), Sec. 5.8 (a deterministic analysis under the recorded settings).
    """

    seed: int = 20_260_219
    held_out_region: str = "II"
    imaging_interval_months: int = 12
    perimetry_interval_months: int = 6
    pairing_window_share: float = 0.75
    calibration_share: float = 0.5


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation settings.

    ``optimizer`` and ``scheduler`` are printed in Algorithm 1; the remaining values are
    engineering defaults because the manuscript reports no schedule constant.
    """

    optimizer: str = "adamw"
    scheduler: str = "cosine"
    learning_rate: float = 0.00058
    weight_decay: float = 0.013
    betas: tuple[float, float] = (0.9, 0.97)
    batch_size: int = 24
    epochs: int = 32
    warmup_fraction: float = 0.08
    grad_clip: float = 1.0
    precision: str = "bf16"
    reweighting_strength: float = 2.35
    consistency_weight: float = 0.65
    world_size: int = 1


@dataclass(frozen=True)
class InferenceConfig:
    """Statistical layer constants printed by the manuscript."""

    bootstrap_replicates: int = 2_000
    specificity_target: float = 0.95
    matched_specificity: float = 0.90
    calibration_bins: int = 15
    nominal_coverage: float = 0.90
    decision_threshold: float = 0.20
    minimum_detectable_difference: dict[str, float] = field(
        default_factory=lambda: {
            "discrimination": 0.030,
            "agreement": 0.05,
            "reader_accuracy_pp": 5.0,
            "deferral_intervals": 1.0,
        }
    )


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    path: str
    model: ModelConfig
    cohort: CohortConfig
    train: TrainConfig
    inference: InferenceConfig
    variant: str = "full"
    notes: str = ""

    def as_mapping(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "variant": self.variant,
            "notes": self.notes,
            "model": self.model.__dict__,
            "cohort": self.cohort.__dict__,
            "train": self.train.__dict__,
            "inference": self.inference.__dict__,
        }


def _build(dataclass_type: type[Any], payload: dict[str, Any] | None) -> Any:
    if not payload:
        return dataclass_type()
    known = {field_name for field_name in dataclass_type.__dataclass_fields__}
    filtered = {key: value for key, value in payload.items() if key in known}
    for key in ("stage_mix", "betas"):
        if key in filtered and isinstance(filtered[key], list):
            filtered[key] = tuple(filtered[key])
    return dataclass_type(**filtered)


def _merge_layers(source: Path, root: Path) -> dict[str, Any]:
    """Merge the component files an experiment declares, then its own sections on top."""

    raw: dict[str, Any] = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    merged: dict[str, Any] = {}
    for relative in raw.get("defaults", []) or []:
        component = (root / "configs" / str(relative)).resolve()
        payload = yaml.safe_load(component.read_text(encoding="utf-8")) or {}
        for section, values in payload.items():
            if isinstance(values, dict):
                merged.setdefault(section, {}).update(values)
            else:
                merged[section] = values
    for section, values in raw.items():
        if section == "defaults":
            continue
        if isinstance(values, dict):
            merged.setdefault(section, {}).update(values)
        else:
            merged[section] = values
    return merged


def load_experiment(path: Path, *, root: Path | None = None) -> ExperimentConfig:
    """Load one experiment file, merging its declared components and filling the rest."""

    source = Path(path).resolve()
    base = Path(root).resolve() if root is not None else source.parents[2]
    raw = _merge_layers(source, base)
    try:
        rendered = source.relative_to(base).as_posix()
    except ValueError:
        rendered = source.name
    return ExperimentConfig(
        name=str(raw.get("name", source.stem)),
        path=rendered,
        variant=str(raw.get("variant", source.stem)),
        notes=str(raw.get("notes", "")),
        model=_build(ModelConfig, raw.get("model")),
        cohort=_build(CohortConfig, raw.get("cohort")),
        train=_build(TrainConfig, raw.get("train")),
        inference=_build(InferenceConfig, raw.get("inference")),
    )


def apply_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Apply ``section.key=value`` overrides on top of an experiment file."""

    if not overrides:
        return config
    raw = config.as_mapping()
    for item in overrides:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"override {item!r} is not of the form section.key=value")
        dotted, literal = item.split("=", 1)
        section, key = dotted.split(".", 1)
        if section not in {"model", "cohort", "train", "inference"}:
            raise ValueError(f"unknown override section {section!r}")
        raw[section][key] = yaml.safe_load(literal)
    return ExperimentConfig(
        name=config.name,
        path=config.path,
        variant=config.variant,
        notes=config.notes,
        model=_build(ModelConfig, raw["model"]),
        cohort=_build(CohortConfig, raw["cohort"]),
        train=_build(TrainConfig, raw["train"]),
        inference=_build(InferenceConfig, raw["inference"]),
    )
