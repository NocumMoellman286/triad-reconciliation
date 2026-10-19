"""Configuration conformance.

Every numeric value the manuscript prints as a design constant must appear in the released
configuration, and nothing may be missing that the reported analyses need.

Ref: Sec. 5.6 (two thousand replicates, four minimum detectable differences, the specificity
levels and the threshold probability), Sec. 5.5 (the consistency weight and the re-weighting
strength the development group fixes), Algorithm 1 (the optimiser and the schedule).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from triad_reconciliation.core.config import InferenceConfig, TrainConfig

REQUIRED_EXPERIMENTS: tuple[str, ...] = (
    "main.yaml",
    "ablation_without_reconciliation_state.yaml",
    "ablation_without_arbitration_head.yaml",
    "ablation_without_attribution_consistency.yaml",
    "ablation_without_agreement_state_reweighting.yaml",
    "ablation_without_monotone_causal_alignment.yaml",
    "ablation_with_numeric_perimetry_input.yaml",
    "ablation_without_calibrated_deferral.yaml",
    "ablation_interaction_state_and_head.yaml",
    "supplementary_workflow_outcomes.yaml",
    "supplementary_silent_mode.yaml",
    "supplementary_strata.yaml",
    "_smoke.yaml",
)


@dataclass(frozen=True)
class ConformanceOutcome:
    name: str
    status: str
    detail: str


def _load(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def check_experiment_files(root: Path) -> ConformanceOutcome:
    directory = Path(root) / "configs" / "experiment"
    present = {path.name for path in directory.glob("*.yaml")} if directory.is_dir() else set()
    missing = sorted(name for name in REQUIRED_EXPERIMENTS if name not in present)
    return ConformanceOutcome(
        name="experiment-files",
        status="PASS" if not missing else "FAIL",
        detail=f"{len(present)} files present, missing {missing}",
    )


def check_printed_constants(root: Path) -> ConformanceOutcome:
    """The four printed statistical constants must match the released defaults."""

    defaults = InferenceConfig()
    expected = {
        "bootstrap_replicates": 2_000,
        "specificity_target": 0.95,
        "matched_specificity": 0.90,
        "nominal_coverage": 0.90,
        "decision_threshold": 0.20,
    }
    mismatched = {
        key: getattr(defaults, key)
        for key, value in expected.items()
        if getattr(defaults, key) != value
    }
    margins = defaults.minimum_detectable_difference
    margin_ok = (
        margins.get("discrimination") == 0.030
        and margins.get("agreement") == 0.05
        and margins.get("reader_accuracy_pp") == 5.0
        and margins.get("deferral_intervals") == 1.0
    )
    main = _load(Path(root) / "configs" / "experiment" / "main.yaml")
    inference = main.get("inference", {}) if isinstance(main.get("inference"), dict) else {}
    replicates = inference.get("bootstrap_replicates", defaults.bootstrap_replicates)
    ok = not mismatched and margin_ok and replicates == 2_000
    return ConformanceOutcome(
        name="printed-statistical-constants",
        status="PASS" if ok else "FAIL",
        detail=f"mismatched {mismatched}, four minimum detectable differences present {margin_ok}, "
        f"replicates from the experiment file {replicates}",
    )


def check_optimiser_and_schedule(root: Path) -> ConformanceOutcome:
    """Algorithm 1 prints AdamW under a cosine schedule; nothing else may be configured."""

    defaults = TrainConfig()
    main = _load(Path(root) / "configs" / "experiment" / "main.yaml")
    train = main.get("train", {}) if isinstance(main.get("train"), dict) else {}
    optimiser = str(train.get("optimizer", defaults.optimizer))
    scheduler = str(train.get("scheduler", defaults.scheduler))
    weight = train.get("consistency_weight", defaults.consistency_weight)
    strength = train.get("reweighting_strength", defaults.reweighting_strength)
    ok = (
        optimiser == "adamw"
        and scheduler == "cosine"
        and isinstance(weight, float)
        and isinstance(strength, float)
    )
    return ConformanceOutcome(
        name="optimiser-and-schedule",
        status="PASS" if ok else "FAIL",
        detail=f"optimiser {optimiser}, schedule {scheduler}, consistency weight {weight}, "
        f"re-weighting strength {strength}",
    )


def check_architecture_constants(root: Path) -> ConformanceOutcome:
    """The alignment margin and the token widths must be declared and self-consistent."""

    payload = _load(Path(root) / "configs" / "model" / "reconciliation.yaml")
    main = payload.get("model", payload) if isinstance(payload.get("model"), dict) else payload
    token_dim = main.get("vision_token_dim")
    language_dim = main.get("language_dim")
    padding = main.get("max_perimetry_tokens")
    ok = token_dim == language_dim and isinstance(padding, int) and padding > 0
    return ConformanceOutcome(
        name="architecture-constants",
        status="PASS" if ok else "FAIL",
        detail=f"vision width {token_dim}, language width {language_dim}, perimetry tokens {padding}",
    )


def check_smoke_is_labelled(root: Path) -> ConformanceOutcome:
    """A smoke file may only exist to carry the unit tests, and must say so."""

    path = Path(root) / "configs" / "experiment" / "_smoke.yaml"
    if not path.is_file():
        return ConformanceOutcome(
            name="smoke-labelling", status="FAIL", detail="no smoke file present"
        )
    body = path.read_text(encoding="utf-8")
    labelled = "unit-test smoke only" in body
    return ConformanceOutcome(
        name="smoke-labelling",
        status="PASS" if labelled else "FAIL",
        detail=(
            "the smoke file declares that it is for unit tests and not for reporting"
            if labelled
            else "the smoke file does not declare its scope"
        ),
    )


def check_no_quick_start_shrink(root: Path) -> ConformanceOutcome:
    """No file other than the smoke file may quietly reduce the reported training budget."""

    directory = Path(root) / "configs" / "experiment"
    offending: list[str] = []
    for path in sorted(directory.glob("*.yaml")):
        if path.name == "_smoke.yaml":
            continue
        payload = _load(path)
        train = payload.get("train", {})
        if not isinstance(train, dict):
            continue
        if "epochs" in train and int(train["epochs"]) < 8:
            offending.append(f"{path.name}: epochs {train['epochs']}")
    return ConformanceOutcome(
        name="no-quick-start",
        status="PASS" if not offending else "FAIL",
        detail=f"files that shrink the budget: {offending}",
    )


def all_conformance(root: Path) -> list[ConformanceOutcome]:
    return [
        check_experiment_files(root),
        check_printed_constants(root),
        check_optimiser_and_schedule(root),
        check_architecture_constants(root),
        check_smoke_is_labelled(root),
        check_no_quick_start_shrink(root),
    ]
