"""Pytest fixtures and the tensor harness shared by the released test suite.

Nothing here builds a cohort. The analysed records are not distributed and this release
generates none, so a test that needs one takes the table a caller supplies and the numerical
tests run on the tensor harness in the package's verification module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from triad_reconciliation.core.config import (
    CohortConfig,
    ExperimentConfig,
    InferenceConfig,
    ModelConfig,
    TrainConfig,
)
from triad_reconciliation.core.seeding import set_seed
from triad_reconciliation.models.triad import TriadModel
from triad_reconciliation.verification.execution import HARNESS_SEED, random_stream_batch

MODEL = ModelConfig(max_visits=4)
TRAIN = TrainConfig(epochs=2, batch_size=8, learning_rate=0.006)

# The smallest table the schema accepts, written as a literal so that the reader has something
# to read. This is parser input and not a cohort: the identifiers are placeholders, the values
# are hand-written, nothing consumes them outside this suite and nothing writes them to the
# release tree.
SMALL_TABLE_LINES: tuple[str, ...] = (
    '{"age_years": 61.0, "ancestry_group": 1, "eye_id": "E1", "high_myopia": false, "patient_id": "P1", "record": "eye", "region": "II", "sex": "female", "site": "A", "stage": "early"}',
    '{"eye_id": "E1", "morphometry": [0.45, 0.43, 1.42, 2.5, 0.12, 84.0, 78.0, 0.91], "patient_id": "P1", "record": "structural", "time_years": 0.0, "volume": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]}',
    '{"duration_s": 280.0, "eye_id": "E1", "fix": 0, "fn": 0.03, "fp": 0.02, "gaze_ok": true, "md": -2.0, "patient_id": "P1", "pd": -1.8, "record": "functional", "sensitivity_db": [30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0], "time_years": 0.0, "vfi": 92.0}',
    '{"arbitration": "concordant", "eye_id": "E1", "index_stream": "functional", "index_years": 0.0, "nearest_years": 0.0, "pair_id": "P1-000", "patient_id": "P1", "progression": 0, "record": "pair"}',
)


@pytest.fixture(scope="session")
def caller_table(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The smallest table the schema accepts, written for this test session."""

    target = tmp_path_factory.mktemp("caller") / "cohort.jsonl"
    target.write_text("\n".join(SMALL_TABLE_LINES) + "\n", encoding="utf-8")
    return target


@pytest.fixture
def model() -> TriadModel:
    set_seed(HARNESS_SEED)
    return TriadModel(MODEL)


@pytest.fixture(scope="session")
def tensor_batch():
    """One tensor batch, drawn from the recorded harness seed."""

    return random_stream_batch(6, MODEL, HARNESS_SEED)


@pytest.fixture(scope="session")
def smoke_config(caller_table: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="smoke",
        path="configs/experiment/_smoke.yaml",
        model=MODEL,
        cohort=CohortConfig(),
        train=TRAIN,
        inference=InferenceConfig(),
    )


@pytest.fixture(scope="session")
def release_root() -> Path:
    return Path(__file__).resolve().parents[1]
