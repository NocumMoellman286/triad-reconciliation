"""Seeding, path resolution and atomic writing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from triad_reconciliation.core.config import ExperimentConfig, apply_overrides, load_experiment
from triad_reconciliation.core.io_text import render_block
from triad_reconciliation.core.jsonio import (
    atomic_write_text,
    digest_paths,
    manifest_digest,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from triad_reconciliation.core.paths import relative_to_root, repository_root
from triad_reconciliation.core.seeding import DEFAULT_SEED, seed_rng, set_seed


def test_set_seed_reproduces_streams() -> None:
    set_seed(11)
    first = (torch.rand(3), np.random.random(3), torch.nn.Linear(2, 2).weight.detach().clone())
    set_seed(11)
    second = (torch.rand(3), np.random.random(3), torch.nn.Linear(2, 2).weight.detach().clone())
    assert torch.equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])
    assert torch.equal(first[2], second[2])


def test_default_seed_is_recorded() -> None:
    assert set_seed() == DEFAULT_SEED
    assert seed_rng(5).integers(0, 10) == seed_rng(5).integers(0, 10)


def test_atomic_write_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "payload.json"
    write_json(target, {"b": 1, "a": 2})
    assert read_json(target) == {"a": 2, "b": 1}
    assert target.read_text(encoding="utf-8").startswith("{")
    assert oct(target.stat().st_mode & 0o777) == "0o644"
    assert sorted(path.name for path in target.parent.iterdir()) == ["payload.json"]


def test_jsonl_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    write_jsonl(target, [{"a": 1}, {"a": 2}])
    assert read_jsonl(target) == [{"a": 1}, {"a": 2}]


def test_manifest_digest_is_order_sensitive_and_excludes_named_entries(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one", encoding="utf-8")
    (tmp_path / "two.txt").write_text("two", encoding="utf-8")
    entries = digest_paths(tmp_path)
    assert set(entries) == {"one.txt", "two.txt"}
    assert manifest_digest(entries) == manifest_digest(dict(entries))
    assert manifest_digest(entries) != manifest_digest(dict(reversed(list(entries.items()))))
    assert set(digest_paths(tmp_path, exclude=frozenset({"one.txt"}))) == {"two.txt"}


def test_digest_skips_unanchored_source_data_directories(tmp_path: Path) -> None:
    nested = tmp_path / "src" / "pkg" / "data"
    nested.mkdir(parents=True)
    (nested / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cohort.jsonl").write_text("{}\n", encoding="utf-8")
    entries = digest_paths(tmp_path)
    assert "src/pkg/data/module.py" in entries
    assert "data/cohort.jsonl" not in entries


def test_repository_root_and_relative_rendering(release_root: Path) -> None:
    assert repository_root() == release_root
    inside = release_root / "pyproject.toml"
    assert relative_to_root(inside, release_root) == "pyproject.toml"
    assert relative_to_root(Path("/elsewhere/file.txt"), release_root) == "file.txt"


def test_render_block_pads_to_the_widest_value() -> None:
    block = render_block("T", ("a", "b"), [["1", "short"], ["2", "a-considerably-longer-value"]])
    rows = [line for line in block.splitlines() if line.startswith(("1", "2"))]
    assert all(len(row) == len(rows[0]) for row in rows)
    assert "a-considerably-longer-value" in rows[-1]


def test_experiment_loading_merges_components(release_root: Path) -> None:
    config = load_experiment(
        release_root / "configs" / "experiment" / "main.yaml", root=release_root
    )
    assert config.model.vision_token_dim == 96
    assert config.train.optimizer == "adamw"
    assert config.inference.bootstrap_replicates == 2000
    assert config.path == "configs/experiment/main.yaml"


def test_overrides_apply_before_defaults(release_root: Path) -> None:
    config = load_experiment(
        release_root / "configs" / "experiment" / "main.yaml", root=release_root
    )
    overridden = apply_overrides(config, ["train.epochs=3", "model.dropout=0.2"])
    assert overridden.train.epochs == 3
    assert overridden.model.dropout == 0.2
    assert overridden.model.vision_token_dim == config.model.vision_token_dim


def test_override_rejects_unknown_sections(release_root: Path) -> None:
    config = load_experiment(
        release_root / "configs" / "experiment" / "main.yaml", root=release_root
    )
    try:
        apply_overrides(config, ["nonsense.key=1"])
    except ValueError as error:
        assert "unknown override section" in str(error)
    else:  # pragma: no cover - the branch must raise
        raise AssertionError("an unknown section must be rejected")


def test_experiment_mapping_is_json_serialisable(release_root: Path) -> None:
    config: ExperimentConfig = load_experiment(
        release_root / "configs" / "experiment" / "main.yaml", root=release_root
    )
    assert json.loads(json.dumps(config.as_mapping()))["name"] == "main"


def test_atomic_write_leaves_no_staging_file(tmp_path: Path) -> None:
    target = tmp_path / "report.txt"
    atomic_write_text(target, "body")
    assert [path.name for path in tmp_path.iterdir()] == ["report.txt"]
