"""Tests for hooks/artifacts.py — run result I/O."""

import json
from pathlib import Path

import pytest

from hooks.types import RunConfig, RunResults, PreRunContext
from hooks import artifacts
from hooks.artifacts import (
    _next_run_id,
    _best_val_bpb,
    _write_run_result,
)


@pytest.fixture
def sample_config():
    return RunConfig(
        matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
        scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
        device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
    )


@pytest.fixture
def sample_results():
    return RunResults(
        val_bpb=1.872, steps=1450, peak_vram_mb=4200.0, final_loss=2.34,
        mfu=0.38, diverged=False, loss_trend="improving", grad_norm_max=3.2,
    )


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr("hooks.artifacts.RESULTS_DIR", tmp_path)
    return tmp_path


class TestNextRunId:
    def test_empty_dir(self, tmp_results):
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        assert _next_run_id(run_dir) == 1

    def test_nonexistent_dir(self, tmp_results):
        run_dir = tmp_results / "runs" / "condition_A" / "seed_99"
        assert _next_run_id(run_dir) == 1

    def test_sequential_files(self, tmp_results, sample_config, sample_results):
        _write_run_result("A", 42, 1, sample_config, sample_results)
        _write_run_result("A", 42, 2, sample_config, sample_results)
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        assert _next_run_id(run_dir) == 3

    def test_gap_in_sequence(self, tmp_results):
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_001.json").write_text("{}")
        (run_dir / "run_005.json").write_text("{}")
        assert _next_run_id(run_dir) == 6


class TestBestValBpb:
    def test_all_diverged_returns_none(self, tmp_results):
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        doc = {"results": {"val_bpb": 999999.0, "diverged": True}}
        (run_dir / "run_001.json").write_text(json.dumps(doc))
        assert _best_val_bpb(run_dir) is None

    def test_mixed_diverged_and_normal(self, tmp_results):
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_001.json").write_text(json.dumps({"results": {"val_bpb": 0.5, "diverged": True}}))
        (run_dir / "run_002.json").write_text(json.dumps({"results": {"val_bpb": 1.85, "diverged": False}}))
        assert _best_val_bpb(run_dir) == 1.85

    def test_empty_dir_returns_none(self, tmp_results):
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        assert _best_val_bpb(run_dir) is None

    def test_corrupt_json_skipped(self, tmp_results):
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_001.json").write_text("not json!!!")
        (run_dir / "run_002.json").write_text(json.dumps({"results": {"val_bpb": 1.90, "diverged": False}}))
        assert _best_val_bpb(run_dir) == 1.90


class TestWriteRunResult:
    def test_all_json_schema_fields_present(self, tmp_results, sample_config, sample_results):
        path = _write_run_result("A", 42, 1, sample_config, sample_results)
        data = json.loads(path.read_text())
        for field in ["run_id", "condition", "seed", "timestamp", "config", "results",
                       "rationale_tag", "rationale", "wasted", "delta_from_best"]:
            assert field in data, f"Missing field: {field}"

    def test_rationale_stripped_from_config(self, tmp_results, sample_results):
        cfg = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
            rationale_tag="test_tag", rationale="test_rationale",
        )
        path = _write_run_result("A", 42, 1, cfg, sample_results)
        data = json.loads(path.read_text())
        assert "rationale_tag" not in data["config"]
        assert data["rationale_tag"] == "test_tag"

    def test_pre_run_context_included(self, tmp_results, sample_config, sample_results):
        ctx = PreRunContext(condition="A", summaries=["test summary"])
        path = _write_run_result("A", 42, 1, sample_config, sample_results, pre_run_context=ctx)
        data = json.loads(path.read_text())
        assert "pre_run_context" in data

    def test_delta_from_best_first_run(self, tmp_results, sample_config, sample_results):
        path = _write_run_result("A", 42, 1, sample_config, sample_results)
        data = json.loads(path.read_text())
        assert data["delta_from_best"] == 0.0
