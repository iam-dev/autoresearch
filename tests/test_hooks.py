"""Tests for mnemebrain_hooks — Conditions A & B (no external deps)."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from mnemebrain_hooks import (
    NullHooks,
    LoggingHooks,
    RunConfig,
    RunResults,
    PreRunContext,
    PredictionResult,
    RecommendationResult,
    create_hooks,
    evidence_weight,
    _config_distance,
    _is_wasted,
    _write_run_result,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    return RunConfig(
        matrix_lr=0.04,
        embedding_lr=0.6,
        unembedding_lr=0.004,
        scalar_lr=0.5,
        wd=0.2,
        depth=4,
        total_batch_size=65536,
        device_batch_size=16,
        warmup_ratio=0.0,
        warmdown_ratio=0.5,
        seed=42,
    )


@pytest.fixture
def sample_results():
    return RunResults(
        val_bpb=1.872,
        steps=1450,
        peak_vram_mb=4200.0,
        final_loss=2.34,
        mfu=0.38,
        diverged=False,
        loss_trend="improving",
        grad_norm_max=3.2,
    )


@pytest.fixture
def diverged_results():
    return RunResults(
        val_bpb=float("nan"),
        steps=342,
        peak_vram_mb=4200.0,
        final_loss=100.0,
        mfu=0.10,
        diverged=True,
        loss_trend="diverging",
        grad_norm_max=9.1,
    )


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    """Redirect RESULTS_DIR to a temp directory."""
    monkeypatch.setattr("mnemebrain_hooks.RESULTS_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

class TestDataContracts:
    def test_run_config_lr_property(self, sample_config):
        assert sample_config.lr == sample_config.matrix_lr == 0.04

    def test_run_config_defaults(self):
        cfg = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
        )
        assert cfg.rationale_tag == ""
        assert cfg.rationale == ""

    def test_prediction_result_defaults(self):
        p = PredictionResult(
            expected_outcome="improve", confidence=0.8,
            similar_runs=[1, 2], risks=["test"],
        )
        assert p.source_run_ids == []

    def test_recommendation_result_defaults(self):
        r = RecommendationResult(
            suggested_change="lower lr", rationale=["test"], risk_level="low",
        )
        assert r.source_run_ids == []

    def test_pre_run_context_defaults(self):
        ctx = PreRunContext(condition="A")
        assert ctx.summaries == []
        assert ctx.similar_runs == []
        assert ctx.contradictions == []
        assert ctx.prediction is None
        assert ctx.recommendation is None


# ---------------------------------------------------------------------------
# Evidence weight formula
# ---------------------------------------------------------------------------

class TestEvidenceWeight:
    def test_improvement_high_weight(self):
        """Strong improvement → weight well above 0.5."""
        w = evidence_weight(1.80, 1.85)  # delta = -0.05
        assert w > 0.8

    def test_regression_low_weight(self):
        """Strong regression → weight well below 0.5."""
        w = evidence_weight(1.90, 1.85)  # delta = +0.05
        assert w < 0.2

    def test_small_improvement_above_half(self):
        """Even small improvements should get weight > 0.5."""
        w = evidence_weight(1.849, 1.85)  # delta = -0.001
        assert w > 0.5

    def test_small_regression_below_half(self):
        """Even small regressions should get weight < 0.5."""
        w = evidence_weight(1.851, 1.85)  # delta = +0.001
        assert w < 0.5

    def test_zero_delta_returns_half(self):
        """No change → weight = 0.5."""
        w = evidence_weight(1.85, 1.85)
        assert w == 0.5

    def test_monotonic(self):
        """Larger improvements should have higher weight."""
        w_small = evidence_weight(1.84, 1.85)
        w_large = evidence_weight(1.80, 1.85)
        assert w_large > w_small


# ---------------------------------------------------------------------------
# Config distance
# ---------------------------------------------------------------------------

class TestConfigDistance:
    def test_identical_configs_zero_distance(self, sample_config):
        from dataclasses import asdict
        cfg = asdict(sample_config)
        assert _config_distance(cfg, cfg) == 0.0

    def test_different_configs_positive_distance(self, sample_config):
        from dataclasses import asdict
        cfg1 = asdict(sample_config)
        cfg2 = dict(cfg1)
        cfg2["matrix_lr"] = 0.1  # significantly different
        assert _config_distance(cfg1, cfg2) > 0.0


# ---------------------------------------------------------------------------
# NullHooks (Condition A)
# ---------------------------------------------------------------------------

class TestNullHooks:
    def test_pre_run_returns_condition_a(self, sample_config):
        hooks = NullHooks(seed=42)
        ctx = hooks.pre_run(sample_config)
        assert ctx.condition == "A"
        assert ctx.summaries == []
        assert ctx.prediction is None

    def test_post_run_writes_json(self, sample_config, sample_results, tmp_results):
        hooks = NullHooks(seed=42)
        hooks.post_run(sample_config, sample_results)

        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        files = list(run_dir.glob("run_*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        assert data["condition"] == "A"
        assert data["seed"] == 42
        assert data["run_id"] == 1
        assert data["results"]["val_bpb"] == 1.872
        assert data["results"]["diverged"] is False
        assert data["wasted"] is False

    def test_post_run_increments_run_id(self, sample_config, sample_results, tmp_results):
        hooks = NullHooks(seed=42)
        hooks.post_run(sample_config, sample_results)
        hooks.post_run(sample_config, sample_results)

        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        files = sorted(run_dir.glob("run_*.json"))
        assert len(files) == 2
        assert json.loads(files[0].read_text())["run_id"] == 1
        assert json.loads(files[1].read_text())["run_id"] == 2

    def test_diverged_run_marked_wasted(self, sample_config, diverged_results, tmp_results):
        hooks = NullHooks(seed=42)
        hooks.post_run(sample_config, diverged_results)

        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        data = json.loads(next(run_dir.glob("run_*.json")).read_text())
        assert data["wasted"] is True


# ---------------------------------------------------------------------------
# LoggingHooks (Condition B)
# ---------------------------------------------------------------------------

class TestLoggingHooks:
    def test_pre_run_returns_condition_b(self, sample_config, tmp_results):
        hooks = LoggingHooks(seed=42)
        ctx = hooks.pre_run(sample_config)
        assert ctx.condition == "B"

    def test_post_run_writes_json_and_jsonl(self, sample_config, sample_results, tmp_results):
        hooks = LoggingHooks(seed=42)
        hooks.post_run(sample_config, sample_results)

        # Check individual run JSON
        run_dir = tmp_results / "runs" / "condition_B" / "seed_42"
        files = list(run_dir.glob("run_*.json"))
        assert len(files) == 1

        # Check JSONL log
        log_path = tmp_results / "experiment_log.jsonl"
        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1

        entry = json.loads(lines[0])
        assert entry["run_id"] == 1
        assert entry["seed"] == 42
        assert "timestamp" in entry
        assert entry["config"]["matrix_lr"] == 0.04
        assert entry["outcome"]["val_bpb"] == 1.872
        assert entry["signals"]["loss_trend"] == "improving"
        assert "summary" in entry

    def test_pre_run_shows_recent_summaries(self, sample_config, sample_results, tmp_results):
        hooks = LoggingHooks(seed=42)
        # Write two runs
        hooks.post_run(sample_config, sample_results)
        hooks.post_run(sample_config, sample_results)

        ctx = hooks.pre_run(sample_config)
        assert len(ctx.summaries) == 2
        assert all("Run" in s for s in ctx.summaries)

    def test_jsonl_schema_has_required_fields(self, sample_config, sample_results, tmp_results):
        hooks = LoggingHooks(seed=42)
        hooks.post_run(sample_config, sample_results)

        log_path = tmp_results / "experiment_log.jsonl"
        entry = json.loads(log_path.read_text().strip())

        # Per plan's Condition B schema (v3 patch 5)
        assert "run_id" in entry
        assert "timestamp" in entry
        assert "seed" in entry
        assert "config" in entry
        assert "outcome" in entry
        assert "signals" in entry
        assert "summary" in entry


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Wasted-run logic
# ---------------------------------------------------------------------------

class TestWastedRun:
    def test_diverged_is_wasted(self, sample_config, diverged_results, tmp_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        seed_dir.mkdir(parents=True, exist_ok=True)
        assert _is_wasted(sample_config, diverged_results, seed_dir, best_val_bpb=1.85) is True

    def test_regression_beyond_epsilon_is_wasted(self, sample_config, tmp_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        seed_dir.mkdir(parents=True, exist_ok=True)
        regressed = RunResults(
            val_bpb=1.90, steps=1000, peak_vram_mb=4200.0, final_loss=2.5,
            mfu=0.38, diverged=False, loss_trend="flat", grad_norm_max=3.0,
        )
        # best is 1.85, regression to 1.90 → delta = 0.05 > epsilon (0.02)
        assert _is_wasted(sample_config, regressed, seed_dir, best_val_bpb=1.85) is True

    def test_improvement_not_wasted(self, sample_config, sample_results, tmp_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        seed_dir.mkdir(parents=True, exist_ok=True)
        # val_bpb=1.872, best=1.90 → improvement
        assert _is_wasted(sample_config, sample_results, seed_dir, best_val_bpb=1.90) is False

    def test_near_bad_config_is_wasted(self, sample_config, sample_results, tmp_results):
        """A run near a previously diverged config is wasted."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        # Write a prior diverged run with same config
        _write_run_result(
            "A", 42, 1, sample_config,
            RunResults(val_bpb=999999.0, steps=100, peak_vram_mb=0, final_loss=999999.0,
                       mfu=0, diverged=True, loss_trend="diverging", grad_norm_max=10.0),
        )
        # Now check a new run with identical config → near bad config
        assert _is_wasted(sample_config, sample_results, seed_dir, best_val_bpb=1.90) is True

    def test_exact_duplicate_no_rationale_is_wasted(self, sample_config, sample_results, tmp_results):
        """Exact duplicate config with no rationale is wasted."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        # Write a prior run with same config
        _write_run_result("A", 42, 1, sample_config, sample_results)
        # Same config again, no rationale
        assert _is_wasted(sample_config, sample_results, seed_dir, best_val_bpb=1.90) is True

    def test_exact_duplicate_with_rationale_not_wasted(self, sample_results, tmp_results):
        """Exact duplicate config WITH rationale is not flagged by dup check."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        cfg = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
        )
        _write_run_result("A", 42, 1, cfg, sample_results)
        # Same config but with rationale
        cfg_with_rationale = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
            rationale_tag="verify_reproducibility",
        )
        assert _is_wasted(cfg_with_rationale, sample_results, seed_dir, best_val_bpb=1.90) is False

    def test_rationale_with_meaningful_step_not_wasted(self, tmp_results):
        """A run near a bad config is NOT wasted if it has rationale + meaningful step."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        # Write a prior diverged run
        bad_config = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
        )
        _write_run_result(
            "A", 42, 1, bad_config,
            RunResults(val_bpb=999999.0, steps=100, peak_vram_mb=0, final_loss=999999.0,
                       mfu=0, diverged=True, loss_trend="diverging", grad_norm_max=10.0),
        )
        # New config: same but with matrix_lr changed by 2x (meaningful step) + rationale
        probe_config = RunConfig(
            matrix_lr=0.02, embedding_lr=0.6, unembedding_lr=0.004,  # 0.04 → 0.02 = 1 octave
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
            rationale_tag="ablate_lr_near_failure",
        )
        good_results = RunResults(
            val_bpb=1.85, steps=1000, peak_vram_mb=4200.0, final_loss=2.3,
            mfu=0.38, diverged=False, loss_trend="improving", grad_norm_max=3.0,
        )
        assert _is_wasted(probe_config, good_results, seed_dir, best_val_bpb=1.90) is False


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class TestCreateHooks:
    def test_default_is_condition_a(self, monkeypatch):
        monkeypatch.delenv("AUTORESEARCH_CONDITION", raising=False)
        hooks = create_hooks()
        assert isinstance(hooks, NullHooks)

    def test_condition_b_from_env(self, monkeypatch):
        monkeypatch.setenv("AUTORESEARCH_CONDITION", "B")
        hooks = create_hooks()
        assert isinstance(hooks, LoggingHooks)

    def test_condition_explicit(self):
        hooks = create_hooks(condition="B", seed=42)
        assert isinstance(hooks, LoggingHooks)

    def test_invalid_condition_raises(self):
        with pytest.raises(ValueError, match="Unknown condition"):
            create_hooks(condition="Z")

    def test_case_insensitive(self):
        hooks = create_hooks(condition="b")
        assert isinstance(hooks, LoggingHooks)


# ---------------------------------------------------------------------------
# Log capture
# ---------------------------------------------------------------------------

class TestLogCapture:
    def test_null_hooks_captures_log(self, sample_config, sample_results, tmp_results):
        hooks = NullHooks(seed=42)
        hooks.pre_run(sample_config)
        hooks.start_log_capture(sample_config)
        print("test training output")
        hooks.stop_log_capture()
        hooks.post_run(sample_config, sample_results)

        log_dir = tmp_results / "logs" / "condition_A" / "seed_42"
        logs = list(log_dir.glob("run_*.log"))
        assert len(logs) == 1
        content = logs[0].read_text()
        assert "test training output" in content

    def test_logging_hooks_captures_log(self, sample_config, sample_results, tmp_results):
        hooks = LoggingHooks(seed=42)
        hooks.pre_run(sample_config)
        hooks.start_log_capture(sample_config)
        print("training step 1")
        hooks.stop_log_capture()
        hooks.post_run(sample_config, sample_results)

        log_dir = tmp_results / "logs" / "condition_B" / "seed_42"
        logs = list(log_dir.glob("run_*.log"))
        assert len(logs) == 1
        assert "training step 1" in logs[0].read_text()

    def test_log_capture_restores_stdout(self, sample_config, tmp_results):
        import sys
        original_stdout = sys.stdout
        hooks = NullHooks(seed=42)
        hooks.start_log_capture(sample_config)
        assert sys.stdout is not original_stdout
        hooks.stop_log_capture()
        assert sys.stdout is original_stdout

    def test_run_id_consistency(self, sample_config, sample_results, tmp_results):
        """Run ID from start_log_capture should match post_run's written JSON."""
        hooks = NullHooks(seed=42)
        hooks.pre_run(sample_config)
        hooks.start_log_capture(sample_config)
        hooks.stop_log_capture()
        hooks.post_run(sample_config, sample_results)

        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        log_dir = tmp_results / "logs" / "condition_A" / "seed_42"
        run_files = sorted(run_dir.glob("run_*.json"))
        log_files = sorted(log_dir.glob("run_*.log"))
        assert len(run_files) == 1
        assert len(log_files) == 1
        # run_001.json should pair with run_001.log
        assert run_files[0].stem == log_files[0].stem
