"""Tests for mnemebrain_hooks — Conditions A & B (no external deps)."""

import json
import math
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from mnemebrain_hooks import (
    NullHooks,
    LoggingHooks,
    PassiveHooks,
    ActiveHooks,
    LogCapture,
    RunConfig,
    RunResults,
    PreRunContext,
    PredictionResult,
    RecommendationResult,
    create_hooks,
    evidence_weight,
    _config_distance,
    _meaningful_step_from,
    _is_exact_duplicate,
    _is_near_bad_config,
    _is_wasted,
    _next_run_id,
    _best_val_bpb,
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
        """Strong improvement -> weight well above 0.5."""
        w = evidence_weight(1.80, 1.85)  # delta = -0.05
        assert w > 0.8

    def test_regression_low_weight(self):
        """Strong regression -> weight well below 0.5."""
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
        """No change -> weight = 0.5."""
        w = evidence_weight(1.85, 1.85)
        assert w == 0.5

    def test_monotonic(self):
        """Larger improvements should have higher weight."""
        w_small = evidence_weight(1.84, 1.85)
        w_large = evidence_weight(1.80, 1.85)
        assert w_large > w_small

    def test_very_large_improvement(self):
        """Very large delta (improvement) still yields weight in [0.5, 1.0]."""
        w = evidence_weight(0.50, 1.85)  # delta = -1.35
        assert 0.5 < w <= 1.0

    def test_very_large_regression(self):
        """Very large delta (regression) still yields weight in [0.0, 0.5)."""
        w = evidence_weight(5.0, 1.85)  # delta = +3.15
        assert 0.0 <= w < 0.5

    def test_asymmetry_improvement_vs_regression(self):
        """Symmetric deltas should produce weights symmetric around 0.5."""
        w_improve = evidence_weight(1.80, 1.85)  # delta = -0.05
        w_regress = evidence_weight(1.90, 1.85)  # delta = +0.05
        # w_improve should be as far above 0.5 as w_regress is below
        assert abs((w_improve - 0.5) - (0.5 - w_regress)) < 1e-10

    def test_weight_bounded_zero_to_one(self):
        """Weight should always be in [0, 1] for various deltas."""
        for delta in [-10.0, -1.0, -0.001, 0.001, 1.0, 10.0]:
            w = evidence_weight(1.85 + delta, 1.85)
            assert 0.0 <= w <= 1.0

    def test_near_zero_delta_still_not_half(self):
        """A delta just above the 1e-8 threshold should not return exactly 0.5."""
        w = evidence_weight(1.85 + 1e-7, 1.85)
        assert w != 0.5


# ---------------------------------------------------------------------------
# Config distance
# ---------------------------------------------------------------------------

class TestConfigDistance:
    def test_identical_configs_zero_distance(self, sample_config):
        cfg = asdict(sample_config)
        assert _config_distance(cfg, cfg) == 0.0

    def test_different_configs_positive_distance(self, sample_config):
        cfg1 = asdict(sample_config)
        cfg2 = dict(cfg1)
        cfg2["matrix_lr"] = 0.1  # significantly different
        assert _config_distance(cfg1, cfg2) > 0.0

    def test_boundary_min_values(self):
        """Distance is zero when both configs are at parameter minimums."""
        cfg_min = {
            "matrix_lr": 0.005, "embedding_lr": 0.05, "scalar_lr": 0.05,
            "unembedding_lr": 0.0005, "wd": 0.0, "depth": 2,
            "total_batch_size": 16384, "warmup_ratio": 0.0, "warmdown_ratio": 0.0,
        }
        assert _config_distance(cfg_min, cfg_min) == 0.0

    def test_boundary_max_values(self):
        """Distance is zero when both configs are at parameter maximums."""
        cfg_max = {
            "matrix_lr": 0.2, "embedding_lr": 2.0, "scalar_lr": 2.0,
            "unembedding_lr": 0.02, "wd": 0.4, "depth": 16,
            "total_batch_size": 524288, "warmup_ratio": 0.2, "warmdown_ratio": 0.8,
        }
        assert _config_distance(cfg_max, cfg_max) == 0.0

    def test_min_to_max_distance(self):
        """Max distance: all params go from min to max. Each normalized dim -> 1.0."""
        cfg_min = {
            "matrix_lr": 0.005, "embedding_lr": 0.05, "scalar_lr": 0.05,
            "unembedding_lr": 0.0005, "wd": 0.0, "depth": 2,
            "total_batch_size": 16384, "warmup_ratio": 0.0, "warmdown_ratio": 0.0,
        }
        cfg_max = {
            "matrix_lr": 0.2, "embedding_lr": 2.0, "scalar_lr": 2.0,
            "unembedding_lr": 0.02, "wd": 0.4, "depth": 16,
            "total_batch_size": 524288, "warmup_ratio": 0.2, "warmdown_ratio": 0.8,
        }
        d = _config_distance(cfg_min, cfg_max)
        # 9 dimensions, each contributing 1.0 -> sqrt(9) = 3.0
        assert abs(d - 3.0) < 1e-6

    def test_log_scale_correctness(self):
        """Verify log-scale normalization: midpoint in log-space maps to 0.5."""
        lo, hi = 0.005, 0.2
        mid = math.sqrt(lo * hi)  # geometric mean = midpoint in log space
        cfg_lo = {"matrix_lr": lo, "embedding_lr": 0.05, "scalar_lr": 0.05,
                  "unembedding_lr": 0.0005, "wd": 0.0, "depth": 2,
                  "total_batch_size": 16384, "warmup_ratio": 0.0, "warmdown_ratio": 0.0}
        cfg_mid = dict(cfg_lo, matrix_lr=mid)
        cfg_hi = dict(cfg_lo, matrix_lr=hi)
        d_lo_mid = _config_distance(cfg_lo, cfg_mid)
        d_mid_hi = _config_distance(cfg_mid, cfg_hi)
        # Should be equal (midpoint in log space)
        assert abs(d_lo_mid - d_mid_hi) < 1e-6

    def test_linear_scale_correctness(self):
        """Verify linear-scale normalization: midpoint is arithmetic mean."""
        cfg_base = {
            "matrix_lr": 0.005, "embedding_lr": 0.05, "scalar_lr": 0.05,
            "unembedding_lr": 0.0005, "wd": 0.0, "depth": 2,
            "total_batch_size": 16384, "warmup_ratio": 0.0, "warmdown_ratio": 0.0,
        }
        cfg_mid_wd = dict(cfg_base, wd=0.2)  # midpoint of [0.0, 0.4]
        cfg_max_wd = dict(cfg_base, wd=0.4)
        d_lo_mid = _config_distance(cfg_base, cfg_mid_wd)
        d_mid_hi = _config_distance(cfg_mid_wd, cfg_max_wd)
        assert abs(d_lo_mid - d_mid_hi) < 1e-6

    def test_below_min_clamped_for_log(self):
        """Values below the log-scale minimum are clamped to lo."""
        cfg_base = {
            "matrix_lr": 0.005, "embedding_lr": 0.05, "scalar_lr": 0.05,
            "unembedding_lr": 0.0005, "wd": 0.0, "depth": 2,
            "total_batch_size": 16384, "warmup_ratio": 0.0, "warmdown_ratio": 0.0,
        }
        cfg_below = dict(cfg_base, matrix_lr=0.001)  # below min of 0.005
        # Should clamp to 0.005 -> distance 0
        assert _config_distance(cfg_base, cfg_below) == 0.0


# ---------------------------------------------------------------------------
# _meaningful_step_from
# ---------------------------------------------------------------------------

class TestMeaningfulStepFrom:
    @pytest.fixture
    def base_cfg(self):
        return {
            "matrix_lr": 0.04, "embedding_lr": 0.6, "unembedding_lr": 0.004,
            "scalar_lr": 0.5, "wd": 0.2, "depth": 4,
            "total_batch_size": 65536, "device_batch_size": 16,
            "warmup_ratio": 0.0, "warmdown_ratio": 0.5,
        }

    def test_no_change_not_meaningful(self, base_cfg):
        assert _meaningful_step_from(base_cfg, base_cfg) is False

    def test_matrix_lr_2x_is_meaningful(self, base_cfg):
        """matrix_lr changed by 2x (1 octave) should be meaningful."""
        changed = dict(base_cfg, matrix_lr=0.08)  # 2x
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_matrix_lr_half_is_meaningful(self, base_cfg):
        """matrix_lr halved (1 octave down) should be meaningful."""
        changed = dict(base_cfg, matrix_lr=0.02)  # 0.5x
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_matrix_lr_just_under_2x_not_meaningful(self, base_cfg):
        """matrix_lr changed by less than 2x should NOT be meaningful."""
        changed = dict(base_cfg, matrix_lr=0.06)  # 1.5x < 2x
        assert _meaningful_step_from(changed, base_cfg) is False

    def test_depth_plus_2_is_meaningful(self, base_cfg):
        """depth changed by +2 should be meaningful."""
        changed = dict(base_cfg, depth=6)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_depth_minus_2_is_meaningful(self, base_cfg):
        """depth changed by -2 should be meaningful."""
        changed = dict(base_cfg, depth=2)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_depth_plus_1_not_meaningful(self, base_cfg):
        """depth changed by only 1 should NOT be meaningful."""
        changed = dict(base_cfg, depth=5)
        assert _meaningful_step_from(changed, base_cfg) is False

    def test_warmup_plus_005_is_meaningful(self, base_cfg):
        """warmup_ratio changed by +0.05 should be meaningful."""
        changed = dict(base_cfg, warmup_ratio=0.05)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_warmup_minus_005_is_meaningful(self, base_cfg):
        """warmup_ratio changed by -0.05 from 0.1 should be meaningful."""
        prior = dict(base_cfg, warmup_ratio=0.1)
        changed = dict(base_cfg, warmup_ratio=0.05)
        assert _meaningful_step_from(changed, prior) is True

    def test_warmup_just_under_005_not_meaningful(self, base_cfg):
        """warmup_ratio changed by < 0.05 should NOT be meaningful."""
        changed = dict(base_cfg, warmup_ratio=0.04)
        assert _meaningful_step_from(changed, base_cfg) is False

    def test_batch_2x_is_meaningful(self, base_cfg):
        """total_batch_size doubled should be meaningful."""
        changed = dict(base_cfg, total_batch_size=131072)  # 2x
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_batch_half_is_meaningful(self, base_cfg):
        """total_batch_size halved should be meaningful."""
        changed = dict(base_cfg, total_batch_size=32768)  # 0.5x
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_batch_just_under_2x_not_meaningful(self, base_cfg):
        """total_batch_size changed by less than 2x should NOT be meaningful."""
        changed = dict(base_cfg, total_batch_size=90000)  # ~1.37x
        assert _meaningful_step_from(changed, base_cfg) is False

    def test_wd_plus_005_is_meaningful(self, base_cfg):
        """wd changed by >= 0.05 should be meaningful."""
        changed = dict(base_cfg, wd=0.26)  # 0.06 delta, clearly above threshold
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_wd_minus_005_is_meaningful(self, base_cfg):
        """wd changed by >= 0.05 should be meaningful."""
        changed = dict(base_cfg, wd=0.14)  # 0.06 delta, clearly above threshold
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_wd_just_under_005_not_meaningful(self, base_cfg):
        """wd changed by < 0.05 should NOT be meaningful."""
        changed = dict(base_cfg, wd=0.22)
        assert _meaningful_step_from(changed, base_cfg) is False


# ---------------------------------------------------------------------------
# _is_exact_duplicate
# ---------------------------------------------------------------------------

class TestIsExactDuplicate:
    def test_no_prior_runs_returns_false(self, tmp_results):
        """Empty seed dir -> no duplicates."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        seed_dir.mkdir(parents=True, exist_ok=True)
        cfg = {
            "matrix_lr": 0.04, "embedding_lr": 0.6, "unembedding_lr": 0.004,
            "scalar_lr": 0.5, "wd": 0.2, "depth": 4,
            "total_batch_size": 65536, "device_batch_size": 16,
            "warmup_ratio": 0.0, "warmdown_ratio": 0.5,
        }
        assert _is_exact_duplicate(cfg, seed_dir) is False

    def test_partial_match_returns_false(self, tmp_results, sample_config, sample_results):
        """A prior run with a different matrix_lr is not a duplicate."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        _write_run_result("A", 42, 1, sample_config, sample_results)
        cfg = asdict(sample_config)
        cfg["matrix_lr"] = 0.08  # different
        assert _is_exact_duplicate(cfg, seed_dir) is False

    def test_exact_match_returns_true(self, tmp_results, sample_config, sample_results):
        """A prior run with identical HP keys is a duplicate."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        _write_run_result("A", 42, 1, sample_config, sample_results)
        cfg = asdict(sample_config)
        assert _is_exact_duplicate(cfg, seed_dir) is True

    def test_ignores_corrupt_json(self, tmp_results):
        """Corrupt JSON files in seed_dir should not cause errors."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "run_001.json").write_text("not valid json{{{")
        cfg = {"matrix_lr": 0.04, "embedding_lr": 0.6, "unembedding_lr": 0.004,
               "scalar_lr": 0.5, "wd": 0.2, "depth": 4,
               "total_batch_size": 65536, "device_batch_size": 16,
               "warmup_ratio": 0.0, "warmdown_ratio": 0.5}
        assert _is_exact_duplicate(cfg, seed_dir) is False


# ---------------------------------------------------------------------------
# _is_near_bad_config
# ---------------------------------------------------------------------------

class TestIsNearBadConfig:
    def test_no_bad_configs_returns_false(self, tmp_results, sample_config, sample_results):
        """No diverged prior runs -> not near bad config."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        _write_run_result("A", 42, 1, sample_config, sample_results)
        result, cfg = _is_near_bad_config(sample_config, seed_dir)
        assert result is False
        assert cfg is None

    def test_identical_bad_config_is_near(self, tmp_results, sample_config):
        """Exact same config as a diverged run -> near bad config."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        diverged = RunResults(
            val_bpb=999999.0, steps=100, peak_vram_mb=0, final_loss=999999.0,
            mfu=0, diverged=True, loss_trend="diverging", grad_norm_max=10.0,
        )
        _write_run_result("A", 42, 1, sample_config, diverged)
        result, cfg = _is_near_bad_config(sample_config, seed_dir)
        assert result is True
        assert cfg is not None

    def test_tau_boundary_just_inside(self, tmp_results):
        """Config just inside tau boundary should be detected as near."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        base = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
        )
        diverged = RunResults(
            val_bpb=999999.0, steps=100, peak_vram_mb=0, final_loss=999999.0,
            mfu=0, diverged=True, loss_trend="diverging", grad_norm_max=10.0,
        )
        _write_run_result("A", 42, 1, base, diverged)
        # Verify the base config is detected (distance=0 < 0.1)
        result, _ = _is_near_bad_config(base, seed_dir, tau=0.1)
        assert result is True

    def test_tau_boundary_just_outside(self, tmp_results):
        """Config well outside tau boundary should NOT be detected as near."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        base = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
        )
        diverged = RunResults(
            val_bpb=999999.0, steps=100, peak_vram_mb=0, final_loss=999999.0,
            mfu=0, diverged=True, loss_trend="diverging", grad_norm_max=10.0,
        )
        _write_run_result("A", 42, 1, base, diverged)
        # Use a very different config so distance > tau
        far_config = RunConfig(
            matrix_lr=0.2, embedding_lr=2.0, unembedding_lr=0.02,
            scalar_lr=2.0, wd=0.4, depth=16, total_batch_size=524288,
            device_batch_size=16, warmup_ratio=0.2, warmdown_ratio=0.8, seed=42,
        )
        result, _ = _is_near_bad_config(far_config, seed_dir, tau=0.1)
        assert result is False

    def test_empty_dir_returns_false(self, tmp_results, sample_config):
        """Non-existent seed_dir -> not near bad config (glob returns empty)."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        seed_dir.mkdir(parents=True, exist_ok=True)
        result, cfg = _is_near_bad_config(sample_config, seed_dir)
        assert result is False
        assert cfg is None


# ---------------------------------------------------------------------------
# _next_run_id
# ---------------------------------------------------------------------------

class TestNextRunId:
    def test_empty_dir(self, tmp_results):
        """Empty directory -> run_id = 1."""
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        assert _next_run_id(run_dir) == 1

    def test_nonexistent_dir(self, tmp_results):
        """Non-existent directory -> run_id = 1."""
        run_dir = tmp_results / "runs" / "condition_A" / "seed_99"
        assert _next_run_id(run_dir) == 1

    def test_sequential_files(self, tmp_results, sample_config, sample_results):
        """After writing 2 runs, next should be 3."""
        _write_run_result("A", 42, 1, sample_config, sample_results)
        _write_run_result("A", 42, 2, sample_config, sample_results)
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        assert _next_run_id(run_dir) == 3

    def test_gap_in_sequence(self, tmp_results):
        """Files run_001.json and run_005.json -> next should be 6 (based on highest)."""
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_001.json").write_text("{}")
        (run_dir / "run_005.json").write_text("{}")
        assert _next_run_id(run_dir) == 6

    def test_non_sequential_files_uses_highest(self, tmp_results):
        """Only run_003.json exists -> next should be 4."""
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_003.json").write_text("{}")
        assert _next_run_id(run_dir) == 4


# ---------------------------------------------------------------------------
# _best_val_bpb
# ---------------------------------------------------------------------------

class TestBestValBpb:
    def test_all_diverged_returns_none(self, tmp_results):
        """All runs diverged -> best_val_bpb = None."""
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        doc = {
            "results": {"val_bpb": 999999.0, "diverged": True},
        }
        (run_dir / "run_001.json").write_text(json.dumps(doc))
        (run_dir / "run_002.json").write_text(json.dumps(doc))
        assert _best_val_bpb(run_dir) is None

    def test_mixed_diverged_and_normal(self, tmp_results):
        """Mix of diverged and non-diverged -> best from non-diverged only."""
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        doc_div = {"results": {"val_bpb": 0.5, "diverged": True}}
        doc_ok = {"results": {"val_bpb": 1.85, "diverged": False}}
        (run_dir / "run_001.json").write_text(json.dumps(doc_div))
        (run_dir / "run_002.json").write_text(json.dumps(doc_ok))
        assert _best_val_bpb(run_dir) == 1.85

    def test_empty_dir_returns_none(self, tmp_results):
        """No run files -> None."""
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        assert _best_val_bpb(run_dir) is None

    def test_corrupt_json_skipped(self, tmp_results):
        """Corrupt JSON files are skipped gracefully."""
        run_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_001.json").write_text("not json!!!")
        doc_ok = {"results": {"val_bpb": 1.90, "diverged": False}}
        (run_dir / "run_002.json").write_text(json.dumps(doc_ok))
        assert _best_val_bpb(run_dir) == 1.90


# ---------------------------------------------------------------------------
# _write_run_result
# ---------------------------------------------------------------------------

class TestWriteRunResult:
    def test_all_json_schema_fields_present(self, tmp_results, sample_config, sample_results):
        """Verify all required top-level JSON fields are present."""
        path = _write_run_result("A", 42, 1, sample_config, sample_results)
        data = json.loads(path.read_text())
        required_fields = [
            "run_id", "condition", "seed", "timestamp", "config", "results",
            "rationale_tag", "rationale", "wasted", "delta_from_best",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_rationale_stripped_from_config(self, tmp_results, sample_results):
        """rationale_tag and rationale should NOT appear in config sub-dict."""
        cfg = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
            rationale_tag="test_tag", rationale="test_rationale",
        )
        path = _write_run_result("A", 42, 1, cfg, sample_results)
        data = json.loads(path.read_text())
        assert "rationale_tag" not in data["config"]
        assert "rationale" not in data["config"]
        # But they should be at the top level
        assert data["rationale_tag"] == "test_tag"
        assert data["rationale"] == "test_rationale"

    def test_pre_run_context_included(self, tmp_results, sample_config, sample_results):
        """pre_run_context should be included when provided."""
        ctx = PreRunContext(condition="A", summaries=["test summary"])
        path = _write_run_result("A", 42, 1, sample_config, sample_results, pre_run_context=ctx)
        data = json.loads(path.read_text())
        assert "pre_run_context" in data
        assert data["pre_run_context"]["condition"] == "A"

    def test_extra_fields_merged(self, tmp_results, sample_config, sample_results):
        """Extra dict should be merged into the top-level document."""
        path = _write_run_result("A", 42, 1, sample_config, sample_results,
                                 extra={"custom_key": "custom_value"})
        data = json.loads(path.read_text())
        assert data["custom_key"] == "custom_value"

    def test_delta_from_best_first_run(self, tmp_results, sample_config, sample_results):
        """First run -> no prior best -> delta_from_best = 0.0."""
        path = _write_run_result("A", 42, 1, sample_config, sample_results)
        data = json.loads(path.read_text())
        assert data["delta_from_best"] == 0.0


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

    def test_seed_fallback_uses_config_seed(self, sample_config, sample_results, tmp_results):
        """When self._seed is None, should use config.seed."""
        hooks = NullHooks(seed=None)
        hooks.post_run(sample_config, sample_results)
        run_dir = tmp_results / "runs" / "condition_A" / f"seed_{sample_config.seed:02d}"
        files = list(run_dir.glob("run_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["seed"] == sample_config.seed


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

    def test_diverged_run_summary_format(self, sample_config, tmp_results):
        """Diverged run summary should mention 'diverged'."""
        hooks = LoggingHooks(seed=42)
        diverged = RunResults(
            val_bpb=float("nan"), steps=342, peak_vram_mb=4200.0,
            final_loss=100.0, mfu=0.10, diverged=True,
            loss_trend="diverging", grad_norm_max=9.1,
        )
        hooks.post_run(sample_config, diverged)
        log_path = tmp_results / "experiment_log.jsonl"
        entry = json.loads(log_path.read_text().strip())
        assert "diverged" in entry["summary"].lower()

    def test_history_lines_limit(self, sample_config, sample_results, tmp_results):
        """history_lines=2 should only return 2 most recent summaries."""
        hooks = LoggingHooks(seed=42, history_lines=2)
        for _ in range(5):
            hooks.post_run(sample_config, sample_results)
        ctx = hooks.pre_run(sample_config)
        assert len(ctx.summaries) == 2

    def test_seed_fallback_uses_config_seed(self, sample_config, sample_results, tmp_results):
        """When self._seed is None, should use config.seed."""
        hooks = LoggingHooks(seed=None)
        hooks.post_run(sample_config, sample_results)
        run_dir = tmp_results / "runs" / "condition_B" / f"seed_{sample_config.seed:02d}"
        files = list(run_dir.glob("run_*.json"))
        assert len(files) == 1


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

    def test_all_four_conditions_return_correct_types(self):
        """All 4 conditions should return the correct hook class with seed param."""
        expected = {
            "A": NullHooks,
            "B": LoggingHooks,
            "C": PassiveHooks,
            "D": ActiveHooks,
        }
        for cond, cls in expected.items():
            hooks = create_hooks(condition=cond, seed=42)
            assert isinstance(hooks, cls), f"Condition {cond} should return {cls.__name__}"

    def test_seed_none_creates_hooks(self):
        """seed=None should still work (seed determined at runtime from config)."""
        hooks = create_hooks(condition="A", seed=None)
        assert isinstance(hooks, NullHooks)

    def test_no_seed_param_creates_hooks(self):
        """Omitting seed entirely should work."""
        hooks = create_hooks(condition="A")
        assert isinstance(hooks, NullHooks)


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

    def test_double_stop_is_safe(self, sample_config, tmp_results):
        """Calling stop_log_capture twice should not raise."""
        capture = LogCapture()
        capture.start("A", 42, 1)
        capture.stop()
        capture.stop()  # second stop should be a no-op

    def test_stop_without_start_is_safe(self):
        """Calling stop without start should not raise."""
        capture = LogCapture()
        capture.stop()  # no-op, should not raise


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
        # best is 1.85, regression to 1.90 -> delta = 0.05 > epsilon (0.02)
        assert _is_wasted(sample_config, regressed, seed_dir, best_val_bpb=1.85) is True

    def test_improvement_not_wasted(self, sample_config, sample_results, tmp_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        seed_dir.mkdir(parents=True, exist_ok=True)
        # val_bpb=1.872, best=1.90 -> improvement
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
        # Now check a new run with identical config -> near bad config
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
            matrix_lr=0.02, embedding_lr=0.6, unembedding_lr=0.004,  # 0.04 -> 0.02 = 1 octave
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
# PassiveHooks / ActiveHooks — ImportError tests
# ---------------------------------------------------------------------------

class TestPassiveHooksImportError:
    def test_passive_hooks_import_error(self, sample_config, tmp_results):
        """PassiveHooks should raise ImportError when mnemebrain_core is missing."""
        hooks = PassiveHooks(seed=42)
        with patch.dict("sys.modules", {"mnemebrain_core": None, "mnemebrain_core.memory": None}):
            with pytest.raises(ImportError, match="mnemebrain-lite"):
                hooks.pre_run(sample_config)


class TestActiveHooksImportError:
    def test_active_hooks_import_error(self, sample_config, tmp_results):
        """ActiveHooks should raise ImportError when mnemebrain is missing."""
        hooks = ActiveHooks(seed=42)
        with patch.dict("sys.modules", {"mnemebrain": None}):
            with pytest.raises(ImportError, match="mnemebrain"):
                hooks.pre_run(sample_config)
