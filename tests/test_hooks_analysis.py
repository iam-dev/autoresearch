"""Tests for hooks/analysis.py — evidence weight, config distance, wasted-run logic."""

import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from hooks.types import RunConfig, RunResults
from hooks import artifacts
from hooks.artifacts import _write_run_result
from hooks.analysis import (
    evidence_weight,
    _config_distance,
    _meaningful_step_from,
    _is_exact_duplicate,
    _is_near_bad_config,
    _is_wasted,
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
def diverged_results():
    return RunResults(
        val_bpb=float("nan"), steps=342, peak_vram_mb=4200.0, final_loss=100.0,
        mfu=0.10, diverged=True, loss_trend="diverging", grad_norm_max=9.1,
    )


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr("hooks.artifacts.RESULTS_DIR", tmp_path)
    return tmp_path


class TestEvidenceWeight:
    def test_improvement_high_weight(self):
        w = evidence_weight(1.80, 1.85)
        assert w > 0.8

    def test_regression_low_weight(self):
        w = evidence_weight(1.90, 1.85)
        assert w < 0.2

    def test_small_improvement_above_half(self):
        w = evidence_weight(1.849, 1.85)
        assert w > 0.5

    def test_small_regression_below_half(self):
        w = evidence_weight(1.851, 1.85)
        assert w < 0.5

    def test_zero_delta_returns_half(self):
        w = evidence_weight(1.85, 1.85)
        assert w == 0.5

    def test_monotonic(self):
        w_small = evidence_weight(1.84, 1.85)
        w_large = evidence_weight(1.80, 1.85)
        assert w_large > w_small

    def test_very_large_improvement(self):
        w = evidence_weight(0.50, 1.85)
        assert 0.5 < w <= 1.0

    def test_very_large_regression(self):
        w = evidence_weight(5.0, 1.85)
        assert 0.0 <= w < 0.5

    def test_asymmetry_improvement_vs_regression(self):
        w_improve = evidence_weight(1.80, 1.85)
        w_regress = evidence_weight(1.90, 1.85)
        assert abs((w_improve - 0.5) - (0.5 - w_regress)) < 1e-10

    def test_weight_bounded_zero_to_one(self):
        for delta in [-10.0, -1.0, -0.001, 0.001, 1.0, 10.0]:
            w = evidence_weight(1.85 + delta, 1.85)
            assert 0.0 <= w <= 1.0

    def test_near_zero_delta_still_not_half(self):
        w = evidence_weight(1.85 + 1e-7, 1.85)
        assert w != 0.5


class TestConfigDistance:
    def test_identical_configs_zero_distance(self, sample_config):
        cfg = asdict(sample_config)
        assert _config_distance(cfg, cfg) == 0.0

    def test_different_configs_positive_distance(self, sample_config):
        cfg1 = asdict(sample_config)
        cfg2 = dict(cfg1)
        cfg2["matrix_lr"] = 0.1
        assert _config_distance(cfg1, cfg2) > 0.0

    def test_boundary_min_values(self):
        cfg_min = {
            "matrix_lr": 0.005, "embedding_lr": 0.05, "scalar_lr": 0.05,
            "unembedding_lr": 0.0005, "wd": 0.0, "depth": 2,
            "total_batch_size": 16384, "warmup_ratio": 0.0, "warmdown_ratio": 0.0,
        }
        assert _config_distance(cfg_min, cfg_min) == 0.0

    def test_boundary_max_values(self):
        cfg_max = {
            "matrix_lr": 0.2, "embedding_lr": 2.0, "scalar_lr": 2.0,
            "unembedding_lr": 0.02, "wd": 0.4, "depth": 16,
            "total_batch_size": 524288, "warmup_ratio": 0.2, "warmdown_ratio": 0.8,
        }
        assert _config_distance(cfg_max, cfg_max) == 0.0

    def test_min_to_max_distance(self):
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
        assert abs(d - 3.0) < 1e-6

    def test_log_scale_correctness(self):
        lo, hi = 0.005, 0.2
        mid = math.sqrt(lo * hi)
        cfg_lo = {"matrix_lr": lo, "embedding_lr": 0.05, "scalar_lr": 0.05,
                  "unembedding_lr": 0.0005, "wd": 0.0, "depth": 2,
                  "total_batch_size": 16384, "warmup_ratio": 0.0, "warmdown_ratio": 0.0}
        cfg_mid = dict(cfg_lo, matrix_lr=mid)
        cfg_hi = dict(cfg_lo, matrix_lr=hi)
        d_lo_mid = _config_distance(cfg_lo, cfg_mid)
        d_mid_hi = _config_distance(cfg_mid, cfg_hi)
        assert abs(d_lo_mid - d_mid_hi) < 1e-6

    def test_linear_scale_correctness(self):
        cfg_base = {
            "matrix_lr": 0.005, "embedding_lr": 0.05, "scalar_lr": 0.05,
            "unembedding_lr": 0.0005, "wd": 0.0, "depth": 2,
            "total_batch_size": 16384, "warmup_ratio": 0.0, "warmdown_ratio": 0.0,
        }
        cfg_mid_wd = dict(cfg_base, wd=0.2)
        cfg_max_wd = dict(cfg_base, wd=0.4)
        d_lo_mid = _config_distance(cfg_base, cfg_mid_wd)
        d_mid_hi = _config_distance(cfg_mid_wd, cfg_max_wd)
        assert abs(d_lo_mid - d_mid_hi) < 1e-6

    def test_below_min_clamped_for_log(self):
        cfg_base = {
            "matrix_lr": 0.005, "embedding_lr": 0.05, "scalar_lr": 0.05,
            "unembedding_lr": 0.0005, "wd": 0.0, "depth": 2,
            "total_batch_size": 16384, "warmup_ratio": 0.0, "warmdown_ratio": 0.0,
        }
        cfg_below = dict(cfg_base, matrix_lr=0.001)
        assert _config_distance(cfg_base, cfg_below) == 0.0


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
        changed = dict(base_cfg, matrix_lr=0.08)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_matrix_lr_half_is_meaningful(self, base_cfg):
        changed = dict(base_cfg, matrix_lr=0.02)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_matrix_lr_just_under_2x_not_meaningful(self, base_cfg):
        changed = dict(base_cfg, matrix_lr=0.06)
        assert _meaningful_step_from(changed, base_cfg) is False

    def test_depth_plus_2_is_meaningful(self, base_cfg):
        changed = dict(base_cfg, depth=6)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_depth_minus_2_is_meaningful(self, base_cfg):
        changed = dict(base_cfg, depth=2)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_depth_plus_1_not_meaningful(self, base_cfg):
        changed = dict(base_cfg, depth=5)
        assert _meaningful_step_from(changed, base_cfg) is False

    def test_warmup_plus_005_is_meaningful(self, base_cfg):
        changed = dict(base_cfg, warmup_ratio=0.05)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_warmup_minus_005_is_meaningful(self, base_cfg):
        prior = dict(base_cfg, warmup_ratio=0.1)
        changed = dict(base_cfg, warmup_ratio=0.05)
        assert _meaningful_step_from(changed, prior) is True

    def test_warmup_just_under_005_not_meaningful(self, base_cfg):
        changed = dict(base_cfg, warmup_ratio=0.04)
        assert _meaningful_step_from(changed, base_cfg) is False

    def test_batch_2x_is_meaningful(self, base_cfg):
        changed = dict(base_cfg, total_batch_size=131072)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_batch_half_is_meaningful(self, base_cfg):
        changed = dict(base_cfg, total_batch_size=32768)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_batch_just_under_2x_not_meaningful(self, base_cfg):
        changed = dict(base_cfg, total_batch_size=90000)
        assert _meaningful_step_from(changed, base_cfg) is False

    def test_wd_plus_005_is_meaningful(self, base_cfg):
        changed = dict(base_cfg, wd=0.26)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_wd_minus_005_is_meaningful(self, base_cfg):
        changed = dict(base_cfg, wd=0.14)
        assert _meaningful_step_from(changed, base_cfg) is True

    def test_wd_just_under_005_not_meaningful(self, base_cfg):
        changed = dict(base_cfg, wd=0.22)
        assert _meaningful_step_from(changed, base_cfg) is False


class TestIsExactDuplicate:
    def test_no_prior_runs_returns_false(self, tmp_results):
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
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        _write_run_result("A", 42, 1, sample_config, sample_results)
        cfg = asdict(sample_config)
        cfg["matrix_lr"] = 0.08
        assert _is_exact_duplicate(cfg, seed_dir) is False

    def test_exact_match_returns_true(self, tmp_results, sample_config, sample_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        _write_run_result("A", 42, 1, sample_config, sample_results)
        cfg = asdict(sample_config)
        assert _is_exact_duplicate(cfg, seed_dir) is True

    def test_ignores_corrupt_json(self, tmp_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "run_001.json").write_text("not valid json{{{")
        cfg = {"matrix_lr": 0.04, "embedding_lr": 0.6, "unembedding_lr": 0.004,
               "scalar_lr": 0.5, "wd": 0.2, "depth": 4,
               "total_batch_size": 65536, "device_batch_size": 16,
               "warmup_ratio": 0.0, "warmdown_ratio": 0.5}
        assert _is_exact_duplicate(cfg, seed_dir) is False


class TestIsNearBadConfig:
    def test_no_bad_configs_returns_false(self, tmp_results, sample_config, sample_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        _write_run_result("A", 42, 1, sample_config, sample_results)
        result, cfg = _is_near_bad_config(sample_config, seed_dir)
        assert result is False
        assert cfg is None

    def test_identical_bad_config_is_near(self, tmp_results, sample_config):
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
        result, _ = _is_near_bad_config(base, seed_dir, tau=0.1)
        assert result is True

    def test_tau_boundary_just_outside(self, tmp_results):
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
        far_config = RunConfig(
            matrix_lr=0.2, embedding_lr=2.0, unembedding_lr=0.02,
            scalar_lr=2.0, wd=0.4, depth=16, total_batch_size=524288,
            device_batch_size=16, warmup_ratio=0.2, warmdown_ratio=0.8, seed=42,
        )
        result, _ = _is_near_bad_config(far_config, seed_dir, tau=0.1)
        assert result is False

    def test_empty_dir_returns_false(self, tmp_results, sample_config):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        seed_dir.mkdir(parents=True, exist_ok=True)
        result, cfg = _is_near_bad_config(sample_config, seed_dir)
        assert result is False
        assert cfg is None


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
        assert _is_wasted(sample_config, regressed, seed_dir, best_val_bpb=1.85) is True

    def test_improvement_not_wasted(self, sample_config, sample_results, tmp_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        seed_dir.mkdir(parents=True, exist_ok=True)
        assert _is_wasted(sample_config, sample_results, seed_dir, best_val_bpb=1.90) is False

    def test_near_bad_config_is_wasted(self, sample_config, sample_results, tmp_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        _write_run_result(
            "A", 42, 1, sample_config,
            RunResults(val_bpb=999999.0, steps=100, peak_vram_mb=0, final_loss=999999.0,
                       mfu=0, diverged=True, loss_trend="diverging", grad_norm_max=10.0),
        )
        assert _is_wasted(sample_config, sample_results, seed_dir, best_val_bpb=1.90) is True

    def test_exact_duplicate_no_rationale_is_wasted(self, sample_config, sample_results, tmp_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        _write_run_result("A", 42, 1, sample_config, sample_results)
        assert _is_wasted(sample_config, sample_results, seed_dir, best_val_bpb=1.90) is True

    def test_exact_duplicate_with_rationale_not_wasted(self, sample_results, tmp_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
        cfg = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
        )
        _write_run_result("A", 42, 1, cfg, sample_results)
        cfg_with_rationale = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
            rationale_tag="verify_reproducibility",
        )
        assert _is_wasted(cfg_with_rationale, sample_results, seed_dir, best_val_bpb=1.90) is False

    def test_rationale_with_meaningful_step_not_wasted(self, tmp_results):
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_42"
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
        probe_config = RunConfig(
            matrix_lr=0.02, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
            rationale_tag="ablate_lr_near_failure",
        )
        good_results = RunResults(
            val_bpb=1.85, steps=1000, peak_vram_mb=4200.0, final_loss=2.3,
            mfu=0.38, diverged=False, loss_trend="improving", grad_norm_max=3.0,
        )
        assert _is_wasted(probe_config, good_results, seed_dir, best_val_bpb=1.90) is False
