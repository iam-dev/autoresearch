"""Tests for results_analyzer — summary and comparison generation."""

import json
from pathlib import Path

import pytest

from results_analyzer import (
    _check_stability,
    _check_telemetry_completeness,
    cohens_d,
    generate_comparison,
    generate_plots,
    generate_summary,
    load_runs,
    load_runs_by_seed,
)


@pytest.fixture
def tmp_results(tmp_path, monkeypatch):
    """Redirect RESULTS_DIR to temp directory and return it."""
    monkeypatch.setattr("results_analyzer.RESULTS_DIR", tmp_path)
    return tmp_path


def _write_run(tmp_results: Path, condition: str, seed: int, run_id: int, val_bpb: float,
               diverged: bool = False, wasted: bool = False) -> None:
    """Write a run result JSON with all required telemetry fields."""
    seed_dir = tmp_results / "runs" / f"condition_{condition}" / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    doc = {
        "run_id": run_id,
        "condition": condition,
        "seed": seed,
        "timestamp": "2026-03-16T14:00:00Z",
        "config": {
            "matrix_lr": 0.04,
            "embedding_lr": 0.6,
            "depth": 4,
            "total_batch_size": 65536,
        },
        "results": {
            "val_bpb": val_bpb,
            "steps": 1000,
            "diverged": diverged,
            "loss_trend": "diverging" if diverged else "improving",
            "grad_norm_max": 3.0,
        },
        "wasted": wasted,
        "delta_from_best": 0.0,
    }
    path = seed_dir / f"run_{run_id:03d}.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")


class TestLoadRuns:
    def test_empty_directory(self, tmp_results):
        assert load_runs("A") == []

    def test_loads_all_runs(self, tmp_results):
        _write_run(tmp_results, "A", 1, 1, 1.90)
        _write_run(tmp_results, "A", 1, 2, 1.85)
        _write_run(tmp_results, "A", 2, 1, 1.88)
        runs = load_runs("A")
        assert len(runs) == 3

    def test_loads_by_seed(self, tmp_results):
        _write_run(tmp_results, "B", 1, 1, 1.90)
        _write_run(tmp_results, "B", 1, 2, 1.85)
        _write_run(tmp_results, "B", 2, 1, 1.88)
        by_seed = load_runs_by_seed("B")
        assert set(by_seed.keys()) == {1, 2}
        assert len(by_seed[1]) == 2
        assert len(by_seed[2]) == 1


class TestGenerateSummary:
    def test_no_data_returns_none(self, tmp_results):
        assert generate_summary("A") is None

    def test_generates_summary(self, tmp_results):
        _write_run(tmp_results, "A", 1, 1, 1.90)
        _write_run(tmp_results, "A", 1, 2, 1.85)
        _write_run(tmp_results, "A", 2, 1, 1.88)
        summary = generate_summary("A")
        assert summary is not None
        assert summary["condition"] == "A"
        assert summary["total_runs"] == 3
        assert summary["best_val_bpb"] == 1.85
        assert len(summary["val_bpb_trajectory"]) == 3
        assert summary["seeds_completed"] == [1, 2]

    def test_writes_summary_file(self, tmp_results):
        _write_run(tmp_results, "B", 1, 1, 1.90)
        generate_summary("B")
        path = tmp_results / "summaries" / "condition_B_summary.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["condition"] == "B"

    def test_handles_diverged_runs(self, tmp_results):
        _write_run(tmp_results, "A", 1, 1, 1.90)
        _write_run(tmp_results, "A", 1, 2, 999999.0, diverged=True)
        summary = generate_summary("A")
        assert summary["diverged_runs"] == 1
        assert summary["best_val_bpb"] == 1.90  # diverged runs excluded from best

    def test_wasted_run_rate(self, tmp_results):
        _write_run(tmp_results, "A", 1, 1, 1.90)
        _write_run(tmp_results, "A", 1, 2, 1.85, wasted=True)
        summary = generate_summary("A")
        assert summary["wasted_runs"] == 1
        assert summary["wasted_run_rate"] == 0.5

    def test_runs_to_threshold(self, tmp_results):
        # Seed 1: best at run 3 (1.85), seed 2: best at run 2 (1.84)
        _write_run(tmp_results, "A", 1, 1, 1.95)
        _write_run(tmp_results, "A", 1, 2, 1.90)
        _write_run(tmp_results, "A", 1, 3, 1.85)  # seed 1 best at run 3
        _write_run(tmp_results, "A", 2, 1, 1.90)
        _write_run(tmp_results, "A", 2, 2, 1.84)  # seed 2 best at run 2
        summary = generate_summary("A")
        # Average: (3 + 2) / 2 = 2.5
        assert summary["runs_to_threshold"] == 2.5

    def test_improvement_rate_negative_means_improving(self, tmp_results):
        _write_run(tmp_results, "A", 1, 1, 1.95)
        _write_run(tmp_results, "A", 1, 2, 1.90)
        _write_run(tmp_results, "A", 1, 3, 1.85)
        summary = generate_summary("A")
        assert summary["improvement_rate"] < 0  # negative = val_bpb decreasing = good

    def test_single_run_std_is_zero(self, tmp_results):
        """A single non-diverged run should have std_val_bpb = 0."""
        _write_run(tmp_results, "A", 1, 1, 1.90)
        summary = generate_summary("A")
        assert summary["std_val_bpb"] == 0.0

    def test_all_diverged_best_is_none(self, tmp_results):
        """When all runs diverged, best_val_bpb should be None."""
        _write_run(tmp_results, "A", 1, 1, 999999.0, diverged=True)
        _write_run(tmp_results, "A", 1, 2, 999999.0, diverged=True)
        summary = generate_summary("A")
        assert summary["best_val_bpb"] is None
        assert summary["mean_val_bpb"] is None


class TestCohensD:
    def test_identical_groups_zero(self):
        assert cohens_d([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.0

    def test_different_groups_positive(self):
        d = cohens_d([2.0, 2.1, 1.9], [1.0, 1.1, 0.9])
        assert d > 0.0

    def test_small_groups_return_zero(self):
        assert cohens_d([1.0], [2.0]) == 0.0

    def test_known_value_one_std_apart(self):
        """Groups with known Cohen's d."""
        # mean diff = 1.0, both std = 0.5, pooled_std = 0.5, d = -1.0/0.5 = -2.0
        group1 = [-0.5, 0.0, 0.5]
        group2 = [0.5, 1.0, 1.5]
        d = cohens_d(group1, group2)
        assert abs(d) - 2.0 < 0.01

    def test_symmetric_sign(self):
        """Swapping groups should flip the sign."""
        g1 = [1.0, 2.0, 3.0]
        g2 = [4.0, 5.0, 6.0]
        d1 = cohens_d(g1, g2)
        d2 = cohens_d(g2, g1)
        assert abs(d1 + d2) < 1e-10

    def test_empty_groups_return_zero(self):
        assert cohens_d([], [1.0, 2.0]) == 0.0
        assert cohens_d([1.0, 2.0], []) == 0.0


class TestGenerateComparison:
    def test_insufficient_data(self, tmp_results):
        _write_run(tmp_results, "A", 1, 1, 1.90)
        assert generate_comparison() is None

    def test_generates_comparison(self, tmp_results):
        # Create data for two conditions with multiple seeds
        for seed in [1, 2, 3]:
            _write_run(tmp_results, "A", seed, 1, 1.90 + seed * 0.01)
            _write_run(tmp_results, "B", seed, 1, 1.85 + seed * 0.01)
        comp = generate_comparison(phase="pilot")
        assert comp is not None
        assert "A" in comp["conditions"]
        assert "B" in comp["conditions"]
        assert comp["phase"] == "pilot"

    def test_writes_comparison_file(self, tmp_results):
        for seed in [1, 2, 3]:
            _write_run(tmp_results, "A", seed, 1, 1.90 + seed * 0.01)
            _write_run(tmp_results, "B", seed, 1, 1.85 + seed * 0.01)
        generate_comparison(phase="pilot")
        path = tmp_results / "comparisons" / "pilot_comparison.json"
        assert path.exists()

    def test_pairwise_tests(self, tmp_results):
        # Add small variance so Cohen's d is computable
        for i, seed in enumerate([1, 2, 3]):
            _write_run(tmp_results, "A", seed, 1, 1.95 + i * 0.01)
            _write_run(tmp_results, "B", seed, 1, 1.85 + i * 0.01)
            _write_run(tmp_results, "C", seed, 1, 1.80 + i * 0.01)
        comp = generate_comparison()
        assert "A_vs_B" in comp["pairwise_tests"]
        assert "B_vs_C" in comp["pairwise_tests"]
        assert comp["pairwise_tests"]["A_vs_B"]["cohens_d"] > 0

    def test_promotion_criteria_in_output(self, tmp_results):
        for i, seed in enumerate([1, 2, 3]):
            _write_run(tmp_results, "A", seed, 1, 1.95 + i * 0.01)
            _write_run(tmp_results, "B", seed, 1, 1.85 + i * 0.01)
            _write_run(tmp_results, "C", seed, 1, 1.80 + i * 0.01)
        comp = generate_comparison(phase="pilot")
        assert "promotion_criteria" in comp
        assert "hooks_stable" in comp["promotion_criteria"]
        assert "telemetry_complete" in comp["promotion_criteria"]
        assert "directional_separation" in comp["promotion_criteria"]
        assert "effect_size_sufficient" in comp["promotion_criteria"]
        assert "stability_details" in comp

    def test_four_conditions(self, tmp_results):
        """Comparison with all 4 conditions should produce A_vs_B, B_vs_C, C_vs_D pairs."""
        for i, seed in enumerate([1, 2, 3]):
            _write_run(tmp_results, "A", seed, 1, 1.95 + i * 0.01)
            _write_run(tmp_results, "B", seed, 1, 1.90 + i * 0.01)
            _write_run(tmp_results, "C", seed, 1, 1.85 + i * 0.01)
            _write_run(tmp_results, "D", seed, 1, 1.80 + i * 0.01)
        comp = generate_comparison()
        assert comp is not None
        assert "A_vs_B" in comp["pairwise_tests"]
        assert "B_vs_C" in comp["pairwise_tests"]
        assert "C_vs_D" in comp["pairwise_tests"]
        assert len(comp["conditions"]) == 4

    def test_promotion_edge_case_all_criteria_met(self, tmp_results):
        """When all criteria are met, promotion_decision should be 'promote'."""
        # Create data with large effect sizes to trigger promotion
        for i, seed in enumerate([1, 2, 3]):
            _write_run(tmp_results, "A", seed, 1, 2.00 + i * 0.01)
            _write_run(tmp_results, "B", seed, 1, 1.50 + i * 0.01)
            _write_run(tmp_results, "C", seed, 1, 1.00 + i * 0.01)
        comp = generate_comparison()
        # With these large separations, effect_size should be > 0.3
        # and directional should be met (B better than A, C better than B)
        assert comp["promotion_criteria"]["effect_size_sufficient"] is True
        assert comp["promotion_criteria"]["hooks_stable"] is True
        assert comp["promotion_criteria"]["telemetry_complete"] is True

    def test_comparison_with_threshold_parameter(self, tmp_results):
        """generate_comparison should accept and use threshold parameter."""
        for i, seed in enumerate([1, 2, 3]):
            _write_run(tmp_results, "A", seed, 1, 1.95 + i * 0.01)
            _write_run(tmp_results, "A", seed, 2, 1.85 + i * 0.01)
            _write_run(tmp_results, "B", seed, 1, 1.90 + i * 0.01)
            _write_run(tmp_results, "B", seed, 2, 1.80 + i * 0.01)
        comp = generate_comparison(threshold=1.90)
        assert comp is not None
        # At least one condition should have mean_runs_to_threshold
        has_threshold = any(
            "mean_runs_to_threshold" in v
            for v in comp["conditions"].values()
        )
        assert has_threshold


class TestRunsToThresholdGlobal:
    def test_with_fixed_threshold(self, tmp_results):
        # Seed 1: run 1=1.95, run 2=1.90, run 3=1.85 -> hits 1.88 at run 2
        # Seed 2: run 1=1.90, run 2=1.84 -> hits 1.88 at run 2
        _write_run(tmp_results, "A", 1, 1, 1.95)
        _write_run(tmp_results, "A", 1, 2, 1.90)
        _write_run(tmp_results, "A", 1, 3, 1.85)
        _write_run(tmp_results, "A", 2, 1, 1.90)
        _write_run(tmp_results, "A", 2, 2, 1.84)
        summary = generate_summary("A", threshold=1.88)
        # Seed 1: first hit at run 3 (1.85 <= 1.88), seed 2: first hit at run 2 (1.84 <= 1.88)
        assert summary["runs_to_threshold"] == 2.5
        assert summary["threshold_used"] == 1.88

    def test_threshold_not_reached(self, tmp_results):
        _write_run(tmp_results, "A", 1, 1, 1.95)
        _write_run(tmp_results, "A", 1, 2, 1.90)
        # Threshold too low -- never reached
        summary = generate_summary("A", threshold=1.50)
        assert summary["runs_to_threshold"] is None

    def test_no_threshold_uses_per_seed_best(self, tmp_results):
        _write_run(tmp_results, "A", 1, 1, 1.95)
        _write_run(tmp_results, "A", 1, 2, 1.85)
        summary = generate_summary("A", threshold=None)
        assert summary["threshold_used"] is None
        assert summary["runs_to_threshold"] is not None  # falls back to per-seed best


class TestStabilityChecks:
    def test_stable_runs(self, tmp_results):
        _write_run(tmp_results, "A", 1, 1, 1.90)
        _write_run(tmp_results, "A", 1, 2, 1.85)
        stable, crashes = _check_stability("A")
        assert stable is True
        assert crashes == 0

    def test_incomplete_telemetry(self, tmp_results):
        # Write a run with missing fields
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_01"
        seed_dir.mkdir(parents=True, exist_ok=True)
        doc = {
            "run_id": 1,
            "condition": "A",
            "seed": 1,
            "timestamp": "2026-03-16T14:00:00Z",
            "config": {"matrix_lr": 0.04},
            "results": {"val_bpb": 1.90, "steps": 1000},  # missing diverged, loss_trend, grad_norm_max
        }
        (seed_dir / "run_001.json").write_text(json.dumps(doc))
        complete, missing = _check_telemetry_completeness("A")
        assert complete is False
        assert len(missing) > 0

    def test_empty_condition_returns_stable(self, tmp_results):
        """An empty condition (no runs) should be considered stable."""
        stable, crashes = _check_stability("A")
        assert stable is True
        assert crashes == 0

    def test_empty_condition_returns_complete(self, tmp_results):
        """An empty condition (no runs) should be considered complete."""
        complete, missing = _check_telemetry_completeness("A")
        assert complete is True
        assert missing == []

    def test_crash_detected_missing_results_fields(self, tmp_results):
        """A run missing required results fields should count as a crash."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_01"
        seed_dir.mkdir(parents=True, exist_ok=True)
        doc = {
            "run_id": 1,
            "condition": "A",
            "seed": 1,
            "timestamp": "2026-03-16T14:00:00Z",
            "config": {"matrix_lr": 0.04, "embedding_lr": 0.6, "depth": 4, "total_batch_size": 65536},
            "results": {"val_bpb": 1.90},  # missing steps, diverged, loss_trend, grad_norm_max
        }
        (seed_dir / "run_001.json").write_text(json.dumps(doc))
        stable, crashes = _check_stability("A")
        assert stable is False
        assert crashes == 1

    def test_telemetry_missing_config_fields(self, tmp_results):
        """A run missing config sub-fields should be flagged as incomplete."""
        seed_dir = tmp_results / "runs" / "condition_A" / "seed_01"
        seed_dir.mkdir(parents=True, exist_ok=True)
        doc = {
            "run_id": 1,
            "condition": "A",
            "seed": 1,
            "timestamp": "2026-03-16T14:00:00Z",
            "config": {"matrix_lr": 0.04},  # missing embedding_lr, depth, total_batch_size
            "results": {
                "val_bpb": 1.90, "steps": 1000, "diverged": False,
                "loss_trend": "improving", "grad_norm_max": 3.0,
            },
        }
        (seed_dir / "run_001.json").write_text(json.dumps(doc))
        complete, missing = _check_telemetry_completeness("A")
        assert complete is False
        assert any("config" in m for m in missing)


class TestPlots:
    def test_generate_plots_no_data(self, tmp_results):
        generate_plots()  # should not crash with no data

    def test_generate_plots_with_data(self, tmp_results):
        for seed in [1, 2, 3]:
            _write_run(tmp_results, "A", seed, 1, 1.90 + seed * 0.01)
            _write_run(tmp_results, "B", seed, 1, 1.85 + seed * 0.01)
        generate_plots()
        plot_dir = tmp_results / "plots"
        assert (plot_dir / "val_bpb_by_condition.png").exists()
        assert (plot_dir / "wasted_run_rate.png").exists()
        assert (plot_dir / "runs_to_threshold.png").exists()

    def test_plot_files_have_nonzero_size(self, tmp_results):
        """Generated plot files should have non-zero size."""
        for seed in [1, 2, 3]:
            _write_run(tmp_results, "A", seed, 1, 1.90 + seed * 0.01)
            _write_run(tmp_results, "B", seed, 1, 1.85 + seed * 0.01)
        generate_plots()
        plot_dir = tmp_results / "plots"
        for name in ["val_bpb_by_condition.png", "wasted_run_rate.png", "runs_to_threshold.png"]:
            plot_path = plot_dir / name
            assert plot_path.exists(), f"{name} should exist"
            assert plot_path.stat().st_size > 0, f"{name} should have non-zero size"
