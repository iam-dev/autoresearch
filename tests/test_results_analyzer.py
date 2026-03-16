"""Tests for results_analyzer — summary and comparison generation."""

import json
from pathlib import Path

import pytest

from results_analyzer import (
    cohens_d,
    generate_comparison,
    generate_plots,
    generate_summary,
    load_runs,
    load_runs_by_seed,
    _check_stability,
    _check_telemetry_completeness,
    RESULTS_DIR,
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


class TestCohensD:
    def test_identical_groups_zero(self):
        assert cohens_d([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 0.0

    def test_different_groups_positive(self):
        d = cohens_d([2.0, 2.1, 1.9], [1.0, 1.1, 0.9])
        assert d > 0.0

    def test_small_groups_return_zero(self):
        assert cohens_d([1.0], [2.0]) == 0.0


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


class TestRunsToThresholdGlobal:
    def test_with_fixed_threshold(self, tmp_results):
        # Seed 1: run 1=1.95, run 2=1.90, run 3=1.85 → hits 1.88 at run 2
        # Seed 2: run 1=1.90, run 2=1.84 → hits 1.88 at run 2
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
        # Threshold too low — never reached
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
