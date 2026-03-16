"""Tests for condition implementations and the create_hooks factory."""

import json
import sys
from unittest.mock import patch

import pytest

from hooks import (
    ActiveHooks,
    LoggingHooks,
    NullHooks,
    PassiveHooks,
    create_hooks,
)
from hooks.types import RunResults


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
        hooks = NullHooks(seed=None)
        hooks.post_run(sample_config, sample_results)
        run_dir = tmp_results / "runs" / "condition_A" / f"seed_{sample_config.seed:02d}"
        files = list(run_dir.glob("run_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["seed"] == sample_config.seed


class TestLoggingHooks:
    def test_pre_run_returns_condition_b(self, sample_config, tmp_results):
        hooks = LoggingHooks(seed=42)
        ctx = hooks.pre_run(sample_config)
        assert ctx.condition == "B"

    def test_post_run_writes_json_and_jsonl(self, sample_config, sample_results, tmp_results):
        hooks = LoggingHooks(seed=42)
        hooks.post_run(sample_config, sample_results)

        run_dir = tmp_results / "runs" / "condition_B" / "seed_42"
        files = list(run_dir.glob("run_*.json"))
        assert len(files) == 1

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

        assert "run_id" in entry
        assert "timestamp" in entry
        assert "seed" in entry
        assert "config" in entry
        assert "outcome" in entry
        assert "signals" in entry
        assert "summary" in entry

    def test_diverged_run_summary_format(self, sample_config, tmp_results):
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
        hooks = LoggingHooks(seed=42, history_lines=2)
        for _ in range(5):
            hooks.post_run(sample_config, sample_results)
        ctx = hooks.pre_run(sample_config)
        assert len(ctx.summaries) == 2

    def test_seed_fallback_uses_config_seed(self, sample_config, sample_results, tmp_results):
        hooks = LoggingHooks(seed=None)
        hooks.post_run(sample_config, sample_results)
        run_dir = tmp_results / "runs" / "condition_B" / f"seed_{sample_config.seed:02d}"
        files = list(run_dir.glob("run_*.json"))
        assert len(files) == 1


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
        hooks = create_hooks(condition="A", seed=None)
        assert isinstance(hooks, NullHooks)

    def test_no_seed_param_creates_hooks(self):
        hooks = create_hooks(condition="A")
        assert isinstance(hooks, NullHooks)


class TestLogCaptureIntegration:
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
        assert run_files[0].stem == log_files[0].stem


class TestPassiveHooksImportError:
    def test_passive_hooks_import_error(self, sample_config, tmp_results):
        hooks = PassiveHooks(seed=42)
        with patch.dict("sys.modules", {"mnemebrain_core": None, "mnemebrain_core.memory": None}), pytest.raises(ImportError, match="mnemebrain-lite"):
            hooks.pre_run(sample_config)


class TestActiveHooksImportError:
    def test_active_hooks_import_error(self, sample_config, tmp_results):
        hooks = ActiveHooks(seed=42)
        with patch.dict("sys.modules", {"mnemebrain": None}), pytest.raises(ImportError, match="mnemebrain"):
            hooks.pre_run(sample_config)
