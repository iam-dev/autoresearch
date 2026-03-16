"""
E2E tests for MnemeBrain integration — requires mnemebrain-lite[embeddings].

These tests exercise the PassiveHooks (Condition C) full lifecycle using the
real BeliefMemory from mnemebrain-lite. They are skipped when the package is
not installed (Python 3.12+ required).

Run with: uv run pytest -m e2e
"""
from __future__ import annotations

import pytest

mnemebrain_core = pytest.importorskip(
    "mnemebrain_core",
    reason="mnemebrain-lite not installed (requires Python 3.12+)",
)

from hooks import PassiveHooks, RunConfig, RunResults, evidence_weight


@pytest.fixture
def good_results():
    return RunResults(
        val_bpb=1.85, steps=1450, peak_vram_mb=0, final_loss=2.3,
        mfu=0.38, diverged=False, loss_trend="improving", grad_norm_max=3.2,
    )


@pytest.fixture
def bad_results():
    return RunResults(
        val_bpb=2.10, steps=1000, peak_vram_mb=0, final_loss=3.0,
        mfu=0.30, diverged=False, loss_trend="flat", grad_norm_max=5.0,
    )


@pytest.mark.e2e
class TestPassiveHooksE2E:
    def test_full_lifecycle(self, sample_config, good_results, tmp_results):
        """pre_run (empty) -> post_run (record) -> pre_run (finds prior)."""
        db_path = str(tmp_results / "beliefs_e2e")
        hooks = PassiveHooks(seed=42, db_path=db_path)

        ctx1 = hooks.pre_run(sample_config)
        assert ctx1.condition == "C"
        assert ctx1.similar_runs == []
        assert ctx1.contradictions == []

        hooks.post_run(sample_config, good_results)

        run_dir = tmp_results / "runs" / "condition_C" / "seed_42"
        files = list(run_dir.glob("run_*.json"))
        assert len(files) == 1

        ctx2 = hooks.pre_run(sample_config)
        assert ctx2.condition == "C"

    def test_multiple_runs_accumulate(self, sample_config, good_results, bad_results, tmp_results):
        """Multiple post_run calls should accumulate run JSONs."""
        db_path = str(tmp_results / "beliefs_acc")
        hooks = PassiveHooks(seed=42, db_path=db_path)

        hooks.post_run(sample_config, good_results)
        hooks.post_run(sample_config, bad_results)

        run_dir = tmp_results / "runs" / "condition_C" / "seed_42"
        files = list(run_dir.glob("run_*.json"))
        assert len(files) == 2

    def test_evidence_weight_integration(self, tmp_results):
        """Evidence weights should reflect improvement vs regression."""
        db_path = str(tmp_results / "beliefs_weight")
        hooks = PassiveHooks(seed=42, db_path=db_path)

        config = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
        )

        r1 = RunResults(val_bpb=1.90, steps=1000, peak_vram_mb=0, final_loss=2.5,
                        mfu=0.35, diverged=False, loss_trend="improving", grad_norm_max=3.0)
        hooks.post_run(config, r1)

        r2 = RunResults(val_bpb=1.85, steps=1200, peak_vram_mb=0, final_loss=2.3,
                        mfu=0.37, diverged=False, loss_trend="improving", grad_norm_max=2.8)
        hooks.post_run(config, r2)

        w = evidence_weight(1.85, 1.90)
        assert w > 0.5

    def test_contradiction_detection(self, tmp_results):
        """Conflicting results for same LR should not crash."""
        db_path = str(tmp_results / "beliefs_contra")
        hooks = PassiveHooks(seed=42, db_path=db_path)

        config = RunConfig(
            matrix_lr=0.06, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
        )

        good = RunResults(val_bpb=1.82, steps=1400, peak_vram_mb=0, final_loss=2.2,
                          mfu=0.38, diverged=False, loss_trend="improving", grad_norm_max=2.5)
        hooks.post_run(config, good)

        bad = RunResults(val_bpb=2.30, steps=800, peak_vram_mb=0, final_loss=3.5,
                         mfu=0.20, diverged=False, loss_trend="diverging", grad_norm_max=8.0)
        hooks.post_run(config, bad)

        ctx = hooks.pre_run(config)
        assert ctx.condition == "C"
