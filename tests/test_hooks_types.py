"""Tests for hooks/types.py — data contract validation."""

from hooks.types import (
    PredictionResult,
    PreRunContext,
    RecommendationResult,
    RunConfig,
)


class TestDataContracts:
    def test_run_config_lr_property(self):
        cfg = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
        )
        assert cfg.lr == cfg.matrix_lr == 0.04

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
