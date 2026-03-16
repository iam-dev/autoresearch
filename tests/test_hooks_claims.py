"""Tests for hooks/claims.py — structured claim generation."""

import pytest

from hooks.claims import ClaimBuilder, StructuredClaim
from hooks.types import RunConfig, RunResults


class TestStructuredClaim:
    def test_to_claim_string_basic(self):
        claim = StructuredClaim(
            config_slice={"matrix_lr": 0.03, "depth": 4},
            outcome_type="achieved",
            context_factors=[],
            metric_value=1.872,
        )
        s = claim.to_claim_string()
        assert "depth=4" in s
        assert "matrix_lr=0.03" in s
        assert "achieved" in s
        assert "val_bpb=1.8720" in s

    def test_to_claim_string_with_context(self):
        claim = StructuredClaim(
            config_slice={"matrix_lr": 0.03},
            outcome_type="diverged",
            context_factors=["warmup_ratio=0.05", "grad_norm_unstable"],
        )
        s = claim.to_claim_string()
        assert "diverged" in s
        assert "[warmup_ratio=0.05, grad_norm_unstable]" in s

    def test_to_claim_string_no_metric(self):
        claim = StructuredClaim(
            config_slice={"matrix_lr": 0.03},
            outcome_type="improves",
            context_factors=[],
        )
        s = claim.to_claim_string()
        assert "val_bpb" not in s

    def test_config_slice_sorted(self):
        claim = StructuredClaim(
            config_slice={"depth": 4, "batch_size": 65536, "matrix_lr": 0.03},
            outcome_type="achieved",
            context_factors=[],
        )
        s = claim.to_claim_string()
        assert s.startswith("batch_size=65536 depth=4 matrix_lr=0.03")


class TestClaimBuilder:
    @pytest.fixture
    def builder(self):
        return ClaimBuilder()

    @pytest.fixture
    def config(self):
        return RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.0, warmdown_ratio=0.5, seed=42,
        )

    @pytest.fixture
    def good_results(self):
        return RunResults(
            val_bpb=1.872, steps=1450, peak_vram_mb=4200.0, final_loss=2.34,
            mfu=0.38, diverged=False, loss_trend="improving", grad_norm_max=3.2,
        )

    @pytest.fixture
    def diverged_results(self):
        return RunResults(
            val_bpb=999999.0, steps=100, peak_vram_mb=0, final_loss=999999.0,
            mfu=0, diverged=True, loss_trend="diverging", grad_norm_max=10.0,
        )

    def test_outcome_claim_achieved(self, builder, config, good_results):
        claim = builder.outcome_claim(config, good_results)
        assert claim.outcome_type == "achieved"
        assert claim.metric_value == good_results.val_bpb
        assert "matrix_lr" in claim.config_slice
        assert "depth" in claim.config_slice

    def test_outcome_claim_diverged(self, builder, config, diverged_results):
        claim = builder.outcome_claim(config, diverged_results)
        assert claim.outcome_type == "diverged"

    def test_outcome_claim_context_factors(self, builder, config, good_results):
        claim = builder.outcome_claim(config, good_results)
        assert f"loss_trend={good_results.loss_trend}" in claim.context_factors

    def test_outcome_claim_unstable_grad(self, builder, config):
        results = RunResults(
            val_bpb=1.9, steps=1000, peak_vram_mb=0, final_loss=2.5,
            mfu=0.3, diverged=False, loss_trend="flat", grad_norm_max=7.0,
        )
        claim = builder.outcome_claim(config, results)
        assert "grad_norm_unstable" in claim.context_factors

    def test_outcome_claim_warmup_context(self, builder, good_results):
        config = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.05, warmdown_ratio=0.5, seed=42,
        )
        claim = builder.outcome_claim(config, good_results)
        assert "warmup_ratio=0.05" in claim.context_factors

    def test_query_claims_returns_multiple(self, builder, config):
        claims = builder.query_claims(config)
        assert len(claims) >= 2

    def test_query_claims_single_param(self, builder, config):
        claims = builder.query_claims(config)
        single = [c for c in claims if len(c.config_slice) == 1]
        assert len(single) >= 1
        assert "matrix_lr" in single[0].config_slice

    def test_query_claims_multi_param(self, builder, config):
        claims = builder.query_claims(config)
        multi = [c for c in claims if len(c.config_slice) >= 2]
        assert len(multi) >= 1

    def test_query_claims_stability(self, builder, config):
        claims = builder.query_claims(config)
        stability = [c for c in claims if c.outcome_type == "stabilized"]
        assert len(stability) >= 1

    def test_query_claims_with_warmup(self, builder):
        config = RunConfig(
            matrix_lr=0.04, embedding_lr=0.6, unembedding_lr=0.004,
            scalar_lr=0.5, wd=0.2, depth=4, total_batch_size=65536,
            device_batch_size=16, warmup_ratio=0.05, warmdown_ratio=0.5, seed=42,
        )
        claims = builder.query_claims(config)
        context_aware = [c for c in claims if "warmup_ratio" in c.config_slice]
        assert len(context_aware) >= 1

    def test_all_claims_serialize(self, builder, config, good_results):
        for claim in builder.query_claims(config):
            s = claim.to_claim_string()
            assert isinstance(s, str) and len(s) > 0
        s = builder.outcome_claim(config, good_results).to_claim_string()
        assert isinstance(s, str) and len(s) > 0
