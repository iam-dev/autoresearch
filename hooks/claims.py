"""Structured claims — composable, typed claim generation for the belief store."""

from __future__ import annotations

from dataclasses import dataclass, field

from hooks.types import RunConfig, RunResults


@dataclass
class StructuredClaim:
    """A typed, composable claim for the mnemebrain belief store."""

    config_slice: dict[str, float | int]
    outcome_type: str       # "achieved" | "improves" | "diverged" | "stabilized"
    context_factors: list[str] = field(default_factory=list)
    metric_value: float | None = None
    metric_name: str = "val_bpb"

    def to_claim_string(self) -> str:
        """Serialize to mnemebrain-compatible claim string."""
        parts = [f"{k}={v}" for k, v in sorted(self.config_slice.items())]
        claim = " ".join(parts) + f" {self.outcome_type}"
        if self.metric_value is not None:
            claim += f" {self.metric_name}={self.metric_value:.4f}"
        if self.context_factors:
            claim += f" [{', '.join(self.context_factors)}]"
        return claim


class ClaimBuilder:
    """Builds structured claims from RunConfig and RunResults."""

    def outcome_claim(self, config: RunConfig, results: RunResults) -> StructuredClaim:
        """Post-run: build a claim recording what happened."""
        if results.diverged:
            outcome_type = "diverged"
        else:
            outcome_type = "achieved"

        context_factors = [f"loss_trend={results.loss_trend}"]
        if results.grad_norm_max > 5.0:
            context_factors.append("grad_norm_unstable")
        if config.warmup_ratio > 0:
            context_factors.append(f"warmup_ratio={config.warmup_ratio}")

        config_slice: dict[str, float | int] = {
            "matrix_lr": config.matrix_lr,
            "depth": config.depth,
        }
        if config.warmup_ratio > 0:
            config_slice["warmup_ratio"] = config.warmup_ratio
        if config.wd > 0:
            config_slice["wd"] = config.wd

        return StructuredClaim(
            config_slice=config_slice,
            outcome_type=outcome_type,
            context_factors=context_factors,
            metric_value=results.val_bpb if not results.diverged else None,
        )

    def query_claims(self, config: RunConfig) -> list[StructuredClaim]:
        """Pre-run: generate claims to query the belief store at multiple granularities."""
        claims = []

        # 1. Single-param: matrix_lr
        claims.append(StructuredClaim(
            config_slice={"matrix_lr": config.matrix_lr},
            outcome_type="improves",
            context_factors=[],
        ))

        # 2. Multi-param: matrix_lr + depth
        claims.append(StructuredClaim(
            config_slice={"matrix_lr": config.matrix_lr, "depth": config.depth},
            outcome_type="improves",
            context_factors=[],
        ))

        # 3. Context-aware: add warmup if non-zero
        if config.warmup_ratio > 0:
            claims.append(StructuredClaim(
                config_slice={
                    "matrix_lr": config.matrix_lr,
                    "depth": config.depth,
                    "warmup_ratio": config.warmup_ratio,
                },
                outcome_type="improves",
                context_factors=[],
            ))

        # 4. Stability query
        claims.append(StructuredClaim(
            config_slice={"matrix_lr": config.matrix_lr, "depth": config.depth},
            outcome_type="stabilized",
            context_factors=[],
        ))

        return claims
