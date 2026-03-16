"""Typed data contracts for the experiment hooks system."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunConfig:
    matrix_lr: float
    embedding_lr: float
    unembedding_lr: float
    scalar_lr: float
    wd: float
    depth: int
    total_batch_size: int
    device_batch_size: int
    warmup_ratio: float
    warmdown_ratio: float
    seed: int
    rationale_tag: str = ""
    rationale: str = ""

    @property
    def lr(self) -> float:
        """Shorthand for matrix_lr — the highest-leverage knob."""
        return self.matrix_lr


@dataclass
class RunResults:
    val_bpb: float
    steps: int
    peak_vram_mb: float
    final_loss: float
    mfu: float
    diverged: bool
    loss_trend: str       # "improving" | "flat" | "diverging"
    grad_norm_max: float


@dataclass
class PredictionResult:
    expected_outcome: str
    confidence: float
    similar_runs: list[int]
    risks: list[str]
    source_run_ids: list[int] = field(default_factory=list)


@dataclass
class RecommendationResult:
    suggested_change: str
    rationale: list[str]
    risk_level: str       # "low" | "medium" | "high"
    source_run_ids: list[int] = field(default_factory=list)


@dataclass
class PreRunContext:
    condition: str        # "A" | "B" | "C" | "D"
    summaries: list[str] = field(default_factory=list)
    similar_runs: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    prediction: PredictionResult | None = None
    recommendation: RecommendationResult | None = None
