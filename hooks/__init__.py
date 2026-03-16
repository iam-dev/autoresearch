"""MnemeBrain experiment hooks — public API."""

import os

from hooks.analysis import (
    _config_distance,
    _is_exact_duplicate,
    _is_near_bad_config,
    _is_wasted,
    _meaningful_step_from,
    evidence_weight,
)

# NOTE: RESULTS_DIR is re-exported as an import-time copy. Monkeypatching
# hooks.artifacts.RESULTS_DIR will NOT update hooks.RESULTS_DIR. Tests and
# condition modules should access hooks.artifacts.RESULTS_DIR directly.
from hooks.artifacts import (
    RESULTS_DIR,
    _best_val_bpb,
    _ensure_dir,
    _next_run_id,
    _write_run_result,
)
from hooks.base import ExperimentHooks
from hooks.claims import ClaimBuilder, StructuredClaim
from hooks.condition_a import NullHooks
from hooks.condition_b import LoggingHooks
from hooks.condition_c import PassiveHooks
from hooks.condition_d import ActiveHooks
from hooks.log_capture import LogCapture
from hooks.types import (
    PredictionResult,
    PreRunContext,
    RecommendationResult,
    RunConfig,
    RunResults,
)

_CONDITION_MAP = {
    "A": NullHooks,
    "B": LoggingHooks,
    "C": PassiveHooks,
    "D": ActiveHooks,
}


def create_hooks(condition: str | None = None, seed: int | None = None) -> ExperimentHooks:
    """Create hooks for the given condition (default: reads AUTORESEARCH_CONDITION env var)."""
    cond = condition or os.environ.get("AUTORESEARCH_CONDITION", "A")
    cond = cond.upper()
    if cond not in _CONDITION_MAP:
        raise ValueError(
            f"Unknown condition '{cond}'. Expected one of: {', '.join(_CONDITION_MAP)}"
        )
    cls = _CONDITION_MAP[cond]
    kwargs: dict = {}
    if seed is not None:
        kwargs["seed"] = seed
    hooks = cls(**kwargs)
    return hooks  # type: ignore[return-value]


__all__ = [
    "RunConfig", "RunResults", "PredictionResult", "RecommendationResult", "PreRunContext",
    "ExperimentHooks",
    "NullHooks", "LoggingHooks", "PassiveHooks", "ActiveHooks",
    "create_hooks",
    "LogCapture",
    "evidence_weight",
    "StructuredClaim", "ClaimBuilder",
    "_config_distance", "_meaningful_step_from", "_is_exact_duplicate",
    "_is_near_bad_config", "_is_wasted", "_next_run_id", "_best_val_bpb",
    "_write_run_result", "_ensure_dir",
    "RESULTS_DIR",
]
