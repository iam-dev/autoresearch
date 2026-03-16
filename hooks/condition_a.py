"""Condition A — NullHooks: no memory, no guidance."""

from __future__ import annotations

from hooks import artifacts
from hooks.log_capture import LogCapture
from hooks.types import PreRunContext, RunConfig, RunResults


class NullHooks:
    """No memory, no guidance. Writes run result JSON for analysis only."""

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._log_capture = LogCapture()
        self._run_id: int | None = None
        self._last_pre_run_ctx: PreRunContext | None = None

    def pre_run(self, config: RunConfig) -> PreRunContext:
        ctx = PreRunContext(condition="A")
        self._last_pre_run_ctx = ctx
        return ctx

    def start_log_capture(self, config: RunConfig) -> None:
        seed = self._seed or config.seed
        seed_dir = artifacts.RESULTS_DIR / "runs" / "condition_A" / f"seed_{seed:02d}"
        self._run_id = artifacts._next_run_id(seed_dir)
        log_dir = artifacts.RESULTS_DIR / "logs" / "condition_A" / f"seed_{seed:02d}"
        self._log_capture.start(log_dir, self._run_id)

    def stop_log_capture(self) -> None:
        self._log_capture.stop()

    def post_run(self, config: RunConfig, results: RunResults) -> None:
        seed = self._seed or config.seed
        seed_dir = artifacts.RESULTS_DIR / "runs" / "condition_A" / f"seed_{seed:02d}"
        run_id = self._run_id or artifacts._next_run_id(seed_dir)
        self._run_id = None
        from hooks.analysis import _is_wasted
        best = artifacts._best_val_bpb(seed_dir)
        wasted = _is_wasted(config, results, seed_dir, best)
        path = artifacts._write_run_result(
            "A", seed, run_id, config, results,
            pre_run_context=self._last_pre_run_ctx, wasted=wasted,
        )
        self._last_pre_run_ctx = None
        print(f"[Condition A] Run {run_id} result saved to {path}")
