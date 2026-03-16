"""Condition B — LoggingHooks: structured JSONL logging."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from hooks import artifacts
from hooks.log_capture import LogCapture
from hooks.types import PreRunContext, RunConfig, RunResults


class LoggingHooks:
    """Structured JSONL logging. Prints recent summaries before each run."""

    def __init__(self, seed: int | None = None, history_lines: int = 5):
        self._seed = seed
        self._history_lines = history_lines
        self._log_capture = LogCapture()
        self._run_id: int | None = None
        self._last_pre_run_ctx: PreRunContext | None = None

    @property
    def _log_path(self):
        return artifacts.RESULTS_DIR / "experiment_log.jsonl"

    def pre_run(self, config: RunConfig) -> PreRunContext:
        summaries = self._read_recent_summaries()
        if summaries:
            print(f"\n[Condition B] Recent run summaries ({len(summaries)}):")
            for s in summaries:
                print(f"  {s}")
            print()
        else:
            print("\n[Condition B] No prior runs recorded.\n")
        ctx = PreRunContext(condition="B", summaries=summaries)
        self._last_pre_run_ctx = ctx
        return ctx

    def start_log_capture(self, config: RunConfig) -> None:
        seed = self._seed or config.seed
        seed_dir = artifacts.RESULTS_DIR / "runs" / "condition_B" / f"seed_{seed:02d}"
        self._run_id = artifacts._next_run_id(seed_dir)
        log_dir = artifacts.RESULTS_DIR / "logs" / "condition_B" / f"seed_{seed:02d}"
        self._log_capture.start(log_dir, self._run_id)

    def stop_log_capture(self) -> None:
        self._log_capture.stop()

    def post_run(self, config: RunConfig, results: RunResults) -> None:
        seed = self._seed or config.seed
        seed_dir = artifacts.RESULTS_DIR / "runs" / "condition_B" / f"seed_{seed:02d}"
        run_id = self._run_id or artifacts._next_run_id(seed_dir)
        self._run_id = None

        from hooks.analysis import _is_wasted
        best = artifacts._best_val_bpb(seed_dir)
        wasted = _is_wasted(config, results, seed_dir, best)
        path = artifacts._write_run_result(
            "B", seed, run_id, config, results,
            pre_run_context=self._last_pre_run_ctx, wasted=wasted,
        )
        self._last_pre_run_ctx = None

        summary = self._make_summary(run_id, seed, config, results)
        artifacts._ensure_dir(self._log_path.parent)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"[Condition B] Run {run_id} result saved to {path}")

    def _make_summary(self, run_id: int, seed: int, config: RunConfig, results: RunResults) -> dict:
        return {
            "run_id": run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "seed": seed,
            "config": {
                "matrix_lr": config.matrix_lr,
                "embedding_lr": config.embedding_lr,
                "wd": config.wd,
                "depth": config.depth,
                "total_batch_size": config.total_batch_size,
            },
            "outcome": {
                "val_bpb": results.val_bpb,
                "diverged": results.diverged,
            },
            "signals": {
                "grad_norm_max": results.grad_norm_max,
                "loss_trend": results.loss_trend,
                "final_step": results.steps,
            },
            "summary": self._one_line_summary(config, results),
        }

    def _one_line_summary(self, config: RunConfig, results: RunResults) -> str:
        if results.diverged:
            return f"matrix_lr {config.matrix_lr} diverged around step {results.steps}"
        return f"matrix_lr {config.matrix_lr} depth={config.depth} → val_bpb={results.val_bpb:.4f} ({results.loss_trend})"

    def _read_recent_summaries(self) -> list[str]:
        if not self._log_path.exists():
            return []
        lines = self._log_path.read_text().strip().split("\n")
        recent = lines[-self._history_lines:]
        summaries = []
        for line in recent:
            try:
                entry = json.loads(line)
                summaries.append(f"Run {entry['run_id']}: {entry['summary']}")
            except (json.JSONDecodeError, KeyError):
                continue
        return summaries
