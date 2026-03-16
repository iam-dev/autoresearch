"""Condition C — PassiveHooks: mnemebrain-lite belief memory (passive)."""

from __future__ import annotations

import os

from hooks import artifacts
from hooks.analysis import evidence_weight
from hooks.claims import ClaimBuilder
from hooks.log_capture import LogCapture
from hooks.types import PreRunContext, RunConfig, RunResults


class PassiveHooks:
    """MnemeBrain belief memory — surfaces contradictions, confidence, similar runs."""

    def __init__(self, seed: int | None = None, db_path: str | None = None):
        self._seed = seed
        self._explicit_db_path = db_path  # None means use default at call time
        self._memory = None
        self._log_capture = LogCapture()
        self._run_id: int | None = None
        self._claim_builder = ClaimBuilder()
        self._last_pre_run_ctx: PreRunContext | None = None

    @property
    def _db_path(self) -> str:
        """Compute db_path lazily to respect monkeypatched RESULTS_DIR."""
        return self._explicit_db_path or str(artifacts.RESULTS_DIR / "beliefs")

    def _get_memory(self):
        if self._memory is None:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            try:
                from mnemebrain_core.memory import BeliefMemory
                self._memory = BeliefMemory(db_path=self._db_path, device="cpu")
            except ImportError as e:
                raise ImportError(
                    "Condition C requires mnemebrain-lite[embeddings]. "
                    "Install with: pip install mnemebrain-lite[embeddings]"
                ) from e
            except TypeError:
                from mnemebrain_core.memory import BeliefMemory
                self._memory = BeliefMemory(db_path=self._db_path)
        return self._memory

    def pre_run(self, config: RunConfig) -> PreRunContext:
        memory = self._get_memory()
        similar_runs: list[str] = []
        contradictions: list[str] = []

        for claim in self._claim_builder.query_claims(config):
            claim_str = claim.to_claim_string()
            label = " ".join(f"{k}={v}" for k, v in sorted(claim.config_slice.items()))
            try:
                explanation = memory.explain(claim=claim_str)
                if hasattr(explanation, 'truth_state') and explanation.truth_state == "BOTH":
                    parts = [f"Contradictory evidence for {label}"]
                    if hasattr(explanation, 'supporting_count'):
                        parts.append(f"{explanation.supporting_count} supporting")
                    if hasattr(explanation, 'attacking_count'):
                        parts.append(f"{explanation.attacking_count} attacking")
                    contradictions.append(
                        ": ".join(parts[:1]) + " — " + ", ".join(parts[1:]) if len(parts) > 1 else parts[0]
                    )
                if hasattr(explanation, 'evidence'):
                    for ev in explanation.evidence[:3]:
                        entry = str(ev)
                        if entry not in similar_runs:
                            similar_runs.append(entry)
            except Exception:
                pass

        similar_runs = similar_runs[:5]

        if contradictions:
            print("\n[Condition C] Contradictions detected:")
            for c in contradictions:
                print(f"  {c}")
        if similar_runs:
            print("\n[Condition C] Similar prior runs:")
            for s in similar_runs:
                print(f"  {s}")
        if not contradictions and not similar_runs:
            print("\n[Condition C] No prior beliefs recorded.\n")

        ctx = PreRunContext(
            condition="C",
            similar_runs=similar_runs,
            contradictions=contradictions,
        )
        self._last_pre_run_ctx = ctx
        return ctx

    def start_log_capture(self, config: RunConfig) -> None:
        seed = self._seed or config.seed
        seed_dir = artifacts.RESULTS_DIR / "runs" / "condition_C" / f"seed_{seed:02d}"
        self._run_id = artifacts._next_run_id(seed_dir)
        log_dir = artifacts.RESULTS_DIR / "logs" / "condition_C" / f"seed_{seed:02d}"
        self._log_capture.start(log_dir, self._run_id)

    def stop_log_capture(self) -> None:
        self._log_capture.stop()

    def post_run(self, config: RunConfig, results: RunResults) -> None:
        seed = self._seed or config.seed
        seed_dir = artifacts.RESULTS_DIR / "runs" / "condition_C" / f"seed_{seed:02d}"
        run_id = self._run_id or artifacts._next_run_id(seed_dir)
        self._run_id = None

        from hooks.analysis import _is_wasted
        best = artifacts._best_val_bpb(seed_dir)
        wasted = _is_wasted(config, results, seed_dir, best)
        path = artifacts._write_run_result(
            "C", seed, run_id, config, results,
            pre_run_context=self._last_pre_run_ctx, wasted=wasted,
        )
        self._last_pre_run_ctx = None

        memory = self._get_memory()
        claim = self._claim_builder.outcome_claim(config, results)

        try:
            from mnemebrain_core.providers.base import EvidenceInput

            weight = evidence_weight(results.val_bpb, best) if best is not None else 0.5
            polarity = "supports" if (best is None or results.val_bpb <= best) else "attacks"

            memory.believe(
                claim=claim.to_claim_string(),
                evidence=[EvidenceInput(
                    source_ref=f"run_{run_id}",
                    content=(
                        f"val_bpb={results.val_bpb:.4f}, steps={results.steps}, "
                        f"diverged={results.diverged}, loss_trend={results.loss_trend}"
                    ),
                    polarity=polarity,
                    weight=weight,
                    reliability=0.95,
                )],
            )
        except Exception as e:
            print(f"[Condition C] Warning: failed to record belief: {e}")

        print(f"[Condition C] Run {run_id} result saved to {path}")
