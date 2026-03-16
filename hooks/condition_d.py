"""Condition D — ActiveHooks: full MnemeBrain with prediction and recommendation."""

from __future__ import annotations

from hooks.types import (
    PreRunContext, PredictionResult, RecommendationResult,
    RunConfig, RunResults,
)
from hooks import artifacts
from hooks.log_capture import LogCapture
from hooks.analysis import evidence_weight, _config_distance
from hooks.claims import ClaimBuilder


class ActiveHooks:
    """Full MnemeBrain with prediction and recommendation."""

    def __init__(self, seed: int | None = None, base_url: str = "http://localhost:8000"):
        self._seed = seed
        self._base_url = base_url
        self._client = None
        self._log_capture = LogCapture()
        self._run_id: int | None = None
        self._claim_builder = ClaimBuilder()
        self._last_pre_run_ctx: PreRunContext | None = None

    def _get_client(self):
        if self._client is None:
            try:
                from mnemebrain import MnemeBrainClient
                self._client = MnemeBrainClient(base_url=self._base_url)
            except ImportError:
                raise ImportError(
                    "Condition D requires mnemebrain. "
                    "Install with: pip install mnemebrain"
                )
        return self._client

    def pre_run(self, config: RunConfig) -> PreRunContext:
        client = self._get_client()
        similar_runs: list[str] = []
        contradictions: list[str] = []
        prediction: PredictionResult | None = None
        recommendation: RecommendationResult | None = None

        try:
            for claim in self._claim_builder.query_claims(config):
                claim_str = claim.to_claim_string()
                label = " ".join(f"{k}={v}" for k, v in sorted(claim.config_slice.items()))
                try:
                    explanation = client.explain(claim=claim_str)
                    if hasattr(explanation, 'truth_state') and explanation.truth_state == "BOTH":
                        contradictions.append(f"Contradictory evidence for {label}")
                    if hasattr(explanation, 'evidence'):
                        for ev in explanation.evidence[:3]:
                            entry = str(ev)
                            if entry not in similar_runs:
                                similar_runs.append(entry)
                except Exception:
                    pass
            similar_runs = similar_runs[:5]

            prediction = self._predict(config, client)
            recommendation = self._recommend(config, client)

        except Exception as e:
            print(f"[Condition D] Warning: belief query failed: {e}")

        if prediction:
            print(f"\n[Condition D] Prediction:")
            print(f"  Expected outcome: {prediction.expected_outcome}")
            print(f"  Confidence: {prediction.confidence:.2f}")
            if prediction.risks:
                print(f"  Risks: {', '.join(prediction.risks)}")

        if recommendation:
            print(f"\n[Condition D] Recommendation:")
            print(f"  {recommendation.suggested_change}")
            print(f"  Risk level: {recommendation.risk_level}")
            for r in recommendation.rationale:
                print(f"    - {r}")

        if contradictions:
            print(f"\n[Condition D] Contradictions:")
            for c in contradictions:
                print(f"  {c}")

        if not prediction and not recommendation and not contradictions:
            print("\n[Condition D] No prior beliefs — first run.\n")

        ctx = PreRunContext(
            condition="D",
            similar_runs=similar_runs,
            contradictions=contradictions,
            prediction=prediction,
            recommendation=recommendation,
        )
        self._last_pre_run_ctx = ctx
        return ctx

    def start_log_capture(self, config: RunConfig) -> None:
        seed = self._seed or config.seed
        seed_dir = artifacts.RESULTS_DIR / "runs" / "condition_D" / f"seed_{seed:02d}"
        self._run_id = artifacts._next_run_id(seed_dir)
        log_dir = artifacts.RESULTS_DIR / "logs" / "condition_D" / f"seed_{seed:02d}"
        self._log_capture.start(log_dir, self._run_id)

    def stop_log_capture(self) -> None:
        self._log_capture.stop()

    def post_run(self, config: RunConfig, results: RunResults) -> None:
        seed = self._seed or config.seed
        seed_dir = artifacts.RESULTS_DIR / "runs" / "condition_D" / f"seed_{seed:02d}"
        run_id = self._run_id or artifacts._next_run_id(seed_dir)
        self._run_id = None

        from hooks.analysis import _is_wasted
        best = artifacts._best_val_bpb(seed_dir)
        wasted = _is_wasted(config, results, seed_dir, best)
        path = artifacts._write_run_result(
            "D", seed, run_id, config, results,
            pre_run_context=self._last_pre_run_ctx, wasted=wasted,
        )
        self._last_pre_run_ctx = None

        client = self._get_client()
        claim = self._claim_builder.outcome_claim(config, results)

        try:
            weight = evidence_weight(results.val_bpb, best) if best is not None else 0.5
            polarity = "supports" if (best is None or results.val_bpb <= best) else "attacks"
            client.believe(
                claim=claim.to_claim_string(),
                evidence=[{
                    "source_ref": f"run_{run_id}",
                    "content": (
                        f"val_bpb={results.val_bpb:.4f}, steps={results.steps}, "
                        f"diverged={results.diverged}, loss_trend={results.loss_trend}"
                    ),
                    "polarity": polarity,
                    "weight": weight,
                    "reliability": 0.95,
                }],
            )
        except Exception as e:
            print(f"[Condition D] Warning: failed to record belief: {e}")

        print(f"[Condition D] Run {run_id} result saved to {path}")

    def _predict(self, config: RunConfig, client) -> PredictionResult | None:
        """Evidence-weight-based prediction for this config."""
        try:
            claims = self._claim_builder.query_claims(config)
            all_beliefs = []
            for claim in claims:
                results = client.search(query=claim.to_claim_string(), limit=5)
                if results:
                    all_beliefs.extend(results)

            if not all_beliefs:
                return None

            similar_ids = []
            diverge_weight = 0.0
            improve_weight = 0.0
            total_weight = 0.0
            risks: list[str] = []

            for belief in all_beliefs:
                w = getattr(belief, 'weight', 0.5)
                total_weight += w

                if hasattr(belief, 'source_ref'):
                    try:
                        rid = int(belief.source_ref.split("_")[1])
                        if rid not in similar_ids:
                            similar_ids.append(rid)
                    except (IndexError, ValueError):
                        pass

                content = str(getattr(belief, 'content', ''))
                if 'diverged' in content.lower():
                    diverge_weight += w
                    if 'warmup_ratio=0.0' in content or 'warmup_ratio=0' in content:
                        risk = "No warmup in similar configs that diverged"
                        if risk not in risks:
                            risks.append(risk)
                else:
                    improve_weight += w

                if 'grad_norm_unstable' in content:
                    risk = "Gradient instability in similar runs"
                    if risk not in risks:
                        risks.append(risk)

            if total_weight == 0:
                return None

            diverge_ratio = diverge_weight / total_weight

            if diverge_ratio > 0.4:
                expected = "diverge"
                confidence = diverge_ratio
            elif diverge_ratio > 0.15:
                expected = "marginal"
                confidence = 0.5
            else:
                expected = "improve"
                confidence = improve_weight / total_weight

            return PredictionResult(
                expected_outcome=expected,
                confidence=round(min(confidence, 0.95), 2),
                similar_runs=similar_ids,
                risks=risks,
                source_run_ids=similar_ids,
            )
        except Exception:
            return None

    def _recommend(self, config: RunConfig, client) -> RecommendationResult | None:
        """Belief-grounded recommendation for next experiment."""
        try:
            best_claim = self._claim_builder.query_claims(config)[1]  # multi-param
            beliefs = client.search(query=best_claim.to_claim_string(), limit=10)
            if not beliefs:
                return None

            source_ids = []
            best_configs: list[dict] = []
            rationale_items: list[str] = []

            for belief in beliefs:
                if hasattr(belief, 'source_ref'):
                    try:
                        rid = int(belief.source_ref.split("_")[1])
                        source_ids.append(rid)
                    except (IndexError, ValueError):
                        pass

                content = str(getattr(belief, 'content', ''))
                claim_str = str(getattr(belief, 'claim', ''))

                if 'diverged' not in content.lower():
                    rationale_items.append(claim_str)
                    for part in claim_str.split():
                        if part.startswith("matrix_lr="):
                            try:
                                lr = float(part.split("=")[1])
                                best_configs.append({"matrix_lr": lr})
                            except ValueError:
                                pass

            if not rationale_items:
                return None

            suggested_change = f"Consider adjusting matrix_lr based on {len(beliefs)} prior observations"
            risk_level = "low"

            if best_configs:
                avg_lr = sum(c["matrix_lr"] for c in best_configs) / len(best_configs)
                if abs(config.matrix_lr - avg_lr) > 0.005:
                    direction = "Reduce" if config.matrix_lr > avg_lr else "Increase"
                    suggested_change = f"{direction} matrix_lr from {config.matrix_lr} to {avg_lr:.4f}"

                has_diverged = any('diverged' in str(getattr(b, 'content', '')).lower() for b in beliefs)
                if has_diverged:
                    risk_level = "medium"

            return RecommendationResult(
                suggested_change=suggested_change,
                rationale=rationale_items[:3],
                risk_level=risk_level,
                source_run_ids=source_ids,
            )
        except Exception:
            return None
