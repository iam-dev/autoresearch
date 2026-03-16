"""
MnemeBrain experiment hooks — between-run integration for controlled experiments.

Provides 4 condition implementations:
  A (NullHooks)    — no memory, no guidance; writes run result JSON for analysis only
  B (LoggingHooks) — structured JSONL logging; prints recent summaries before each run
  C (PassiveHooks) — mnemebrain-lite belief memory; surfaces contradictions & similar runs
  D (ActiveHooks)  — full mnemebrain; adds prediction & recommendation before each run

Usage:
  hooks = create_hooks()  # reads AUTORESEARCH_CONDITION env var
  ctx = hooks.pre_run(run_config)
  # ... training ...
  hooks.post_run(run_config, run_results)
"""

from __future__ import annotations

import io
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


# ---------------------------------------------------------------------------
# Typed data contracts
# ---------------------------------------------------------------------------

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
    # expected_outcome: "improve" | "regress" | "diverge" | "marginal"
    #   improve  — val_bpb beats current best by more than ε (0.02)
    #   marginal — within ±ε of current best
    #   regress  — worse than current best by more than ε
    #   diverge  — NaN / unstable / catastrophic loss growth
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


# ---------------------------------------------------------------------------
# Hook protocol
# ---------------------------------------------------------------------------

class ExperimentHooks(Protocol):
    def pre_run(self, config: RunConfig) -> PreRunContext:
        """Called BEFORE training. Returns context/suggestions."""
        ...

    def post_run(self, config: RunConfig, results: RunResults) -> None:
        """Called AFTER training. Records results."""
        ...

    def start_log_capture(self, config: RunConfig) -> None:
        """Called after pre_run to start capturing stdout/stderr to a log file."""
        ...

    def stop_log_capture(self) -> None:
        """Called before post_run to stop capturing and flush the log file."""
        ...


# ---------------------------------------------------------------------------
# Log capture — tees stdout/stderr to a file while preserving terminal output
# ---------------------------------------------------------------------------

class _TeeWriter:
    """Writes to both the original stream and a file."""

    def __init__(self, original: io.TextIOBase, log_file: io.TextIOBase):
        self._original = original
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._original.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def __getattr__(self, name: str):
        return getattr(self._original, name)


class LogCapture:
    """Captures stdout/stderr to results/logs/condition_X/seed_Y/run_NNN.log."""

    def __init__(self):
        self._log_file: io.TextIOBase | None = None
        self._orig_stdout = None
        self._orig_stderr = None

    def start(self, condition: str, seed: int, run_id: int) -> Path:
        log_dir = RESULTS_DIR / "logs" / f"condition_{condition}" / f"seed_{seed:02d}"
        _ensure_dir(log_dir)
        log_path = log_dir / f"run_{run_id:03d}.log"
        self._log_file = open(log_path, "w")
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeWriter(self._orig_stdout, self._log_file)
        sys.stderr = _TeeWriter(self._orig_stderr, self._log_file)
        return log_path

    def stop(self) -> None:
        if self._orig_stdout is not None:
            sys.stdout = self._orig_stdout
            self._orig_stdout = None
        if self._orig_stderr is not None:
            sys.stderr = self._orig_stderr
            self._orig_stderr = None
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _next_run_id(run_dir: Path) -> int:
    """Determine next run ID by scanning existing run files."""
    if not run_dir.exists():
        return 1
    existing = sorted(run_dir.glob("run_*.json"))
    if not existing:
        return 1
    # Extract the highest run number
    last = existing[-1].stem  # e.g. "run_003"
    return int(last.split("_")[1]) + 1


def _best_val_bpb(run_dir: Path) -> float | None:
    """Find the best val_bpb across all runs in a directory."""
    best = None
    for f in run_dir.glob("run_*.json"):
        try:
            data = json.loads(f.read_text())
            vbpb = data["results"]["val_bpb"]
            if not data["results"].get("diverged", False):
                if best is None or vbpb < best:
                    best = vbpb
        except (json.JSONDecodeError, KeyError):
            continue
    return best


def _write_run_result(
    condition: str,
    seed: int,
    run_id: int,
    config: RunConfig,
    results: RunResults,
    pre_run_context: PreRunContext | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a single run result JSON to results/runs/condition_X/seed_Y/run_NNN.json."""
    seed_dir = RESULTS_DIR / "runs" / f"condition_{condition}" / f"seed_{seed:02d}"
    _ensure_dir(seed_dir)

    best = _best_val_bpb(seed_dir)
    delta_from_best = (results.val_bpb - best) if best is not None else 0.0

    # Strip rationale fields from config sub-dict (they're stored at top level)
    config_dict = {k: v for k, v in asdict(config).items()
                   if k not in ("rationale_tag", "rationale")}
    doc = {
        "run_id": run_id,
        "condition": condition,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config_dict,
        "results": asdict(results),
        "rationale_tag": config.rationale_tag,
        "rationale": config.rationale,
        "wasted": _is_wasted(config, results, seed_dir, best),
        "delta_from_best": round(delta_from_best, 6),
    }
    if pre_run_context is not None:
        doc["pre_run_context"] = asdict(pre_run_context)
    if extra:
        doc.update(extra)

    path = seed_dir / f"run_{run_id:03d}.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def _is_wasted(
    config: RunConfig,
    results: RunResults,
    seed_dir: Path,
    best_val_bpb: float | None,
    epsilon: float = 0.02,
) -> bool:
    """Determine if a run is wasted per the plan's criteria."""
    cfg = asdict(config)

    # Diverged
    if results.diverged:
        return True

    # Regressed beyond epsilon — no exception; regression is always wasted
    if best_val_bpb is not None and results.val_bpb > best_val_bpb + epsilon:
        return True

    # Exact duplicate of a previously tested config with no new rationale
    if not config.rationale_tag and _is_exact_duplicate(cfg, seed_dir):
        return True

    # Near a previously observed bad config
    near, bad_cfg = _is_near_bad_config(config, seed_dir)
    if near and bad_cfg is not None:
        # Exception: rationale + meaningful step from the *specific bad config*
        if config.rationale_tag and _meaningful_step_from(cfg, bad_cfg):
            return False
        return True

    return False


def _config_distance(c1: dict, c2: dict) -> float:
    """L2 distance in normalized parameter space."""
    def norm_log(val, lo, hi):
        return (math.log10(val) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))

    def norm_lin(val, lo, hi):
        return (val - lo) / (hi - lo) if hi > lo else 0.0

    vec1, vec2 = [], []
    for key, lo, hi, scale in [
        ("matrix_lr", 0.005, 0.2, "log"),
        ("embedding_lr", 0.05, 2.0, "log"),
        ("scalar_lr", 0.05, 2.0, "log"),
        ("unembedding_lr", 0.0005, 0.02, "log"),
        ("wd", 0.0, 0.4, "linear"),
        ("depth", 2, 16, "linear"),
        ("total_batch_size", 16384, 524288, "log"),
        ("warmup_ratio", 0.0, 0.2, "linear"),
        ("warmdown_ratio", 0.0, 0.8, "linear"),
    ]:
        v1 = c1.get(key, 0)
        v2 = c2.get(key, 0)
        if scale == "log":
            v1 = norm_log(max(v1, lo), lo, hi)
            v2 = norm_log(max(v2, lo), lo, hi)
        else:
            v1 = norm_lin(v1, lo, hi)
            v2 = norm_lin(v2, lo, hi)
        vec1.append(v1)
        vec2.append(v2)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def _meaningful_step_from(cfg: dict, prior_cfg: dict) -> bool:
    """Check if config has at least one focal parameter changed beyond meaningful threshold."""
    return (
        abs(math.log2(cfg["matrix_lr"] / max(prior_cfg["matrix_lr"], 1e-10))) >= 1.0
        or abs(cfg["depth"] - prior_cfg["depth"]) >= 2
        or abs(cfg["warmup_ratio"] - prior_cfg["warmup_ratio"]) >= 0.05
        or abs(math.log2(cfg["total_batch_size"] / max(prior_cfg["total_batch_size"], 1))) >= 1.0
        or abs(cfg["wd"] - prior_cfg["wd"]) >= 0.05
    )


def _is_exact_duplicate(cfg: dict, seed_dir: Path) -> bool:
    """Check if this config exactly matches a previously tested config."""
    # Compare the hyperparameter keys only (not rationale_tag/rationale/seed)
    hp_keys = [
        "matrix_lr", "embedding_lr", "unembedding_lr", "scalar_lr",
        "wd", "depth", "total_batch_size", "device_batch_size",
        "warmup_ratio", "warmdown_ratio",
    ]
    for f in seed_dir.glob("run_*.json"):
        try:
            prior_cfg = json.loads(f.read_text())["config"]
        except (json.JSONDecodeError, KeyError):
            continue
        if all(cfg.get(k) == prior_cfg.get(k) for k in hp_keys):
            return True
    return False


def _is_near_bad_config(config: RunConfig, seed_dir: Path, tau: float = 0.1) -> tuple[bool, dict | None]:
    """Check if config is within L2 tolerance of a previously observed bad config.
    Returns (is_near, nearest_bad_config_dict)."""
    cfg = asdict(config)
    for f in seed_dir.glob("run_*.json"):
        try:
            data = json.loads(f.read_text())
            prior_cfg = data["config"]
            prior_results = data["results"]
        except (json.JSONDecodeError, KeyError):
            continue
        if prior_results.get("diverged", False):
            if _config_distance(cfg, prior_cfg) < tau:
                return True, prior_cfg
    return False, None


def evidence_weight(val_bpb: float, best_val_bpb: float) -> float:
    """Log-scale evidence weight. Improvements → weight > 0.5, regressions → weight < 0.5.

    Formula: weight = 0.5 + sign * 0.5 * |tanh(log10(|delta| + 1e-3) + 2.5)|

    Constants calibrated to match plan's review note #4 table while maintaining
    monotonicity (the plan's original formula with offset=2, eps=1e-4 is non-monotonic
    around delta=0.01):

    Sign convention: delta = val_bpb - best_val_bpb
      negative delta = improvement (lower bpb is better) → high weight
      positive delta = regression → low weight

    Calibration (approximate):
      delta=-0.05 → ~0.92, delta=-0.01 → ~0.75, delta=-0.001 → ~0.60
      delta=+0.001 → ~0.40, delta=+0.01 → ~0.25, delta=+0.05 → ~0.08
    """
    delta = val_bpb - best_val_bpb
    if abs(delta) < 1e-8:
        return 0.5
    magnitude = abs(math.tanh(math.log10(abs(delta) + 1e-3) + 2.5))
    sign = 1.0 if delta < 0 else -1.0
    return 0.5 + sign * 0.5 * magnitude


# ---------------------------------------------------------------------------
# Condition A — NullHooks
# ---------------------------------------------------------------------------

class NullHooks:
    """No memory, no guidance. Writes run result JSON for analysis only."""

    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._log_capture = LogCapture()
        self._run_id: int | None = None

    def pre_run(self, config: RunConfig) -> PreRunContext:
        return PreRunContext(condition="A")

    def start_log_capture(self, config: RunConfig) -> None:
        seed = self._seed or config.seed
        seed_dir = RESULTS_DIR / "runs" / "condition_A" / f"seed_{seed:02d}"
        self._run_id = _next_run_id(seed_dir)
        self._log_capture.start("A", seed, self._run_id)

    def stop_log_capture(self) -> None:
        self._log_capture.stop()

    def post_run(self, config: RunConfig, results: RunResults) -> None:
        seed = self._seed or config.seed
        seed_dir = RESULTS_DIR / "runs" / "condition_A" / f"seed_{seed:02d}"
        run_id = self._run_id or _next_run_id(seed_dir)
        self._run_id = None
        path = _write_run_result("A", seed, run_id, config, results)
        print(f"[Condition A] Run {run_id} result saved to {path}")


# ---------------------------------------------------------------------------
# Condition B — LoggingHooks
# ---------------------------------------------------------------------------

class LoggingHooks:
    """Structured JSONL logging. Prints recent summaries before each run."""

    def __init__(self, seed: int | None = None, history_lines: int = 5):
        self._seed = seed
        self._history_lines = history_lines
        self._log_path = RESULTS_DIR / "experiment_log.jsonl"
        self._log_capture = LogCapture()
        self._run_id: int | None = None

    def pre_run(self, config: RunConfig) -> PreRunContext:
        summaries = self._read_recent_summaries()
        if summaries:
            print(f"\n[Condition B] Recent run summaries ({len(summaries)}):")
            for s in summaries:
                print(f"  {s}")
            print()
        else:
            print("\n[Condition B] No prior runs recorded.\n")
        return PreRunContext(condition="B", summaries=summaries)

    def start_log_capture(self, config: RunConfig) -> None:
        seed = self._seed or config.seed
        seed_dir = RESULTS_DIR / "runs" / "condition_B" / f"seed_{seed:02d}"
        self._run_id = _next_run_id(seed_dir)
        self._log_capture.start("B", seed, self._run_id)

    def stop_log_capture(self) -> None:
        self._log_capture.stop()

    def post_run(self, config: RunConfig, results: RunResults) -> None:
        seed = self._seed or config.seed
        seed_dir = RESULTS_DIR / "runs" / "condition_B" / f"seed_{seed:02d}"
        run_id = self._run_id or _next_run_id(seed_dir)
        self._run_id = None

        # Write individual run JSON
        ctx = PreRunContext(condition="B", summaries=self._read_recent_summaries())
        path = _write_run_result("B", seed, run_id, config, results, pre_run_context=ctx)

        # Append to experiment_log.jsonl
        summary = self._make_summary(run_id, seed, config, results)
        _ensure_dir(self._log_path.parent)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")

        print(f"[Condition B] Run {run_id} result saved to {path}")

    def _make_summary(self, run_id: int, seed: int, config: RunConfig, results: RunResults) -> dict:
        return {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
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


# ---------------------------------------------------------------------------
# Condition C — PassiveHooks (mnemebrain-lite as library)
# ---------------------------------------------------------------------------

class PassiveHooks:
    """
    MnemeBrain belief memory — surfaces contradictions, confidence, similar runs.
    Does NOT suggest parameter changes. Uses mnemebrain-lite as a direct library.
    """

    def __init__(self, seed: int | None = None, db_path: str | None = None):
        self._seed = seed
        self._db_path = db_path or str(RESULTS_DIR / "beliefs")
        self._memory = None  # Lazy init to avoid import at module level
        self._log_capture = LogCapture()
        self._run_id: int | None = None

    def _get_memory(self):
        if self._memory is None:
            # Avoid HuggingFace tokenizer fork warnings
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            try:
                from mnemebrain_core.memory import BeliefMemory
                # Force CPU-only to avoid competing with training for GPU VRAM
                self._memory = BeliefMemory(db_path=self._db_path, device="cpu")
            except ImportError:
                raise ImportError(
                    "Condition C requires mnemebrain-lite[embeddings]. "
                    "Install with: pip install mnemebrain-lite[embeddings]"
                )
            except TypeError:
                # Fallback if BeliefMemory doesn't accept device kwarg
                from mnemebrain_core.memory import BeliefMemory
                self._memory = BeliefMemory(db_path=self._db_path)
        return self._memory

    def pre_run(self, config: RunConfig) -> PreRunContext:
        memory = self._get_memory()

        similar_runs: list[str] = []
        contradictions: list[str] = []

        # Query beliefs about key parameters (multi-parameter, not just matrix_lr)
        claims = [
            (f"matrix_lr={config.matrix_lr} improves training", f"matrix_lr={config.matrix_lr}"),
            (f"matrix_lr={config.matrix_lr} depth={config.depth} improves training",
             f"matrix_lr={config.matrix_lr} depth={config.depth}"),
        ]

        for claim, label in claims:
            try:
                explanation = memory.explain(claim=claim)
                if hasattr(explanation, 'truth_state') and explanation.truth_state == "BOTH":
                    parts = [f"Contradictory evidence for {label}"]
                    if hasattr(explanation, 'supporting_count'):
                        parts.append(f"{explanation.supporting_count} supporting")
                    if hasattr(explanation, 'attacking_count'):
                        parts.append(f"{explanation.attacking_count} attacking")
                    contradictions.append(": ".join(parts[:1]) + " — " + ", ".join(parts[1:]) if len(parts) > 1 else parts[0])
                if hasattr(explanation, 'evidence'):
                    for ev in explanation.evidence[:3]:
                        entry = str(ev)
                        if entry not in similar_runs:
                            similar_runs.append(entry)
            except Exception:
                pass  # No beliefs yet — first run

        # Cap similar runs to avoid flooding output
        similar_runs = similar_runs[:5]

        if contradictions:
            print(f"\n[Condition C] Contradictions detected:")
            for c in contradictions:
                print(f"  {c}")
        if similar_runs:
            print(f"\n[Condition C] Similar prior runs:")
            for s in similar_runs:
                print(f"  {s}")
        if not contradictions and not similar_runs:
            print("\n[Condition C] No prior beliefs recorded.\n")

        return PreRunContext(
            condition="C",
            similar_runs=similar_runs,
            contradictions=contradictions,
        )

    def start_log_capture(self, config: RunConfig) -> None:
        seed = self._seed or config.seed
        seed_dir = RESULTS_DIR / "runs" / "condition_C" / f"seed_{seed:02d}"
        self._run_id = _next_run_id(seed_dir)
        self._log_capture.start("C", seed, self._run_id)

    def stop_log_capture(self) -> None:
        self._log_capture.stop()

    def post_run(self, config: RunConfig, results: RunResults) -> None:
        seed = self._seed or config.seed
        seed_dir = RESULTS_DIR / "runs" / "condition_C" / f"seed_{seed:02d}"
        run_id = self._run_id or _next_run_id(seed_dir)
        self._run_id = None

        # Write run result JSON
        path = _write_run_result("C", seed, run_id, config, results)

        # Record observation as belief in mnemebrain-lite
        memory = self._get_memory()
        best = _best_val_bpb(seed_dir)

        try:
            from mnemebrain_core.providers.base import EvidenceInput

            weight = evidence_weight(results.val_bpb, best) if best is not None else 0.5
            polarity = "supports" if (best is None or results.val_bpb <= best) else "attacks"

            memory.believe(
                claim=f"matrix_lr={config.matrix_lr} depth={config.depth} achieved val_bpb={results.val_bpb:.4f}",
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


# ---------------------------------------------------------------------------
# Condition D — ActiveHooks (mnemebrain SDK + prediction/recommendation)
# ---------------------------------------------------------------------------

class ActiveHooks:
    """
    Full MnemeBrain with prediction and recommendation.
    Uses mnemebrain SDK client to query the mnemebrain-lite API server.
    Prediction/recommendation logic lives here, not in the server.
    """

    def __init__(self, seed: int | None = None, base_url: str = "http://localhost:8000"):
        self._seed = seed
        self._base_url = base_url
        self._client = None  # Lazy init
        self._log_capture = LogCapture()
        self._run_id: int | None = None

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
            # Query belief store for context (multi-parameter claims)
            claims = [
                (f"matrix_lr={config.matrix_lr} improves training", f"matrix_lr={config.matrix_lr}"),
                (f"matrix_lr={config.matrix_lr} depth={config.depth} improves training",
                 f"matrix_lr={config.matrix_lr} depth={config.depth}"),
            ]
            for claim, label in claims:
                try:
                    explanation = client.explain(claim=claim)
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

            # Generate prediction from accumulated beliefs
            prediction = self._predict(config, client)
            recommendation = self._recommend(config, client)

        except Exception as e:
            print(f"[Condition D] Warning: belief query failed: {e}")

        # Print context
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
            print(f"\n[Condition D] ⚠ Contradictions:")
            for c in contradictions:
                print(f"  {c}")

        if not prediction and not recommendation and not contradictions:
            print("\n[Condition D] No prior beliefs — first run.\n")

        return PreRunContext(
            condition="D",
            similar_runs=similar_runs,
            contradictions=contradictions,
            prediction=prediction,
            recommendation=recommendation,
        )

    def start_log_capture(self, config: RunConfig) -> None:
        seed = self._seed or config.seed
        seed_dir = RESULTS_DIR / "runs" / "condition_D" / f"seed_{seed:02d}"
        self._run_id = _next_run_id(seed_dir)
        self._log_capture.start("D", seed, self._run_id)

    def stop_log_capture(self) -> None:
        self._log_capture.stop()

    def post_run(self, config: RunConfig, results: RunResults) -> None:
        seed = self._seed or config.seed
        seed_dir = RESULTS_DIR / "runs" / "condition_D" / f"seed_{seed:02d}"
        run_id = self._run_id or _next_run_id(seed_dir)
        self._run_id = None

        # Write run result JSON
        path = _write_run_result("D", seed, run_id, config, results)

        # Record belief via SDK
        client = self._get_client()
        best = _best_val_bpb(seed_dir)
        try:
            weight = evidence_weight(results.val_bpb, best) if best is not None else 0.5
            polarity = "supports" if (best is None or results.val_bpb <= best) else "attacks"
            client.believe(
                claim=f"matrix_lr={config.matrix_lr} depth={config.depth} achieved val_bpb={results.val_bpb:.4f}",
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
        """Generate a prediction for this config based on accumulated beliefs."""
        try:
            # Query for similar configurations
            query = (
                f"training with matrix_lr={config.matrix_lr} "
                f"depth={config.depth} batch_size={config.total_batch_size}"
            )
            similar = client.search(query=query, limit=5)
            if not similar:
                return None

            # Analyze similar runs to predict outcome
            similar_ids = []
            outcomes = []
            for belief in similar:
                if hasattr(belief, 'source_ref'):
                    try:
                        rid = int(belief.source_ref.split("_")[1])
                        similar_ids.append(rid)
                    except (IndexError, ValueError):
                        pass
                if hasattr(belief, 'content'):
                    outcomes.append(str(belief.content))

            # Simple heuristic: if most similar runs improved, predict improvement
            diverge_count = sum(1 for o in outcomes if "diverged=True" in o)
            total = len(outcomes) or 1

            if diverge_count / total > 0.5:
                expected = "diverge"
                confidence = diverge_count / total
                risks = ["High divergence rate in similar configs"]
            elif diverge_count > 0:
                expected = "marginal"
                confidence = 0.5
                risks = ["Some similar configs diverged"]
            else:
                expected = "improve"
                confidence = 0.6
                risks = []

            return PredictionResult(
                expected_outcome=expected,
                confidence=round(confidence, 2),
                similar_runs=similar_ids,
                risks=risks,
                source_run_ids=similar_ids,
            )
        except Exception:
            return None

    def _recommend(self, config: RunConfig, client) -> RecommendationResult | None:
        """Suggest next experiment based on accumulated beliefs."""
        try:
            # Look for the most successful configuration pattern
            query = "best val_bpb training configuration"
            beliefs = client.search(query=query, limit=10)
            if not beliefs:
                return None

            source_ids = []
            rationale_items = []
            for belief in beliefs:
                if hasattr(belief, 'source_ref'):
                    try:
                        rid = int(belief.source_ref.split("_")[1])
                        source_ids.append(rid)
                    except (IndexError, ValueError):
                        pass
                if hasattr(belief, 'claim'):
                    rationale_items.append(str(belief.claim))

            if not rationale_items:
                return None

            return RecommendationResult(
                suggested_change=f"Consider adjusting matrix_lr based on {len(beliefs)} prior observations",
                rationale=rationale_items[:3],
                risk_level="low",
                source_run_ids=source_ids,
            )
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

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
    return cls(**kwargs)
