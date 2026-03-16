# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This project is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## [Unreleased]

### Added

- **`hooks/claims.py`** — new `StructuredClaim` and `ClaimBuilder` for typed, composable belief store queries at multiple granularities (single-param, multi-param, context-aware, stability)
- **`docs/experiment-protocol.md`** — pre-registration-grade experiment protocol covering hypotheses, design (Latin square, 8 seeds, 320 runs), statistical analysis plan (Wilcoxon signed-rank, Holm-Bonferroni correction), power analysis, validity threats, and reproduction protocol
- **`docs/decision-policy.md`** — documents how Condition C (passive/information-only) and Condition D (active/advisory) use beliefs to influence decisions, with explicit advisory framing and operator boundary guarantees

### Changed

- **Hooks refactored from monolith to package** — split 905-line `mnemebrain_hooks.py` into 11 focused modules under `hooks/`:
  - `hooks/types.py` — all dataclasses (RunConfig, RunResults, PredictionResult, RecommendationResult, PreRunContext)
  - `hooks/base.py` — ExperimentHooks Protocol
  - `hooks/log_capture.py` — standalone _TeeWriter and LogCapture (no project imports)
  - `hooks/artifacts.py` — RESULTS_DIR, run result I/O (pure I/O, no analysis deps)
  - `hooks/analysis.py` — evidence_weight, config distance, wasted-run detection
  - `hooks/claims.py` — StructuredClaim, ClaimBuilder (new)
  - `hooks/condition_a.py` — NullHooks
  - `hooks/condition_b.py` — LoggingHooks
  - `hooks/condition_c.py` — PassiveHooks (now uses ClaimBuilder for belief queries)
  - `hooks/condition_d.py` — ActiveHooks (strengthened: evidence-weight-based prediction, specific parameter recommendations)
  - `hooks/__init__.py` — public API, create_hooks() factory, __all__
- `mnemebrain_hooks.py` — replaced with thin backward-compatibility shim (`from hooks import *`)
- Condition D `_predict()` — now uses evidence-weight ratios instead of simple count-based heuristics
- Condition D `_recommend()` — generates specific parameter adjustment suggestions based on belief store evidence
- LogCapture API — `start()` now takes `(log_dir, run_id)` instead of `(condition, seed, run_id)`, making it fully standalone
- `_write_run_result()` — accepts explicit `wasted` parameter instead of computing it internally, avoiding circular deps between artifacts and analysis
- Tests split from single `test_hooks.py` into 6 focused files mirroring package structure: `test_hooks_types.py`, `test_hooks_log_capture.py`, `test_hooks_artifacts.py`, `test_hooks_analysis.py`, `test_hooks_claims.py`, `test_hooks_conditions.py`
- Shared test fixtures (sample_config, sample_results, diverged_results, tmp_results) moved to `tests/conftest.py`
- `tests/test_e2e_mnemebrain.py` — updated imports and monkeypatch target to `hooks.artifacts.RESULTS_DIR`
- Repo-wide ruff lint cleanup (import sorting, unused imports, style fixes)
- `scripts/` — moved `launch.sh` and `persona.md` from project root

---

## [0.1.0] - 2026-03-16

### Added

- **MnemeBrain experiment framework** — 4 experimental conditions (A/B/C/D) for controlled comparison of structured belief memory in autonomous ML experimentation
  - Condition A (NullHooks): no memory baseline, writes run result JSON for analysis only
  - Condition B (LoggingHooks): structured JSONL logging with recent summaries printed before each run
  - Condition C (PassiveHooks): mnemebrain-lite belief memory with Belnap four-valued logic, contradiction surfacing, and similar-run retrieval
  - Condition D (ActiveHooks): full mnemebrain with pre-run predictions and parameter recommendations
- `mnemebrain_hooks.py` — ExperimentHooks protocol and all 4 condition implementations with typed data contracts
- `results_analyzer.py` — generates per-condition summaries, cross-condition comparisons (Cohen's d effect sizes), pilot promotion criteria evaluation, and matplotlib plots
- `results/` directory structure for organized experiment artifacts (per-condition, per-seed run JSONs, logs, summaries, comparisons, plots)
- Evidence weight formula using log-scale tanh for Belnap belief polarity (improvements get high weight, regressions get low weight)
- Wasted-run detection with exception for deliberate ablation probes (rationale_tag + meaningful parameter step)
- Config distance metric (L2 in normalized parameter space) for near-bad-config detection
- Log capture system that tees stdout/stderr to per-run log files during training
- `AUTORESEARCH_CONDITION` environment variable to select experiment condition (A/B/C/D)
- `AUTORESEARCH_SEED` environment variable to set the random seed
- `tests/test_hooks.py` — unit tests for hook implementations (later split into 6 files)
- `tests/test_results_analyzer.py` — unit tests for the results analyzer
- Experimental design document at `docs/plan-mnemebrain-conditions.md` (conditions, protocols, belief schema, outcome metrics, counterbalancing)
- Sidecar documentation at `docs/mnemebrain-sidecar.md`
- **Multi-platform support** — automatic detection of CUDA, MPS (Apple Silicon), and CPU backends
  - `platform_config.py` — hardware detection singleton with recommended defaults per platform
  - `configs/cuda.toml`, `configs/mps.toml`, `configs/cpu.toml` — platform-specific training recipes
  - MPS compatibility fixes for training on Apple Silicon
- **Model registry** — pluggable model architecture system
  - `models/__init__.py` — registry with `create_model()` factory
  - `models/base.py` — TrainableModel protocol
  - `models/nanochat.py` — default GPT (RoPE, GQA, value embeddings, Muon optimizer)
  - `models/gpt2.py` — standard GPT-2 variant (LayerNorm, GELU, AdamW)
  - `AUTORESEARCH_MODEL` environment variable to select architecture
- `scripts/persona.md` — ML researcher persona for autonomous agent sessions
- `scripts/launch.sh` — convenience launcher with persona injection and autonomous mode
- `Dockerfile` and `docker-compose.yml` for containerized training

### Changed

- `train.py` — added MnemeBrain hook integration (~15 lines): imports hooks, creates RunConfig before training, calls `pre_run` before the loop and `post_run` after evaluation, captures diverged runs before exit
- `train.py` — gradient norm sampled every 50 steps for telemetry; loss trend derived from final-20% comparison; SEED extracted as configurable constant
- `train.py` — platform-aware setup: autocast context, device selection, and compile gating based on `platform_config.py`
- `pyproject.toml` — added optional dependencies for experiment conditions (`mnemebrain-lite[embeddings]`, `mnemebrain`)
