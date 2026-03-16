# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

This project is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## [Unreleased]

### Added

- **MnemeBrain experiment framework** — 4 experimental conditions (A/B/C/D) for controlled comparison of structured belief memory in autonomous ML experimentation
  - Condition A (NullHooks): no memory baseline, writes run result JSON for analysis only
  - Condition B (LoggingHooks): structured JSONL logging with recent summaries printed before each run
  - Condition C (PassiveHooks): mnemebrain-lite belief memory with Belnap four-valued logic, contradiction surfacing, and similar-run retrieval
  - Condition D (ActiveHooks): full mnemebrain with pre-run predictions and parameter recommendations
- `mnemebrain_hooks.py` — ExperimentHooks protocol and all 4 condition implementations with typed data contracts (RunConfig, RunResults, PreRunContext, PredictionResult, RecommendationResult)
- `results_analyzer.py` — generates per-condition summaries, cross-condition comparisons (Cohen's d effect sizes), pilot promotion criteria evaluation, and matplotlib plots
- `results/` directory structure for organized experiment artifacts (per-condition, per-seed run JSONs, logs, summaries, comparisons, plots)
- Evidence weight formula using log-scale tanh for Belnap belief polarity (improvements get high weight, regressions get low weight)
- Wasted-run detection with exception for deliberate ablation probes (rationale_tag + meaningful parameter step)
- Config distance metric (L2 in normalized parameter space) for near-bad-config detection
- Log capture system that tees stdout/stderr to per-run log files during training
- `AUTORESEARCH_CONDITION` environment variable to select experiment condition (A/B/C/D)
- `AUTORESEARCH_SEED` environment variable to set the random seed
- `tests/test_hooks.py` — unit tests for hook implementations
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
- `persona.md` — ML researcher persona for autonomous agent sessions
- `launch.sh` — convenience launcher with persona injection and autonomous mode
- `Dockerfile` and `docker-compose.yml` for containerized training

### Changed

- `train.py` — added MnemeBrain hook integration (~15 lines): imports hooks, creates RunConfig before training, calls `pre_run` before the loop and `post_run` after evaluation, captures diverged runs before exit
- `train.py` — gradient norm sampled every 50 steps for telemetry; loss trend derived from final-20% comparison; SEED extracted as configurable constant
- `train.py` — platform-aware setup: autocast context, device selection, and compile gating based on `platform_config.py`
- `pyproject.toml` — added optional dependencies for experiment conditions (`mnemebrain-lite[embeddings]`, `mnemebrain`)
