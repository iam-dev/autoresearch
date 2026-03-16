# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069).

> **This is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch)** that adds **MnemeBrain** — a structured belief memory system for controlled ML experimentation. See [MnemeBrain Experiment Framework](#mnemebrain-experiment-framework) below.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), and one of:
- **Linux** with an NVIDIA GPU (tested on H100) — full performance
- **macOS Apple Silicon** (M1/M2/M3/M4) — MPS backend, reduced scale
- **macOS Intel / Linux CPU** — CPU fallback, for development and testing

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies (platform-detected automatically)
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py

# Optional: select a different model architecture
AUTORESEARCH_MODEL=gpt2 uv run train.py

# Optional: override platform defaults
AUTORESEARCH_DEPTH=4 AUTORESEARCH_DEVICE_BATCH=8 uv run train.py
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

See [docs/platform-support.md](docs/platform-support.md) for detailed platform information and [docs/model-selection.md](docs/model-selection.md) for available model architectures.

## Running the agent

The easiest way to launch is with the included `launch.sh` script, which activates the ML researcher persona automatically:

```bash
# Interactive mode — opens Claude Code with the researcher persona
./launch.sh

# One-shot mode — pass a prompt directly
./launch.sh "Hi have a look at program.md and let's kick off a new experiment! let's do the setup first."
```

The script uses `--append-system-prompt-file` to inject `persona.md` (ML researcher identity, experiment prioritization, anti-patterns) on top of the default Claude Code system prompt. It also enables `--allow-dangerously-skip-permissions` so the agent can run autonomously without approval prompts.

Alternatively, spin up Claude/Codex manually in this repo and prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

---

## MnemeBrain Experiment Framework

This fork adds a controlled experiment framework to answer the question: **does structured belief memory improve autonomous ML experimentation?**

The core claim is not "MnemeBrain improves model training" but rather: **"Structured belief memory reduces wasted experiments and reaches good configurations faster than no memory, plain logs, or passive memory alone."**

The full experimental design is documented in [docs/plan-mnemebrain-conditions.md](docs/plan-mnemebrain-conditions.md).

### The 4 experimental conditions

| Condition | Name | What it does |
|-----------|------|-------------|
| **A** | No Memory | Plain `train.py`. No sidecar, no history, no suggestions. Raw human baseline. |
| **B** | Structured Logging | Same training, but post-run telemetry stored as structured JSON. Recent summaries printed before the next run. Controls for "more observability." |
| **C** | Passive Belief Memory | [mnemebrain-lite](https://pypi.org/project/mnemebrain-lite/) stores beliefs across runs using Belnap four-valued logic (TRUE, FALSE, BOTH, NEITHER). Surfaces contradictions, confidence scores, and similar prior runs. Does NOT suggest parameter changes. |
| **D** | Active Prediction & Recommendation | Everything in C, plus pre-run predictions ("this config will likely diverge") and recommendations ("reduce lr from 0.03 to 0.02 based on 4 similar runs"). |

**Critical design rule:** All conditions run identical training code. MnemeBrain only influences what happens BETWEEN runs — never during training. Once `uv run train.py` starts, it runs identically regardless of condition. This isolates the memory effect from optimizer effects.

### What the comparisons isolate

| Comparison | What it tells you |
|------------|------------------|
| A vs B | Does telemetry/observability alone help? |
| B vs C | Does structured belief memory beat plain logs? |
| C vs D | Does active reasoning beat passive memory? |
| A vs D | Total effect of the full system |

### Running each condition

Set the `AUTORESEARCH_CONDITION` environment variable:

```bash
# Condition A — no memory (default if unset)
AUTORESEARCH_CONDITION=A uv run train.py

# Condition B — structured logging
AUTORESEARCH_CONDITION=B uv run train.py

# Condition C — passive belief memory (requires mnemebrain-lite)
pip install mnemebrain-lite[embeddings]
AUTORESEARCH_CONDITION=C uv run train.py

# Condition D — active prediction/recommendation (requires mnemebrain server)
pip install mnemebrain-lite[embeddings] mnemebrain
CUDA_VISIBLE_DEVICES="" mnemebrain &    # start sidecar on CPU only, port 8000
AUTORESEARCH_CONDITION=D uv run train.py
```

Conditions A and B have no external dependencies. Condition C uses `mnemebrain-lite` as a direct Python library (no server needed). Condition D requires the mnemebrain-lite API server running as a sidecar.

For Intel Mac users, use `mnemebrain-lite[openai]` instead of `[embeddings]` since PyTorch no longer ships x86_64 macOS wheels for sentence-transformers.

See [docs/mnemebrain-sidecar.md](docs/mnemebrain-sidecar.md) for detailed setup instructions.

### Results directory

All experiment results are written to `results/` (git-ignored):

```
results/
├── runs/                          # Individual run results
│   ├── condition_A/
│   │   ├── seed_01/
│   │   │   ├── run_001.json       # Full run result (config + results + telemetry)
│   │   │   ├── run_002.json
│   │   │   └── ...
│   │   └── seed_02/
│   ├── condition_B/
│   ├── condition_C/
│   └── condition_D/
├── logs/                          # Raw training stdout/stderr per run
├── experiment_log.jsonl           # Condition B append-only structured log
├── summaries/                     # Per-condition aggregate summaries (auto-generated)
├── comparisons/                   # Cross-condition analysis
└── plots/                         # Visualization outputs
```

Each run produces a single JSON file with the full config, results (val_bpb, steps, diverged, loss trend, grad norm), wasted-run classification, and pre-run context for that condition.

### Analyzing results

Use `results_analyzer.py` to generate summaries, cross-condition comparisons, and plots:

```bash
# Generate all summaries + comparison
python results_analyzer.py

# Single condition summary
python results_analyzer.py --condition B

# Cross-condition comparison (pilot phase)
python results_analyzer.py --compare --phase pilot

# Set a fixed threshold for runs-to-threshold metric
python results_analyzer.py --threshold 1.85

# Generate plots (requires matplotlib)
python results_analyzer.py --plot
```

The analyzer computes: best/mean/std val_bpb, wasted-run rate, improvement rate, runs-to-threshold, val_bpb trajectory, and pairwise Cohen's d effect sizes. Pilot promotion criteria (hook stability, telemetry completeness, directional separation, effect size > 0.3) are evaluated automatically.

### Key files (MnemeBrain additions)

```
mnemebrain_hooks.py    — ExperimentHooks protocol + 4 condition implementations (NullHooks, LoggingHooks, PassiveHooks, ActiveHooks)
results_analyzer.py    — Generates summaries, comparisons, and plots from run JSONs
tests/test_hooks.py    — Unit tests for hook implementations
tests/test_results_analyzer.py — Unit tests for the analyzer
docs/plan-mnemebrain-conditions.md — Full experimental design (conditions, protocols, belief schema, metrics)
docs/mnemebrain-sidecar.md — mnemebrain-lite setup and troubleshooting
```

---

## Project structure

```
prepare.py          — constants, data prep + runtime utilities (do not modify)
train.py            — training loop + hyperparameters (agent modifies this)
program.md          — agent instructions
platform_config.py  — auto-detects CUDA/MPS/CPU, sets recommended defaults
mnemebrain_hooks.py — experiment hooks (A/B/C/D conditions)
results_analyzer.py — results analysis and plotting
configs/            — platform-specific training recipes (cuda.toml, mps.toml, cpu.toml)
models/             — model architectures (nanochat, gpt2)
  __init__.py       — registry + create_model() factory
  base.py           — TrainableModel protocol
  nanochat.py       — default GPT (RoPE, GQA, value embeddings, Muon optimizer)
  gpt2.py           — standard GPT-2 variant (LayerNorm, GELU, AdamW)
tests/              — unit tests (run with: uv sync --extra dev && uv run pytest tests/ -m unit)
docs/               — platform support, model selection, experiment design, sidecar setup
persona.md          — persona activation primer for autonomous research
Dockerfile          — containerized training (CUDA)
docker-compose.yml  — multi-service Docker setup
launch.sh           — convenience launcher script
pyproject.toml      — dependencies (platform-conditional)
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.
- **Between-run only.** MnemeBrain hooks execute before and after training, never during. The training loop is identical across all conditions.

## Platform support

Autoresearch now runs on multiple platforms. Platform detection is automatic via `platform_config.py`:

| Platform | Device | Flash Attention | torch.compile | Recommended for |
|---|---|---|---|---|
| Linux + NVIDIA GPU | CUDA | Yes (Hopper+) | Yes | Production training |
| macOS Apple Silicon | MPS | No (SDPA fallback) | No | Development, small experiments |
| macOS Intel / CPU | CPU | No (SDPA fallback) | No | Testing, CI |

Training hyperparameters (batch size, learning rates, window pattern, etc.) are loaded automatically from `configs/{platform}.toml` — CUDA gets H100-class defaults, MPS gets Apple Silicon defaults (from the [miolini fork](https://github.com/miolini/autoresearch-macos)), CPU gets minimal test settings. Override via environment variables:

```bash
AUTORESEARCH_DEPTH=4 AUTORESEARCH_DEVICE_BATCH=8 uv run train.py
```

See [docs/platform-support.md](docs/platform-support.md) for full details.

### Tips for smaller compute

For Macbooks and small GPUs, the MPS/CPU config files already set conservative defaults. You can further tune by:

1. Use a dataset with less entropy, e.g. [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean).
2. Decrease `vocab_size` (4096, 2048, 1024, or even byte-level 256).
3. Lower `MAX_SEQ_LEN` in `prepare.py` (down to 256 etc.) and increase `DEVICE_BATCH_SIZE` to compensate.
4. Decrease `EVAL_TOKENS` in `prepare.py` for faster validation.
5. Lower `DEPTH` (the platform auto-detects a reasonable default).
6. Edit `configs/mps.toml` or `configs/cpu.toml` to change defaults permanently.

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)

## License

MIT
