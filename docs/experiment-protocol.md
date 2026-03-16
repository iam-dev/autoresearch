# Experiment Protocol: MnemeBrain Controlled Experiment

> Pre-registration-grade protocol for the MnemeBrain controlled experiment.
> Reference: `docs/plan-mnemebrain-conditions.md` for condition details.

## 1. Hypotheses

- **H0 (sanity check):** Condition A (no memory) produces results no worse than random hyperparameter search over the same budget.
- **H1 (primary):** Condition C (passive MnemeBrain) achieves lower best val_bpb than Condition B (structured logging) within the same run budget, indicating that belief-structured memory provides value beyond structured note-taking.
- **H2 (secondary):** Condition D (active MnemeBrain) achieves lower best val_bpb than Condition C, indicating that prediction and recommendation provide additional value beyond passive belief surfacing.
- **H3 (exploratory):** Condition D produces a lower wasted-run rate than Conditions A-C, indicating that active guidance helps avoid unproductive experiments.

## 2. Design

- **Conditions:** 4 (A=NullHooks, B=LoggingHooks, C=PassiveHooks, D=ActiveHooks)
- **Seeds:** 8 independent random seeds per condition
- **Runs per seed-condition pair:** 10
- **Total runs:** 4 x 8 x 10 = 320
- **Run budget:** 5 minutes wall-clock per run (enforced by training script)
- **Assignment:** Latin square design — each seed cycles through conditions in a balanced order to control for learning effects across the session
- **Blinding:** Condition labels are randomized per session to reduce operator bias

## 3. Primary Metrics

| Metric | Definition | Unit |
|--------|-----------|------|
| Best val_bpb | Lowest validation bits-per-byte across all 10 runs for a seed-condition pair | bpb |
| Runs-to-threshold | Number of runs needed to reach a pre-defined val_bpb threshold (set after pilot) | count |
| Wasted-run rate | Fraction of runs flagged as wasted by `_is_wasted()` | proportion |

## 4. Statistical Analysis Plan

### Primary test

Wilcoxon signed-rank test (paired by seed) for each pairwise comparison:
- B vs C (H1)
- C vs D (H2)
- A vs D (exploratory)

Effect size: matched-pairs rank-biserial correlation coefficient.

### Robustness check

Paired t-test + Cohen's d_z for each comparison, reported alongside the non-parametric results.

### Correction

Holm-Bonferroni correction across the 3 pairwise tests. Family-wise alpha = 0.05, two-tailed.

## 5. Power Analysis

Under a paired parametric approximation, n=8 seeds gives approximately 80% power to detect an effect size of d=0.8 (large effect) at alpha=0.05 two-tailed. This is an estimate; if pilot effect sizes are smaller, document the required n for 80% power and consider increasing the seed count before the full experiment.

## 6. Validity Threats and Mitigations

| Threat | Mitigation |
|--------|-----------|
| **Learning confound** — operator improves over time regardless of condition | Latin square: each seed cycles through conditions in balanced order |
| **Operator bias** — knowing the condition influences choices | Blinded labels: condition names randomized per session |
| **Hardware variability** — GPU thermals, background processes | Same machine, same GPU, minimal background load |
| **Seed sensitivity** — some seeds may be inherently easier | Minimum 8 seeds; paired analysis controls for per-seed difficulty |
| **Belief store contamination** — cross-seed information leakage | Separate belief DB per condition x seed pair. Each seed-condition gets its own isolated belief store directory, preventing cross-seed contamination in Conditions C and D |
| **Software version drift** — mnemebrain-lite updates during experiment | Pin mnemebrain-lite version in pyproject.toml; record version in run metadata |
| **Time-of-day effects** — thermal throttling, fatigue | Randomize run scheduling within sessions |

## 7. Stopping Rules

- **Pilot phase:** 40 runs (1 seed x 4 conditions x 10 runs). Used to calibrate the val_bpb threshold for runs-to-threshold metric and estimate effect sizes for power analysis.
- **Full experiment:** 320 runs (8 seeds x 4 conditions x 10 runs).
- **No early stopping:** All planned runs are executed regardless of interim results. The experiment is designed to be analyzed only after all runs complete.

## 8. Reproduction Protocol

### Environment setup

```bash
# Clone and set up
git clone <repo-url> && cd autoresearch
uv sync --extra dev

# Pin mnemebrain-lite version
uv pip install mnemebrain-lite[embeddings]==<version>
```

### Running a single condition-seed pair

```bash
export AUTORESEARCH_CONDITION=C
export AUTORESEARCH_SEED=42

for run in $(seq 1 10); do
    uv run python train.py --seed $AUTORESEARCH_SEED
done
```

### Directory structure

```
results/
  runs/
    condition_A/
      seed_01/
        run_001.json
        run_002.json
        ...
      seed_02/
        ...
    condition_B/
      ...
    condition_C/
      ...
    condition_D/
      ...
  logs/
    condition_A/
      seed_01/
        run_001.log
        ...
  beliefs/  # Condition C: one DB per condition x seed
    condition_C_seed_01/
    condition_C_seed_02/
    ...
  experiment_log.jsonl  # Condition B
```

### Analysis

```bash
uv run python results_analyzer.py --results-dir results/
```
