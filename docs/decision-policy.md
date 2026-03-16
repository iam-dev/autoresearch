# Decision Policy: How Beliefs Influence Experimental Choices

> Documents how each condition uses (or doesn't use) accumulated knowledge to inform hyperparameter decisions.

## Condition C — Passive (Information Only)

Condition C surfaces information but has **no algorithmic decision policy**. The operator makes all choices.

### What C provides before each run

- **Similar prior runs:** Semantic similarity search finds runs with similar configs
- **Contradictions:** Belnap four-valued logic identifies parameters with conflicting evidence (e.g., "matrix_lr=0.04 has both supporting and attacking evidence")
- **Evidence strength:** Weighted confidence from `evidence_weight()` — improvements get weight > 0.5, regressions get weight < 0.5

### What C does NOT do

- Suggest parameter changes
- Rank candidate configurations
- Predict outcomes
- Modify any training parameters

The operator reads the surfaced information and decides what to try next. This isolates the value of structured belief memory from the value of algorithmic guidance.

## Condition D — Active (Semi-Structured Advisory)

Condition D adds prediction and recommendation on top of C's belief surfacing. The pipeline:

### 1. Claim generation (`ClaimBuilder.query_claims()`)

Before each run, generates multi-granularity query claims:
- Single-parameter: `matrix_lr=0.04 improves`
- Multi-parameter: `matrix_lr=0.04 depth=4 improves`
- Context-aware: includes `warmup_ratio` when non-zero
- Stability: `matrix_lr=0.04 depth=4 stabilized`

### 2. Prediction (`_predict()`)

Uses evidence-weight ratios (not simple counts) to predict the outcome:
- Queries the belief store with each claim
- Accumulates `diverge_weight` and `improve_weight` from evidence
- Thresholds: diverge ratio > 0.4 predicts "diverge", > 0.15 predicts "marginal", else "improve"
- The 0.4 threshold is a **provisional pilot heuristic**, not a theory-backed constant. It is expected to be calibrated during the pilot phase.
- Returns confidence (capped at 0.95), similar run IDs, and risk factors

### 3. Recommendation (`_recommend()`)

Generates specific parameter adjustment suggestions:
- Queries multi-param claims for the best evidence
- Extracts successful config parameters from belief content and claim strings
- Computes average successful `matrix_lr` and suggests adjustment direction
- Note: still relies partially on `belief.content` and `belief.claim` string inspection for extracting parameter values. This is semi-structured, reducing but not eliminating stringly-typed logic.

### 4. Risk escalation

- **Gradient instability:** Flagged when `grad_norm_unstable` appears in similar runs
- **No warmup near divergence:** Flagged when similar diverged configs had `warmup_ratio=0`
- **Contradictions:** Risk level escalated to "medium" when both diverged and successful runs exist for similar configs

## Advisory Framing (Critical)

**Recommendations are advisory and NEVER mutate `train.py` or the live run automatically.** The operator makes all final decisions. This is essential to maintain the "between-run only" experimental contract.

### Operator boundary

No Condition D code path:
- Modifies hyperparameters
- Starts training runs
- Alters training behavior
- Writes to any training configuration file

The separation between suggestion and execution is absolute. Condition D outputs text to stdout; the operator reads it and decides. This design ensures that the experiment measures the value of *informed human decisions*, not autonomous hyperparameter optimization.

### Why this matters

If Condition D were allowed to automatically adjust parameters, the experiment would conflate two effects:
1. The value of belief-structured memory for informing decisions
2. The value of automated hyperparameter search

By keeping the operator in the loop, we isolate effect (1), which is the core research question.
