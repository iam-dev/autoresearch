# Condition A Summary (No Memory)

| Metric | Value |
|---|---|
| Total runs | 13 |
| Best val_bpb | 2.733 |
| Mean val_bpb | 2.821 |
| Std val_bpb | 0.120 |
| Wasted runs | 5 (38.5%) |
| Diverged runs | 0 |
| Runs to threshold | 12 |
| Improvement | 3.005 → 2.733 (9.1% reduction) |

## val_bpb trajectory

3.005 → 3.131 → 2.890 → 2.766 → 2.784 → 2.780 → 2.817 → 2.753 → 2.755 → 2.756 → 2.754 → 2.733 → 2.745

## Insight

The 38.5% wasted run rate is the key metric for the MnemeBrain experiment. Condition A (no memory) serves as the control — the agent has no structured memory of what worked/failed. Conditions B/C/D will be compared against this to measure whether structured logging, belief memory, or active recommendations reduce the wasted run rate and reach good configs faster. The `runs_to_threshold` of 12 (out of 13) and the non-monotonic trajectory show the agent exploring without guidance.
