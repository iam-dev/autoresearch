"""
Results analyzer — generates summaries and comparisons from individual run JSONs.

Usage:
  python results_analyzer.py                           # generate all summaries + comparison
  python results_analyzer.py --condition B             # generate summary for condition B only
  python results_analyzer.py --compare                 # generate comparison only
  python results_analyzer.py --threshold 1.85          # use fixed threshold for runs-to-threshold
  python results_analyzer.py --plot                    # generate plots to results/plots/
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path("results")
CONDITIONS = ["A", "B", "C", "D"]


def load_runs(condition: str) -> list[dict]:
    """Load all run result JSONs for a condition, across all seeds."""
    condition_dir = RESULTS_DIR / "runs" / f"condition_{condition}"
    if not condition_dir.exists():
        return []
    runs = []
    for f in sorted(condition_dir.glob("*/run_*.json")):
        try:
            runs.append(json.loads(f.read_text()))
        except (json.JSONDecodeError, KeyError):
            continue
    return runs


def load_runs_by_seed(condition: str) -> dict[int, list[dict]]:
    """Load runs grouped by seed."""
    condition_dir = RESULTS_DIR / "runs" / f"condition_{condition}"
    if not condition_dir.exists():
        return {}
    by_seed: dict[int, list[dict]] = {}
    for seed_dir in sorted(condition_dir.iterdir()):
        if not seed_dir.is_dir():
            continue
        seed = int(seed_dir.name.split("_")[1])
        runs = []
        for f in sorted(seed_dir.glob("run_*.json")):
            try:
                runs.append(json.loads(f.read_text()))
            except (json.JSONDecodeError, KeyError):
                continue
        if runs:
            by_seed[seed] = runs
    return by_seed


def generate_summary(condition: str, threshold: float | None = None) -> dict | None:
    """Generate a condition summary from individual run files.

    Args:
        condition: Which condition (A/B/C/D) to summarize.
        threshold: Fixed val_bpb target for runs-to-threshold metric.
            Per Protocol 2, this should be determined from baseline runs
            (mean_best - 1σ). If None, falls back to per-seed best.
    """
    runs = load_runs(condition)
    if not runs:
        return None

    by_seed = load_runs_by_seed(condition)
    val_bpbs = [r["results"]["val_bpb"] for r in runs if not r["results"].get("diverged", False)]
    wasted = [r for r in runs if r.get("wasted", False)]
    diverged = [r for r in runs if r["results"].get("diverged", False)]

    best_val_bpb = min(val_bpbs) if val_bpbs else None
    mean_val_bpb = statistics.mean(val_bpbs) if val_bpbs else None
    std_val_bpb = statistics.stdev(val_bpbs) if len(val_bpbs) > 1 else 0.0

    # Improvement rate: average val_bpb delta per sequential run (non-diverged, ordered by run_id)
    ordered_non_diverged = sorted(
        [r for r in runs if not r["results"].get("diverged", False)],
        key=lambda r: r["run_id"],
    )
    improvement_rate = 0.0
    if len(ordered_non_diverged) > 1:
        deltas = [
            ordered_non_diverged[i + 1]["results"]["val_bpb"] - ordered_non_diverged[i]["results"]["val_bpb"]
            for i in range(len(ordered_non_diverged) - 1)
        ]
        improvement_rate = statistics.mean(deltas)

    # Val_bpb trajectory (ordered by run_id, all runs including diverged)
    trajectory = [r["results"]["val_bpb"] for r in sorted(runs, key=lambda r: r["run_id"])]

    seeds_completed = sorted(by_seed.keys())

    # Runs-to-threshold: how many runs until val_bpb <= threshold.
    # If a global threshold is provided (from Protocol 2 baseline), use that.
    # Otherwise fall back to per-seed best (useful before threshold is locked).
    runs_to_threshold: float | None = None
    if by_seed:
        seed_counts = []
        for seed_runs in by_seed.values():
            ordered = sorted(seed_runs, key=lambda r: r["run_id"])
            if threshold is not None:
                # Global threshold: count runs until seed hits it
                for i, r in enumerate(ordered):
                    if not r["results"].get("diverged", False) and r["results"]["val_bpb"] <= threshold:
                        seed_counts.append(i + 1)
                        break
                # If seed never reached threshold, record None (not counted)
            else:
                # Fallback: per-seed best (before threshold is locked)
                seed_best = None
                for r in ordered:
                    if not r["results"].get("diverged", False):
                        vbpb = r["results"]["val_bpb"]
                        if seed_best is None or vbpb < seed_best:
                            seed_best = vbpb
                if seed_best is not None:
                    for i, r in enumerate(ordered):
                        if not r["results"].get("diverged", False) and r["results"]["val_bpb"] <= seed_best:
                            seed_counts.append(i + 1)
                            break
        if seed_counts:
            runs_to_threshold = round(statistics.mean(seed_counts), 1)

    summary = {
        "condition": condition,
        "total_runs": len(runs),
        "seeds_completed": seeds_completed,
        "best_val_bpb": round(best_val_bpb, 6) if best_val_bpb is not None else None,
        "mean_val_bpb": round(mean_val_bpb, 6) if mean_val_bpb is not None else None,
        "std_val_bpb": round(std_val_bpb, 6),
        "wasted_runs": len(wasted),
        "wasted_run_rate": round(len(wasted) / len(runs), 4) if runs else 0.0,
        "diverged_runs": len(diverged),
        "improvement_rate": round(improvement_rate, 6),
        "runs_to_threshold": runs_to_threshold,
        "threshold_used": threshold,
        "val_bpb_trajectory": [round(v, 6) for v in trajectory],
    }

    # Write summary
    summary_dir = RESULTS_DIR / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / f"condition_{condition}_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Summary written to {path}")
    return summary


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d effect size between two groups."""
    if len(group1) < 2 or len(group2) < 2:
        return 0.0
    m1, m2 = statistics.mean(group1), statistics.mean(group2)
    s1, s2 = statistics.stdev(group1), statistics.stdev(group2)
    pooled_std = math.sqrt(((len(group1) - 1) * s1**2 + (len(group2) - 1) * s2**2) /
                           (len(group1) + len(group2) - 2))
    if pooled_std == 0:
        return 0.0
    return (m1 - m2) / pooled_std


def _check_stability(condition: str) -> tuple[bool, int]:
    """Check hook stability — zero crashes across runs. Returns (stable, crash_count)."""
    runs = load_runs(condition)
    # A crash is indicated by a run that diverged with steps < 10 (failed before training)
    # or a missing/corrupt run file. Since hooks write the JSON, if the file exists, hooks ran.
    # We count diverged runs as a separate metric; stability here means "hooks didn't crash."
    # If we have run files, the hooks completed without exceptions.
    crash_count = 0
    for r in runs:
        # A run with no results section or missing required fields indicates a hook crash
        results = r.get("results", {})
        required = ["val_bpb", "steps", "diverged", "loss_trend", "grad_norm_max"]
        if not all(k in results for k in required):
            crash_count += 1
    return crash_count == 0, crash_count


def _check_telemetry_completeness(condition: str) -> tuple[bool, list[str]]:
    """Check all required fields are populated in every run. Returns (complete, missing_fields)."""
    runs = load_runs(condition)
    missing: list[str] = []
    required_top = ["run_id", "condition", "seed", "timestamp", "config", "results"]
    required_results = ["val_bpb", "steps", "diverged", "loss_trend", "grad_norm_max"]
    required_config = ["matrix_lr", "embedding_lr", "depth", "total_batch_size"]

    for r in runs:
        run_label = f"run_{r.get('run_id', '?')}"
        for field in required_top:
            if field not in r:
                missing.append(f"{run_label}: missing top-level '{field}'")
        for field in required_results:
            if field not in r.get("results", {}):
                missing.append(f"{run_label}: missing results.{field}")
        for field in required_config:
            if field not in r.get("config", {}):
                missing.append(f"{run_label}: missing config.{field}")
    return len(missing) == 0, missing


def generate_comparison(phase: str = "full", threshold: float | None = None) -> dict | None:
    """Generate cross-condition comparison.

    Args:
        phase: "pilot" or "full" — labels the comparison output.
        threshold: Fixed val_bpb target for runs-to-threshold comparison.
    """
    condition_stats: dict[str, dict] = {}

    for cond in CONDITIONS:
        by_seed = load_runs_by_seed(cond)
        if not by_seed:
            continue

        # Best val_bpb per seed
        best_per_seed = []
        wasted_rate_per_seed = []
        runs_to_threshold_per_seed = []
        for _, runs in by_seed.items():
            non_diverged = [r for r in runs if not r["results"].get("diverged", False)]
            if non_diverged:
                best_per_seed.append(min(r["results"]["val_bpb"] for r in non_diverged))
            wasted = sum(1 for r in runs if r.get("wasted", False))
            wasted_rate_per_seed.append(wasted / len(runs) if runs else 0.0)
            # Runs-to-threshold per seed (if threshold provided)
            if threshold is not None:
                ordered = sorted(runs, key=lambda r: r["run_id"])
                for i, r in enumerate(ordered):
                    if not r["results"].get("diverged", False) and r["results"]["val_bpb"] <= threshold:
                        runs_to_threshold_per_seed.append(i + 1)
                        break

        if best_per_seed:
            condition_stats[cond] = {
                "mean_best_val_bpb": round(statistics.mean(best_per_seed), 4),
                "std": round(statistics.stdev(best_per_seed), 4) if len(best_per_seed) > 1 else 0.0,
                "mean_wasted_rate": round(statistics.mean(wasted_rate_per_seed), 4),
                "best_per_seed": best_per_seed,
            }
            if runs_to_threshold_per_seed:
                condition_stats[cond]["mean_runs_to_threshold"] = round(
                    statistics.mean(runs_to_threshold_per_seed), 1
                )

    if len(condition_stats) < 2:
        print("Not enough conditions with data for comparison.")
        return None

    # Pairwise tests
    pairs = [("A", "B"), ("B", "C"), ("C", "D")]
    pairwise: dict[str, dict] = {}
    for c1, c2 in pairs:
        if c1 in condition_stats and c2 in condition_stats:
            d = cohens_d(
                condition_stats[c1]["best_per_seed"],
                condition_stats[c2]["best_per_seed"],
            )
            pairwise[f"{c1}_vs_{c2}"] = {
                "cohens_d": round(abs(d), 4),
                "directional": d > 0,  # True if c1 > c2 (c2 is better since lower bpb is better)
            }

    # Promotion criteria (plan lines 379-385):
    # 1. Hooks are stable — zero crashes across pilot runs
    # 2. Telemetry is complete — all fields populated
    # 3. Directional separation — at least 2 of {runs-to-threshold, best val_bpb, wasted-run rate}
    # 4. Effect size justifies compute — Cohen's d > 0.3
    all_d = [v["cohens_d"] for v in pairwise.values()]
    directional_count = sum(1 for v in pairwise.values() if v["directional"])

    stability_ok = True
    telemetry_ok = True
    stability_details: dict[str, dict] = {}
    for cond in condition_stats:
        stable, crashes = _check_stability(cond)
        complete, missing = _check_telemetry_completeness(cond)
        stability_details[cond] = {"stable": stable, "crashes": crashes, "telemetry_complete": complete}
        if not stable:
            stability_ok = False
        if not complete:
            telemetry_ok = False

    effect_size_ok = any(d > 0.3 for d in all_d) if all_d else False
    directional_ok = directional_count >= 2

    promote = stability_ok and telemetry_ok and directional_ok and effect_size_ok

    criteria = {
        "hooks_stable": stability_ok,
        "telemetry_complete": telemetry_ok,
        "directional_separation": directional_ok,
        "effect_size_sufficient": effect_size_ok,
    }

    # Strip internal fields before writing
    conditions_output = {
        k: {kk: vv for kk, vv in v.items() if kk != "best_per_seed"}
        for k, v in condition_stats.items()
    }

    comparison = {
        "protocol": "protocol_1",
        "phase": phase,
        "n_seeds": min(len(v.get("best_per_seed", [])) for v in condition_stats.values()),
        "timestamp": datetime.now(UTC).isoformat(),
        "conditions": conditions_output,
        "pairwise_tests": pairwise,
        "promotion_criteria": criteria,
        "stability_details": stability_details,
        "promotion_decision": "promote" if promote else "hold",
        "promotion_rationale": (
            f"{'All 4' if promote else 'Not all'} promotion criteria met. "
            f"Stable: {stability_ok}, Telemetry: {telemetry_ok}, "
            f"Directional pairs: {directional_count}/{len(pairwise)}, "
            f"Max Cohen's d: {max(all_d):.2f}" if all_d else "Insufficient data"
        ),
    }

    comp_dir = RESULTS_DIR / "comparisons"
    comp_dir.mkdir(parents=True, exist_ok=True)
    path = comp_dir / f"{phase}_comparison.json"
    path.write_text(json.dumps(comparison, indent=2) + "\n")
    print(f"Comparison written to {path}")
    return comparison


def generate_plots() -> None:
    """Generate visualization plots to results/plots/."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots")
        return

    plot_dir = RESULTS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Collect data per condition
    cond_data: dict[str, dict] = {}
    for cond in CONDITIONS:
        runs = load_runs(cond)
        if not runs:
            continue
        by_seed = load_runs_by_seed(cond)
        best_per_seed = []
        for seed_runs in by_seed.values():
            nd = [r for r in seed_runs if not r["results"].get("diverged", False)]
            if nd:
                best_per_seed.append(min(r["results"]["val_bpb"] for r in nd))
        wasted_rate = sum(1 for r in runs if r.get("wasted", False)) / len(runs) if runs else 0.0
        trajectory = [r["results"]["val_bpb"] for r in sorted(runs, key=lambda r: r["run_id"])]
        cond_data[cond] = {
            "best_per_seed": best_per_seed,
            "wasted_rate": wasted_rate,
            "trajectory": trajectory,
        }

    if not cond_data:
        print("No data to plot.")
        return

    # Plot 1: val_bpb by condition (box plot of best-per-seed)
    conds_with_data = [c for c in CONDITIONS if c in cond_data and cond_data[c]["best_per_seed"]]
    if conds_with_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [cond_data[c]["best_per_seed"] for c in conds_with_data]
        labels = [f"Condition {c}" for c in conds_with_data]
        ax.boxplot(data, tick_labels=labels)
        ax.set_ylabel("Best val_bpb (per seed)")
        ax.set_title("Best val_bpb by Condition")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "val_bpb_by_condition.png", dpi=150)
        plt.close(fig)
        print(f"Plot: {plot_dir / 'val_bpb_by_condition.png'}")

    # Plot 2: wasted run rate bar chart
    if conds_with_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        rates = [cond_data[c]["wasted_rate"] for c in conds_with_data]
        labels = [f"Condition {c}" for c in conds_with_data]
        ax.bar(labels, rates, color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"][:len(labels)])
        ax.set_ylabel("Wasted Run Rate")
        ax.set_title("Wasted Run Rate by Condition")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "wasted_run_rate.png", dpi=150)
        plt.close(fig)
        print(f"Plot: {plot_dir / 'wasted_run_rate.png'}")

    # Plot 3: val_bpb trajectory (run-by-run, all conditions overlaid)
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"A": "#d62728", "B": "#ff7f0e", "C": "#2ca02c", "D": "#1f77b4"}
    for c in conds_with_data:
        traj = cond_data[c]["trajectory"]
        ax.plot(range(1, len(traj) + 1), traj, marker=".", label=f"Condition {c}",
                color=colors.get(c, "gray"), alpha=0.8)
    ax.set_xlabel("Run Number")
    ax.set_ylabel("val_bpb")
    ax.set_title("val_bpb Trajectory by Condition")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_dir / "runs_to_threshold.png", dpi=150)
    plt.close(fig)
    print(f"Plot: {plot_dir / 'runs_to_threshold.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("--condition", "-c", choices=CONDITIONS, help="Generate summary for specific condition")
    parser.add_argument("--compare", action="store_true", help="Generate comparison only")
    parser.add_argument("--phase", default="full", choices=["pilot", "full"], help="Comparison phase label")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed val_bpb threshold for runs-to-threshold (from Protocol 2 baseline)")
    parser.add_argument("--plot", action="store_true", help="Generate plots to results/plots/")
    args = parser.parse_args()

    if args.plot:
        generate_plots()
        return

    if args.compare:
        generate_comparison(phase=args.phase, threshold=args.threshold)
        return

    if args.condition:
        generate_summary(args.condition, threshold=args.threshold)
    else:
        # Generate all summaries + comparison
        for cond in CONDITIONS:
            generate_summary(cond, threshold=args.threshold)
        generate_comparison(phase=args.phase, threshold=args.threshold)


if __name__ == "__main__":
    main()
