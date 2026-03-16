"""Analysis functions — config distance, wasted-run detection, evidence weighting."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

from hooks.types import RunConfig, RunResults


def evidence_weight(val_bpb: float, best_val_bpb: float) -> float:
    """Log-scale evidence weight. Improvements -> weight > 0.5, regressions -> weight < 0.5.

    Formula: weight = 0.5 + sign * 0.5 * |tanh(log10(|delta| + 1e-3) + 2.5)|

    Sign convention: delta = val_bpb - best_val_bpb
      negative delta = improvement (lower bpb is better) -> high weight
      positive delta = regression -> low weight
    """
    delta = val_bpb - best_val_bpb
    if abs(delta) < 1e-8:
        return 0.5
    magnitude = abs(math.tanh(math.log10(abs(delta) + 1e-3) + 2.5))
    sign = 1.0 if delta < 0 else -1.0
    return 0.5 + sign * 0.5 * magnitude


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
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2, strict=True)))


def _meaningful_step_from(cfg: dict, prior_cfg: dict) -> bool:
    """Check if config has at least one focal parameter changed beyond threshold."""
    return (
        abs(math.log2(cfg["matrix_lr"] / max(prior_cfg["matrix_lr"], 1e-10))) >= 1.0
        or abs(cfg["depth"] - prior_cfg["depth"]) >= 2
        or abs(cfg["warmup_ratio"] - prior_cfg["warmup_ratio"]) >= 0.05
        or abs(math.log2(cfg["total_batch_size"] / max(prior_cfg["total_batch_size"], 1))) >= 1.0
        or abs(cfg["wd"] - prior_cfg["wd"]) >= 0.05
    )


def _is_exact_duplicate(cfg: dict, seed_dir: Path) -> bool:
    """Check if this config exactly matches a previously tested config."""
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
        if prior_results.get("diverged", False) and _config_distance(cfg, prior_cfg) < tau:
            return True, prior_cfg
    return False, None


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

    # Regressed beyond epsilon
    if best_val_bpb is not None and results.val_bpb > best_val_bpb + epsilon:
        return True

    # Exact duplicate of a previously tested config with no new rationale
    if not config.rationale_tag and _is_exact_duplicate(cfg, seed_dir):
        return True

    # Near a previously observed bad config
    near, bad_cfg = _is_near_bad_config(config, seed_dir)
    if near and bad_cfg is not None:
        # Exception: rationale + meaningful step from the *specific bad config*
        return not (config.rationale_tag and _meaningful_step_from(cfg, bad_cfg))

    return False
