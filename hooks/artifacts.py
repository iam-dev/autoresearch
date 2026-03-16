"""Run result I/O — writing, reading, and querying run artifact files."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from hooks.types import PreRunContext, RunConfig, RunResults

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
    last = existing[-1].stem
    return int(last.split("_")[1]) + 1


def _best_val_bpb(run_dir: Path) -> float | None:
    """Find the best val_bpb across all runs in a directory."""
    best = None
    for f in run_dir.glob("run_*.json"):
        try:
            data = json.loads(f.read_text())
            vbpb = data["results"]["val_bpb"]
            if not data["results"].get("diverged", False) and (best is None or vbpb < best):
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
    wasted: bool | None = None,
) -> Path:
    """Write a single run result JSON.

    Args:
        wasted: If provided, uses this value. If None, defaults to False.
            Keeping wasted-run logic out of artifacts.py avoids circular deps
            and keeps I/O separate from analysis.
    """
    seed_dir = RESULTS_DIR / "runs" / f"condition_{condition}" / f"seed_{seed:02d}"
    _ensure_dir(seed_dir)

    best = _best_val_bpb(seed_dir)
    delta_from_best = (results.val_bpb - best) if best is not None else 0.0

    config_dict = {k: v for k, v in asdict(config).items()
                   if k not in ("rationale_tag", "rationale")}
    doc = {
        "run_id": run_id,
        "condition": condition,
        "seed": seed,
        "timestamp": datetime.now(UTC).isoformat(),
        "config": config_dict,
        "results": asdict(results),
        "rationale_tag": config.rationale_tag,
        "rationale": config.rationale,
        "wasted": wasted if wasted is not None else False,
        "delta_from_best": round(delta_from_best, 6),
    }
    if pre_run_context is not None:
        doc["pre_run_context"] = asdict(pre_run_context)
    if extra:
        doc.update(extra)

    path = seed_dir / f"run_{run_id:03d}.json"
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path
