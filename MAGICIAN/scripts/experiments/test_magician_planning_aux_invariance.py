#!/usr/bin/env python3
"""MAGICIAN planning-level auxiliary compression trajectory-invariance experiment.

Tests whether compressing planning-level depth/mask significantly alters MAGICIAN's
next-view selection and trajectory compared to clean (r=1.0) input.

Two compression modes are evaluated:

  block_dropout       rho_plan in {1.00, 0.75, 0.50, 0.25, 0.10}
                      nominal_data_ratio   = rho_plan
                      nominal_compression  = 1 / rho_plan

  downsample_upsample scale in {1.0, 0.5, 0.25, 0.125}
                      nominal_data_ratio   = scale^2
                      nominal_compression  = 1 / scale^2

The baseline for both modes is the clean block_dropout trajectory (rho=1.0, seed=0).
Trajectories for each compressed run are compared step-by-step against this baseline.

Usage
-----
  # Full experiment:
  python scripts/experiments/test_magician_planning_aux_invariance.py

  # Reuse existing MAGICIAN runs:
  python scripts/experiments/test_magician_planning_aux_invariance.py --skip_existing

  # Smoke test (fast):
  python scripts/experiments/test_magician_planning_aux_invariance.py --smoke

  # Full one-liner (as specified in Section 11):
  python scripts/experiments/test_magician_planning_aux_invariance.py \\
    --scene fushimi --output_root results/planning_aux_invariance_fushimi \\
    --steps 100 --seed 0 \\
    --compression_modes block_dropout downsample_upsample \\
    --rho_plan_list 1.0 0.75 0.5 0.25 0.1 \\
    --downsample_scales 1.0 0.5 0.25 0.125

Outputs (default: results/planning_aux_invariance_fushimi/)
-----------------------------------------------------------
  summary.csv          -- one row per run, all metrics
  summary.md           -- human-readable analysis table and conclusions
  step_comparison.csv  -- per-step metrics for every non-baseline run
  compression_stats.csv-- per-step depth-retention stats from LMDB timing records
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import run_raux_block_sweep_fushimi as raux  # noqa: E402

# ---------------------------------------------------------------------------
# Trajectory-preserving thresholds (Section 6)
# ---------------------------------------------------------------------------
THRESHOLDS: dict[str, float] = {
    "top1_consistency_min": 0.95,
    "mean_position_error_max": 0.5,    # metres
    "mean_angle_error_deg_max": 5.0,   # degrees
    "final_coverage_drop_rel_max": 0.01,
    "auc_drop_rel_max": 0.01,
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "planning_aux_invariance_fushimi"
DEFAULT_RHO_PLAN_LIST = [1.0, 0.75, 0.50, 0.25, 0.10]
DEFAULT_DOWNSAMPLE_SCALES = [1.0, 0.5, 0.25, 0.125]
DEFAULT_COMPRESSION_MODES = ["block_dropout", "downsample_upsample"]
BASELINE_RHO = 1.0
LMDB_KEY = "fushimi/0"
DSUP_LOG_DIR = ROOT / "results" / "dsup_sweep_logs"


# ---------------------------------------------------------------------------
# Nominal ratio helpers
# ---------------------------------------------------------------------------

def nominal_data_ratio(mode: str, rho: float | None = None, scale: float | None = None) -> float:
    if mode in ("block_dropout", "pixel_dropout"):
        return float(rho)
    elif mode == "downsample_upsample":
        return float(scale) ** 2
    raise ValueError(f"Unknown mode: {mode!r}")


def nominal_compression_ratio(ndr: float) -> float:
    return 1.0 / ndr if ndr > 0 else math.inf


# ---------------------------------------------------------------------------
# LMDB / run management
# ---------------------------------------------------------------------------

def _dsup_run_id(scale: float, seed: int) -> str:
    stag = f"{scale:.3f}".replace(".", "p")
    return f"dsup_fushimi_scale{stag}_seed{seed}"


def _write_dsup_config(scale: float, seed: int) -> Path:
    base = json.loads(raux.BASE_CONFIG.read_text())
    rid = _dsup_run_id(scale, seed)
    base.update({
        "test_scenes": ["fushimi"],
        "results_json_name": f"{rid}.json",
        "lmdb_dir_name": f"{rid}_lmdb",
        "random_seed": int(seed),
        "torch_seed": int(seed),
        "memory_dir_name": f"test_memory_{rid}",
        "max_trajectories_per_scene": 1,
        "export_hda_roi_dataset": False,
        "hda_roi_skip_existing_trajectories": False,
        "r_aux": float(scale),
        "degrade_mode": "downsample_upsample",
        "degrade_seed": int(seed),
        "degrade_block_size": 16,
    })
    config_path = raux.CONFIG_DIR / f"_{rid}.json"
    config_path.write_text(json.dumps(base, indent=2) + "\n")
    return config_path


def _dsup_run_one(scale: float, seed: int, skip_existing: bool) -> dict:
    """Ensure MAGICIAN has been run with downsample_upsample at given scale; return LMDB record."""
    rid = _dsup_run_id(scale, seed)
    lmdb_dir = raux.SCENE_RESULTS_DIR / f"{rid}_lmdb"
    if skip_existing and lmdb_dir.exists():
        record = raux.load_lmdb_record(lmdb_dir, key=LMDB_KEY)
        n = len(record.get("coverage", []))
        timing = record.get("timing", [])
        ret = float(np.mean([t.get("depth_retention_ratio", 0.0) for t in timing])) if timing else math.nan
        print(f"  [skip] {rid}: {n} steps, retention={ret:.4f}")
        return record

    config_path = _write_dsup_config(scale, seed)
    DSUP_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DSUP_LOG_DIR / f"{rid}.log"
    cmd = [
        str(raux.PYTHON),
        "test_magician_planning.py",
        "-c", config_path.name,
        "--r_aux", f"{scale:.6f}",
        "--degrade_mode", "downsample_upsample",
        "--degrade_seed", str(seed),
        "--degrade_block_size", "16",
    ]
    print(f"Running {rid}", flush=True)
    print(f"  log: {log_path}", flush=True)
    with log_path.open("w") as log:
        log.write("Command: " + " ".join(cmd) + "\n")
        log.flush()
        completed = subprocess.run(cmd, cwd=raux.ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"Run failed for {rid}; see {log_path}")
    record = raux.load_lmdb_record(lmdb_dir, key=LMDB_KEY)
    n = len(record.get("coverage", []))
    print(f"  Done: {n} steps", flush=True)
    return record


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _auc(coverage: list[float]) -> float:
    n = len(coverage)
    if n < 2:
        return float(coverage[0]) if n == 1 else 0.0
    arr = np.asarray(coverage, dtype=np.float64)
    return float(np.trapz(arr, np.arange(n)) / (n - 1))


def _angle_diff_deg(a: float, b: float, wrap: float = 360.0) -> float:
    d = (a - b) % wrap
    return d if d <= wrap / 2 else d - wrap


def _actual_keep_ratio(record: dict, n: int) -> float:
    timing = record.get("timing", [])[:n]
    if not timing:
        return math.nan
    ratios = [float(t.get("depth_retention_ratio", 0.0)) for t in timing]
    return float(np.mean(ratios))


def compare_trajectories(
    baseline_record: dict,
    compressed_record: dict,
    run_name: str,
    mode: str,
    rho: float | None,
    scale: float | None,
    n_steps: int | None,
    pose_match_threshold: float,
    seed: int,
    scene: str,
) -> tuple[list[dict], dict[str, Any]]:
    """Compare two MAGICIAN LMDB records step by step against the clean baseline."""
    x_base = np.asarray(baseline_record["X_cam_history"], dtype=np.float64)
    v_base = np.asarray(baseline_record["V_cam_history"], dtype=np.float64)
    cov_base = [float(v) for v in baseline_record.get("coverage", [])]

    x_comp = np.asarray(compressed_record["X_cam_history"], dtype=np.float64)
    v_comp = np.asarray(compressed_record["V_cam_history"], dtype=np.float64)
    cov_comp = [float(v) for v in compressed_record.get("coverage", [])]

    n = min(len(cov_base), len(cov_comp))
    if n_steps is not None:
        n = min(n, int(n_steps))

    ndr = nominal_data_ratio(mode, rho=rho, scale=scale)
    ncr = nominal_compression_ratio(ndr)
    actual_kr = _actual_keep_ratio(compressed_record, n)

    step_rows: list[dict] = []
    for i in range(n):
        pose_b = x_base[i + 1]   # position chosen at planning step i
        pose_c = x_comp[i + 1]
        pose_err = float(np.linalg.norm(pose_b - pose_c))

        az_err = _angle_diff_deg(float(v_base[i + 1, 0]), float(v_comp[i + 1, 0]))
        el_err = float(v_base[i + 1, 1]) - float(v_comp[i + 1, 1])
        ang_err = float(math.sqrt(az_err ** 2 + el_err ** 2))

        top1 = int(pose_err < pose_match_threshold)
        cov_diff = cov_comp[i] - cov_base[i]

        step_rows.append({
            "run_name": run_name,
            "scene": scene,
            "compression_mode": mode,
            "rho_plan": rho if rho is not None else "NA",
            "downsample_scale": scale if scale is not None else "NA",
            "step_i": i,
            "pose_base_x": float(pose_b[0]),
            "pose_base_y": float(pose_b[1]),
            "pose_base_z": float(pose_b[2]),
            "pose_comp_x": float(pose_c[0]),
            "pose_comp_y": float(pose_c[1]),
            "pose_comp_z": float(pose_c[2]),
            "pose_error_m": pose_err,
            "angle_error_deg": ang_err,
            "top1_consistent": top1,
            "coverage_baseline": cov_base[i],
            "coverage_compressed": cov_comp[i],
            "coverage_diff": cov_diff,
        })

    pose_errors = np.asarray([r["pose_error_m"] for r in step_rows])
    angle_errors = np.asarray([r["angle_error_deg"] for r in step_rows])
    top1_vals = np.asarray([r["top1_consistent"] for r in step_rows], dtype=float)

    # Coverage gain errors (per-step delta comparison)
    gains_base = [cov_base[0]] + [cov_base[i] - cov_base[i - 1] for i in range(1, n)]
    gains_comp = [cov_comp[0]] + [cov_comp[i] - cov_comp[i - 1] for i in range(1, n)]
    gain_errs = [abs(gc - gb) for gc, gb in zip(gains_comp, gains_base)]
    mean_cov_gain_err = float(np.mean(gain_errs)) if gain_errs else math.nan
    final_base = cov_base[n - 1] if n > 0 else math.nan
    mean_rel_cov_gain_err = mean_cov_gain_err / max(abs(final_base), 1e-9) if not math.isnan(final_base) else math.nan

    final_comp = cov_comp[n - 1] if n > 0 else math.nan
    auc_base = _auc(cov_base[:n])
    auc_comp = _auc(cov_comp[:n])

    first_div = next((i for i, r in enumerate(step_rows) if r["top1_consistent"] == 0), n)

    cov_drop_abs = final_base - final_comp if n > 0 else math.nan
    cov_drop_rel = cov_drop_abs / max(abs(final_base), 1e-9) if n > 0 else math.nan
    auc_drop_abs = auc_base - auc_comp
    auc_drop_rel = auc_drop_abs / max(abs(auc_base), 1e-9)

    top1_consistency = float(np.mean(top1_vals)) if len(top1_vals) > 0 else math.nan

    summ: dict[str, Any] = {
        "run_name": run_name,
        "scene": scene,
        "compression_mode": mode,
        "rho_plan": rho if rho is not None else "NA",
        "downsample_scale": scale if scale is not None else "NA",
        "nominal_data_ratio": ndr,
        "nominal_compression_ratio": ncr,
        "actual_keep_ratio": actual_kr,
        "steps": n,
        "seed": seed,
        "top1_consistency": top1_consistency,
        "top3_consistency": "NA",   # not available without storing full beam candidates
        "mean_position_error": float(np.mean(pose_errors)) if len(pose_errors) > 0 else math.nan,
        "max_position_error": float(np.max(pose_errors)) if len(pose_errors) > 0 else math.nan,
        "median_position_error": float(np.median(pose_errors)) if len(pose_errors) > 0 else math.nan,
        "mean_angle_error_deg": float(np.mean(angle_errors)) if len(angle_errors) > 0 else math.nan,
        "max_angle_error_deg": float(np.max(angle_errors)) if len(angle_errors) > 0 else math.nan,
        "median_angle_error_deg": float(np.median(angle_errors)) if len(angle_errors) > 0 else math.nan,
        "final_coverage_clean": final_base,
        "final_coverage_compressed": final_comp,
        "coverage_drop_abs": cov_drop_abs,
        "coverage_drop_rel": cov_drop_rel,
        "AUC_clean": auc_base,
        "AUC_compressed": auc_comp,
        "AUC_drop_abs": auc_drop_abs,
        "AUC_drop_rel": auc_drop_rel,
        "mean_coverage_gain_error": mean_cov_gain_err,
        "mean_relative_coverage_gain_error": mean_rel_cov_gain_err,
        "first_divergence_step": first_div,
        "trajectory_preserving": "TBD",  # filled in after
    }

    # Trajectory-preserving determination
    tp = _determine_trajectory_preserving(summ)
    summ["trajectory_preserving"] = tp
    return step_rows, summ


def _determine_trajectory_preserving(summ: dict) -> bool:
    """Return True only if ALL threshold criteria are met."""
    def _f(key: str) -> float:
        v = summ.get(key, math.nan)
        try:
            return float(v)
        except (TypeError, ValueError):
            return math.nan

    top1 = _f("top1_consistency")
    pos_err = _f("mean_position_error")
    ang_err = _f("mean_angle_error_deg")
    cov_drop_rel = _f("coverage_drop_rel")
    auc_drop_rel = _f("AUC_drop_rel")

    if any(math.isnan(v) for v in [top1, pos_err, ang_err, cov_drop_rel, auc_drop_rel]):
        return False
    return (
        top1 >= THRESHOLDS["top1_consistency_min"]
        and pos_err <= THRESHOLDS["mean_position_error_max"]
        and ang_err <= THRESHOLDS["mean_angle_error_deg_max"]
        and cov_drop_rel <= THRESHOLDS["final_coverage_drop_rel_max"]
        and auc_drop_rel <= THRESHOLDS["auc_drop_rel_max"]
    )


def _baseline_summary_row(
    baseline_record: dict,
    run_name: str,
    mode: str,
    rho: float | None,
    scale: float | None,
    n_steps: int | None,
    seed: int,
    scene: str,
) -> dict[str, Any]:
    """Build a reference row for the clean baseline (self-comparison = identity)."""
    cov = [float(v) for v in baseline_record.get("coverage", [])]
    n = min(len(cov), n_steps) if n_steps is not None else len(cov)
    auc_val = _auc(cov[:n])
    ndr = nominal_data_ratio(mode, rho=rho, scale=scale)
    return {
        "run_name": run_name,
        "scene": scene,
        "compression_mode": mode,
        "rho_plan": rho if rho is not None else "NA",
        "downsample_scale": scale if scale is not None else "NA",
        "nominal_data_ratio": ndr,
        "nominal_compression_ratio": nominal_compression_ratio(ndr),
        "actual_keep_ratio": 1.0,
        "steps": n,
        "seed": seed,
        "top1_consistency": 1.0,
        "top3_consistency": "NA",
        "mean_position_error": 0.0,
        "max_position_error": 0.0,
        "median_position_error": 0.0,
        "mean_angle_error_deg": 0.0,
        "max_angle_error_deg": 0.0,
        "median_angle_error_deg": 0.0,
        "final_coverage_clean": cov[n - 1] if n > 0 else math.nan,
        "final_coverage_compressed": cov[n - 1] if n > 0 else math.nan,
        "coverage_drop_abs": 0.0,
        "coverage_drop_rel": 0.0,
        "AUC_clean": auc_val,
        "AUC_compressed": auc_val,
        "AUC_drop_abs": 0.0,
        "AUC_drop_rel": 0.0,
        "mean_coverage_gain_error": 0.0,
        "mean_relative_coverage_gain_error": 0.0,
        "first_divergence_step": n,
        "trajectory_preserving": True,
    }


def build_compression_stats(
    record: dict,
    run_name: str,
    mode: str,
    n_steps: int | None,
) -> list[dict]:
    timing = record.get("timing", [])
    if n_steps is not None:
        timing = timing[:int(n_steps)]
    rows = []
    for i, t in enumerate(timing):
        rows.append({
            "run_name": run_name,
            "compression_mode": mode,
            "step_i": i,
            "n_valid_before": t.get("n_valid_depth_points_before_degrade", ""),
            "n_valid_after": t.get("n_valid_depth_points", ""),
            "depth_retention_ratio": t.get("depth_retention_ratio", ""),
            "r_aux": t.get("r_aux", ""),
            "degrade_mode": t.get("degrade_mode", ""),
            "degrade_seed": t.get("degrade_seed", ""),
            "degrade_block_size": t.get("degrade_block_size", ""),
            "new_frames_processed": t.get("new_frames_processed", ""),
        })
    return rows


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _fmt(v: Any, key: str = "") -> str:
    if v == "NA" or v is None:
        return "NA"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf"
        if "ratio" in key or "consistency" in key or "auc_drop_rel" == key.lower() or "coverage_drop_rel" == key.lower():
            return f"{v:.4f}"
        if "error" in key or "coverage" in key.lower() or key.lower().startswith("auc") or "drop_abs" in key:
            return f"{v:.6f}"
        if key in ("rho_plan", "downsample_scale", "nominal_data_ratio"):
            return f"{v:.4f}"
        if key == "nominal_compression_ratio":
            return f"{v:.2f}x"
        return f"{v:.6f}"
    return str(v)


def _make_table(rows: list[dict], columns: list[str]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |"]
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(c, "NA"), c) for c in columns) + " |")
    return lines


def _find_max_safe_compression(rows: list[dict]) -> dict[str, Any]:
    """For each mode, find the highest nominal_compression_ratio where trajectory_preserving=True."""
    from collections import defaultdict
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("compression_mode", ""))].append(row)

    result: dict[str, Any] = {}
    for mode, mode_rows in by_mode.items():
        safe = [r for r in mode_rows if r.get("trajectory_preserving") is True]
        if not safe:
            result[mode] = {"max_safe_ncr": math.nan, "max_safe_ndr": math.nan}
            continue
        # Sort by nominal_compression_ratio descending; find max where tp=True
        def _ncr(r: dict) -> float:
            v = r.get("nominal_compression_ratio", 0.0)
            try:
                return float(v) if str(v) != "inf" else float("inf")
            except (TypeError, ValueError):
                return 0.0
        safe_sorted = sorted(safe, key=_ncr, reverse=True)
        best = safe_sorted[0]
        result[mode] = {
            "max_safe_ncr": _ncr(best),
            "max_safe_ndr": float(best.get("nominal_data_ratio", math.nan)),
            "max_safe_run": str(best.get("run_name", "")),
        }
    return result


def _first_divergence_by_mode(rows: list[dict]) -> dict[str, Any]:
    """For each mode, find the first (lowest compression) run where trajectory_preserving=False."""
    from collections import defaultdict
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_mode[str(row.get("compression_mode", ""))].append(row)

    result: dict[str, Any] = {}
    for mode, mode_rows in by_mode.items():
        non_tp = [
            r for r in mode_rows
            if r.get("trajectory_preserving") is False
        ]
        if not non_tp:
            result[mode] = None
            continue

        def _ncr_sort(r: dict) -> float:
            v = r.get("nominal_compression_ratio", 0.0)
            try:
                return float(v) if str(v) != "inf" else float("inf")
            except (TypeError, ValueError):
                return 0.0

        first = sorted(non_tp, key=_ncr_sort)[0]
        result[mode] = first
    return result


def _write_summary_md(
    path: Path,
    summary_rows: list[dict],
    args: argparse.Namespace,
    baseline_record: dict,
) -> None:
    cov_base = [float(v) for v in baseline_record.get("coverage", [])]
    n_total = len(cov_base)
    n_compared = min(n_total, args.steps) if args.steps else n_total
    auc_base = _auc(cov_base[:n_compared])
    final_cov_base = cov_base[n_compared - 1] if n_compared > 0 else math.nan

    block_rows = [r for r in summary_rows if str(r.get("compression_mode")) == "block_dropout"]
    dsup_rows = [r for r in summary_rows if str(r.get("compression_mode")) == "downsample_upsample"]
    all_non_base = [r for r in summary_rows if r.get("trajectory_preserving") != "NA"]

    safe_info = _find_max_safe_compression(
        [r for r in summary_rows if r.get("run_name", "").split("_seed")[0] != "clean_baseline"]
    )
    first_div_info = _first_divergence_by_mode(
        [r for r in summary_rows if float(r.get("nominal_compression_ratio", 1.0)) > 1.0
         if str(r.get("nominal_compression_ratio")) != "inf"]
    )

    block_cols = [
        "run_name", "rho_plan", "nominal_compression_ratio", "actual_keep_ratio",
        "top1_consistency", "first_divergence_step",
        "mean_position_error", "mean_angle_error_deg",
        "final_coverage_clean", "final_coverage_compressed", "coverage_drop_rel",
        "AUC_clean", "AUC_compressed", "AUC_drop_rel",
        "trajectory_preserving",
    ]
    dsup_cols = [
        "run_name", "downsample_scale", "nominal_compression_ratio", "actual_keep_ratio",
        "top1_consistency", "first_divergence_step",
        "mean_position_error", "mean_angle_error_deg",
        "final_coverage_clean", "final_coverage_compressed", "coverage_drop_rel",
        "AUC_clean", "AUC_compressed", "AUC_drop_rel",
        "trajectory_preserving",
    ]

    # Build recommendation
    def _mode_rec(mode: str) -> str:
        info = safe_info.get(mode, {})
        ncr = info.get("max_safe_ncr", math.nan)
        ndr = info.get("max_safe_ndr", math.nan)
        if math.isnan(float(ncr) if not isinstance(ncr, float) else ncr):
            return "No trajectory-preserving run found. Planning-level compression appears unsafe at all tested levels."
        if float(ncr) <= 1.0:
            return "Only the clean baseline is trajectory-preserving. All tested compressions cause divergence."
        ndr_pct = f"{float(ndr)*100:.1f}%"
        return (
            f"Maximum safe compression ratio: **{float(ncr):.1f}x** "
            f"(nominal data ratio = {ndr_pct}). "
            f"Run: `{info.get('max_safe_run', 'N/A')}`."
        )

    def _first_div_str(mode: str) -> str:
        row = first_div_info.get(mode)
        if row is None:
            return "No trajectory divergence detected at any tested compression level."
        ncr = row.get("nominal_compression_ratio", "?")
        rn = row.get("run_name", "?")
        fds = row.get("first_divergence_step", "?")
        return (
            f"First divergence at **{ncr}x** compression (run `{rn}`). "
            f"Trajectories begin to differ at step {fds}."
        )

    # High-compression but still trajectory-preserving?
    high_cr_safe = [
        r for r in all_non_base
        if r.get("trajectory_preserving") is True
        and float(r.get("nominal_compression_ratio", 1.0)) >= 4.0
    ]

    lines = [
        "# MAGICIAN Planning-Level Compression Trajectory-Invariance Experiment",
        "",
        "## 1. Experiment Purpose",
        "Validate that compressing planning-level depth/mask (block_dropout or",
        "downsample_upsample) does not significantly alter MAGICIAN's next-view",
        "selection or resulting trajectory compared to clean input.",
        "This experiment is specific to the **Fushimi** scene. Results cannot be",
        "generalised to other scenes without additional experiments.",
        "",
        "## 2. Data Paths",
        f"- Scene data:   `{ROOT / 'data' / 'Macarons++' / 'macarons++' / 'fushimi'}`",
        f"- LMDB root:    `{raux.SCENE_RESULTS_DIR}`",
        f"- Output root:  `{args.output_root}`",
        "",
        "## 3. MAGICIAN Planner Entry",
        f"- Script:       `{ROOT / 'test_magician_planning.py'}`",
        f"- Config base:  `{raux.BASE_CONFIG}`",
        f"- Seed:         `{args.seed}`",
        f"- Steps total:  `{n_total}` (compared: `{n_compared}`)",
        "",
        "## 4. Clean Baseline Definition",
        "The clean baseline is the `block_dropout rho=1.0, seed=0` trajectory.",
        "All other runs (both block_dropout rho<1.0 and downsample_upsample scale<1.0)",
        "are compared against this baseline.",
        f"- Baseline final coverage: `{_fmt(final_cov_base, 'final_coverage')}`",
        f"- Baseline AUC ({n_compared} steps): `{_fmt(auc_base, 'AUC')}`",
        "",
        "## 5. Trajectory-Preserving Thresholds",
        f"A run is **trajectory_preserving = True** only if ALL of the following hold:",
        f"| Metric | Threshold |",
        f"| --- | --- |",
        f"| top1_consistency | >= {THRESHOLDS['top1_consistency_min']} |",
        f"| mean_position_error (m) | <= {THRESHOLDS['mean_position_error_max']} |",
        f"| mean_angle_error (deg) | <= {THRESHOLDS['mean_angle_error_deg_max']} |",
        f"| coverage_drop_rel | <= {THRESHOLDS['final_coverage_drop_rel_max']} |",
        f"| AUC_drop_rel | <= {THRESHOLDS['auc_drop_rel_max']} |",
        "",
        "top3_consistency is reported as NA (beam search candidates are not stored in LMDB).",
        "",
    ]

    lines += ["## 6. block_dropout Results", ""]
    if block_rows:
        lines += _make_table(block_rows, block_cols)
    else:
        lines += ["*(no block_dropout runs)*"]
    lines += [""]

    lines += ["## 7. downsample_upsample Results", ""]
    if dsup_rows:
        lines += _make_table(dsup_rows, dsup_cols)
    else:
        lines += ["*(no downsample_upsample runs)*"]
    lines += [""]

    lines += [
        "## 8. Maximum Safe Compression",
        "",
        f"### block_dropout",
        f"{_mode_rec('block_dropout')}",
        "",
        f"### downsample_upsample",
        f"{_mode_rec('downsample_upsample')}",
        "",
        "## 9. First Level Where Trajectory Changes",
        "",
        f"### block_dropout",
        f"{_first_div_str('block_dropout')}",
        "",
        f"### downsample_upsample",
        f"{_first_div_str('downsample_upsample')}",
        "",
    ]

    if high_cr_safe:
        lines += [
            "## 10. High-Compression Runs That Are Still Trajectory-Preserving",
            "",
        ]
        for r in high_cr_safe:
            lines.append(
                f"- `{r['run_name']}`: NCR={_fmt(r.get('nominal_compression_ratio'), 'nominal_compression_ratio')}, "
                f"top1={_fmt(r.get('top1_consistency'), 'consistency')}, "
                f"AUC_drop_rel={_fmt(r.get('AUC_drop_rel'), 'AUC_drop_rel')}"
            )
        lines.append("")
    else:
        lines += [
            "## 10. High-Compression Runs That Are Still Trajectory-Preserving",
            "",
            "No runs with nominal_compression_ratio >= 4x were found to be trajectory-preserving.",
            "",
        ]

    # Recommendation
    recs: list[str] = []
    for mode in ["block_dropout", "downsample_upsample"]:
        info = safe_info.get(mode, {})
        ncr = info.get("max_safe_ncr", math.nan)
        ndr = info.get("max_safe_ndr", math.nan)
        try:
            ncr_f = float(ncr)
            ndr_f = float(ndr)
        except (TypeError, ValueError):
            ncr_f = math.nan
            ndr_f = math.nan
        if not math.isnan(ncr_f) and ncr_f > 1.0:
            recs.append(
                f"- **{mode}**: up to **{ncr_f:.1f}x** compression "
                f"(retain >= {ndr_f * 100:.1f}% of planning-level depth/mask pixels)"
            )
        else:
            recs.append(f"- **{mode}**: no safe compression level found in this experiment")

    lines += [
        "## 11. Recommended Maximum Planning-Level Compression",
        "",
        "Based on Fushimi scene only. Do not extrapolate to other scenes without further experiments.",
        "",
    ] + recs + [""]

    # Caveat about strong compression
    any_divergent = any(
        r.get("trajectory_preserving") is False for r in all_non_base
    )
    if any_divergent:
        lines += [
            "## 12. Strong Compression Causes Trajectory Divergence",
            "",
            "At least one tested compression level causes trajectory divergence "
            "(trajectory_preserving = False). This means MAGICIAN's beam search "
            "selects different next views under heavy depth/mask compression, "
            "leading to a meaningfully different exploration trajectory.",
            "The divergence level and metrics are reported accurately in the tables above.",
            "",
        ]
    else:
        lines += [
            "## 12. Trajectory Stability Note",
            "",
            "All tested compression levels are trajectory-preserving in this experiment. "
            "However, this is limited to the Fushimi scene and the tested compression levels. "
            "Stronger compression or different scenes may not preserve the trajectory.",
            "",
        ]

    lines += [
        "## Output Files",
        f"- `summary.csv`           -- all metrics, one row per run",
        f"- `step_comparison.csv`   -- per-step detail for non-baseline runs",
        f"- `compression_stats.csv` -- per-step depth-retention from LMDB timing",
        f"- `summary.md`            -- this file",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MAGICIAN planning-level compression trajectory-invariance experiment."
    )
    p.add_argument("--scene", default="fushimi", choices=["fushimi"])
    p.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--compression_modes", nargs="+", default=DEFAULT_COMPRESSION_MODES,
                   choices=["block_dropout", "pixel_dropout", "downsample_upsample"],
                   help="Compression modes to evaluate.")
    p.add_argument("--rho_plan_list", nargs="+", type=float, default=DEFAULT_RHO_PLAN_LIST,
                   help="Retention ratios for block_dropout / pixel_dropout.")
    p.add_argument("--downsample_scales", nargs="+", type=float, default=DEFAULT_DOWNSAMPLE_SCALES,
                   help="Linear scale factors for downsample_upsample mode.")
    p.add_argument("--steps", type=int, default=None,
                   help="Limit comparison to first N planning steps (None = all).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--block_size", type=int, default=16,
                   help="Block size for block_dropout.")
    p.add_argument("--pose_error_threshold", type=float, default=1.0,
                   help="Metres below which two poses are considered matching (top1).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Reuse existing LMDB results without re-running MAGICIAN.")
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: block_dropout=[1.0,0.5], downsample=[1.0,0.25], steps=10.")
    args = p.parse_args()

    if args.smoke:
        if args.rho_plan_list == DEFAULT_RHO_PLAN_LIST:
            args.rho_plan_list = [1.0, 0.50]
        if args.downsample_scales == DEFAULT_DOWNSAMPLE_SCALES:
            args.downsample_scales = [1.0, 0.25]
        if args.steps is None:
            args.steps = 10

    # Ensure baseline rho is included for block modes
    for mode in args.compression_modes:
        if mode in ("block_dropout", "pixel_dropout"):
            if BASELINE_RHO not in args.rho_plan_list:
                args.rho_plan_list = [BASELINE_RHO] + list(args.rho_plan_list)
            break

    return args


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scene:              {args.scene}")
    print(f"[INFO] Compression modes:  {args.compression_modes}")
    print(f"[INFO] Seed:               {args.seed}")
    print(f"[INFO] Steps limit:        {args.steps}")
    print(f"[INFO] Output root:        {output_root}")

    # ------------------------------------------------------------------
    # Step 1: ensure clean baseline exists (block_dropout rho=1.0, seed=0)
    # ------------------------------------------------------------------
    print(f"\n[INFO] Ensuring clean baseline (block_dropout rho=1.0, seed={args.seed}) ...")
    baseline_summary = raux.run_one(
        BASELINE_RHO, args.seed,
        skip_existing=args.skip_existing,
        block_size=args.block_size,
    )
    baseline_lmdb = raux.SCENE_RESULTS_DIR / f"{raux.run_id(BASELINE_RHO, args.seed)}_lmdb"
    baseline_record = raux.load_lmdb_record(baseline_lmdb, key=LMDB_KEY)
    cov_base_all = baseline_record.get("coverage", [])
    print(
        f"  Baseline: steps={len(cov_base_all)}, "
        f"final_cov={baseline_summary['final_coverage']:.6f}, "
        f"auc={baseline_summary['auc']:.4f}"
    )

    all_summary_rows: list[dict] = []
    all_step_rows: list[dict] = []
    all_comp_stats: list[dict] = []

    # ------------------------------------------------------------------
    # Step 2: block_dropout / pixel_dropout modes
    # ------------------------------------------------------------------
    for mode in args.compression_modes:
        if mode not in ("block_dropout", "pixel_dropout"):
            continue

        magician_mode = "block" if mode == "block_dropout" else "pixel_dropout"
        print(f"\n{'='*70}")
        print(f"[MODE] {mode}  (MAGICIAN degrade_mode={magician_mode!r})")
        print(f"  rho values: {args.rho_plan_list}")

        for rho in args.rho_plan_list:
            rtag = f"{rho:.2f}".replace(".", "p")
            run_name = f"{mode}_r{rtag}_seed{args.seed}"

            if rho == BASELINE_RHO:
                # Reference row: self-comparison
                row = _baseline_summary_row(
                    baseline_record, run_name=f"clean_baseline_seed{args.seed}",
                    mode=mode, rho=rho, scale=None,
                    n_steps=args.steps, seed=args.seed, scene=args.scene,
                )
                all_summary_rows.append(row)
                all_comp_stats.extend(build_compression_stats(
                    baseline_record, run_name=f"clean_baseline_seed{args.seed}",
                    mode=mode, n_steps=args.steps,
                ))
                print(f"  rho={rho:.2f}: [baseline]")
                continue

            print(f"\n  rho={rho:.2f}  run_name={run_name}")
            if mode == "block_dropout":
                summary_row = raux.run_one(
                    rho, args.seed,
                    skip_existing=args.skip_existing,
                    block_size=args.block_size,
                )
                rid = raux.run_id(rho, args.seed)
                lmdb_dir = raux.SCENE_RESULTS_DIR / f"{rid}_lmdb"
            else:
                raise NotImplementedError(
                    "pixel_dropout requires a separate MAGICIAN mode; "
                    "add _pixel_run_one() helper analogous to _dsup_run_one()."
                )

            compressed_record = raux.load_lmdb_record(lmdb_dir, key=LMDB_KEY)
            step_rows, summ = compare_trajectories(
                baseline_record=baseline_record,
                compressed_record=compressed_record,
                run_name=run_name,
                mode=mode,
                rho=rho,
                scale=None,
                n_steps=args.steps,
                pose_match_threshold=args.pose_error_threshold,
                seed=args.seed,
                scene=args.scene,
            )
            all_step_rows.extend(step_rows)
            all_comp_stats.extend(build_compression_stats(
                compressed_record, run_name=run_name, mode=mode, n_steps=args.steps,
            ))
            all_summary_rows.append(summ)
            print(
                f"    top1={summ['top1_consistency']:.4f}  "
                f"first_div={summ['first_divergence_step']}  "
                f"pos_err={summ['mean_position_error']:.4f}m  "
                f"auc_drop={summ['AUC_drop_rel']:.4f}  "
                f"tp={summ['trajectory_preserving']}"
            )

    # ------------------------------------------------------------------
    # Step 3: downsample_upsample mode
    # ------------------------------------------------------------------
    if "downsample_upsample" in args.compression_modes:
        print(f"\n{'='*70}")
        print(f"[MODE] downsample_upsample")
        print(f"  scales: {args.downsample_scales}")

        for scale in args.downsample_scales:
            stag = f"{scale:.3f}".replace(".", "p")
            run_name = f"dsup_scale{stag}_seed{args.seed}"

            if scale >= 1.0:
                # Scale=1.0 is identity; use baseline record
                row = _baseline_summary_row(
                    baseline_record, run_name=run_name,
                    mode="downsample_upsample", rho=None, scale=scale,
                    n_steps=args.steps, seed=args.seed, scene=args.scene,
                )
                all_summary_rows.append(row)
                print(f"  scale={scale:.3f}: [identity, uses baseline record]")
                continue

            print(f"\n  scale={scale:.3f}  run_name={run_name}")
            compressed_record = _dsup_run_one(scale, args.seed, skip_existing=args.skip_existing)
            step_rows, summ = compare_trajectories(
                baseline_record=baseline_record,
                compressed_record=compressed_record,
                run_name=run_name,
                mode="downsample_upsample",
                rho=None,
                scale=scale,
                n_steps=args.steps,
                pose_match_threshold=args.pose_error_threshold,
                seed=args.seed,
                scene=args.scene,
            )
            all_step_rows.extend(step_rows)
            all_comp_stats.extend(build_compression_stats(
                compressed_record, run_name=run_name, mode="downsample_upsample", n_steps=args.steps,
            ))
            all_summary_rows.append(summ)
            print(
                f"    top1={summ['top1_consistency']:.4f}  "
                f"first_div={summ['first_divergence_step']}  "
                f"pos_err={summ['mean_position_error']:.4f}m  "
                f"auc_drop={summ['AUC_drop_rel']:.4f}  "
                f"tp={summ['trajectory_preserving']}"
            )

    # ------------------------------------------------------------------
    # Step 4: write outputs
    # ------------------------------------------------------------------
    _write_csv(output_root / "summary.csv", all_summary_rows)
    _write_csv(output_root / "step_comparison.csv", all_step_rows)
    _write_csv(output_root / "compression_stats.csv", all_comp_stats)
    _write_summary_md(output_root / "summary.md", all_summary_rows, args, baseline_record)

    print(f"\n[OK] summary.csv         -> {output_root / 'summary.csv'}")
    print(f"[OK] step_comparison.csv  -> {output_root / 'step_comparison.csv'}")
    print(f"[OK] compression_stats.csv-> {output_root / 'compression_stats.csv'}")
    print(f"[OK] summary.md           -> {output_root / 'summary.md'}")
    print(f"\nDone.  {len(all_summary_rows)} run(s) summarised.")


if __name__ == "__main__":
    main()
