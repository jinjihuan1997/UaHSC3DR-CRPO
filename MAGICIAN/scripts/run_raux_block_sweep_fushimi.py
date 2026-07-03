#!/usr/bin/env python3
"""Run fushimi r_aux sweep with block-wise depth/mask loss."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import subprocess
import sys
from pathlib import Path

import lmdb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "test"
RESULTS_DIR = ROOT / "results"
SCENE_RESULTS_DIR = RESULTS_DIR / "scene_exploration"
LOG_DIR = RESULTS_DIR / "raux_block_sweep_logs"
BASE_CONFIG = CONFIG_DIR / "test_in_default_scenes_config.json"
PYTHON = Path(sys.executable)
SCENE = "fushimi"
R_VALUES = [1.0, 0.95, 0.90, 0.85]
SEEDS = [0, 1, 2, 3, 4]
DEGRADE_MODE = "block"
DEGRADE_BLOCK_SIZE = 16


def r_tag(r: float) -> str:
    return f"{r:.2f}".replace(".", "p")


def run_id(r: float, seed: int) -> str:
    return f"raux_block_fushimi_r{r_tag(r)}_seed{seed}"


def auc(coverage: list[float]) -> float:
    n = len(coverage)
    if n < 2:
        return float(coverage[0]) if n == 1 else 0.0
    arr = np.asarray(coverage, dtype=np.float64)
    return float(np.trapz(arr, np.arange(n)) / (n - 1))


def load_lmdb_record(lmdb_dir: Path, key: str = "fushimi/0") -> dict:
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False)
    try:
        with env.begin() as txn:
            raw = txn.get(key.encode("utf-8"))
            if raw is None:
                raise RuntimeError(f"Missing LMDB key {key} in {lmdb_dir}")
            return pickle.loads(raw)
    finally:
        env.close()


def write_config(r: float, seed: int, block_size: int = DEGRADE_BLOCK_SIZE) -> Path:
    base = json.loads(BASE_CONFIG.read_text())
    rid = run_id(r, seed)
    base.update(
        {
            "test_scenes": [SCENE],
            "results_json_name": f"{rid}.json",
            "lmdb_dir_name": f"{rid}_lmdb",
            "random_seed": int(seed),
            "torch_seed": int(seed),
            "memory_dir_name": f"test_memory_{rid}",
            "max_trajectories_per_scene": 1,
            "export_hda_roi_dataset": False,
            "hda_roi_skip_existing_trajectories": False,
            "r_aux": float(r),
            "degrade_mode": DEGRADE_MODE,
            "degrade_seed": int(seed),
            "degrade_block_size": int(block_size),
        }
    )
    config_path = CONFIG_DIR / f"_{rid}.json"
    config_path.write_text(json.dumps(base, indent=2) + "\n")
    return config_path


def summarize_record(
    r: float,
    seed: int,
    record: dict,
    already_ran: bool = False,
    block_size: int = DEGRADE_BLOCK_SIZE,
) -> dict:
    coverage = [float(v) for v in record.get("coverage", [])]
    timing = record.get("timing", []) or []
    n_valid = [int(t.get("n_valid_depth_points", 0)) for t in timing if "n_valid_depth_points" in t]
    n_valid_before = [
        int(t.get("n_valid_depth_points_before_degrade", 0))
        for t in timing
        if "n_valid_depth_points_before_degrade" in t
    ]
    ratios = [float(t.get("depth_retention_ratio", 0.0)) for t in timing if "depth_retention_ratio" in t]
    x_hist = np.asarray(record.get("X_cam_history", []), dtype=np.float64)
    v_hist = np.asarray(record.get("V_cam_history", []), dtype=np.float64)
    summary = {
        "r_aux": float(r),
        "seed": int(seed),
        "degrade_mode": DEGRADE_MODE,
        "degrade_block_size": int(block_size),
        "final_coverage": float(coverage[-1]) if coverage else 0.0,
        "auc": auc(coverage),
        "n_valid_pts_mean": float(np.mean(n_valid)) if n_valid else 0.0,
        "n_valid_pts_before_mean": float(np.mean(n_valid_before)) if n_valid_before else 0.0,
        "depth_retention_ratio_mean": float(np.mean(ratios)) if ratios else 0.0,
        "coverage": coverage,
        "n_valid_depth_points": n_valid,
        "n_valid_depth_points_before_degrade": n_valid_before,
        "X_cam_history": x_hist.tolist(),
        "V_cam_history": v_hist.tolist(),
        "already_ran": bool(already_ran),
    }
    traj_path = RESULTS_DIR / f"traj_fushimi_block_{r:.2f}_{seed}.json"
    traj_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"  final={summary['final_coverage']:.6f} auc={summary['auc']:.3f} "
        f"valid={summary['n_valid_pts_mean']:.1f} retention={summary['depth_retention_ratio_mean']:.4f}",
        flush=True,
    )
    return summary


def run_one(r: float, seed: int, skip_existing: bool, block_size: int = DEGRADE_BLOCK_SIZE) -> dict:
    rid = run_id(r, seed)
    lmdb_dir = SCENE_RESULTS_DIR / f"{rid}_lmdb"
    if skip_existing and lmdb_dir.exists():
        return summarize_record(r, seed, load_lmdb_record(lmdb_dir), already_ran=True, block_size=block_size)

    config_path = write_config(r, seed, block_size=block_size)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{rid}.log"
    cmd = [
        str(PYTHON),
        "test_magician_planning.py",
        "-c",
        config_path.name,
        "--r_aux",
        f"{r:.2f}",
        "--degrade_mode",
        DEGRADE_MODE,
        "--degrade_seed",
        str(seed),
        "--degrade_block_size",
        str(block_size),
    ]
    print("Running", rid, flush=True)
    print("  log:", log_path, flush=True)
    with log_path.open("w") as log:
        log.write("Command: " + " ".join(cmd) + "\n")
        log.flush()
        completed = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f"Run failed for {rid}; see {log_path}")
    return summarize_record(r, seed, load_lmdb_record(lmdb_dir), already_ran=False, block_size=block_size)


def write_csv(rows: list[dict]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "raux_block_sweep_fushimi.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "r_aux",
                "seed",
                "degrade_mode",
                "degrade_block_size",
                "final_coverage",
                "auc",
                "n_valid_pts_mean",
                "depth_retention_ratio_mean",
            ],
        )
        writer.writeheader()
        for row in sorted(rows, key=lambda x: (x["r_aux"], x["seed"])):
            writer.writerow({key: row[key] for key in writer.fieldnames})
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fushimi r_aux sweep with block depth/mask loss.")
    parser.add_argument("--smoke", action="store_true", help="Run only r=1.00 and r=0.85 with seed 0.")
    parser.add_argument("--r-values", nargs="+", default=None, help="R values, e.g. --r-values 1.0 0.9 0.8")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds, e.g. --seeds 0 1 2")
    parser.add_argument("--block-size", type=int, default=DEGRADE_BLOCK_SIZE)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    r_values = [float(v) for v in args.r_values] if args.r_values else R_VALUES
    seeds = args.seeds if args.seeds else SEEDS
    jobs = [(1.0, 0), (0.85, 0)] if args.smoke else [(r, seed) for r in r_values for seed in seeds]
    rows = [run_one(r, seed, skip_existing=args.skip_existing, block_size=args.block_size) for r, seed in jobs]
    csv_path = write_csv(rows)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
