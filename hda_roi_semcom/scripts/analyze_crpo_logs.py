"""Analyze CRPO-PPO CSV logs for basic training diagnostics."""
import argparse
import csv
import math
from collections import Counter

import numpy as np


def _to_float(row, key):
    try:
        return float(row.get(key, "nan"))
    except ValueError:
        return float("nan")


def _avg(rows, key):
    vals = np.asarray([_to_float(r, key) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _split(rows):
    n = max(1, int(math.ceil(0.1 * len(rows))))
    return rows[:n], rows[-n:]


def _warn_constant(rows, key, warnings):
    vals = np.asarray([_to_float(r, key) for r in rows], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size and float(vals.max() - vals.min()) < 1e-9:
        warnings.append(f"{key} never changes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_csv")
    args = ap.parse_args()

    with open(args.log_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty log: {args.log_csv}")

    first, last = _split(rows)
    mode_counts = Counter(r.get("constraint_mode", "") for r in rows)
    print(f"number of iterations: {len(rows)}")
    print(
        "mode counts: "
        f"reward={mode_counts.get('reward', 0)} "
        f"constraint_rgb={mode_counts.get('constraint_rgb', 0)} "
        f"constraint_depth={mode_counts.get('constraint_depth', 0)}"
    )
    pairs = [
        ("avg_map_reward", "reward"),
        ("avg_q_3d", "q_3d"),
        ("avg_cost_rgb", "cost_rgb"),
        ("avg_cost_depth", "cost_depth"),
        ("excess_R", "excess_rgb"),
        ("excess_D", "excess_depth"),
        ("relative_excess_R", "relative_excess_rgb"),
        ("relative_excess_D", "relative_excess_depth"),
        ("avg_Q_rgb", "q_rgb"),
        ("avg_R_depth", "r_depth"),
        ("avg_reconstruction_gain", "reconstruction_gain"),
        ("avg_k_d", "k_d"),
        ("avg_beta_d", "beta_d"),
    ]
    summary = {}
    for key, label in pairs:
        a = _avg(first, key)
        b = _avg(last, key)
        summary[label] = (a, b)
        print(f"first 10% avg {label}: {a:.6g}")
        print(f"last 10% avg {label}: {b:.6g}")

    warnings = []
    if summary["q_3d"][1] <= summary["q_3d"][0]:
        warnings.append("q_3d does not improve")
    if summary["cost_rgb"][1] > summary["cost_rgb"][0] and summary["cost_depth"][1] > summary["cost_depth"][0]:
        warnings.append("both costs increase")
    if mode_counts.get("constraint_rgb", 0) + mode_counts.get("constraint_depth", 0) == 0:
        warnings.append("constraint modes never appear")
    if mode_counts.get("reward", 0) == 0:
        warnings.append("reward mode never appears")
    _warn_constant(rows, "avg_beta_d", warnings)
    _warn_constant(rows, "avg_k_d", warnings)
    for key in ("avg_Q_rgb", "avg_R_depth"):
        vals = np.asarray([_to_float(r, key) for r in rows], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size and (np.all(vals == 0.0) or np.all(vals == 1.0)):
            warnings.append(f"{key} is always exactly {vals[0]:.0f}")
    for key in ("avg_snr_rgb_db", "avg_snr_depth_db"):
        vals = np.asarray([_to_float(r, key) for r in rows], dtype=np.float64)
        if np.any(~np.isfinite(vals)):
            warnings.append(f"{key} contains NaN/Inf")

    for warning in warnings:
        print(f"[warn] {warning}")


if __name__ == "__main__":
    main()
