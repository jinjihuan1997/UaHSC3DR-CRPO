"""Fit saturation surrogates independently per trajectory."""

import argparse
import csv
import json
import os
import re
from collections import defaultdict

import numpy as np

from evaluate_geometry_saturation_surrogate import (
    _diagnostics,
    _finite_rows,
    _features,
    _fit_nonnegative,
    _predict,
    _public_fit,
    _target,
)


COMPOSITE_REQUIRED = [
    "q_rgb", "r_depth", "psnr", "ssim", "lpips",
    "chamfer", "fscore", "completeness",
]


def _read_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row = dict(row)
                row["_source_file"] = path
                rows.append(row)
    return rows


def _finite_composite_rows(rows):
    out = []
    for row in rows:
        try:
            if all(np.isfinite(float(row[name])) for name in COMPOSITE_REQUIRED):
                out.append(row)
        except (KeyError, ValueError):
            continue
    return out


def _col(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def _basis_composite(rows):
    return {
        name: (
            float(np.nanpercentile(_col(rows, name), 5)),
            float(np.nanpercentile(_col(rows, name), 95)),
        )
        for name in ["psnr", "ssim", "lpips", "chamfer", "fscore", "completeness"]
    }


def _norm_high(rows, name, basis):
    lo, hi = basis[name]
    return np.clip((_col(rows, name) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def _norm_low(rows, name, basis):
    lo, hi = basis[name]
    return np.clip((hi - _col(rows, name)) / max(hi - lo, 1e-12), 0.0, 1.0)


def _target_composite(rows, basis, app_weights=(0.4, 0.4, 0.2), geo_weights=(0.35, 0.35, 0.30)):
    psnr_w, ssim_w, lpips_w = app_weights
    fscore_w, comp_w, chamfer_w = geo_weights
    q_app = (
        psnr_w * _norm_high(rows, "psnr", basis)
        + ssim_w * _norm_high(rows, "ssim", basis)
        + lpips_w * _norm_low(rows, "lpips", basis)
    )
    q_geo = (
        fscore_w * _norm_high(rows, "fscore", basis)
        + comp_w * _norm_high(rows, "completeness", basis)
        + chamfer_w * _norm_low(rows, "chamfer", basis)
    )
    return np.clip(0.45 * q_app + 0.45 * q_geo + 0.10 * q_app * q_geo, 0.0, 1.0)


def _linear_features(rows, include_joint):
    q_rgb = np.clip(_col(rows, "q_rgb"), 0.0, 1.0)
    r_depth = np.clip(_col(rows, "r_depth"), 0.0, 1.0)
    feats = [q_rgb, r_depth]
    if include_joint:
        feats.append(q_rgb * r_depth)
    return np.stack(feats, axis=1)


def _public_linear_fit(fit):
    weights = list(fit["weights"])
    out = {
        "bias": float(fit["bias"]),
        "w_rgb": float(weights[0]) if len(weights) > 0 else 0.0,
        "w_depth": float(weights[1]) if len(weights) > 1 else 0.0,
        "w_joint": float(weights[2]) if len(weights) > 2 else 0.0,
        "clip_ratio": float(fit.get("clip_ratio", 0.0)),
    }
    return out


def _predict_linear(rows, fit):
    raw = float(fit["bias"]) + _linear_features(rows, fit["include_joint"]) @ np.asarray(fit["weights"])
    return np.clip(raw, 0.0, 1.0)


def _fit_for_target(rows, lambda_rgb_grid, lambda_depth_grid, include_joint, target_mode, app_weights, geo_weights):
    if target_mode == "geometry":
        from evaluate_geometry_saturation_surrogate import _fit
        return _fit(rows, lambda_rgb_grid, lambda_depth_grid, include_joint)

    basis = _basis_composite(rows)
    y = _target_composite(rows, basis, app_weights, geo_weights)
    best = None
    for lambda_rgb in lambda_rgb_grid:
        for lambda_depth in lambda_depth_grid:
            x = _features(rows, lambda_rgb, lambda_depth, include_joint)
            bias, weights, raw, pred = _fit_nonnegative(x, y)
            diag = _diagnostics(y, pred)
            candidate = {
                "bias": bias,
                "weights": weights,
                "lambda_rgb": float(lambda_rgb),
                "lambda_depth": float(lambda_depth),
                "include_joint": include_joint,
                "basis": basis,
                "app_weights": tuple(float(x) for x in app_weights),
                "geo_weights": tuple(float(x) for x in geo_weights),
                "clip_ratio": float(np.mean((raw < 0.0) | (raw > 1.0))),
                **diag,
            }
            key = (candidate["r2"], -candidate["rmse"], candidate["spearman"])
            best_key = None if best is None else (best["r2"], -best["rmse"], best["spearman"])
            if best is None or key > best_key:
                best = candidate
    return best


def _fit_linear_for_target(rows, target_mode, include_joint, app_weights, geo_weights, basis):
    if target_mode == "geometry":
        y = _target(rows, basis)
    else:
        y = _target_composite(rows, basis, app_weights, geo_weights)
    x = _linear_features(rows, include_joint)
    bias, weights, raw, pred = _fit_nonnegative(x, y)
    return {
        "bias": bias,
        "weights": weights,
        "include_joint": include_joint,
        "basis": basis,
        "app_weights": tuple(float(x) for x in app_weights),
        "geo_weights": tuple(float(x) for x in geo_weights),
        "clip_ratio": float(np.mean((raw < 0.0) | (raw > 1.0))),
        **_diagnostics(y, pred),
    }


def _target_for_mode(rows, fit, target_mode):
    if target_mode == "geometry":
        return _target(rows, fit["basis"])
    return _target_composite(rows, fit["basis"], fit["app_weights"], fit["geo_weights"])


def _trajectory_id(row):
    if row.get("trajectory_id"):
        return row["trajectory_id"]
    match = re.search(r"traj\d+", row.get("condition", ""))
    if match:
        return match.group(0)
    raise ValueError(f"cannot infer trajectory from row: {row.get('condition')}")


def _traj_key(traj):
    match = re.fullmatch(r"traj(\d+)", traj)
    return int(match.group(1)) if match else traj


def _write_plot(path, results):
    if not path:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    trajs = [r["trajectory_id"] for r in results]
    fit_r2 = [r["fit"]["r2"] for r in results]
    fit_sp = [r["fit"]["spearman"] for r in results]
    fit_rmse = [r["fit"]["rmse"] for r in results]
    depth_w = [r["fit"]["w_depth"] for r in results]
    rgb_w = [r["fit"]["w_rgb"] for r in results]
    joint_w = [r["fit"]["w_joint"] for r in results]
    has_test = any("test" in r for r in results)
    test_r2   = [r["test"]["r2"]       if "test" in r else None for r in results]
    test_sp   = [r["test"]["spearman"] if "test" in r else None for r in results]
    test_rmse = [r["test"]["rmse"]     if "test" in r else None for r in results]

    nrows = 3 if has_test else 2
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 4 * nrows), dpi=170)

    x = np.arange(len(trajs))
    bar_w = 0.38

    ax = axes[0, 0]
    if has_test:
        ax.bar(x - bar_w / 2, fit_r2, width=bar_w, label="train", color="#4b83c4")
        ax.bar(x + bar_w / 2, [v if v is not None else 0 for v in test_r2],
               width=bar_w, label="test", color="#f0a050")
        ax.set_xticks(x); ax.set_xticklabels(trajs, rotation=45)
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.bar(trajs, fit_r2, color="#4b83c4")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("R²  (train vs test)" if has_test else "per-trajectory fit R²")
    ax.set_ylim(min(-0.05, min(fit_r2) - 0.05), 1.05)
    ax.tick_params(axis="x", rotation=45)

    ax = axes[0, 1]
    if has_test:
        ax.bar(x - bar_w / 2, fit_sp, width=bar_w, label="train", color="#5aa36f")
        ax.bar(x + bar_w / 2, [v if v is not None else 0 for v in test_sp],
               width=bar_w, label="test", color="#f0a050")
        ax.set_xticks(x); ax.set_xticklabels(trajs, rotation=45)
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.bar(trajs, fit_sp, color="#5aa36f")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Spearman  (train vs test)" if has_test else "per-trajectory Spearman")
    ax.set_ylim(min(-0.05, min(fit_sp) - 0.05), 1.05)
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1, 0]
    if has_test:
        ax.bar(x - bar_w / 2, fit_rmse, width=bar_w, label="train", color="#c47c46")
        ax.bar(x + bar_w / 2, [v if v is not None else 0 for v in test_rmse],
               width=bar_w, label="test", color="#f0a050")
        ax.set_xticks(x); ax.set_xticklabels(trajs, rotation=45)
        ax.legend(frameon=False, fontsize=8)
    else:
        ax.bar(trajs, fit_rmse, color="#c47c46")
    ax.set_title("RMSE  (train vs test)" if has_test else "per-trajectory RMSE")
    ax.tick_params(axis="x", rotation=45)

    ax = axes[1, 1]
    ax.bar(x - 0.25, rgb_w, width=0.25, label="w_rgb")
    ax.bar(x, depth_w, width=0.25, label="w_depth")
    ax.bar(x + 0.25, joint_w, width=0.25, label="w_joint")
    ax.set_xticks(x); ax.set_xticklabels(trajs, rotation=45)
    ax.set_title("nonnegative weights")
    ax.legend(frameon=False)

    if has_test:
        ax = axes[2, 0]
        delta_r2 = [tr - (te if te is not None else tr) for tr, te in zip(fit_r2, test_r2)]
        colors = ["#e05050" if d > 0.05 else "#5aa36f" for d in delta_r2]
        ax.bar(trajs, delta_r2, color=colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("R² gap (train − test)")
        ax.tick_params(axis="x", rotation=45)

        ax = axes[2, 1]
        delta_sp = [tr - (te if te is not None else tr) for tr, te in zip(fit_sp, test_sp)]
        colors = ["#e05050" if d > 0.05 else "#5aa36f" for d in delta_sp]
        ax.bar(trajs, delta_sp, color=colors)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title("Spearman gap (train − test)")
        ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_yaml(path, results):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("per_trajectory_surrogates:\n")
        for item in results:
            fit = item["fit"]
            f.write(f"  {item['trajectory_id']}:\n")
            f.write(f"    target: {item['target_mode']}\n")
            f.write("    model: saturation\n")
            for key in ["bias", "w_rgb", "w_depth", "w_joint", "lambda_rgb", "lambda_depth"]:
                f.write(f"    {key}: {float(fit[key]):.8f}\n")
            f.write(f"    include_joint: {str(fit['include_joint']).lower()}\n")
            f.write(f"    r2: {float(fit['r2']):.8f}\n")
            f.write(f"    spearman: {float(fit['spearman']):.8f}\n")
            f.write(f"    rmse: {float(fit['rmse']):.8f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-yaml", default=None)
    ap.add_argument("--plot-out", default=None)
    ap.add_argument("--min-samples", type=int, default=20)
    ap.add_argument("--target", choices=["geometry", "composite"], default="geometry")
    ap.add_argument("--app-weights", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                    help="Q_app = w_psnr*PSNR + w_ssim*SSIM + w_lpips*(1-LPIPS). "
                         "Default matches paper setting (0.6/0.2/0.2).")
    ap.add_argument("--geo-weights", type=float, nargs=3, default=[0.35, 0.35, 0.30])
    ap.add_argument("--allow-joint", action="store_true")
    ap.add_argument("--lambda-rgb-grid", type=float, nargs="+", default=[1, 2, 3, 4, 6, 8, 10, 12])
    ap.add_argument("--lambda-depth-grid", type=float, nargs="+", default=[0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12])
    ap.add_argument("--test-fraction", type=float, default=0.0,
                    help="Fraction of each trajectory's conditions held out for test evaluation (0 = no split).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.target == "composite":
        for name, weights in [("app", args.app_weights), ("geo", args.geo_weights)]:
            total = sum(weights)
            if total <= 0.0:
                raise SystemExit(f"--{name}-weights must have a positive sum")
            if not np.isclose(total, 1.0):
                raise SystemExit(f"--{name}-weights must sum to 1.0, got {total:.8f}")
    if not 0.0 <= args.test_fraction < 1.0:
        raise SystemExit("--test-fraction must be in [0, 1)")

    raw_rows = _read_rows(args.metrics)
    rows = _finite_composite_rows(raw_rows) if args.target == "composite" else _finite_rows(raw_rows)
    by_traj = defaultdict(list)
    for row in rows:
        by_traj[_trajectory_id(row)].append(row)

    rng = np.random.default_rng(args.seed)

    results = []
    for traj in sorted(by_traj, key=_traj_key):
        traj_rows = by_traj[traj]
        if len(traj_rows) < args.min_samples:
            continue

        if args.test_fraction > 0.0:
            idx = rng.permutation(len(traj_rows))
            n_test = max(1, int(round(len(traj_rows) * args.test_fraction)))
            test_idx = set(idx[:n_test].tolist())
            train_rows = [r for i, r in enumerate(traj_rows) if i not in test_idx]
            test_rows  = [r for i, r in enumerate(traj_rows) if i in test_idx]
        else:
            train_rows = traj_rows
            test_rows  = []

        fit = _fit_for_target(
            train_rows,
            args.lambda_rgb_grid,
            args.lambda_depth_grid,
            args.allow_joint,
            args.target,
            args.app_weights,
            args.geo_weights,
        )
        linear_fit = _fit_linear_for_target(
            train_rows,
            args.target,
            args.allow_joint,
            args.app_weights,
            args.geo_weights,
            fit["basis"],
        )
        y_train = _target_for_mode(train_rows, fit, args.target)
        pred_train = _predict(train_rows, fit)
        public = _public_fit(fit)
        public.update(_diagnostics(y_train, pred_train))
        linear_public = _public_linear_fit(linear_fit)
        linear_public.update(_diagnostics(y_train, _predict_linear(train_rows, linear_fit)))

        entry = {
            "trajectory_id": traj,
            "samples": len(traj_rows),
            "train_samples": len(train_rows),
            "test_samples": len(test_rows),
            "target_mode": args.target,
            "fit": public,
            "linear_fit": linear_public,
        }

        if test_rows:
            y_test = _target_for_mode(test_rows, fit, args.target)
            pred_test = _predict(test_rows, fit)
            entry["test"] = _diagnostics(y_test, pred_test)
            entry["linear_test"] = _diagnostics(y_test, _predict_linear(test_rows, linear_fit))

        results.append(entry)

    if not results:
        raise SystemExit("no trajectories met --min-samples")

    best = max(results, key=lambda r: (r["fit"]["r2"], r["fit"]["spearman"], -r["fit"]["rmse"]))
    out = {
        "target": (
            (
                "per-trajectory composite Q_gt = clip(0.45 Q_app + 0.45 Q_geo + "
                f"0.10 Q_app Q_geo, 0, 1), app_weights={args.app_weights}, "
                f"geo_weights={args.geo_weights}"
            )
            if args.target == "composite"
            else "per-trajectory geometry-only q_geometry from chamfer/fscore/completeness"
        ),
        "metrics": args.metrics,
        "num_trajectories": len(results),
        "best_trajectory": best["trajectory_id"],
        "selection_rule": "max fit R2, then Spearman, then lower RMSE",
        "trajectories": results,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    _write_yaml(args.out_yaml, results)
    _write_plot(args.plot_out, results)
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
