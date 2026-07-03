"""Select a stable, explainable per-scene GSFusion saturation surrogate."""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict

import numpy as np


METRIC_COLUMNS = [
    "q_rgb", "r_depth", "psnr", "ssim", "lpips",
    "chamfer", "fscore", "completeness",
]


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _finite_rows(rows):
    out = []
    for row in rows:
        if all(math.isfinite(float(row[name])) for name in METRIC_COLUMNS):
            out.append(row)
    return out


def _col(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def _basis(rows):
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


def _target(rows, basis):
    q_render = (
        0.4 * _norm_high(rows, "psnr", basis)
        + 0.4 * _norm_high(rows, "ssim", basis)
        + 0.2 * _norm_low(rows, "lpips", basis)
    )
    q_geometry = (
        0.35 * _norm_high(rows, "fscore", basis)
        + 0.35 * _norm_high(rows, "completeness", basis)
        + 0.30 * _norm_low(rows, "chamfer", basis)
    )
    return np.clip(0.45 * q_render + 0.45 * q_geometry + 0.10 * q_render * q_geometry, 0.0, 1.0)


def _rankdata(x):
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def _spearman(x, y):
    if len(x) < 2:
        return float("nan")
    rx = _rankdata(x)
    ry = _rankdata(y)
    sx = rx.std()
    sy = ry.std()
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def _features(rows, lambda_rgb, lambda_depth, include_joint):
    q = np.clip(_col(rows, "q_rgb"), 0.0, 1.0)
    r = np.clip(_col(rows, "r_depth"), 0.0, 1.0)
    g_rgb = 1.0 - np.exp(-float(lambda_rgb) * q)
    g_depth = 1.0 - np.exp(-float(lambda_depth) * r)
    cols = [g_rgb, g_depth]
    if include_joint:
        cols.append(g_rgb * g_depth)
    return np.column_stack(cols)


def _fit_nonnegative(x, y, min_weights):
    best = None
    best_sse = float("inf")
    n_features = x.shape[1]
    y_adjusted = y - x @ min_weights
    for mask in range(1 << n_features):
        active = [idx for idx in range(n_features) if mask & (1 << idx)]
        design = np.ones((len(y), 1 + len(active)), dtype=np.float64)
        if active:
            design[:, 1:] = x[:, active]
        coef, *_ = np.linalg.lstsq(design, y_adjusted, rcond=None)
        if active and np.any(coef[1:] < 0.0):
            continue
        weights = min_weights.copy()
        if active:
            weights[active] += coef[1:]
        raw = float(coef[0]) + x @ weights
        pred = np.clip(raw, 0.0, 1.0)
        sse = float(np.square(pred - y).sum())
        if sse < best_sse:
            best_sse = sse
            best = (float(coef[0]), weights, raw, pred)
    return best


def _diagnostics(y, pred):
    rmse = float(np.sqrt(np.square(pred - y).mean()))
    mae = float(np.abs(pred - y).mean())
    r2 = float(1.0 - np.square(pred - y).sum() / max(np.square(y - y.mean()).sum(), 1e-12))
    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "spearman": _spearman(y, pred),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
    }


def _traj_id(row):
    m = re.search(r"traj(\d+)", row["condition"])
    return m.group(1) if m else row["condition"]


def _folds_by_trajectory(rows):
    by_traj = defaultdict(list)
    for row in rows:
        by_traj[_traj_id(row)].append(row)
    if len(by_traj) < 3:
        return []
    folds = []
    for traj, valid_rows in sorted(by_traj.items()):
        train_rows = [row for key, values in by_traj.items() if key != traj for row in values]
        if train_rows and valid_rows:
            folds.append((traj, train_rows, valid_rows))
    return folds


def _evaluate_params(train_rows, valid_rows, lambda_rgb, lambda_depth, include_joint, min_weights):
    train_basis = _basis(train_rows)
    y_train = _target(train_rows, train_basis)
    fit = _fit_nonnegative(
        _features(train_rows, lambda_rgb, lambda_depth, include_joint),
        y_train,
        min_weights,
    )
    bias, weights, _, _ = fit
    y_valid = _target(valid_rows, train_basis)
    pred = np.clip(
        bias + _features(valid_rows, lambda_rgb, lambda_depth, include_joint) @ weights,
        0.0,
        1.0,
    )
    return _diagnostics(y_valid, pred)


def _select_params(rows, lambda_rgb_grid, lambda_depth_grid, include_joint, min_weights):
    folds = _folds_by_trajectory(rows)
    if not folds:
        raise ValueError("need at least three trajectories for trajectory-held-out validation")
    best = None
    for lambda_rgb in lambda_rgb_grid:
        for lambda_depth in lambda_depth_grid:
            diagnostics = [
                _evaluate_params(train_rows, valid_rows, lambda_rgb, lambda_depth, include_joint, min_weights)
                for _, train_rows, valid_rows in folds
            ]
            mean = {
                key: float(np.nanmean([diag[key] for diag in diagnostics]))
                for key in ["rmse", "mae", "r2", "spearman"]
            }
            candidate = {
                "lambda_rgb": float(lambda_rgb),
                "lambda_depth": float(lambda_depth),
                "cv": mean,
            }
            if best is None:
                best = candidate
                continue
            key = (candidate["cv"]["spearman"], candidate["cv"]["r2"], -candidate["cv"]["rmse"])
            best_key = (best["cv"]["spearman"], best["cv"]["r2"], -best["cv"]["rmse"])
            if key > best_key:
                best = candidate
    return best


def _fit_final(rows, lambda_rgb, lambda_depth, include_joint, min_weights):
    basis = _basis(rows)
    y = _target(rows, basis)
    bias, weights, raw, pred = _fit_nonnegative(
        _features(rows, lambda_rgb, lambda_depth, include_joint),
        y,
        min_weights,
    )
    result = {
        "bias": bias,
        "w_rgb": float(weights[0]),
        "w_depth": float(weights[1]),
        "w_joint": float(weights[2]) if include_joint else 0.0,
        "lambda_rgb": float(lambda_rgb),
        "lambda_depth": float(lambda_depth),
        "include_joint": include_joint,
        "clip_ratio": float(np.mean((raw < 0.0) | (raw > 1.0))),
        "basis": basis,
        **_diagnostics(y, pred),
    }
    return result


def _evaluate_test(rows, fit):
    y = _target(rows, fit["basis"])
    pred = _predict(rows, fit)
    return _diagnostics(y, pred)


def _predict(rows, fit):
    weights = [fit["w_rgb"], fit["w_depth"]]
    include_joint = bool(fit["include_joint"])
    if include_joint:
        weights.append(fit["w_joint"])
    return np.clip(
        fit["bias"] + _features(rows, fit["lambda_rgb"], fit["lambda_depth"], include_joint) @ np.asarray(weights),
        0.0,
        1.0,
    )


def _bootstrap(rows, fit, n_bootstrap, seed):
    if n_bootstrap <= 0:
        return {}
    rng = np.random.default_rng(seed)
    vals = defaultdict(list)
    min_weights = np.zeros(3 if fit["include_joint"] else 2, dtype=np.float64)
    for _ in range(n_bootstrap):
        sample = [rows[idx] for idx in rng.integers(0, len(rows), size=len(rows))]
        cur = _fit_final(
            sample,
            fit["lambda_rgb"],
            fit["lambda_depth"],
            fit["include_joint"],
            min_weights,
        )
        for key in ["bias", "w_rgb", "w_depth", "w_joint", "r2", "spearman", "rmse"]:
            vals[key].append(cur[key])
    return {
        key: {
            "mean": float(np.mean(values)),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
        for key, values in vals.items()
    }


def _write_yaml(path, scene, fit):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("mapping_quality:\n")
        f.write("  mapping_quality_mode: gsfusion_surrogate\n")
        f.write("  gsfusion_weights:\n")
        f.write("    model: scene_saturation\n")
        f.write(f"    scene: {scene}\n")
        f.write(f"    bias: {fit['bias']:.8f}\n")
        f.write(f"    w_rgb: {fit['w_rgb']:.8f}\n")
        f.write(f"    w_depth: {fit['w_depth']:.8f}\n")
        f.write(f"    w_joint: {fit['w_joint']:.8f}\n")
        f.write(f"    lambda_rgb: {fit['lambda_rgb']:.8f}\n")
        f.write(f"    lambda_depth: {fit['lambda_depth']:.8f}\n")
        f.write(f"    include_joint: {str(fit['include_joint']).lower()}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-metrics", default="outputs/multiscene_surrogate/gsfusion_real_metrics_multiscene_fit.csv")
    ap.add_argument("--test-metrics", default="outputs/multiscene_surrogate/gsfusion_real_metrics_multiscene_test.csv")
    ap.add_argument("--out-json", default="outputs/multiscene_surrogate/stable_scene_surrogate_selection.json")
    ap.add_argument("--out-yaml", default="outputs/multiscene_surrogate/gsfusion_surrogate_weights_stable_scene.yaml")
    ap.add_argument("--lambda-rgb-grid", type=float, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--lambda-depth-grid", type=float, nargs="+", default=[0.5, 1, 1.5, 2, 3])
    ap.add_argument("--min-samples", type=int, default=48)
    ap.add_argument("--allow-joint", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fit_rows = _finite_rows(_read_rows(args.fit_metrics))
    test_rows = _finite_rows(_read_rows(args.test_metrics)) if args.test_metrics else []
    fit_by_scene = defaultdict(list)
    test_by_scene = defaultdict(list)
    for row in fit_rows:
        fit_by_scene[row["source_scene"]].append(row)
    for row in test_rows:
        test_by_scene[row["source_scene"]].append(row)

    include_joint = bool(args.allow_joint)
    min_weights = np.zeros(3 if include_joint else 2, dtype=np.float64)
    results = []
    for scene, rows in sorted(fit_by_scene.items()):
        if len(rows) < args.min_samples:
            continue
        try:
            selected = _select_params(
                rows,
                args.lambda_rgb_grid,
                args.lambda_depth_grid,
                include_joint,
                min_weights,
            )
        except ValueError:
            continue
        fit = _fit_final(rows, selected["lambda_rgb"], selected["lambda_depth"], include_joint, min_weights)
        result = {
            "scene": scene,
            "n_train": len(rows),
            "n_test": len(test_by_scene.get(scene, [])),
            "cv": selected["cv"],
            "fit": {key: value for key, value in fit.items() if key != "basis"},
        }
        if scene in test_by_scene:
            result["heldout_test"] = _evaluate_test(test_by_scene[scene], fit)
        result["bootstrap"] = _bootstrap(rows, fit, args.bootstrap, args.seed)
        result["_fit_with_basis"] = fit
        results.append(result)

    if not results:
        raise SystemExit("no eligible scenes")

    candidates = [result for result in results if "heldout_test" in result] or results

    def score(result):
        heldout = result.get("heldout_test")
        primary = heldout or result["cv"]
        return (
            primary["spearman"],
            primary["r2"],
            -primary["rmse"],
            result["cv"]["spearman"],
        )

    results.sort(key=score, reverse=True)
    candidates.sort(key=score, reverse=True)
    best = candidates[0]
    best_fit = best.pop("_fit_with_basis")
    for result in results:
        result.pop("_fit_with_basis", None)

    out = {
        "selection_rule": "if any scene has held-out test metrics, rank those first; otherwise rank by trajectory-held-out CV",
        "model": "additive_saturation" if not include_joint else "saturation_with_joint",
        "target": "scene-local q_render/q_geometry composite",
        "best_scene": best["scene"],
        "best_function": {key: value for key, value in best_fit.items() if key != "basis"},
        "scenes": results,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    _write_yaml(args.out_yaml, best["scene"], best_fit)
    print(json.dumps({
        "best_scene": best["scene"],
        "best_function": out["best_function"],
        "heldout_test": best.get("heldout_test"),
        "cv": best["cv"],
        "wrote_json": args.out_json,
        "wrote_yaml": args.out_yaml,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
