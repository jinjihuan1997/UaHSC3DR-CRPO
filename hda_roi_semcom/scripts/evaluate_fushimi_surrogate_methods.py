"""Evaluate fushimi surrogate variants using existing GSFusion metrics."""

import csv
import json
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_COLUMNS = [
    "q_rgb", "r_depth", "psnr", "ssim", "lpips",
    "chamfer", "fscore", "completeness",
]
FIT_CSV = "outputs/multiscene_surrogate/gsfusion_real_metrics_multiscene_fit.csv"
TEST_CSV = "outputs/multiscene_surrogate/gsfusion_real_metrics_multiscene_test.csv"
OUT_JSON = "outputs/multiscene_surrogate/fushimi_surrogate_method_tests.json"
OUT_PNG = "outputs/multiscene_surrogate/plots/fushimi_surrogate_method_tests.png"


def read_rows(path, scene):
    with open(path, newline="") as f:
        rows = [row for row in csv.DictReader(f) if row["source_scene"] == scene]
    return [
        row for row in rows
        if all(math.isfinite(float(row[name])) for name in METRIC_COLUMNS)
    ]


def col(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def basis(rows):
    return {
        name: (np.nanpercentile(col(rows, name), 5), np.nanpercentile(col(rows, name), 95))
        for name in ["psnr", "ssim", "lpips", "chamfer", "fscore", "completeness"]
    }


def norm_high(rows, name, b):
    lo, hi = b[name]
    return np.clip((col(rows, name) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def norm_low(rows, name, b):
    lo, hi = b[name]
    return np.clip((hi - col(rows, name)) / max(hi - lo, 1e-12), 0.0, 1.0)


def target(rows, b):
    q_render = (
        0.4 * norm_high(rows, "psnr", b)
        + 0.4 * norm_high(rows, "ssim", b)
        + 0.2 * norm_low(rows, "lpips", b)
    )
    q_geometry = (
        0.35 * norm_high(rows, "fscore", b)
        + 0.35 * norm_high(rows, "completeness", b)
        + 0.30 * norm_low(rows, "chamfer", b)
    )
    return np.clip(0.45 * q_render + 0.45 * q_geometry + 0.10 * q_render * q_geometry, 0.0, 1.0)


def traj_id(row):
    m = re.search(r"traj(\d+)", row["condition"])
    return m.group(1) if m else row["condition"]


def rankdata(x):
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def diagnostics(y, pred):
    return {
        "rmse": float(np.sqrt(np.square(pred - y).mean())),
        "mae": float(np.abs(pred - y).mean()),
        "r2": float(1.0 - np.square(pred - y).sum() / max(np.square(y - y.mean()).sum(), 1e-12)),
        "spearman": spearman(y, pred),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
    }


def classify(diag):
    if diag["spearman"] >= 0.6 and diag["rmse"] <= 0.10:
        return "good"
    if diag["spearman"] >= 0.35 and diag["rmse"] <= 0.16:
        return "medium"
    return "bad"


def sat_features(rows, lambda_rgb, lambda_depth, mode):
    q = np.clip(col(rows, "q_rgb"), 0.0, 1.0)
    r = np.clip(col(rows, "r_depth"), 0.0, 1.0)
    gr = 1.0 - np.exp(-lambda_rgb * q)
    gd = 1.0 - np.exp(-lambda_depth * r)
    if mode == "rgb":
        return np.column_stack([gr])
    if mode == "additive":
        return np.column_stack([gr, gd])
    if mode == "joint":
        return np.column_stack([gr, gd, gr * gd])
    raise ValueError(mode)


def linear_features(rows, mode):
    q = np.clip(col(rows, "q_rgb"), 0.0, 1.0)
    r = np.clip(col(rows, "r_depth"), 0.0, 1.0)
    if mode == "linear":
        return np.column_stack([q, r])
    if mode == "quadratic":
        return np.column_stack([q, r, q * r, q * q, r * r])
    raise ValueError(mode)


def fit_nonnegative(x, y):
    best = None
    best_sse = float("inf")
    for mask in range(1 << x.shape[1]):
        active = [idx for idx in range(x.shape[1]) if mask & (1 << idx)]
        design = np.ones((len(y), 1 + len(active)), dtype=np.float64)
        if active:
            design[:, 1:] = x[:, active]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        if active and np.any(coef[1:] < 0.0):
            continue
        weights = np.zeros(x.shape[1], dtype=np.float64)
        if active:
            weights[active] = coef[1:]
        raw = float(coef[0]) + x @ weights
        pred = np.clip(raw, 0.0, 1.0)
        sse = float(np.square(pred - y).sum())
        if sse < best_sse:
            best_sse = sse
            best = (float(coef[0]), weights)
    return best


def fit_ridge(x, y, alpha=1e-3):
    design = np.column_stack([np.ones(len(y)), x])
    eye = np.eye(design.shape[1])
    eye[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + alpha * eye, design.T @ y)
    return float(coef[0]), coef[1:]


def predict_linear(rows, params, feature_mode):
    bias, weights = params
    return np.clip(bias + linear_features(rows, feature_mode) @ weights, 0.0, 1.0)


def predict_sat(rows, params, mode):
    bias, weights, lambda_rgb, lambda_depth = params
    return np.clip(bias + sat_features(rows, lambda_rgb, lambda_depth, mode) @ weights, 0.0, 1.0)


def pava(y):
    levels = []
    weights = []
    counts = []
    for value in y:
        levels.append(float(value))
        weights.append(1.0)
        counts.append(1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            total_w = weights[-2] + weights[-1]
            avg = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / total_w
            total_count = counts[-2] + counts[-1]
            levels[-2:] = [avg]
            weights[-2:] = [total_w]
            counts[-2:] = [total_count]
    return np.repeat(levels, counts)


def fit_isotonic_q(rows, y):
    q = np.clip(col(rows, "q_rgb"), 0.0, 1.0)
    order = np.argsort(q)
    x_sorted = q[order]
    y_iso = pava(y[order])
    uniq_x = []
    uniq_y = []
    for value in np.unique(x_sorted):
        mask = x_sorted == value
        uniq_x.append(float(value))
        uniq_y.append(float(y_iso[mask].mean()))
    return np.asarray(uniq_x), np.asarray(uniq_y)


def predict_isotonic(rows, params):
    x, y = params
    return np.clip(np.interp(np.clip(col(rows, "q_rgb"), 0.0, 1.0), x, y, left=y[0], right=y[-1]), 0.0, 1.0)


def fit_calibration(raw_oof, y_oof):
    design = np.column_stack([np.ones(len(raw_oof)), raw_oof])
    coef, *_ = np.linalg.lstsq(design, y_oof, rcond=None)
    return float(coef[0]), float(coef[1])


def apply_calibration(pred, calib):
    a, b = calib
    return np.clip(a + b * pred, 0.0, 1.0)


def folds(rows):
    by_traj = {}
    for row in rows:
        by_traj.setdefault(traj_id(row), []).append(row)
    return [
        ([row for key, values in by_traj.items() if key != valid_key for row in values], valid_rows)
        for valid_key, valid_rows in sorted(by_traj.items())
    ]


def select_saturation(train_rows, mode, y_train):
    best = None
    for lambda_rgb in [1.0, 2.0, 3.0, 4.0]:
        for lambda_depth in [0.5, 1.0, 1.5, 2.0, 3.0]:
            fold_scores = []
            for tr, va in folds(train_rows):
                b = basis(tr)
                y_tr = target(tr, b)
                y_va = target(va, b)
                params = (*fit_nonnegative(sat_features(tr, lambda_rgb, lambda_depth, mode), y_tr), lambda_rgb, lambda_depth)
                pred = predict_sat(va, params, mode)
                fold_scores.append(diagnostics(y_va, pred))
            score = (
                np.nanmean([s["spearman"] for s in fold_scores]),
                np.nanmean([s["r2"] for s in fold_scores]),
                -np.nanmean([s["rmse"] for s in fold_scores]),
            )
            if best is None or score > best[0]:
                best = (score, lambda_rgb, lambda_depth)
    lambda_rgb, lambda_depth = best[1], best[2]
    params = (*fit_nonnegative(sat_features(train_rows, lambda_rgb, lambda_depth, mode), y_train), lambda_rgb, lambda_depth)
    return params


def oof_predictions(train_rows, model_name):
    out_pred = []
    out_y = []
    for tr, va in folds(train_rows):
        b = basis(tr)
        y_tr = target(tr, b)
        y_va = target(va, b)
        if model_name == "rgb_saturation":
            params = select_saturation(tr, "rgb", y_tr)
            pred = predict_sat(va, params, "rgb")
        else:
            raise ValueError(model_name)
        out_pred.extend(pred.tolist())
        out_y.extend(y_va.tolist())
    return np.asarray(out_pred), np.asarray(out_y)


def main():
    train_rows = read_rows(FIT_CSV, "fushimi")
    test_rows = read_rows(TEST_CSV, "fushimi")
    b_train = basis(train_rows)
    y_train = target(train_rows, b_train)
    y_test = target(test_rows, b_train)

    methods = []

    for name, mode in [
        ("rgb_saturation", "rgb"),
        ("rgb_depth_additive_saturation", "additive"),
        ("rgb_depth_joint_saturation", "joint"),
    ]:
        params = select_saturation(train_rows, mode, y_train)
        pred = predict_sat(test_rows, params, mode)
        diag = diagnostics(y_test, pred)
        methods.append({
            "method": name,
            "params": {
                "bias": params[0],
                "weights": params[1].tolist(),
                "lambda_rgb": params[2],
                "lambda_depth": params[3],
            },
            "test": {**diag, "grade": classify(diag)},
        })

    base_params = select_saturation(train_rows, "rgb", y_train)
    oof_pred, oof_y = oof_predictions(train_rows, "rgb_saturation")
    calib = fit_calibration(oof_pred, oof_y)
    pred = apply_calibration(predict_sat(test_rows, base_params, "rgb"), calib)
    diag = diagnostics(y_test, pred)
    methods.append({
        "method": "rgb_saturation_oof_affine_calibration",
        "params": {
            "base": {
                "bias": base_params[0],
                "weights": base_params[1].tolist(),
                "lambda_rgb": base_params[2],
                "lambda_depth": base_params[3],
            },
            "calibration": {"a": calib[0], "b": calib[1]},
        },
        "test": {**diag, "grade": classify(diag)},
    })

    for name, feature_mode, fitter in [
        ("linear_qr_nonnegative", "linear", fit_nonnegative),
        ("quadratic_qr_ridge_upper_bound", "quadratic", fit_ridge),
    ]:
        params = fitter(linear_features(train_rows, feature_mode), y_train)
        pred = predict_linear(test_rows, params, feature_mode)
        diag = diagnostics(y_test, pred)
        methods.append({
            "method": name,
            "params": {"bias": params[0], "weights": params[1].tolist()},
            "test": {**diag, "grade": classify(diag)},
        })

    iso_params = fit_isotonic_q(train_rows, y_train)
    pred = predict_isotonic(test_rows, iso_params)
    diag = diagnostics(y_test, pred)
    methods.append({
        "method": "q_rgb_isotonic_upper_bound",
        "params": {"knots": len(iso_params[0])},
        "test": {**diag, "grade": classify(diag)},
    })

    methods.sort(key=lambda m: (m["test"]["grade"] == "good", m["test"]["spearman"], -m["test"]["rmse"]), reverse=True)
    out = {
        "scene": "fushimi",
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "target_basis": "fit_fushimi train percentiles applied to test_fushimi",
        "grading": {
            "good": "Spearman >= 0.60 and RMSE <= 0.10",
            "medium": "Spearman >= 0.35 and RMSE <= 0.16",
            "bad": "otherwise",
        },
        "methods": methods,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)

    names = [m["method"] for m in methods]
    spears = [m["test"]["spearman"] for m in methods]
    rmses = [m["test"]["rmse"] for m in methods]
    grades = [m["test"]["grade"] for m in methods]
    colors = [{"good": "#2ca02c", "medium": "#ff7f0e", "bad": "#d62728"}[g] for g in grades]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)
    y_pos = np.arange(len(names))
    axes[0].barh(y_pos, spears, color=colors)
    axes[0].axvline(0.35, color="0.4", linestyle="--", linewidth=1)
    axes[0].axvline(0.60, color="0.4", linestyle=":", linewidth=1)
    axes[0].set_yticks(y_pos, names)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("held-out Spearman")
    axes[0].set_title("ranking quality")
    axes[0].grid(True, axis="x", alpha=0.25)

    axes[1].barh(y_pos, rmses, color=colors)
    axes[1].axvline(0.10, color="0.4", linestyle=":", linewidth=1)
    axes[1].axvline(0.16, color="0.4", linestyle="--", linewidth=1)
    axes[1].set_yticks(y_pos, [])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("held-out RMSE")
    axes[1].set_title("absolute error")
    axes[1].grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG)
    print(json.dumps(out, indent=2, allow_nan=False))
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
