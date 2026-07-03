"""Fit/test a geometry-only saturation surrogate from GSFusion metrics."""

import argparse
import csv
import json
import math
import os

import numpy as np


REQUIRED = ["q_rgb", "r_depth", "chamfer", "fscore", "completeness"]


def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _finite_rows(rows):
    out = []
    for row in rows:
        try:
            if all(math.isfinite(float(row[name])) for name in REQUIRED):
                out.append(row)
        except (KeyError, ValueError):
            continue
    return out


def _col(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


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
    if rx.std() <= 1e-12 or ry.std() <= 1e-12:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (rx.std() * ry.std()))


def _diagnostics(y, pred):
    return {
        "rmse": float(np.sqrt(np.square(pred - y).mean())),
        "mae": float(np.abs(pred - y).mean()),
        "r2": float(1.0 - np.square(pred - y).sum() / max(np.square(y - y.mean()).sum(), 1e-12)),
        "spearman": _spearman(y, pred),
        "target_mean": float(y.mean()),
        "target_std": float(y.std()),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
    }


def _basis(rows):
    return {
        name: (
            float(np.nanpercentile(_col(rows, name), 5)),
            float(np.nanpercentile(_col(rows, name), 95)),
        )
        for name in ["chamfer", "fscore", "completeness"]
    }


def _norm_high(x, lo, hi):
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def _norm_low(x, lo, hi):
    return np.clip((hi - x) / max(hi - lo, 1e-12), 0.0, 1.0)


def _target(rows, basis):
    return np.clip(
        0.35 * _norm_high(_col(rows, "fscore"), *basis["fscore"])
        + 0.35 * _norm_high(_col(rows, "completeness"), *basis["completeness"])
        + 0.30 * _norm_low(_col(rows, "chamfer"), *basis["chamfer"]),
        0.0,
        1.0,
    )


def _features(rows, lambda_rgb, lambda_depth, include_joint):
    q = np.clip(_col(rows, "q_rgb"), 0.0, 1.0)
    r = np.clip(_col(rows, "r_depth"), 0.0, 1.0)
    g_rgb = 1.0 - np.exp(-float(lambda_rgb) * q)
    g_depth = 1.0 - np.exp(-float(lambda_depth) * r)
    cols = [g_rgb, g_depth]
    if include_joint:
        cols.append(g_rgb * g_depth)
    return np.column_stack(cols)


def _fit_nonnegative(x, y):
    best = None
    best_sse = float("inf")
    n_features = x.shape[1]
    for mask in range(1 << n_features):
        active = [idx for idx in range(n_features) if mask & (1 << idx)]
        design = np.ones((len(y), 1 + len(active)), dtype=np.float64)
        if active:
            design[:, 1:] = x[:, active]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        if active and np.any(coef[1:] < 0.0):
            continue
        weights = np.zeros(n_features, dtype=np.float64)
        if active:
            weights[active] = coef[1:]
        raw = float(coef[0]) + x @ weights
        pred = np.clip(raw, 0.0, 1.0)
        sse = float(np.square(pred - y).sum())
        if sse < best_sse:
            best_sse = sse
            best = (float(coef[0]), weights, raw, pred)
    return best


def _fit(rows, lambda_rgb_grid, lambda_depth_grid, include_joint):
    basis = _basis(rows)
    y = _target(rows, basis)
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
                "clip_ratio": float(np.mean((raw < 0.0) | (raw > 1.0))),
                **diag,
            }
            key = (candidate["r2"], -candidate["rmse"], candidate["spearman"])
            best_key = None if best is None else (best["r2"], -best["rmse"], best["spearman"])
            if best is None or key > best_key:
                best = candidate
    return best


def _predict(rows, fit):
    pred = fit["bias"] + _features(
        rows,
        fit["lambda_rgb"],
        fit["lambda_depth"],
        fit["include_joint"],
    ) @ fit["weights"]
    return np.clip(pred, 0.0, 1.0)


def _write_plot(path, fit_rows, test_rows, fit, fit_target, fit_pred, test_target, test_pred):
    if not path:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=170)
    axes[0].scatter(fit_target, fit_pred, s=24, alpha=0.75)
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=1)
    axes[0].set_title(f"fit n={len(fit_rows)}")
    axes[0].set_xlabel("real geometry quality")
    axes[0].set_ylabel("predicted")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    axes[1].scatter(test_target, test_pred, s=28, alpha=0.75, color="#c46a32")
    axes[1].plot([0, 1], [0, 1], color="black", linewidth=1)
    axes[1].set_title(f"held-out test n={len(test_rows)}")
    axes[1].set_xlabel("real geometry quality")
    axes[1].set_ylabel("predicted")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    q = np.linspace(0, 1, 150)
    weights = fit["weights"]
    for r_depth, label in [(0.25, "Rdep=.25"), (0.5, "Rdep=.50"), (0.75, "Rdep=.75"), (1.0, "Rdep=1")]:
        g_rgb = 1.0 - np.exp(-fit["lambda_rgb"] * q)
        g_depth = 1.0 - np.exp(-fit["lambda_depth"] * r_depth)
        pred = fit["bias"] + weights[0] * g_rgb + weights[1] * g_depth
        if fit["include_joint"]:
            pred = pred + weights[2] * g_rgb * g_depth
        axes[2].plot(q, np.clip(pred, 0, 1), label=label)
    axes[2].set_title("saturation curves")
    axes[2].set_xlabel("Qrgb")
    axes[2].set_ylabel("Ksur geometry")
    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
    axes[2].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _public_fit(fit):
    out = {key: value for key, value in fit.items() if key not in ("weights", "basis")}
    out["w_rgb"] = float(fit["weights"][0])
    out["w_depth"] = float(fit["weights"][1])
    out["w_joint"] = float(fit["weights"][2]) if fit["include_joint"] else 0.0
    out["basis"] = fit["basis"]
    return out


def _write_yaml(path, fit):
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("mapping_quality:\n")
        f.write("  mapping_quality_mode: gsfusion_surrogate\n")
        f.write("  gsfusion_weights:\n")
        f.write("    model: saturation\n")
        f.write(f"    bias: {fit['bias']:.8f}\n")
        f.write(f"    w_rgb: {float(fit['weights'][0]):.8f}\n")
        f.write(f"    w_depth: {float(fit['weights'][1]):.8f}\n")
        f.write(f"    w_joint: {float(fit['weights'][2]) if fit['include_joint'] else 0.0:.8f}\n")
        f.write(f"    lambda_rgb: {fit['lambda_rgb']:.8f}\n")
        f.write(f"    lambda_depth: {fit['lambda_depth']:.8f}\n")
        f.write(f"    include_joint: {str(fit['include_joint']).lower()}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-metrics", required=True)
    ap.add_argument("--test-metrics", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-yaml", default=None)
    ap.add_argument("--plot-out", default=None)
    ap.add_argument("--allow-joint", action="store_true")
    ap.add_argument("--lambda-rgb-grid", type=float, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--lambda-depth-grid", type=float, nargs="+", default=[0.5, 1, 1.5, 2, 3])
    args = ap.parse_args()

    fit_input = _read_rows(args.fit_metrics)
    test_input = _read_rows(args.test_metrics)
    fit_rows = _finite_rows(fit_input)
    test_rows = _finite_rows(test_input)
    if len(fit_rows) < 3:
        raise SystemExit(f"need at least 3 finite fit rows, got {len(fit_rows)}")
    if not test_rows:
        raise SystemExit(f"no finite test rows in {args.test_metrics}")

    fit = _fit(fit_rows, args.lambda_rgb_grid, args.lambda_depth_grid, args.allow_joint)
    fit_target = _target(fit_rows, fit["basis"])
    fit_pred = _predict(fit_rows, fit)
    test_target = _target(test_rows, fit["basis"])
    test_pred = _predict(test_rows, fit)
    test_diag = _diagnostics(test_target, test_pred)

    out = {
        "target": "geometry-only q_geometry from chamfer/fscore/completeness",
        "input_samples": {"fit": len(fit_input), "test": len(test_input)},
        "valid_samples": {"fit": len(fit_rows), "test": len(test_rows)},
        "dropped_samples": {"fit": len(fit_input) - len(fit_rows), "test": len(test_input) - len(test_rows)},
        "fit": _public_fit(fit),
        "heldout_test": test_diag,
    }
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, allow_nan=False)
    _write_yaml(args.out_yaml, fit)
    _write_plot(args.plot_out, fit_rows, test_rows, fit, fit_target, fit_pred, test_target, test_pred)
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
