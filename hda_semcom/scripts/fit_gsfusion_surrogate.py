"""Fit GSFusion surrogate weights from offline reconstruction metrics.

Expected CSV columns:
  q_rgb,r_depth

Use q_3d_real directly if present. Otherwise the script expects:
  psnr,ssim,lpips,chamfer,fscore,completeness

It fits either a linear surrogate:
  q_3d_real ~= bias + w_rgb*q_rgb + w_depth*r_depth
                  + w_joint*q_rgb*r_depth
or a saturation surrogate:
  q_3d_real ~= bias + w_rgb*g_rgb + w_depth*g_depth
                  + w_joint*g_rgb*g_depth
where g_rgb = 1 - exp(-lambda_rgb*q_rgb) and
g_depth = 1 - exp(-lambda_depth*r_depth).
Modality weights are nonnegative. The bias is unconstrained.
"""

import argparse
import csv
import json
import os

import numpy as np


def _col(rows, name):
    if name not in rows[0]:
        raise SystemExit(f"missing required column: {name}")
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)


def _filter_finite_rows(rows, required_columns):
    invalid_counts = {name: 0 for name in required_columns}
    valid = []
    for row in rows:
        keep = True
        for name in required_columns:
            try:
                value = float(row[name])
            except (KeyError, ValueError):
                invalid_counts[name] += 1
                keep = False
                continue
            if not np.isfinite(value):
                invalid_counts[name] += 1
                keep = False
        if keep:
            valid.append(row)
    invalid_counts = {name: count for name, count in invalid_counts.items() if count}
    return valid, invalid_counts


def _norm_high_good(x):
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def _norm_low_good(x):
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    return np.clip((hi - x) / max(hi - lo, 1e-12), 0.0, 1.0)


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


def _fit_bias_nonnegative_lstsq(x, y, min_weights=None):
    best = None
    best_sse = float("inf")
    n_features = x.shape[1]
    if min_weights is None:
        min_weights = np.zeros(n_features, dtype=np.float64)
    min_weights = np.asarray(min_weights, dtype=np.float64)
    if min_weights.shape != (n_features,):
        raise ValueError(f"min_weights must have shape {(n_features,)}, got {min_weights.shape}")
    y_adjusted = y - x @ min_weights
    for mask in range(0, 1 << n_features):
        active = [idx for idx in range(n_features) if mask & (1 << idx)]
        design = np.ones((x.shape[0], 1 + len(active)), dtype=np.float64)
        if active:
            design[:, 1:] = x[:, active]
        coef, *_ = np.linalg.lstsq(design, y_adjusted, rcond=None)
        if active and np.any(coef[1:] < 0.0):
            continue
        w = min_weights.copy()
        if active:
            w[active] += coef[1:]
        pred = coef[0] + x @ w
        sse = float(np.square(pred - y).sum())
        if sse < best_sse:
            best_sse = sse
            best = (float(coef[0]), w)
    if best is None:
        best = (float((y - x @ min_weights).mean()), min_weights.copy())
    return best


def _diagnostics(y, raw_pred):
    pred = np.clip(raw_pred, 0.0, 1.0)
    rmse = float(np.sqrt(np.square(pred - y).mean()))
    mae = float(np.abs(pred - y).mean())
    denom = float(np.square(y - y.mean()).sum())
    r2 = 1.0 - float(np.square(pred - y).sum()) / max(denom, 1e-12)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": _spearman(y, pred),
        "clip_ratio": float(np.mean((raw_pred < 0.0) | (raw_pred > 1.0))),
    }, pred


def _linear_features(q_rgb, r_depth):
    return np.column_stack([q_rgb, r_depth, q_rgb * r_depth])


def _saturation_features(q_rgb, r_depth, lambda_rgb, lambda_depth):
    g_rgb = 1.0 - np.exp(-float(lambda_rgb) * q_rgb)
    g_depth = 1.0 - np.exp(-float(lambda_depth) * r_depth)
    return np.column_stack([g_rgb, g_depth, g_rgb * g_depth])


def _fit_model(x, y, min_weights=None):
    bias, weights = _fit_bias_nonnegative_lstsq(x, y, min_weights=min_weights)
    raw_pred = bias + x @ weights
    diag, pred = _diagnostics(y, raw_pred)
    return {
        "bias": bias,
        "weights": weights,
        "raw_pred": raw_pred,
        "pred": pred,
        **diag,
    }


def _fit_linear(q_rgb, r_depth, y, min_weights=None):
    fit = _fit_model(_linear_features(q_rgb, r_depth), y, min_weights=min_weights)
    fit.update({
        "model": "linear",
        "feature_names": ["q_rgb", "r_depth", "q_rgb_times_r_depth"],
    })
    return fit


def _fit_saturation(q_rgb, r_depth, y, lambda_rgb_grid, lambda_depth_grid, min_weights=None):
    best = None
    for lambda_rgb in lambda_rgb_grid:
        for lambda_depth in lambda_depth_grid:
            x = _saturation_features(q_rgb, r_depth, lambda_rgb, lambda_depth)
            fit = _fit_model(x, y, min_weights=min_weights)
            fit.update({
                "model": "saturation",
                "lambda_rgb": float(lambda_rgb),
                "lambda_depth": float(lambda_depth),
                "feature_names": ["g_rgb", "g_depth", "g_rgb_times_g_depth"],
            })
            if best is None:
                best = fit
                continue
            # Prefer explained variance, then RMSE, then less clipping.
            key = (fit["r2"], -fit["rmse"], -fit["clip_ratio"])
            best_key = (best["r2"], -best["rmse"], -best["clip_ratio"])
            if key > best_key:
                best = fit
    return best


def _write_weights(path, fit):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    weights = fit["weights"]
    with open(path, "w") as f:
        f.write("mapping_quality:\n")
        f.write("  mapping_quality_mode: gsfusion_surrogate\n")
        f.write("  gsfusion_weights:\n")
        f.write(f"    model: {fit['model']}\n")
        f.write(f"    bias: {fit['bias']:.8f}\n")
        f.write(f"    w_rgb: {weights[0]:.8f}\n")
        f.write(f"    w_depth: {weights[1]:.8f}\n")
        f.write(f"    w_joint: {weights[2]:.8f}\n")
        if fit["model"] == "saturation":
            f.write(f"    lambda_rgb: {fit['lambda_rgb']:.8f}\n")
            f.write(f"    lambda_depth: {fit['lambda_depth']:.8f}\n")
        f.write("    feature_names:\n")
        for name in fit["feature_names"]:
            f.write(f"      - {name}\n")
        f.write(f"# fit_mae: {fit['mae']:.8f}\n")
        f.write(f"# fit_rmse: {fit['rmse']:.8f}\n")
        f.write(f"# fit_r2: {fit['r2']:.8f}\n")
        f.write(f"# fit_spearman: {fit['spearman']:.8f}\n")
        f.write(f"# fit_clip_ratio: {fit['clip_ratio']:.8f}\n")


def _write_plot(path, y, pred):
    if not path:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=160)
    ax.scatter(y, pred, s=24, alpha=0.75)
    ax.plot([0, 1], [0, 1], color="black", linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("real q_3d")
    ax.set_ylabel("predicted q_3d")
    ax.set_title("GSFusion surrogate fit")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fit_json(fit):
    out = {
        "model": fit["model"],
        "bias": fit["bias"],
        "w_rgb": float(fit["weights"][0]),
        "w_depth": float(fit["weights"][1]),
        "w_joint": float(fit["weights"][2]),
        "feature_names": fit["feature_names"],
        "rmse": fit["rmse"],
        "mae": fit["mae"],
        "r2": fit["r2"],
        "spearman": fit["spearman"],
        "clip_ratio": fit["clip_ratio"],
        "best_lambda_rgb": None,
        "best_lambda_depth": None,
    }
    if fit["model"] == "saturation":
        out["lambda_rgb"] = fit["lambda_rgb"]
        out["lambda_depth"] = fit["lambda_depth"]
        out["best_lambda_rgb"] = fit["lambda_rgb"]
        out["best_lambda_depth"] = fit["lambda_depth"]
    return out


def _print_fit(label, fit):
    print(f"[{label}] model={fit['model']}")
    if fit["model"] == "saturation":
        print(f"[{label}] best_lambda_rgb={fit['lambda_rgb']:.8f}")
        print(f"[{label}] best_lambda_depth={fit['lambda_depth']:.8f}")
    else:
        print(f"[{label}] best_lambda_rgb=N/A")
        print(f"[{label}] best_lambda_depth=N/A")
    print(f"[{label}] bias={fit['bias']:.8f}")
    print(f"[{label}] w_rgb={fit['weights'][0]:.8f}")
    print(f"[{label}] w_depth={fit['weights'][1]:.8f}")
    print(f"[{label}] w_joint={fit['weights'][2]:.8f}")
    print(f"[{label}] mae={fit['mae']:.8f}")
    print(f"[{label}] rmse={fit['rmse']:.8f}")
    print(f"[{label}] r2={fit['r2']:.8f}")
    print(f"[{label}] spearman={fit['spearman']:.8f}")
    print(f"[{label}] clip_ratio={fit['clip_ratio']:.8f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_csv", help="CSV exported from GSFusion offline evaluation.")
    ap.add_argument("--out", default="outputs/gsfusion_surrogate_weights.yaml")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--plot-out", default=None)
    ap.add_argument("--model", "--fit-mode", dest="model",
                    choices=["linear", "saturation"], default="linear",
                    help="Surrogate model to write to --out. Both models are fitted and reported.")
    ap.add_argument("--lambda-grid", type=float, nargs="+",
                    default=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
                    help="Positive lambda values to grid-search for both saturation branches unless branch-specific grids are set.")
    ap.add_argument("--lambda-rgb-grid", type=float, nargs="+", default=None,
                    help="Positive lambda values to grid-search for g_rgb.")
    ap.add_argument("--lambda-depth-grid", type=float, nargs="+", default=None,
                    help="Positive lambda values to grid-search for g_depth.")
    ap.add_argument("--min-w-rgb", type=float, default=0.0)
    ap.add_argument("--min-w-depth", type=float, default=0.0)
    ap.add_argument("--min-w-joint", type=float, default=0.0)
    ap.add_argument("--app-weights", type=float, nargs=3, default=[0.6, 0.2, 0.2],
                    metavar=("W_PSNR", "W_SSIM", "W_LPIPS"),
                    help="Weights for Q_app = w_psnr*PSNR + w_ssim*SSIM + w_lpips*(1-LPIPS). Must sum to 1. "
                         "Default matches paper setting (0.6 PSNR + 0.2 SSIM + 0.2 LPIPS).")
    ap.add_argument("--geo-weights", type=float, nargs=3, default=[0.35, 0.35, 0.30],
                    metavar=("W_FSCORE", "W_COMP", "W_CHAMFER"),
                    help="Weights for Q_geo = w_fscore*fscore + w_comp*completeness + w_chamfer*(1-chamfer). Must sum to 1.")
    args = ap.parse_args()

    with open(args.metrics_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty metrics CSV: {args.metrics_csv}")

    if "q_3d_real" in rows[0]:
        required_columns = ["q_rgb", "r_depth", "q_3d_real"]
    else:
        required_columns = [
            "q_rgb", "r_depth", "psnr", "ssim", "lpips",
            "chamfer", "fscore", "completeness",
        ]
    original_samples = len(rows)
    rows, invalid_counts = _filter_finite_rows(rows, required_columns)
    dropped_samples = original_samples - len(rows)
    if not rows:
        raise SystemExit(
            f"no finite samples in {args.metrics_csv}; "
            f"dropped {dropped_samples}/{original_samples} rows"
        )
    if dropped_samples:
        print(
            f"[warn] dropped {dropped_samples}/{original_samples} rows with non-finite surrogate inputs: "
            f"{invalid_counts}",
            flush=True,
        )

    q_rgb = np.clip(_col(rows, "q_rgb"), 0.0, 1.0)
    r_depth = np.clip(_col(rows, "r_depth"), 0.0, 1.0)

    if "q_3d_real" in rows[0]:
        y = np.clip(_col(rows, "q_3d_real"), 0.0, 1.0)
    else:
        wp, ws, wl = args.app_weights
        wf, wc, wch = args.geo_weights
        q_render = (
            wp * _norm_high_good(_col(rows, "psnr"))
            + ws * _norm_high_good(_col(rows, "ssim"))
            + wl * _norm_low_good(_col(rows, "lpips"))
        )
        q_geometry = (
            wf  * _norm_high_good(_col(rows, "fscore"))
            + wc  * _norm_high_good(_col(rows, "completeness"))
            + wch * _norm_low_good(_col(rows, "chamfer"))
        )
        y = np.clip(0.45 * q_render + 0.45 * q_geometry + 0.10 * q_render * q_geometry, 0.0, 1.0)

    lambda_rgb_grid = args.lambda_rgb_grid or args.lambda_grid
    lambda_depth_grid = args.lambda_depth_grid or args.lambda_grid
    if any(val <= 0.0 for val in lambda_rgb_grid):
        raise SystemExit("lambda RGB grid values must be positive")
    if any(val <= 0.0 for val in lambda_depth_grid):
        raise SystemExit("lambda depth grid values must be positive")
    min_weights = np.asarray([args.min_w_rgb, args.min_w_depth, args.min_w_joint], dtype=np.float64)
    if np.any(min_weights < 0.0):
        raise SystemExit("minimum surrogate weights must be nonnegative")

    fits = {
        "linear": _fit_linear(q_rgb, r_depth, y, min_weights=min_weights),
        "saturation": _fit_saturation(
            q_rgb,
            r_depth,
            y,
            lambda_rgb_grid,
            lambda_depth_grid,
            min_weights=min_weights,
        ),
    }
    selected = fits[args.model]

    _write_weights(args.out, selected)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump({
                **_fit_json(selected),
                "input_samples": original_samples,
                "valid_samples": len(rows),
                "dropped_samples": dropped_samples,
                "invalid_counts": invalid_counts,
                "selected_model": args.model,
                "minimum_weights": {
                    "w_rgb": float(min_weights[0]),
                    "w_depth": float(min_weights[1]),
                    "w_joint": float(min_weights[2]),
                },
                "lambda_rgb_grid": [float(x) for x in lambda_rgb_grid],
                "lambda_depth_grid": [float(x) for x in lambda_depth_grid],
                "diagnostics": {
                    "linear": _fit_json(fits["linear"]),
                    "saturation": _fit_json(fits["saturation"]),
                },
            }, f, indent=2, allow_nan=False)
    _write_plot(args.plot_out, y, selected["pred"])
    _print_fit("linear", fits["linear"])
    _print_fit("saturation", fits["saturation"])
    print(f"selected_model={args.model}")
    if selected["r2"] < 0.0:
        print("WARNING: selected surrogate has weak explanatory power for real GSFusion metrics.")
    print(f"wrote={args.out}")
    if args.json_out:
        print(f"wrote={args.json_out}")
    if args.plot_out:
        print(f"wrote={args.plot_out}")


if __name__ == "__main__":
    main()
