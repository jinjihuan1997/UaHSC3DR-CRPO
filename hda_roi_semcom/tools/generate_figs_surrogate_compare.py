#!/usr/bin/env python3
"""Generate Figures/Fig3_surrogate_model_compare.pdf."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT          = Path(__file__).resolve().parents[1]
METRICS_CSV   = ROOT / "outputs/eiffel15_surrogate_test/gsfusion_real_metrics_eiffel15_full_nocrop_all.csv"
SURROGATE_JSON = ROOT / "outputs/eiffel15_surrogate_test/eiffel15_per_traj_composite_testeval.json"
OUT_DIR       = ROOT / "Figures"
EXTRA_OUT_DIR = ROOT / "paper_results/figures/saturation_surrogate"
FIGURE_NAME   = "Fig3_surrogate_model_compare"

# Nature NPG vivid
C_LINEAR = "#E64B35"   # vivid vermillion  — linear (baseline)
C_SAT    = "#4DBBD5"   # vivid sky blue    — saturation (proposed)


def set_style() -> None:
    plt.rcParams.update({
        "font.family":      "serif",
        "font.serif":       ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size":        12,
        "font.weight":      "bold",
        "axes.labelsize":   13,
        "axes.labelweight": "bold",
        "xtick.labelsize":  11,
        "ytick.labelsize":  11,
        "legend.fontsize":  10.5,
        "axes.linewidth":   1.2,
        "lines.linewidth":  2.0,
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "mathtext.fontset": "stix",
        "savefig.bbox":     "tight",
    })


# ── data helpers (same as plot_surrogate_function_comparison.py) ──────────────

def norm_high(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((v - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def norm_low(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((hi - v) / max(hi - lo, 1e-12), 0.0, 1.0)


def composite_target(df: pd.DataFrame, fit: dict) -> np.ndarray:
    b = fit["basis"]
    pw, sw, lw = fit["app_weights"]
    fw, cw, chw = fit["geo_weights"]
    q_app = (pw * norm_high(df["psnr"].to_numpy(float),         *b["psnr"])
           + sw * norm_high(df["ssim"].to_numpy(float),         *b["ssim"])
           + lw * norm_low (df["lpips"].to_numpy(float),        *b["lpips"]))
    q_geo = (fw * norm_high(df["fscore"].to_numpy(float),       *b["fscore"])
           + cw * norm_high(df["completeness"].to_numpy(float), *b["completeness"])
           + chw * norm_low(df["chamfer"].to_numpy(float),      *b["chamfer"]))
    return np.clip(0.45 * q_app + 0.45 * q_geo + 0.10 * q_app * q_geo, 0.0, 1.0)


def fit_nonnegative(x: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    best_sse, best = float("inf"), None
    for mask in range(1 << x.shape[1]):
        active = [i for i in range(x.shape[1]) if mask & (1 << i)]
        design = np.ones((len(y), 1 + len(active)))
        if active:
            design[:, 1:] = x[:, active]
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        if active and np.any(coef[1:] < 0.0):
            continue
        w = np.zeros(x.shape[1])
        if active:
            w[active] = coef[1:]
        pred = np.clip(float(coef[0]) + x @ w, 0.0, 1.0)
        sse = float(np.square(pred - y).sum())
        if sse < best_sse:
            best_sse, best = sse, (float(coef[0]), w, pred)
    if best is None:
        raise RuntimeError("fit failed")
    return best


def sat_predict(q: np.ndarray, r: np.ndarray, fit: dict) -> np.ndarray:
    g_q = 1.0 - np.exp(-float(fit["lambda_rgb"])   * q)
    g_r = 1.0 - np.exp(-float(fit["lambda_depth"]) * r)
    return np.clip(float(fit["bias"])
                   + float(fit["w_rgb"])   * g_q
                   + float(fit["w_depth"]) * g_r
                   + float(fit["w_joint"]) * g_q * g_r, 0.0, 1.0)


def r2_rmse(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    ss_res = float(np.square(pred - y).sum())
    ss_tot = float(np.square(y - y.mean()).sum())
    return (1.0 - ss_res / max(ss_tot, 1e-12)), float(np.sqrt(np.square(pred - y).mean()))


def main() -> None:
    obj      = json.loads(SURROGATE_JSON.read_text())
    traj     = obj["best_trajectory"]
    sat_fit  = next(r for r in obj["trajectories"] if r["trajectory_id"] == traj)["fit"]

    df = pd.read_csv(METRICS_CSV)
    df["trajectory_id"] = df["condition"].str.extract(r"(traj\d+)")[0]
    df = df[df["trajectory_id"] == traj].copy()

    y   = composite_target(df, sat_fit)
    q   = np.clip(df["q_rgb"].to_numpy(float),   0.0, 1.0)
    r   = np.clip(df["r_depth"].to_numpy(float), 0.0, 1.0)

    feats = np.column_stack([q, r, q * r])
    _, _, lin_pred = fit_nonnegative(feats, y)
    sat_pred       = sat_predict(q, r, sat_fit)

    r2_lin,  rmse_lin  = r2_rmse(y, lin_pred)
    r2_sat,  rmse_sat  = r2_rmse(y, sat_pred)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXTRA_OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()

    fig, ax = plt.subplots(figsize=(4.2, 4.2))

    # Extend limits slightly so boundary points aren't clipped by the axes frame
    margin = 0.03
    ax.set_xlim(-margin, 1 + margin)
    ax.set_ylim(-margin, 1 + margin)

    # Per-point alpha: α = 0.25 + 0.75 * Q_true  (origin→transparent, far→opaque)
    import matplotlib.colors as mcolors

    def rgba_array(hex_color: str, alpha_arr: np.ndarray) -> np.ndarray:
        rgb = np.array(mcolors.to_rgb(hex_color))
        return np.column_stack([np.tile(rgb, (len(alpha_arr), 1)), alpha_arr])

    pt_alpha  = np.clip(0.25 + 0.75 * y, 0.0, 1.0)   # gradient by Q_true
    fc_linear = rgba_array(C_LINEAR, pt_alpha)
    fc_sat    = rgba_array(C_SAT,    pt_alpha)

    # y = x reference — lowest zorder so all points render on top
    ax.plot([0, 1], [0, 1], color="0.45", ls="--", lw=1.6, zorder=1,
            label="Reference ($y = x$)")

    # Linear — open-edge circles, RGBA gradient alpha
    ax.scatter(y, lin_pred,
               facecolors=fc_linear, edgecolors="none",
               s=48, marker="o", zorder=5,
               label="Linear")

    # Saturation — filled circles, RGBA gradient alpha — highest zorder
    ax.scatter(y, sat_pred,
               facecolors=fc_sat, edgecolors="none",
               s=48, marker="o", zorder=6,
               label="Saturation")

    ax.set_xlabel(r"Ground-truth $Q_{3\mathrm{d}}$")
    ax.set_ylabel(r"Predicted $\hat{Q}_{3\mathrm{d}}$")
    ax.set_aspect("equal")
    ax.grid(True, color="0.84", linewidth=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", width=1.1, length=4.2)

    # Legend inside — lower right (data dense upper right, sparse lower right)
    ax.legend(
        frameon=True, framealpha=0.90, edgecolor="0.80",
        fancybox=False, loc="lower right",
        handletextpad=0.5, labelspacing=0.4, borderpad=0.55,
    )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{FIGURE_NAME}.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{FIGURE_NAME}.png", dpi=600, bbox_inches="tight")
    fig.savefig(EXTRA_OUT_DIR / "surrogate_model_compare.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(EXTRA_OUT_DIR / "surrogate_model_compare.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"traj={traj}  n={len(y)}")
    print(f"  Linear:     R²={r2_lin:.4f}  RMSE={rmse_lin:.4f}")
    print(f"  Saturation: R²={r2_sat:.4f}  RMSE={rmse_sat:.4f}")
    print(f"  → {OUT_DIR.relative_to(ROOT)}/{FIGURE_NAME}.pdf")
    print(f"  → {EXTRA_OUT_DIR.relative_to(ROOT)}/surrogate_model_compare.pdf")


if __name__ == "__main__":
    main()
