#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotting_utils import METHOD_ORDER, moving_average, save_figure, set_ieee_style
from result_file_discovery import discover_result_files, write_discovery_summary
from table_utils import save_table_bundle, write_caption


METHOD_MAP = {
    "fixed-balanced": "Fixed-balanced allocation",
    "fixed_balanced": "Fixed-balanced allocation",
    "Fixed-balanced allocation": "Fixed-balanced allocation",
    "RGB-priority": "RGB-priority allocation",
    "rgb-priority": "RGB-priority allocation",
    "rgb_priority": "RGB-priority allocation",
    "Depth-priority": "Depth-priority allocation",
    "depth-priority": "Depth-priority allocation",
    "depth_priority": "Depth-priority allocation",
    "random": "Random allocation",
    "Random": "Random allocation",
    "PPO-penalty": "PPO-penalty",
    "ppo-penalty": "PPO-penalty",
    "penalty": "PPO-penalty",
    "Lagrangian-PPO": "Lagrangian-PPO",
    "lagrangian": "Lagrangian-PPO",
    "CRPO-PPO": "CRPO-guided PPO",
    "CRPO-guided PPO": "CRPO-guided PPO",
    "crpo": "CRPO-guided PPO",
}


@dataclass
class Generated:
    figures: list[Path] = field(default_factory=list)
    tables: list[Path] = field(default_factory=list)
    missing: list[dict[str, Any]] = field(default_factory=list)
    used_sources: dict[str, list[Path]] = field(default_factory=dict)

    def add_missing(self, name: str, section: str, reason: str, needed: str, command: str | None = None) -> None:
        self.missing.append(
            {
                "name": name,
                "section": section,
                "reason": reason,
                "needed": needed,
                "command": command or "",
            }
        )


def standardize_method(value: Any) -> str:
    text = str(value)
    return METHOD_MAP.get(text, text)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_existing(root: Path, candidates: list[str]) -> Path | None:
    for rel in candidates:
        path = root / rel
        if path.exists():
            return path
    return None


def metric_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.quantile(0.05)
    hi = values.quantile(0.95)
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(float(lo), float(hi)):
        return pd.Series(np.nan, index=series.index)
    score = (values - lo) / (hi - lo)
    score = score.clip(0.0, 1.0)
    return score if higher_is_better else 1.0 - score


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower_to_col = {c.lower(): c for c in df.columns}
    for name in names:
        if name in df.columns:
            return name
        col = lower_to_col.get(name.lower())
        if col is not None:
            return col
    return None


def add_reconstruction_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    psnr = first_col(out, ["psnr", "psnr_gt", "psnr_clean", "psnr_render"])
    ssim = first_col(out, ["ssim", "ssim_gt", "ssim_clean", "ssim_render"])
    lpips = first_col(out, ["lpips", "lpips_gt", "lpips_clean", "lpips_render"])
    fscore = first_col(out, ["fscore", "f_score", "f-score"])
    comp = first_col(out, ["completeness", "complete"])
    chamfer = first_col(out, ["chamfer", "chamfer_distance", "cd"])
    q_rgb = first_col(out, ["q_rgb", "Q_rgb", "avg_q_rgb"])
    r_depth = first_col(out, ["r_depth", "R_depth", "avg_r_depth"])

    missing = [name for name, col in {
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS": lpips,
        "F-score": fscore,
        "Completeness": comp,
        "Chamfer": chamfer,
    }.items() if col is None]
    if missing:
        raise ValueError(f"Cannot compute Q_gt; missing columns: {', '.join(missing)}")

    out["Qbar_psnr"] = metric_score(out[psnr], True)
    out["Qbar_ssim"] = metric_score(out[ssim], True)
    out["Qbar_lpips"] = metric_score(out[lpips], False)
    out["Qbar_fscore"] = metric_score(out[fscore], True)
    out["Qbar_comp"] = metric_score(out[comp], True)
    out["Qbar_chamfer"] = metric_score(out[chamfer], False)
    out["Q_app"] = 0.6 * out["Qbar_psnr"] + 0.2 * out["Qbar_ssim"] + 0.2 * out["Qbar_lpips"]
    out["Q_geo"] = 0.35 * out["Qbar_fscore"] + 0.35 * out["Qbar_comp"] + 0.30 * out["Qbar_chamfer"]
    out["Q_gt"] = (0.45 * out["Q_app"] + 0.45 * out["Q_geo"] + 0.10 * out["Q_app"] * out["Q_geo"]).clip(0.0, 1.0)

    if q_rgb and q_rgb not in out:
        out["q_rgb"] = out[q_rgb]
    elif q_rgb:
        out["q_rgb"] = out[q_rgb]
    if r_depth:
        out["r_depth"] = out[r_depth]
    return out.dropna(subset=["Q_gt"])


def saturation_feature(x: pd.Series | np.ndarray, lam: float) -> pd.Series | np.ndarray:
    return 1.0 - np.exp(-float(lam) * np.asarray(x, dtype=float))


def predict_from_params(df: pd.DataFrame, params: dict[str, Any], model: str) -> pd.Series:
    q = pd.to_numeric(df["q_rgb"], errors="coerce")
    r = pd.to_numeric(df["r_depth"], errors="coerce")
    if model == "saturation":
        q_feat = saturation_feature(q, params.get("lambda_rgb", 8.0))
        r_feat = saturation_feature(r, params.get("lambda_depth", 8.0))
    else:
        q_feat = q
        r_feat = r
    pred = (
        float(params.get("bias", params.get("b", 0.0)))
        + float(params.get("w_rgb", params.get("wr", 0.0))) * q_feat
        + float(params.get("w_depth", params.get("wd", 0.0))) * r_feat
        + float(params.get("w_joint", params.get("wj", 0.0))) * q_feat * r_feat
    )
    return pd.Series(np.asarray(pred), index=df.index).clip(0.0, 1.0)


def regression_metrics(y: pd.Series, pred: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"y": y, "pred": pred}).dropna()
    if frame.empty:
        return {"RMSE": np.nan, "MAE": np.nan, "R2": np.nan, "Spearman": np.nan}
    err = frame["pred"] - frame["y"]
    sst = ((frame["y"] - frame["y"].mean()) ** 2).sum()
    r2 = np.nan if math.isclose(float(sst), 0.0) else 1.0 - float((err ** 2).sum() / sst)
    return {
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAE": float(err.abs().mean()),
        "R2": float(r2),
        "Spearman": float(frame["pred"].corr(frame["y"], method="spearman")),
    }


def extract_surrogate_params(data: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Return (linear_params, saturation_params, best_trajectory_id).

    Supports two JSON shapes:
    - Per-trajectory format: {"best_trajectory": "traj13", "trajectories": [{...}]}
      → extracts saturation fit and optional linear baseline for best_trajectory.
    - Legacy global format: {"linear": {...}, "saturation": {...}, ...}
    """
    # Per-trajectory format (current main pipeline)
    if "best_trajectory" in data and "trajectories" in data:
        best_traj = data["best_trajectory"]
        traj_entry = next(
            (t for t in data["trajectories"] if t.get("trajectory_id") == best_traj), None
        )
        if traj_entry is None:
            return None, None, best_traj
        linear = traj_entry.get("linear_fit") or traj_entry.get("linear")
        fit = traj_entry.get("fit", {})
        return linear, fit, best_traj

    # Legacy global format
    diagnostics = data.get("diagnostics") if isinstance(data.get("diagnostics"), dict) else {}
    linear = data.get("linear") or data.get("linear_surrogate") or diagnostics.get("linear")
    saturation = data.get("saturation") or data.get("saturation_surrogate") or diagnostics.get("saturation")
    if not saturation and data.get("model") == "saturation":
        saturation = data
    return linear, saturation, None


def generate_surrogate_outputs(root: Path, out: Path, gen: Generated) -> None:
    json_path = find_existing(
        root,
        [
            # per-trajectory composite surrogates (current main pipeline, app60202 = 0.6/0.2/0.2)
            "outputs/eiffel15_surrogate_test/per_trajectory/eiffel_per_trajectory_composite_app60202_rerun_surrogates.json",
            "outputs/eiffel15_surrogate_test/per_trajectory/eiffel_per_trajectory_composite_app60202_surrogates.json",
            # legacy global fit (kept as last-resort fallback only)
            "outputs/eiffel15_surrogate_test/eiffel15_composite_surrogate_full_nocrop.json",
            "outputs/multiscene_surrogate/gsfusion_surrogate_fit_multiscene_saturation.json",
        ],
    )
    metrics_path = find_existing(
        root,
        [
            # clean-reference metrics first (vs GT, not vs degraded input)
            "outputs/eiffel15_surrogate_test/gsfusion_real_metrics_eiffel15_full_nocrop_all.csv",
            "outputs/eiffel15_surrogate_test/gsfusion_real_metrics_eiffel15_full_nocrop_clean_fit.csv",
            "outputs/eiffel15_surrogate_test/gsfusion_real_metrics_eiffel15_full_nocrop_fit.csv",
            "outputs/multiscene_surrogate/gsfusion_real_metrics_multiscene_fit.csv",
        ],
    )
    if not json_path or not metrics_path:
        gen.add_missing(
            "surrogate_validation_metrics",
            "6.2",
            "Surrogate JSON or real GSFusion metrics CSV was not found.",
            "A surrogate fit JSON and a real metrics CSV with q_rgb, r_depth, PSNR, SSIM, LPIPS, Chamfer, F-score, and completeness.",
            "python scripts/fit_gsfusion_surrogate.py ...",
        )
        return

    data = read_json(json_path)
    linear, saturation, best_traj = extract_surrogate_params(data)
    if not saturation:
        gen.add_missing(
            "surrogate_validation_metrics",
            "6.2",
            f"`{json_path}` does not contain saturation surrogate parameters.",
            "A fit JSON with saturation surrogate parameters (per-trajectory or global format).",
        )
        return

    df = pd.read_csv(metrics_path)
    # For per-trajectory format, filter metrics to the best trajectory only
    if best_traj is not None:
        traj_col = next((c for c in df.columns if c in ("trajectory_id", "traj_id")), None)
        if traj_col is None:
            # Try to extract trajectory id from condition string
            df["_traj"] = df["condition"].str.extract(r"(traj\d+)") if "condition" in df.columns else None
            traj_col = "_traj"
        if traj_col and traj_col in df.columns:
            df = df[df[traj_col] == best_traj].copy()
        if df.empty:
            gen.add_missing(
                "surrogate_validation_metrics", "6.2",
                f"Metrics CSV contains no rows for best_trajectory={best_traj!r}.",
                f"Ensure the metrics CSV includes rows for {best_traj}.",
            )
            return

    try:
        df = add_reconstruction_labels(df)
    except ValueError as exc:
        gen.add_missing("surrogate_prediction_scatter", "6.2", str(exc), "Complete real metrics columns.", None)
        return
    if "q_rgb" not in df.columns or "r_depth" not in df.columns:
        gen.add_missing(
            "surrogate_prediction_scatter",
            "6.2",
            "Metrics CSV does not contain q_rgb and r_depth columns.",
            "Per-condition q_rgb and r_depth values aligned with real GSFusion metrics.",
        )
        return

    sat_pred = predict_from_params(df, saturation, "saturation")
    sat_m = regression_metrics(df["Q_gt"], sat_pred)
    lin_pred = predict_from_params(df, linear, "linear") if linear is not None else None
    lin_m = regression_metrics(df["Q_gt"], lin_pred) if lin_pred is not None else None

    trajectory_note = f" The metrics were filtered to best_trajectory={best_traj}." if best_traj else ""

    table_rows = [{"Model": "Saturation surrogate", "Samples": len(df), **sat_m}]
    if lin_m is not None:
        table_rows.insert(0, {"Model": "Linear surrogate", "Samples": len(df), **lin_m})
    table = pd.DataFrame(table_rows)
    table_paths = save_table_bundle(table, out / "tables", "surrogate_validation_metrics")
    gen.tables.extend(table_paths.values())
    gen.used_sources["surrogate_validation_metrics"] = [json_path, metrics_path]
    write_caption(
        out / "tables" / "surrogate_validation_metrics_caption.md",
        "Surrogate validation metrics",
        [json_path, metrics_path],
        "The table reports how well stored surrogate parameters predict the real reconstruction label computed from existing real GSFusion metrics."
        + trajectory_note,
    )

    param_rows = []
    meanings = {
        "bias": "Intercept term of the surrogate function",
        "b": "Intercept term of the surrogate function",
        "w_rgb": "Weight of the RGB reception quality term",
        "wr": "Weight of the RGB reception quality term",
        "w_depth": "Weight of the depth partial transmission term",
        "wd": "Weight of the depth partial transmission term",
        "w_joint": "Weight of the RGB-depth coupling term",
        "wj": "Weight of the RGB-depth coupling term",
        "lambda_rgb": "RGB saturation rate",
        "lambda_depth": "Depth saturation rate",
    }
    models_to_report = [("Saturation surrogate", saturation)]
    if linear is not None:
        models_to_report.insert(0, ("Linear surrogate", linear))
    for model_name, params in models_to_report:
        for key in ["bias", "b", "w_rgb", "wr", "w_depth", "wd", "w_joint", "wj", "lambda_rgb", "lambda_depth"]:
            if key in params:
                param_rows.append(
                    {
                        "Model": model_name,
                        "Parameter": key,
                        "Meaning": meanings.get(key, key),
                        "Value": params[key],
                    }
                )
    param_df = pd.DataFrame(param_rows)
    param_paths = save_table_bundle(param_df, out / "tables", "surrogate_model_parameters")
    gen.tables.extend(param_paths.values())
    gen.used_sources["surrogate_model_parameters"] = [json_path]
    write_caption(
        out / "tables" / "surrogate_model_parameters_caption.md",
        "Surrogate model parameters",
        [json_path],
        "The stored saturation surrogate contains the RGB, depth, and coupling terms used to explain reconstruction-aware quality.",
    )

    set_ieee_style()
    scatter_panels = [(sat_pred, "Saturation surrogate", sat_m)]
    if lin_pred is not None:
        scatter_panels.insert(0, (lin_pred, "Linear surrogate", lin_m))
    fig, axes = plt.subplots(1, len(scatter_panels), figsize=(3.55 * len(scatter_panels), 3.0),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (pred, title, met) in zip(axes[0], scatter_panels):
        ax.scatter(df["Q_gt"], pred, s=10, alpha=0.7)
        lims = [0.0, 1.0]
        ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel(r"Real quality label $Q_{\mathrm{gt}}$")
        ax.grid(True, alpha=0.3)
        ax.text(
            0.03,
            0.97,
            f"RMSE={met['RMSE']:.3f}\nR2={met['R2']:.3f}\nSp={met['Spearman']:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 2},
        )
    axes[0][0].set_ylabel(r"Predicted quality $\hat{Q}_{\mathrm{rec}}$")
    fig_paths = save_figure(fig, out / "figures", "surrogate_prediction_scatter")
    gen.figures.extend(fig_paths.values())
    gen.used_sources["surrogate_prediction_scatter"] = [json_path, metrics_path]
    write_caption(
        out / "figures" / "surrogate_prediction_scatter_caption.md",
        "Surrogate prediction scatter",
        [json_path, metrics_path],
        "Points closer to the y=x line indicate better agreement between surrogate predictions and real GSFusion-style reconstruction labels."
        + trajectory_note,
    )

    fig, ax = plt.subplots(figsize=(3.55, 2.6))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0)
    if lin_pred is not None:
        ax.scatter(df["Q_gt"], lin_pred - df["Q_gt"], s=10, alpha=0.65, label="Linear surrogate")
    ax.scatter(df["Q_gt"], sat_pred - df["Q_gt"], s=10, alpha=0.65, label="Saturation surrogate")
    ax.set_xlabel(r"Real quality label $Q_{\mathrm{gt}}$")
    ax.set_ylabel("Prediction residual")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True)
    fig_paths = save_figure(fig, out / "figures", "surrogate_residual_plot")
    gen.figures.extend(fig_paths.values())
    gen.used_sources["surrogate_residual_plot"] = [json_path, metrics_path]
    write_caption(
        out / "figures" / "surrogate_residual_plot_caption.md",
        "Surrogate residual plot",
        [json_path, metrics_path],
        "Residual structure reveals whether a surrogate systematically over- or under-predicts the real reconstruction label."
        + trajectory_note,
    )

    q_grid = np.linspace(0.0, 1.0, 80)
    r_grid = np.linspace(0.0, 1.0, 80)
    q_mesh, r_mesh = np.meshgrid(q_grid, r_grid)
    surf_df = pd.DataFrame({"q_rgb": q_mesh.ravel(), "r_depth": r_mesh.ravel()})
    z = predict_from_params(surf_df, saturation, "saturation").to_numpy().reshape(q_mesh.shape)
    fig, ax = plt.subplots(figsize=(3.55, 2.8))
    im = ax.imshow(z, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
    ax.set_xlabel(r"$Q_{\mathrm{rgb}}$")
    ax.set_ylabel(r"$R_{\mathrm{depth}}$")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\hat{Q}_{\mathrm{rec}}$")
    fig_paths = save_figure(fig, out / "figures", "qrgb_rdepth_quality_surface")
    gen.figures.extend(fig_paths.values())
    gen.used_sources["qrgb_rdepth_quality_surface"] = [json_path]
    write_caption(
        out / "figures" / "qrgb_rdepth_quality_surface_caption.md",
        "RGB-depth surrogate quality surface",
        [json_path],
        "The fitted saturation surface visualizes how RGB quality and depth transmission jointly affect predicted reconstruction quality.",
    )


def parse_thresholds_from_path(path: Path) -> tuple[float | None, float | None]:
    match = re.search(r"q(\d+)_d(\d+)", str(path))
    if not match:
        return None, None
    return float(match.group(1)) / 100.0, float(match.group(2)) / 100.0


def generate_training_outputs(root: Path, out: Path, gen: Generated) -> None:
    eval_dir = find_existing(
        root,
        [
            "outputs/eiffel15_surrogate_test/eval_composite_app60202_q070_d035",
            "outputs/eiffel15_surrogate_test/eval_q070_d046_app60202",
            "outputs/eiffel15_surrogate_test/eval",
            "outputs/fushimi_surrogate_test/eval",
        ],
    )
    if not eval_dir:
        gen.add_missing("training_reward_curve", "6.3", "No evaluation directory with training logs was found.", "CRPO training log CSV.")
        return
    crpo_log = None
    for name in ["crpo_train_log.csv", "crpo_fushimi_train_log.csv"]:
        if (eval_dir / name).exists():
            crpo_log = eval_dir / name
            break
    if not crpo_log:
        gen.add_missing("training_reward_curve", "6.3", f"No CRPO training log in `{eval_dir}`.", "A CRPO train log CSV.")
        return

    df = pd.read_csv(crpo_log)
    x_col = "global_step" if "global_step" in df.columns else "iteration"
    x = df[x_col]
    set_ieee_style()

    reward_col = first_col(df, ["avg_map_reward", "avg_q_3d", "avg_episode_reward"])
    if reward_col:
        fig, ax = plt.subplots(figsize=(3.55, 2.5))
        ax.plot(x, df[reward_col], color="0.65", label="Raw")
        ax.plot(x, moving_average(df[reward_col]), color="tab:blue", label="Moving average")
        ax.set_xlabel("Environment steps" if x_col == "global_step" else "PPO updates")
        ax.set_ylabel("Average surrogate reward")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True)
        paths = save_figure(fig, out / "figures", "training_reward_curve")
        gen.figures.extend(paths.values())
        gen.used_sources["training_reward_curve"] = [crpo_log]
        write_caption(
            out / "figures" / "training_reward_curve_caption.md",
            "Training reward curve",
            [crpo_log],
            "The CRPO-guided PPO reward curve shows the training trajectory of the reconstruction-aware reward available in the log.",
        )
    else:
        gen.add_missing("training_reward_curve", "6.3", "Reward column not found.", "avg_map_reward, avg_q_3d, or avg_episode_reward.")

    cost_rgb = first_col(df, ["J_C_R", "avg_cost_rgb"])
    cost_depth = first_col(df, ["J_C_D", "avg_cost_depth"])
    if cost_rgb and cost_depth:
        fig, ax = plt.subplots(figsize=(3.55, 2.5))
        ax.plot(x, df[cost_rgb], label=r"$C_{\mathrm{rgb}}$")
        ax.plot(x, df[cost_depth], label=r"$C_{\mathrm{depth}}$")
        if "epsilon_R" in df.columns:
            ax.axhline(df["epsilon_R"].dropna().iloc[-1], color="tab:blue", linestyle="--", linewidth=1.0)
        if "epsilon_D" in df.columns:
            ax.axhline(df["epsilon_D"].dropna().iloc[-1], color="tab:orange", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Environment steps" if x_col == "global_step" else "PPO updates")
        ax.set_ylabel("Constraint cost")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True)
        paths = save_figure(fig, out / "figures", "training_constraint_costs")
        gen.figures.extend(paths.values())
        gen.used_sources["training_constraint_costs"] = [crpo_log]
        write_caption(
            out / "figures" / "training_constraint_costs_caption.md",
            "Training constraint costs",
            [crpo_log],
            "The logged RGB and depth costs show when CRPO-guided PPO is optimizing reward versus rectifying long-term constraints.",
        )
    else:
        gen.add_missing("training_constraint_costs", "6.3", "Constraint cost columns not found.", "J_C_R/J_C_D or avg_cost_rgb/avg_cost_depth.")

    q_col = first_col(df, ["avg_Q_rgb", "avg_q_rgb"])
    r_col = first_col(df, ["avg_R_depth", "avg_r_depth"])
    if q_col and r_col:
        q_thr, d_thr = parse_thresholds_from_path(eval_dir)
        fig, ax = plt.subplots(figsize=(3.55, 2.5))
        ax.plot(x, df[q_col], label=r"$Q_{\mathrm{rgb}}$")
        ax.plot(x, df[r_col], label=r"$R_{\mathrm{depth}}$")
        if q_thr is not None:
            ax.axhline(q_thr, color="tab:blue", linestyle="--", linewidth=1.0)
        if d_thr is not None:
            ax.axhline(d_thr, color="tab:orange", linestyle="--", linewidth=1.0)
        ax.set_xlabel("Environment steps" if x_col == "global_step" else "PPO updates")
        ax.set_ylabel("Average quality")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True)
        paths = save_figure(fig, out / "figures", "training_quality_curves")
        gen.figures.extend(paths.values())
        gen.used_sources["training_quality_curves"] = [crpo_log]
        write_caption(
            out / "figures" / "training_quality_curves_caption.md",
            "Training quality curves",
            [crpo_log],
            "The quality curves show whether the learned policy moves RGB reception and depth transmission toward their task thresholds.",
        )
    else:
        gen.add_missing("training_quality_curves", "6.3", "Quality columns not found.", "avg_Q_rgb and avg_R_depth.")

    if "policy_entropy" in df.columns:
        fig, ax = plt.subplots(figsize=(3.55, 2.5))
        ax.plot(x, df["policy_entropy"], color="tab:green")
        ax.set_xlabel("Environment steps" if x_col == "global_step" else "PPO updates")
        ax.set_ylabel("Policy entropy")
        ax.grid(True, alpha=0.3)
        paths = save_figure(fig, out / "figures", "policy_entropy_curve")
        gen.figures.extend(paths.values())
        gen.used_sources["policy_entropy_curve"] = [crpo_log]
        write_caption(
            out / "figures" / "policy_entropy_curve_caption.md",
            "Policy entropy curve",
            [crpo_log],
            "The entropy curve indicates how exploration changes as CRPO-guided PPO training progresses.",
        )
    else:
        gen.add_missing("policy_entropy_curve", "6.3", "policy_entropy column not found.", "A policy_entropy column in the train log.")


    agg_path = eval_dir / "aggregate_results_test.csv"
    if not agg_path.exists():
        agg_path = eval_dir / "aggregate_results.csv"
    if agg_path.exists():
        agg = pd.read_csv(agg_path)
        if "method" in agg.columns and "J_C_R" in agg.columns and "J_C_D" in agg.columns:
            agg["Method"] = agg["method"].map(standardize_method)
            eps_r = float(df["epsilon_R"].dropna().iloc[-1]) if "epsilon_R" in df.columns and not df["epsilon_R"].dropna().empty else 0.0
            eps_d = float(df["epsilon_D"].dropna().iloc[-1]) if "epsilon_D" in df.columns and not df["epsilon_D"].dropna().empty else 0.0
            rows = []
            for method in METHOD_ORDER:
                sub = agg[agg["Method"] == method]
                if sub.empty:
                    continue
                avg_r = float(sub["J_C_R"].mean())
                avg_d = float(sub["J_C_D"].mean())
                rgb_ok = avg_r <= eps_r if eps_r > 0 else avg_r <= 0
                dep_ok = avg_d <= eps_d if eps_d > 0 else avg_d <= 0
                rows.append(
                    {
                        "Method": method,
                        "Average RGB constraint cost": avg_r,
                        "Average depth constraint cost": avg_d,
                        "RGB feasible": rgb_ok,
                        "Depth feasible": dep_ok,
                        "Overall feasible": rgb_ok and dep_ok,
                    }
                )
            table = pd.DataFrame(rows)
            paths = save_table_bundle(table, out / "tables", "test_constraint_satisfaction")
            gen.tables.extend(paths.values())
            gen.used_sources["test_constraint_satisfaction"] = [agg_path, crpo_log]
            write_caption(
                out / "tables" / "test_constraint_satisfaction_caption.md",
                "Test constraint satisfaction",
                [agg_path, crpo_log],
                f"Feasibility is evaluated using the logged CRPO thresholds epsilon_R={eps_r:g} and epsilon_D={eps_d:g}.",
            )
        else:
            gen.add_missing("test_constraint_satisfaction", "6.3", "Aggregate result file lacks method/J_C_R/J_C_D columns.", "Method-level test constraint costs.")
    else:
        gen.add_missing("test_constraint_satisfaction", "6.3", "No aggregate test CSV was found.", "aggregate_results_test.csv.")


def generate_resource_outputs(root: Path, out: Path, gen: Generated) -> None:
    eval_dir = find_existing(
        root,
        [
            "outputs/eiffel15_surrogate_test/eval_composite_app60202_q070_d035",
            "outputs/eiffel15_surrogate_test/eval_q070_d046_app60202",
            "outputs/eiffel15_surrogate_test/eval",
            "outputs/fushimi_surrogate_test/eval",
        ],
    )
    if not eval_dir:
        gen.add_missing("resource_allocation_statistics", "6.5", "No evaluation directory found.", "Per-trajectory or per-slot evaluation logs.")
        return
    per_path = eval_dir / "per_traj_results_test.csv"
    if not per_path.exists():
        alt = list(eval_dir.glob("per_traj_results*_test.csv"))
        per_path = alt[0] if alt else per_path
    if not per_path.exists():
        gen.add_missing("resource_allocation_statistics", "6.5", "No per-trajectory test CSV found.", "per_traj_results_test.csv with avg_kd and avg_beta_d.")
        return
    df = pd.read_csv(per_path)
    required = ["method", "avg_kd", "avg_beta_d", "avg_q_rgb", "avg_r_depth"]
    if not all(c in df.columns for c in required):
        gen.add_missing("resource_allocation_statistics", "6.5", f"`{per_path}` lacks one of {required}.", "Method-level allocation and quality columns.")
        return
    df["Method"] = df["method"].map(standardize_method)
    rows = []
    for method in METHOD_ORDER:
        sub = df[df["Method"] == method]
        if sub.empty:
            continue
        rows.append(
            {
                "Method": method,
                "Avg. depth subcarriers (k_d)": sub["avg_kd"].mean(),
                "Std. depth subcarriers (k_d)": sub["avg_kd"].std(ddof=0),
                "Avg. depth power ratio (rho_d)": sub["avg_beta_d"].mean(),
                "Std. depth power ratio (rho_d)": sub["avg_beta_d"].std(ddof=0),
                "Avg. RGB SNR": "N/A",
                "Avg. depth SNR": "N/A",
                "Avg. (Q_rgb)": sub["avg_q_rgb"].mean(),
                "Avg. (R_depth)": sub["avg_r_depth"].mean(),
            }
        )
    table = pd.DataFrame(rows)
    paths = save_table_bundle(table, out / "tables", "resource_allocation_statistics")
    gen.tables.extend(paths.values())
    gen.used_sources["resource_allocation_statistics"] = [per_path]
    write_caption(
        out / "tables" / "resource_allocation_statistics_caption.md",
        "Resource allocation statistics",
        [per_path],
        "The table summarizes method-level allocation statistics from held-out trajectory aggregates; SNR columns are marked N/A because per-slot SNR was not present in this evaluation file.",
    )

    gen.add_missing(
        "action_distribution",
        "6.5",
        "Only per-trajectory averages were found; no per-slot selected k_d/rho_d log was available.",
        "Per-slot evaluation logs with selected k_d and rho_d for each method.",
        "Run the policy evaluator with per-slot action logging enabled.",
    )
    gen.add_missing(
        "allocation_vs_channel_quality",
        "6.5",
        "No per-slot channel quality columns were available in the evaluation outputs.",
        "Per-slot channel/SNR plus selected k_d and rho_d.",
        "Run the policy evaluator with per-slot SNR/channel logging enabled.",
    )
    gen.add_missing(
        "qrgb_rdepth_tradeoff_scatter",
        "6.5",
        "No per-slot Q_rgb and R_depth records were available; per-trajectory means are insufficient for the requested slot-level scatter.",
        "Per-slot Q_rgb and R_depth for each evaluated method.",
        "Run the policy evaluator with per-slot quality logging enabled.",
    )
    gen.add_missing(
        "per_slot_policy_trajectory",
        "6.5",
        "No representative per-slot policy trajectory file was found.",
        "One episode-level time series containing k_d, rho_d, Q_rgb, and R_depth.",
        "Run one test episode with detailed time-slot logging enabled.",
    )


def generate_real_mapping_outputs(root: Path, gen: Generated) -> None:
    candidate_files = list((root / "outputs").rglob("*real_metrics*.csv")) if (root / "outputs").exists() else []
    usable = []
    for path in candidate_files:
        try:
            df = pd.read_csv(path, nrows=5)
        except Exception:
            continue
        cols = {c.lower() for c in df.columns}
        if "method" in cols and {"psnr", "ssim", "lpips"}.issubset(cols) and (
            "chamfer" in cols or "chamfer_distance" in cols
        ):
            usable.append(path)
    if usable:
        gen.add_missing(
            "main_3d_mapping_quality_comparison",
            "6.4",
            "Method-level real metrics candidates exist but automatic generation is disabled until column semantics are verified.",
            "A method-level real final reconstruction metrics CSV with method, PSNR, SSIM, LPIPS, Chamfer, F-score, completeness.",
        )
    else:
        gen.add_missing(
            "main_3d_mapping_quality_comparison",
            "6.4",
            "No method-level real final reconstruction metrics CSV was found. Existing real metrics appear to be per degraded GSFusion condition, not per evaluated policy method.",
            "A CSV with rows for Fixed-balanced, RGB-priority, Depth-priority, Random, PPO-penalty, Lagrangian-PPO, and CRPO-guided PPO containing real final PSNR/SSIM/LPIPS/Chamfer/F-score/completeness.",
            "Run final method-level GSFusion reconstruction evaluation for each policy/baseline and export a method-level real metrics CSV.",
        )
    for name in [
        "main_quality_bar_chart",
        "reconstruction_metric_comparison",
        "qualitative_reconstruction_comparison",
        "geometry_error_visualization",
    ]:
        gen.add_missing(
            name,
            "6.4",
            "Required method-level real 3D reconstruction outputs were not found.",
            "Aligned method-level real metrics and/or same-view qualitative renderings for each method.",
            "Run final method-level GSFusion reconstruction and rendering evaluation for each method.",
        )


def generate_ablation_outputs(gen: Generated) -> None:
    for name in ["ablation_summary", "ablation_quality_bar_chart", "ablation_constraint_bar_chart"]:
        gen.add_missing(
            name,
            "6.6",
            "No dedicated ablation result files were found for no-CRPO, no-joint-term, subcarrier-only, power-only, or no-task-progress variants.",
            "Ablation CSVs with variant, Q_gt/Q_app/Q_geo, C_rgb, C_depth, and feasibility.",
            "Run ablation experiments and export an ablation_summary.csv with the requested variants.",
        )


def write_missing_report(gen: Generated, out: Path) -> None:
    path = out / "missing_report" / "missing_required_results.md"
    lines = ["# Missing Required Results", ""]
    if not gen.missing:
        lines.append("No required outputs are missing.")
    for item in gen.missing:
        lines.extend(
            [
                f"## {item['name']}",
                "",
                f"- Section: {item['section']}",
                f"- Reason: {item['reason']}",
                f"- Needed data: {item['needed']}",
            ]
        )
        if item["command"]:
            lines.append(f"- Recommended command/script: `{item['command']}`")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_final_report(root: Path, out: Path, files: list[Path], gen: Generated) -> None:
    lines = ["# Result Generation Report", ""]
    lines.append(f"Project root: `{root}`")
    lines.append(f"Discovered result files: {len(files)}")
    lines.append(f"Generated figure files: {len(gen.figures)}")
    lines.append(f"Generated table files: {len(gen.tables)}")
    lines.append(f"Missing required outputs: {len(gen.missing)}")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    if gen.figures:
        for path in gen.figures:
            lines.append(f"- `{path.relative_to(root) if path.is_relative_to(root) else path}`")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Generated Tables")
    lines.append("")
    if gen.tables:
        for path in gen.tables:
            lines.append(f"- `{path.relative_to(root) if path.is_relative_to(root) else path}`")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Source Files Used")
    lines.append("")
    if gen.used_sources:
        for name, paths in gen.used_sources.items():
            lines.append(f"### {name}")
            for path in paths:
                lines.append(f"- `{path.relative_to(root) if path.is_relative_to(root) else path}`")
            lines.append("")
    else:
        lines.append("- None")
    lines.append("## Missing Results")
    lines.append("")
    if gen.missing:
        for item in gen.missing:
            lines.append(f"- **{item['name']}** (Section {item['section']}): {item['reason']}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Recommended Next Commands")
    lines.append("")
    commands = [item["command"] for item in gen.missing if item.get("command")]
    if commands:
        for command in sorted(set(commands)):
            lines.append(f"- `{command}`")
    else:
        lines.append("- No additional commands are required.")
    (out / "result_generation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_possible_file(path: Path) -> str:
    """Best-effort loader used only to validate the script can inspect common formats."""
    try:
        if path.suffix == ".csv":
            pd.read_csv(path, nrows=1)
            return "csv"
        if path.suffix in {".json", ".jsonl"}:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                f.readline()
            return "json"
        if path.suffix == ".npz":
            np.load(path)
            return "npz"
        if path.suffix == ".npy":
            np.load(path, mmap_mode="r")
            return "npy"
        if path.suffix == ".pkl":
            with path.open("rb") as f:
                pickle.load(f)
            return "pkl"
    except Exception:
        return "unreadable"
    return "listed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate IEEE-style paper results from existing project outputs.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, default=Path("paper_results"))
    args = parser.parse_args()

    root = args.project_root.resolve()
    out = (root / args.out_dir).resolve() if not args.out_dir.is_absolute() else args.out_dir
    for sub in ["figures", "tables", "data_summary", "missing_report"]:
        (out / sub).mkdir(parents=True, exist_ok=True)

    files = discover_result_files(root)
    write_discovery_summary(files, root, out / "data_summary" / "discovered_result_files.md")

    gen = Generated()
    generate_surrogate_outputs(root, out, gen)
    generate_training_outputs(root, out, gen)
    generate_resource_outputs(root, out, gen)
    generate_real_mapping_outputs(root, gen)
    generate_ablation_outputs(gen)
    write_missing_report(gen, out)
    write_final_report(root, out, files, gen)

    print(f"Discovered files: {len(files)}")
    print(f"Generated figures: {len(gen.figures) // 2} figure groups ({len(gen.figures)} files)")
    print(f"Generated tables: {len(gen.tables) // 3} table groups ({len(gen.tables)} files)")
    print(f"Missing required outputs: {len(gen.missing)}")
    print(f"Report: {out / 'result_generation_report.md'}")


if __name__ == "__main__":
    main()
