#!/usr/bin/env python3
"""Summarize method-level real GSFusion mapping quality for paper Section 6.4."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from plotting_utils import save_figure
from table_utils import save_table_bundle, write_caption


# Nature NPG vivid palette — 7 slots matching METHOD_ORDER
_NATURE_COLORS = {
    "Fixed-balanced allocation":  "#ADB6B6",  # neutral gray   (baseline)
    "RGB-priority allocation":    "#4DBBD5",  # sky blue       (baseline)
    "Depth-priority allocation":  "#00A087",  # teal           (baseline)
    "Random allocation":          "#8491B4",  # lavender       (baseline)
    "PPO-penalty":                "#F39B7F",  # salmon         (baseline)
    "Lagrangian-PPO":             "#3C5488",  # navy           (baseline)
    "CRPO-guided PPO":            "#E64B35",  # vermillion     (proposed)
}
_FALLBACK_COLORS = ["#4DBBD5", "#00A087", "#3C5488", "#F39B7F", "#8491B4", "#91D1C2", "#ADB6B6"]

_DISPLAY_LABELS: dict[str, str] = {
    "PPO-penalty": r"PPO-penalty ($\mu_R\!=\!1.0,\ \mu_D\!=\!1.0$)",
}

QUALITY_FIGURE_NAME_MAP = {
    "quality_q_gt": "Fig5a_quality_q_gt",
    "quality_q_app": "Fig5b_quality_q_app",
    "quality_q_geo": "Fig5c_quality_q_geo",
    "metric_psnr": "Fig6a_metric_psnr",
    "metric_ssim": "Fig6b_metric_ssim",
    "metric_lpips": "Fig6c_metric_lpips",
    "metric_chamfer": "Fig6d_metric_chamfer",
    "metric_fscore": "Fig6e_metric_fscore",
    "metric_completeness": "Fig6f_metric_completeness",
}

QUALITATIVE_FIGURE_NAME_MAP = {
    "CRPO-guided PPO": "Fig7a_crpo_guided_ppo",
    "Lagrangian-PPO": "Fig7b_lagrangian_ppo",
    "PPO-penalty": "Fig7c_ppo_penalty",
    "Random allocation": "Fig7d_random_allocation",
    "Depth-priority allocation": "Fig7e_depth_priority_allocation",
    "Reference": "Fig7f_reference",
}


def _display_label(m: str) -> str:
    return _DISPLAY_LABELS.get(m, m)


def _slug(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_")


def _fmt_val(v: float) -> str:
    if not np.isfinite(v):
        return ""
    return f"{v:.1f}" if abs(v) >= 10 else f"{v:.3f}"


def _set_nature_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "font.weight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 10,
        "grid.linewidth": 0.75,
        "grid.color": "0.84",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _method_colors(methods: list[str]) -> list[str]:
    return [
        _NATURE_COLORS.get(m, _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)])
        for i, m in enumerate(methods)
    ]


METHOD_ORDER = [
    "Fixed-balanced allocation",
    "RGB-priority allocation",
    "Depth-priority allocation",
    "Random allocation",
    "PPO-penalty",
    "Lagrangian-PPO",
    "CRPO-guided PPO",
]


REQUIRED_METRIC_COLUMNS = [
    "condition",
    "psnr",
    "ssim",
    "lpips",
    "chamfer",
    "fscore",
    "completeness",
]


def _compute_basis(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Compute p5/p95 normalization bounds from a reference dataframe."""
    basis = {}
    for col in ["psnr", "ssim", "lpips", "chamfer", "fscore", "completeness"]:
        vals = pd.to_numeric(df[col], errors="coerce")
        basis[col] = (float(vals.quantile(0.05)), float(vals.quantile(0.95)))
    return basis


def _metric_score(
    series: pd.Series,
    higher_is_better: bool,
    lo: float | None = None,
    hi: float | None = None,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if lo is None:
        lo = float(values.quantile(0.05))
    if hi is None:
        hi = float(values.quantile(0.95))
    if not np.isfinite(lo) or not np.isfinite(hi) or math.isclose(lo, hi):
        return pd.Series(np.nan, index=series.index)
    score = ((values - lo) / (hi - lo)).clip(0.0, 1.0)
    return score if higher_is_better else 1.0 - score


def _add_quality_labels(
    df: pd.DataFrame,
    basis: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    b = basis or {}
    out["Qbar_psnr"] = _metric_score(out["psnr"], True, *b.get("psnr", (None, None)))
    out["Qbar_ssim"] = _metric_score(out["ssim"], True, *b.get("ssim", (None, None)))
    out["Qbar_lpips"] = _metric_score(out["lpips"], False, *b.get("lpips", (None, None)))
    out["Qbar_fscore"] = _metric_score(out["fscore"], True, *b.get("fscore", (None, None)))
    out["Qbar_comp"] = _metric_score(out["completeness"], True, *b.get("completeness", (None, None)))
    out["Qbar_chamfer"] = _metric_score(out["chamfer"], False, *b.get("chamfer", (None, None)))
    out["Q_app"] = 0.6 * out["Qbar_psnr"] + 0.2 * out["Qbar_ssim"] + 0.2 * out["Qbar_lpips"]
    out["Q_geo"] = 0.35 * out["Qbar_fscore"] + 0.35 * out["Qbar_comp"] + 0.30 * out["Qbar_chamfer"]
    out["Q_gt"] = (0.45 * out["Q_app"] + 0.45 * out["Q_geo"] + 0.10 * out["Q_app"] * out["Q_geo"]).clip(0, 1)
    return out


def _read_conditions(path: Path) -> pd.DataFrame:
    index = json.loads(path.read_text())
    rows = []
    for cond in index:
        rows.append({
            "condition": cond["condition"],
            "Method": cond.get("method", cond.get("method_key", "")),
            "Trajectory": cond.get("trajectory_id", ""),
            "training_seed": cond.get("training_seed", np.nan),
            "shadow_seed": cond.get("shadow_seed", np.nan),
            "action_seed": cond.get("action_seed", np.nan),
            "Avg. Q_rgb": cond.get("q_rgb", np.nan),
            "Avg. R_depth": cond.get("r_depth", np.nan),
            "Avg. k_d": cond.get("avg_k_d", np.nan),
            "Avg. rho_d": cond.get("avg_beta_d", np.nan),
            "sequence_path": cond.get("sequence_path", ""),
            "output_path": cond.get("output_path", ""),
            "frames": cond.get("frames", []),
        })
    return pd.DataFrame(rows)


def _order_methods(df: pd.DataFrame) -> pd.DataFrame:
    order = {name: i for i, name in enumerate(METHOD_ORDER)}
    out = df.copy()
    out["_order"] = out["Method"].map(order).fillna(999)
    return out.sort_values(["_order", "Method"]).drop(columns=["_order"])


def _save_main_table(
    df: pd.DataFrame,
    out_dir: Path,
    basis: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    labels = _add_quality_labels(df, basis)
    agg_cols = [
        "psnr", "ssim", "lpips", "chamfer", "fscore", "completeness",
        "Q_app", "Q_geo", "Q_gt", "Avg. Q_rgb", "Avg. R_depth", "Avg. k_d", "Avg. rho_d",
    ]
    rename_map = {
        "psnr": "PSNR ↑",
        "ssim": "SSIM ↑",
        "lpips": "LPIPS ↓",
        "chamfer": "Chamfer ↓",
        "fscore": "F-score ↑",
        "completeness": "Completeness ↑",
        "Q_app": "Q_app ↑",
        "Q_geo": "Q_geo ↑",
        "Q_gt": "Q_gt ↑",
    }
    table = labels.groupby("Method", as_index=False)[agg_cols].mean()
    table = _order_methods(table)
    table = table.rename(columns=rename_map)
    # Cross-trajectory std for error bars (ddof=1 over the held-out trajectories).
    std_table = labels.groupby("Method", as_index=False)[agg_cols].std(ddof=1)
    std_table = _order_methods(std_table)
    std_table = std_table.rename(columns=rename_map)
    std_table = std_table.set_index("Method").reindex(table["Method"]).reset_index()
    save_table_bundle(table, out_dir / "tables", "main_3d_mapping_quality_comparison")
    write_caption(
        out_dir / "tables" / "main_3d_mapping_quality_comparison_caption.md",
        "Main 3D mapping quality comparison",
        [Path("conditions_index.json"), Path("gsfusion_real_metrics.csv")],
        "The table reports real GSFusion reconstruction metrics aggregated over held-out policy-generated RGB-D sequences.",
    )
    return table, std_table


def _hbar_panel(
    ax: plt.Axes,
    values: np.ndarray,
    methods: list[str],
    colors: list[str],
    xlabel: str,
    xlim: tuple[float, float] | None = None,
    errors: np.ndarray | None = None,
) -> None:
    y = np.arange(len(methods))
    err_kw = dict(ecolor="0.3", elinewidth=1.1, capsize=2.5, capthick=1.1)
    bars = ax.barh(
        y, values, color=colors, edgecolor="white", linewidth=0.6, height=0.66,
        xerr=errors if errors is not None else None, error_kw=err_kw,
    )
    for bar, m in zip(bars, methods):
        if m == "CRPO-guided PPO":
            bar.set_edgecolor("black")
            bar.set_linewidth(1.4)

    # Auto-extend x-range to fit value labels on the right (past the error bar)
    finite = values[np.isfinite(values)]
    if errors is not None:
        errs = np.where(np.isfinite(errors), errors, 0.0)
        reach = values + errs
    else:
        errs = np.zeros_like(values)
        reach = values
    max_v = float(reach[np.isfinite(reach)].max()) if len(finite) else 1.0
    x_end = max(xlim[1] if xlim else 0.0, max_v * 1.30)
    ax.set_xlim(0.0, x_end)

    for bar, v, e in zip(bars, values, errs):
        if np.isfinite(v):
            ax.text(
                v + e + x_end * 0.015,
                bar.get_y() + bar.get_height() / 2,
                _fmt_val(v),
                va="center", ha="left", fontsize=9.5, fontweight="bold", color="0.25",
            )

    ax.set_xlabel(xlabel, labelpad=3)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _apply_ylabels(axes_left: list[plt.Axes], methods: list[str]) -> None:
    y = np.arange(len(methods))
    labels = [_display_label(m) for m in methods]
    for ax in axes_left:
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", length=0, pad=4)


def _plot_quality_bars(table: pd.DataFrame, std_table: pd.DataFrame, out_dir: Path) -> None:
    _set_nature_style()
    methods = table["Method"].tolist()
    colors = _method_colors(methods)
    specs = [
        ("Q_gt ↑",  r"$Q_\mathrm{gt}$ ↑", "quality_q_gt"),
        ("Q_app ↑", r"$Q_\mathrm{app}$ ↑", "quality_q_app"),
        ("Q_geo ↑", r"$Q_\mathrm{geo}$ ↑", "quality_q_geo"),
    ]
    for col, xlabel, name in specs:
        mapped_name = QUALITY_FIGURE_NAME_MAP[name]
        fig, ax = plt.subplots(figsize=(4.6, 3.3))
        _hbar_panel(ax, table[col].values, methods, colors, xlabel, xlim=(0, 1.0),
                    errors=std_table[col].values)
        _apply_ylabels([ax], methods)
        save_figure(fig, out_dir, mapped_name)
        write_caption(
            out_dir / f"{mapped_name}_caption.md",
            xlabel,
            [out_dir / "tables" / "main_3d_mapping_quality_comparison.csv"],
            f"Normalized {xlabel} across methods. CRPO-guided PPO (proposed) is outlined in black.",
        )


def _plot_metric_comparison(table: pd.DataFrame, std_table: pd.DataFrame, out_dir: Path) -> None:
    _set_nature_style()
    methods = table["Method"].tolist()
    colors = _method_colors(methods)
    metrics = [
        ("PSNR ↑",         "PSNR (dB) ↑", "metric_psnr"),
        ("SSIM ↑",         "SSIM ↑", "metric_ssim"),
        ("LPIPS ↓",        "LPIPS ↓", "metric_lpips"),
        ("Chamfer ↓",      "Chamfer ↓", "metric_chamfer"),
        ("F-score ↑",      "F-score ↑", "metric_fscore"),
        ("Completeness ↑", "Completeness ↑", "metric_completeness"),
    ]
    for col, xlabel, name in metrics:
        mapped_name = QUALITY_FIGURE_NAME_MAP[name]
        fig, ax = plt.subplots(figsize=(4.8, 3.3))
        _hbar_panel(ax, table[col].values, methods, colors, xlabel,
                    errors=std_table[col].values)
        _apply_ylabels([ax], methods)
        save_figure(fig, out_dir, mapped_name)
        direction = "Lower values indicate better reconstruction." if col.endswith("↓") else "Higher values indicate better reconstruction."
        write_caption(
            out_dir / f"{mapped_name}_caption.md",
            xlabel,
            [out_dir / "tables" / "main_3d_mapping_quality_comparison.csv"],
            f"Raw GSFusion {xlabel} across methods. {direction} CRPO-guided PPO (proposed) is outlined in black.",
        )


def _label_image(img: Image.Image, label: str, scale: float = 2.0) -> Image.Image:
    """Scale the image without adding text overlays."""
    img = img.convert("RGB")
    if scale > 1.0:
        img = img.resize(
            (int(round(img.width * scale)), int(round(img.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return img


def _thumbnail_array(path: Path, size: tuple[int, int] = (160, 90)) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size, Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _best_matching_render(clean_path: Path, render_dir: Path) -> tuple[Path, float]:
    target = _thumbnail_array(clean_path)
    best_path = None
    best_score = -float("inf")
    for path in sorted(render_dir.glob("frame*.png")):
        cand = _thumbnail_array(path)
        score = -float(np.mean(np.square(target - cand)))
        if score > best_score:
            best_score = score
            best_path = path
    if best_path is None:
        raise SystemExit(f"no render_eval frames under {render_dir}")
    return best_path, best_score


def _make_qualitative(
    df: pd.DataFrame,
    out_dir: Path,
    frame_idx: int,
    scale: float,
    trajectory: str | None,
) -> None:
    if trajectory:
        view_df = df[df["Trajectory"].astype(str) == str(trajectory)].copy()
        if view_df.empty:
            raise SystemExit(f"no qualitative rows for trajectory={trajectory}")
    else:
        counts = df.groupby("Trajectory")["Method"].nunique().sort_values(ascending=False)
        if counts.empty:
            return
        trajectory = str(counts.index[0])
        view_df = df[df["Trajectory"].astype(str) == trajectory].copy()

    by_method = _order_methods(view_df).drop_duplicates("Method")
    first_frames = by_method.iloc[0]["frames"]
    if not first_frames:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    clean_path = first_frames[min(frame_idx, len(first_frames) - 1)].get("clean_image")
    if not clean_path:
        return
    clean = Path(clean_path)
    if not clean.exists():
        raise SystemExit(f"missing clean reference frame: {clean}")
    img = _label_image(Image.open(clean), "Reference", scale)
    ref_name = QUALITATIVE_FIGURE_NAME_MAP["Reference"]
    ref_png = out_dir / f"{ref_name}.png"
    img.save(ref_png, compress_level=0)
    img.save(out_dir / f"{ref_name}.pdf")
    saved.append(ref_png)
    rows_by_method = {row["Method"]: row for _, row in by_method.iterrows()}
    for method, fig_name in QUALITATIVE_FIGURE_NAME_MAP.items():
        if method == "Reference":
            continue
        if method not in rows_by_method:
            raise SystemExit(f"missing qualitative row for method={method}")
        row = rows_by_method[method]
        render_dir = Path(row["output_path"]) / "render_eval"
        render, _ = _best_matching_render(clean, render_dir)
        img = _label_image(Image.open(render), _display_label(row["Method"]), scale)
        out_png = out_dir / f"{fig_name}.png"
        img.save(out_png, compress_level=0)
        img.save(out_dir / f"{fig_name}.pdf")
        saved.append(out_png)
    write_caption(
        out_dir / "Fig7_qualitative_reconstruction_caption.md",
        "Qualitative reconstruction comparison, separated views",
        [Path("conditions_index.json")],
        f"GSFusion renderings matched to target camera view: trajectory {trajectory}, frame {frame_idx}.",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conditions-dir", default="outputs/eiffel15_mapping_quality/conditions")
    ap.add_argument("--metrics", default="outputs/eiffel15_mapping_quality/gsfusion_real_metrics.csv")
    ap.add_argument("--out-dir", default="Figures")
    ap.add_argument("--qualitative-frame", type=int, default=75)
    ap.add_argument("--qualitative-scale", type=float, default=4.0)
    ap.add_argument("--qualitative-trajectory", default="traj10")
    ap.add_argument("--qualitative-n-random", type=int, default=1,
                    help="Randomly pick N frames for qualitative comparison (overrides --qualitative-frame when >1)")
    args = ap.parse_args()

    cond_path = Path(args.conditions_dir) / "conditions_index.json"
    metrics_path = Path(args.metrics)
    if not cond_path.exists():
        raise SystemExit(f"missing conditions index: {cond_path}")
    if not metrics_path.exists():
        raise SystemExit(f"missing metrics CSV: {metrics_path}")
    cond_df = _read_conditions(cond_path)
    metric_df = pd.read_csv(metrics_path)
    missing_cols = [col for col in REQUIRED_METRIC_COLUMNS if col not in metric_df.columns]
    if missing_cols:
        raise SystemExit(
            "metrics CSV is missing required columns: "
            f"{missing_cols}. Source: {metrics_path}"
        )
    df = cond_df.merge(metric_df, on="condition", how="inner")
    if df.empty:
        raise SystemExit("no overlapping conditions between conditions_index.json and metrics CSV")
    # Compute normalization basis from the full metrics CSV so the p5/p95 bounds
    # are stable regardless of how many conditions each method contributes.
    basis = _compute_basis(metric_df)
    out_dir = Path(args.out_dir)
    table_dir = out_dir / "tables"
    table, std_table = _save_main_table(df, out_dir, basis)
    _plot_quality_bars(table, std_table, out_dir)
    _plot_metric_comparison(table, std_table, out_dir)

    # Determine frame indices for qualitative comparison
    import random as _random
    traj = args.qualitative_trajectory
    traj_df = df[df["Trajectory"].astype(str) == str(traj)] if traj else df
    frame_count = max((len(r) for r in traj_df["frames"] if r), default=102)
    if args.qualitative_n_random > 1:
        _random.seed(42)
        frame_indices = sorted(_random.sample(range(frame_count), min(args.qualitative_n_random, frame_count)))
    else:
        frame_indices = [args.qualitative_frame]

    # Remove old per-frame subfolders before regenerating
    import shutil
    for old_dir in out_dir.glob("frame??????"):
        if old_dir.is_dir():
            shutil.rmtree(old_dir)
    for fi in frame_indices:
        _make_qualitative(df, out_dir, fi, args.qualitative_scale, traj)
    print(f"wrote {table_dir / 'main_3d_mapping_quality_comparison.csv'}")
    print(f"wrote figures under {out_dir}")


if __name__ == "__main__":
    main()
