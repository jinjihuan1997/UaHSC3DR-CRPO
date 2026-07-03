#!/usr/bin/env python3
"""Generate Section 6.5 resource allocation behavior results from per-slot logs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting_utils import save_figure
from table_utils import save_table_bundle, write_caption


METHOD_ORDER = [
    "Fixed-balanced allocation",
    "RGB-priority allocation",
    "Depth-priority allocation",
    "Random allocation",
    "PPO-penalty",
    "Lagrangian-PPO",
    "CRPO-guided PPO",
]

COLORS = {
    "Fixed-balanced allocation": "#ADB6B6",
    "RGB-priority allocation": "#4DBBD5",
    "Depth-priority allocation": "#00A087",
    "Random allocation": "#8491B4",
    "PPO-penalty": "#F39B7F",
    "Lagrangian-PPO": "#3C5488",
    "CRPO-guided PPO": "#E64B35",
}

PPO_PENALTY_PARAMS = r"$\mu_R=1.0,\ \mu_D=1.0$"

REQUIRED_COLUMNS = [
    "method",
    "trajectory_id",
    "slot",
    "k_d",
    "beta_d",
    "total_snr_db",
    "snr_rgb_db",
    "snr_depth_db",
    "q_rgb",
    "r_depth",
    "cost_rgb",
    "cost_depth",
]


def _display_label(method: str) -> str:
    if method == "PPO-penalty":
        return f"PPO-penalty ({PPO_PENALTY_PARAMS})"
    return method


def _method_order(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else 999


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    methods = sorted(df["method"].dropna().unique(), key=_method_order)
    return [m for m in METHOD_ORDER if m in methods] + [m for m in methods if m not in METHOD_ORDER]


def _set_style() -> None:
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
        "legend.fontsize": 9.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # All four spines visible (IEEE box style)
        "axes.linewidth": 1.2,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        # Inward ticks on all sides
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.2,
        "ytick.major.size": 4.2,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        # Dotted light grid
        "axes.grid": True,
        "grid.linewidth": 0.75,
        "grid.color": "0.78",
        "grid.linestyle": ":",
        "grid.alpha": 0.8,
        # Legend with white background and thin border
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.fancybox": False,
        "legend.edgecolor": "0.65",
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.8,
        "legend.handletextpad": 0.5,
        # Output
        "lines.linewidth": 2.2,
        "lines.markersize": 5.0,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _write_fig_caption(out_dir: Path, name: str, title: str, source: Path, conclusion: str) -> None:
    write_caption(out_dir / f"{name}_caption.md", title, [source], conclusion)


def _load_per_slot(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"missing per-slot policy CSV: {path}")
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")
    for col in REQUIRED_COLUMNS:
        if col not in {"method", "trajectory_id"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["method", "trajectory_id", "slot", "k_d", "beta_d", "q_rgb", "r_depth"])
    if df.empty:
        raise SystemExit(f"no usable per-slot rows in {path}")
    return df


def _resource_table(df: pd.DataFrame, out_dir: Path, source: Path) -> pd.DataFrame:
    rows = []
    for method in _ordered_methods(df):
        sub = df[df["method"] == method]
        rows.append({
            "Method": method,
            "Avg. depth subcarriers (k_d)": sub["k_d"].mean(),
            "Std. depth subcarriers (k_d)": sub["k_d"].std(ddof=0),
            "Avg. depth power ratio (rho_d)": sub["beta_d"].mean(),
            "Std. depth power ratio (rho_d)": sub["beta_d"].std(ddof=0),
            "Avg. total SNR (dB)": sub["total_snr_db"].mean(),
            "Avg. RGB SNR (dB)": sub["snr_rgb_db"].mean(),
            "Avg. depth SNR (dB)": sub["snr_depth_db"].mean(),
            "Avg. Q_rgb": sub["q_rgb"].mean(),
            "Avg. R_depth": sub["r_depth"].mean(),
            "Avg. C_rgb": sub["cost_rgb"].mean(),
            "Avg. C_depth": sub["cost_depth"].mean(),
        })
    table = pd.DataFrame(rows)
    save_table_bundle(table, out_dir / "tables", "resource_allocation_statistics")
    write_caption(
        out_dir / "tables" / "resource_allocation_statistics_caption.md",
        "Resource allocation statistics",
        [source],
        "The table summarizes per-slot resource allocation, channel quality, received RGB quality, depth availability, and constraint costs.",
    )
    return table


def _plot_action_distribution(df: pd.DataFrame, out_dir: Path, source: Path, column: str, name: str, xlabel: str) -> None:
    _set_style()
    methods = _ordered_methods(df)
    x_vals = sorted(df[column].dropna().unique())
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for method in methods:
        sub = df[df["method"] == method]
        probs = sub[column].value_counts(normalize=True).reindex(x_vals, fill_value=0.0).sort_index()
        ax.plot(
            x_vals,
            probs.values,
            marker="o",
            linewidth=2.2,
            markersize=5.0,
            color=COLORS.get(method),
            label=_display_label(method),
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Selection probability")
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks(x_vals)
    ax.legend(ncol=2, loc="best")
    save_figure(fig, out_dir, name)
    _write_fig_caption(
        out_dir,
        name,
        xlabel,
        source,
        f"The figure shows how often each method selects each {xlabel} value over all held-out slots.",
    )


def _channel_groups(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bins = pd.qcut(out["total_snr_db"], q=3, labels=False, duplicates="drop")
    label_map = {
        0: "Low channel quality",
        1: "Medium channel quality",
        2: "High channel quality",
    }
    out["channel_group"] = bins.map(label_map)
    return out.dropna(subset=["channel_group"])


def _plot_channel_group_metric(
    df: pd.DataFrame,
    out_dir: Path,
    source: Path,
    value_col: str,
    name: str,
    ylabel: str,
) -> None:
    _set_style()
    grouped = _channel_groups(df)
    methods = _ordered_methods(grouped)
    groups = ["Low channel quality", "Medium channel quality", "High channel quality"]
    x = np.arange(len(groups))
    width = 0.78 / max(len(methods), 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for i, method in enumerate(methods):
        sub = grouped[grouped["method"] == method]
        values = sub.groupby("channel_group", observed=True)[value_col].mean().reindex(groups)
        ax.bar(
            x + (i - (len(methods) - 1) / 2) * width,
            values.values,
            width=width,
            color=COLORS.get(method),
            label=_display_label(method),
            edgecolor="white",
            linewidth=0.7,
        )
    ax.set_xlabel("Channel-quality group")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Medium", "High"])
    ax.legend(ncol=2, loc="best")
    save_figure(fig, out_dir, name)
    _write_fig_caption(
        out_dir,
        name,
        ylabel,
        source,
        f"The figure groups time slots by total SNR terciles and reports average {ylabel} for each method.",
    )


def _plot_tradeoff_scatter(df: pd.DataFrame, out_dir: Path, source: Path, q_thr: float, r_thr: float) -> None:
    _set_style()
    methods = _ordered_methods(df)
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.fill_between(
        [q_thr, 1.02],
        [r_thr, r_thr],
        [1.02, 1.02],
        color="#E8F4EA",
        alpha=0.75,
        zorder=0,
    )
    ax.axvline(q_thr, color="0.25", linestyle="--", linewidth=1.3)
    ax.axhline(r_thr, color="0.25", linestyle="--", linewidth=1.3)
    for method in methods:
        sub = df[df["method"] == method]
        ax.scatter(
            sub["q_rgb"],
            sub["r_depth"],
            s=18,
            alpha=0.35 if method != "CRPO-guided PPO" else 0.65,
            color=COLORS.get(method),
            label=_display_label(method),
            edgecolors="none",
        )
    ax.set_xlabel(r"$Q_\mathrm{rgb}$")
    ax.set_ylabel(r"$R_\mathrm{depth}$")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.legend(ncol=1, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    save_figure(fig, out_dir, "qrgb_rdepth_tradeoff_scatter")
    _write_fig_caption(
        out_dir,
        "qrgb_rdepth_tradeoff_scatter",
        "RGB-depth feasibility scatter",
        source,
        f"The dashed lines indicate Q_thr={q_thr:g} and R_thr={r_thr:g}; the upper-right region is jointly feasible.",
    )


def _plot_crpo_time_series(
    df: pd.DataFrame,
    out_dir: Path,
    source: Path,
    trajectory: str,
    method: str,
    value_col: str,
    name: str,
    ylabel: str,
    threshold: float | None = None,
) -> None:
    _set_style()
    sub = df[(df["trajectory_id"].astype(str) == trajectory) & (df["method"] == method)].sort_values("slot")
    if sub.empty:
        raise SystemExit(f"no rows for method={method}, trajectory={trajectory}")
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(sub["slot"], sub[value_col], color=COLORS.get(method), linewidth=2.2)
    if threshold is not None:
        ax.axhline(threshold, color="0.25", linestyle="--", linewidth=1.3,
                   label=f"Threshold = {threshold:g}")
        ax.legend(loc="best")
    ax.set_xlabel("Time slot")
    ax.set_ylabel(ylabel)
    save_figure(fig, out_dir, name)
    _write_fig_caption(
        out_dir,
        name,
        ylabel,
        source,
        f"The figure shows the per-slot {ylabel} trajectory of {method} on {trajectory}.",
    )


def main() -> None:
    global PPO_PENALTY_PARAMS

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-slot", default="outputs/eiffel15_mapping_quality/conditions/per_slot_policy.csv")
    ap.add_argument("--out-dir", default="paper_results/figures/resource_allocation_behavior")
    ap.add_argument("--q-thr", type=float, default=0.70)
    ap.add_argument("--r-thr", type=float, default=0.35)
    ap.add_argument("--trajectory", default="traj10")
    ap.add_argument("--policy-method", default="CRPO-guided PPO")
    ap.add_argument("--penalty-params", default=PPO_PENALTY_PARAMS)
    args = ap.parse_args()

    PPO_PENALTY_PARAMS = args.penalty_params

    source = Path(args.per_slot)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_per_slot(source)
    table = _resource_table(df, out_dir, source)
    _plot_action_distribution(df, out_dir, source, "k_d", "action_distribution_kd", r"Depth subcarriers $k_d$")
    _plot_action_distribution(df, out_dir, source, "beta_d", "action_distribution_rho", r"Depth power ratio $\rho_d$")
    _plot_channel_group_metric(df, out_dir, source, "k_d", "allocation_vs_channel_kd", r"Average $k_d$")
    _plot_channel_group_metric(df, out_dir, source, "beta_d", "allocation_vs_channel_rho", r"Average $\rho_d$")
    _plot_channel_group_metric(df, out_dir, source, "q_rgb", "allocation_vs_channel_qrgb", r"Average $Q_\mathrm{rgb}$")
    _plot_channel_group_metric(df, out_dir, source, "r_depth", "allocation_vs_channel_rdepth", r"Average $R_\mathrm{depth}$")
    _plot_tradeoff_scatter(df, out_dir, source, args.q_thr, args.r_thr)
    _plot_crpo_time_series(df, out_dir, source, args.trajectory, args.policy_method, "total_snr_db", f"per_slot_snr_{args.trajectory}", "Total SNR (dB)")
    _plot_crpo_time_series(df, out_dir, source, args.trajectory, args.policy_method, "k_d", f"per_slot_kd_{args.trajectory}", r"$k_d$")
    _plot_crpo_time_series(df, out_dir, source, args.trajectory, args.policy_method, "beta_d", f"per_slot_rho_{args.trajectory}", r"$\rho_d$")
    _plot_crpo_time_series(df, out_dir, source, args.trajectory, args.policy_method, "q_rgb", f"per_slot_qrgb_{args.trajectory}", r"$Q_\mathrm{rgb}$", args.q_thr)
    _plot_crpo_time_series(df, out_dir, source, args.trajectory, args.policy_method, "r_depth", f"per_slot_rdepth_{args.trajectory}", r"$R_\mathrm{depth}$", args.r_thr)

    print(f"wrote table: {out_dir / 'tables' / 'resource_allocation_statistics.csv'}")
    print(f"wrote figures under: {out_dir}")
    print(f"methods: {', '.join(table['Method'].tolist())}")


if __name__ == "__main__":
    main()
