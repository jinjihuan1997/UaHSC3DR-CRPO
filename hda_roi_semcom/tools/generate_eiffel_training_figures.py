#!/usr/bin/env python3
"""Generate IEEE-style multi-seed training diagnostics for Fig. 4."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
MULTISEED = ROOT / "outputs/eiffel15_surrogate_test/multiseed_q070_d035"
PENALTY_SWEEP = ROOT / "outputs/eiffel15_surrogate_test/penalty_multiseed_sweep_q070_d035"
FIG_DIR = ROOT / "paper_results/figures"
TABLE_DIR = ROOT / "paper_results/tables"

SMOOTH = 19
SHADE = "stderr"
SHADE_ALPHA = 0.06
Q_THR = 0.70
R_THR = 0.35
EPS_RGB = 0.05
EPS_DEPTH = 0.05

# Fixed y-axis ranges for all single figures — prevents threshold lines from
# distorting axis scale and ensures consistent appearance across paper versions.
METRICS = {
    "q_rec": {
        "column": "avg_q_3d",
        "ylabel": r"Average $\hat{Q}_{\mathrm{rec}}$",
        "stem": "Fig4a_eiffel_training_reward_curve",
        "ylim": (0.70, 0.86),
    },
    "J_C_R": {
        "column": "J_C_R",
        "ylabel": r"Average $J_{C_R}$",
        "stem": "Fig4b_eiffel_training_rgb_constraint_costs",
        "threshold": EPS_RGB,
        "threshold_label": r"$\epsilon_R^{\mathrm{eval}}=0.05$",
        "ylim": (0.015, 0.185),
    },
    "J_C_D": {
        "column": "J_C_D",
        "ylabel": r"Average $J_{C_D}$",
        "stem": "Fig4c_eiffel_training_depth_constraint_costs",
        "threshold": EPS_DEPTH,
        "threshold_label": r"$\epsilon_D^{\mathrm{eval}}=0.05$",
        "ylim": (0.030, 0.080),
    },
    "Q_rgb": {
        "column": "avg_Q_rgb",
        "ylabel": r"Average $Q_{\mathrm{rgb}}$",
        "stem": "Fig4d_eiffel_training_qrgb_curve",
        "threshold": Q_THR,
        "threshold_label": r"$Q_{\mathrm{thr}}=0.70$",
        "threshold_below": True,
        "ylim": (0.57, 0.78),
    },
    "R_depth": {
        "column": "avg_R_depth",
        "ylabel": r"Average $R_{\mathrm{dep}}$",
        "stem": "Fig4e_eiffel_training_rdepth_curve",
        "threshold": R_THR,
        "threshold_label": r"$R_{\mathrm{thr}}=0.35$",
        "ylim": (0.44, 0.57),
    },
    "entropy": {
        "column": "policy_entropy",
        "ylabel": "Policy entropy",
        "stem": "Fig4f_eiffel_policy_entropy_curve",
        "ylim": (0.90, 3.25),
    },
}

METHOD_STYLES = {
    "CRPO-PPO": {
        "label": "CRPO-guided PPO (proposed)",
        "end_label": "CRPO-guided PPO (proposed)",
        "color": "#E64B35",
        "linestyle": "-",
        "linewidth": 1.4,
    },
    "Lagrangian-PPO": {
        "label": "Lagrangian-PPO",
        "end_label": "Lagrangian-PPO",
        "color": "#3C5488",
        "linestyle": "--",
        "linewidth": 1.4,
    },
    "PPO-penalty-mu05-mu10": {
        "label": r"PPO-penalty $(\mu_R=0.5,\mu_D=1.0)$",
        "end_label": "PPO-penalty",
        "color": "#00A087",
        "linestyle": "-.",
        "linewidth": 1.4,
    },
    "PPO-penalty-mu10-mu10": {
        "label": r"PPO-penalty $(\mu_R=1.0,\mu_D=1.0)$",
        "end_label": "PPO-penalty",
        "color": "#4DBBD5",
        "linestyle": (0, (4, 1)),
        "linewidth": 1.4,
    },
    "PPO-penalty-mu15-mu10": {
        "label": r"PPO-penalty $(\mu_R=1.5,\mu_D=1.0)$",
        "end_label": "PPO-penalty",
        "color": "#F39B7F",
        "linestyle": (0, (1, 1)),
        "linewidth": 1.4,
    },
}

BASE_SOURCES = {
    "CRPO-PPO": (MULTISEED, "crpo_train_log.csv"),
    "Lagrangian-PPO": (MULTISEED, "lagrangian_train_log.csv"),
    "PPO-penalty-mu10-mu10": (MULTISEED, "penalty_train_log.csv"),
}

PENALTY_SOURCES = {
    "PPO-penalty-mu05-mu10": (PENALTY_SWEEP / "mu05_mu10", "penalty_train_log.csv"),
    "PPO-penalty-mu10-mu10": (MULTISEED, "penalty_train_log.csv"),
    "PPO-penalty-mu15-mu10": (PENALTY_SWEEP / "mu15_mu10", "penalty_train_log.csv"),
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.9,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })


def smooth(series: pd.Series) -> pd.Series:
    return series.rolling(window=SMOOTH, min_periods=1, center=True).mean()


def load_method_logs(
    sources: dict[str, tuple[Path, str]], *, require_complete: bool = False
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for method, (root, filename) in sources.items():
        seed_dirs = sorted(root.glob("seed*")) if root.exists() else []
        frames = []
        missing = []
        for seed in range(5):
            path = root / f"seed{seed}" / filename
            if not path.exists():
                missing.append(str(path.relative_to(ROOT)))
                continue
            df = pd.read_csv(path)
            if df.empty:
                missing.append(str(path.relative_to(ROOT)))
                continue
            df = df.drop_duplicates("global_step", keep="last").sort_values("global_step")
            df["train_seed"] = seed
            frames.append(df)
        if missing and require_complete:
            raise FileNotFoundError(f"{method} incomplete:\n" + "\n".join(missing))
        if frames:
            data[method] = pd.concat(frames, ignore_index=True)
            if missing:
                print(f"[partial] {method}: {len(frames)}/5 seeds ({len(missing)} missing)")
            print(f"[source]  {method}: {root.relative_to(ROOT)}/seed*/{filename}")
        elif seed_dirs:
            print(f"[skip]    {method}: no usable logs under {root.relative_to(ROOT)}")
    return data


def summarize_curve(
    data: dict[str, pd.DataFrame], metric_key: str
) -> dict[str, pd.DataFrame]:
    col = METRICS[metric_key]["column"]
    summaries: dict[str, pd.DataFrame] = {}
    for method, df in data.items():
        pieces = []
        for seed, sub in df.groupby("train_seed"):
            sub = sub.sort_values("global_step").copy()
            sub["value"] = smooth(pd.to_numeric(sub[col], errors="coerce"))
            sub["train_seed"] = seed
            pieces.append(sub[["global_step", "train_seed", "value"]])
        all_seed = pd.concat(pieces, ignore_index=True)
        grouped = all_seed.groupby("global_step")["value"]
        summary = grouped.agg(mean="mean", std="std", count="count").reset_index()
        summary["std"] = summary["std"].fillna(0.0)
        summary["stderr"] = summary["std"] / summary["count"].clip(lower=1).pow(0.5)
        summaries[method] = summary
    return summaries


def apply_axis_style(ax: plt.Axes, ylabel: str, *, xlabel: bool = True) -> None:
    if xlabel:
        ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(100_000))
    ax.grid(True, color="0.88", linewidth=0.55, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_threshold(ax: plt.Axes, metric_key: str) -> None:
    """Draw threshold line only when it falls within current ylim; otherwise annotate."""
    cfg = METRICS[metric_key]
    if "threshold" not in cfg:
        return
    y = float(cfg["threshold"])
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    y_span = ymax - ymin

    if ymin <= y <= ymax:
        ax.axhline(y, color="0.38", linestyle=":", linewidth=0.9, zorder=1)
        x_label = xmin + 0.97 * (xmax - xmin)
        below = cfg.get("threshold_below", False)
        if below:
            y_text = max(y - 0.025 * y_span, ymin + 0.03 * y_span)
            va = "top"
        else:
            y_text = min(
                max(y + 0.025 * y_span, ymin + 0.05 * y_span),
                ymax - 0.08 * y_span,
            )
            va = "bottom"
        ax.text(
            x_label, y_text, cfg["threshold_label"],
            color="0.28", fontsize=8.2, ha="right", va=va,
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, linewidth=0),
        )
    else:
        # Annotate that threshold is off-axis; do not modify ylim
        below = y < ymin
        arrow = "↓" if below else "↑"  # ↓ or ↑
        corner_y = ymin + 0.04 * y_span if below else ymax - 0.04 * y_span
        corner_va = "bottom" if below else "top"
        ax.text(
            xmin + 0.97 * (xmax - xmin), corner_y,
            f"{cfg['threshold_label']} {arrow}",
            color="0.50", fontsize=7.5, ha="right", va=corner_va, style="italic",
        )


def add_line_end_labels(
    ax: plt.Axes, summaries: dict[str, pd.DataFrame], methods: list[str]
) -> None:
    """Place method name at each curve's right end; push labels apart vertically."""
    entries = []
    for method in methods:
        if method not in summaries:
            continue
        df = summaries[method]
        last_y = float(df["mean"].iloc[-1])
        entries.append((last_y, method))

    if not entries:
        return

    entries.sort(key=lambda e: e[0])
    ymin, ymax = ax.get_ylim()
    y_span = ymax - ymin
    min_spacing = 0.065 * y_span  # minimum gap between label centres

    positions = [e[0] for e in entries]
    for _ in range(60):
        changed = False
        for i in range(1, len(positions)):
            gap = positions[i] - positions[i - 1]
            if gap < min_spacing:
                push = (min_spacing - gap) / 2.0
                positions[i - 1] -= push
                positions[i] += push
                changed = True
        if not changed:
            break

    for i, (orig_y, method) in enumerate(entries):
        style = METHOD_STYLES[method]
        # x=1.0 in axes fraction (right spine), y in data coordinates
        ax.annotate(
            style["end_label"],
            xy=(1.0, positions[i]),
            xycoords=("axes fraction", "data"),
            xytext=(6, 0),
            textcoords="offset points",
            color=style["color"],
            fontsize=8.2,
            va="center",
            ha="left",
            clip_on=False,
        )


def plot_summary_lines(
    ax: plt.Axes,
    summaries: dict[str, pd.DataFrame],
    methods: list[str],
) -> None:
    for method in methods:
        if method not in summaries:
            continue
        df = summaries[method]
        style = METHOD_STYLES[method]
        x = df["global_step"].to_numpy(float)
        y = df["mean"].to_numpy(float)
        ax.plot(x, y, color=style["color"], linestyle=style["linestyle"],
                linewidth=style["linewidth"], zorder=3)


def handles_for(methods: list[str]) -> list[Line2D]:
    return [
        Line2D([0], [0],
               color=METHOD_STYLES[m]["color"],
               linestyle=METHOD_STYLES[m]["linestyle"],
               linewidth=METHOD_STYLES[m]["linewidth"],
               label=METHOD_STYLES[m]["label"])
        for m in methods
    ]


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)


def choose_best_penalty() -> tuple[str, pd.DataFrame]:
    rows = []
    source_map = {
        "PPO-penalty-mu05-mu10": PENALTY_SWEEP / "mu05_mu10",
        "PPO-penalty-mu10-mu10": MULTISEED,
        "PPO-penalty-mu15-mu10": PENALTY_SWEEP / "mu15_mu10",
    }
    for method, root in source_map.items():
        parts = []
        target_method = "PPO-penalty" if method == "PPO-penalty-mu10-mu10" else method
        for seed in range(5):
            path = root / f"seed{seed}" / "per_traj_results_test.csv"
            if not path.exists():
                break
            df = pd.read_csv(path)
            if "method" in df.columns and target_method in set(df["method"]):
                df = df[df["method"] == target_method].copy()
            df["train_seed"] = seed
            parts.append(df)
        if len(parts) != 5:
            continue
        raw = pd.concat(parts, ignore_index=True)
        rows.append({
            "method": method,
            "avg_q3d": raw["avg_q3d"].mean(),
            "J_C_R": raw["J_C_R"].mean(),
            "J_C_D": raw["J_C_D"].mean(),
            "feasible": bool(
                raw["J_C_R"].mean() <= EPS_RGB and raw["J_C_D"].mean() <= EPS_DEPTH
            ),
        })
    if not rows:
        raise FileNotFoundError("No complete PPO-penalty test summaries found.")
    table = pd.DataFrame(rows).sort_values(["feasible", "avg_q3d"], ascending=[False, False])
    return str(table.iloc[0]["method"]), table


def write_last20_summary(data: dict[str, pd.DataFrame], methods: list[str]) -> Path:
    rows = []
    for method in methods:
        df = data[method]
        for metric_key, metric_cfg in METRICS.items():
            col = metric_cfg["column"]
            per_seed_last20, per_seed_final = [], []
            for _, sub in df.groupby("train_seed"):
                sub = sub.sort_values("global_step")
                values = pd.to_numeric(sub[col], errors="coerce")
                per_seed_last20.append(float(values.tail(20).mean()))
                per_seed_final.append(float(values.iloc[-1]))
            s = pd.Series(per_seed_last20, dtype=float)
            rows.append({
                "method": METHOD_STYLES[method]["label"],
                "method_key": method,
                "metric": metric_key,
                "last20_mean": s.mean(),
                "last20_std": s.std(ddof=1) if len(s) > 1 else 0.0,
                "last20_stderr": (s.std(ddof=1) / math.sqrt(len(s))) if len(s) > 1 else 0.0,
                "final_value": pd.Series(per_seed_final, dtype=float).mean(),
                "num_seeds": len(s),
            })
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / "table_training_last20_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def make_single_figures(data: dict[str, pd.DataFrame], methods: list[str]) -> list[str]:
    """One publication-quality figure per metric with legend inside the axes."""
    generated = []
    for metric_key, cfg in METRICS.items():
        summaries = summarize_curve({m: data[m] for m in methods if m in data}, metric_key)
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        plot_summary_lines(ax, summaries, methods)
        apply_axis_style(ax, cfg["ylabel"])
        ax.set_ylim(cfg["ylim"])
        add_threshold(ax, metric_key)
        ax.legend(
            handles=handles_for([m for m in methods if m in summaries]),
            loc="best",
            fontsize=7.2,
            ncol=1,
            frameon=True,
            fancybox=False,
            framealpha=0.92,
            edgecolor="0.75",
            handlelength=1.4,
            handletextpad=0.3,
            borderpad=0.3,
            labelspacing=0.15,
        )
        fig.tight_layout()
        save_figure(fig, cfg["stem"])
        generated += [str(FIG_DIR / f"{cfg['stem']}.pdf"),
                      str(FIG_DIR / f"{cfg['stem']}.png")]
    return generated


def make_combined_figure(data: dict[str, pd.DataFrame], methods: list[str]) -> list[str]:
    """2×3 combined figure spanning double columns. figsize=(7, 6), ratio ~6:7 (h:w)."""
    metric_items = list(METRICS.items())
    fig, axes = plt.subplots(2, 3, figsize=(7, 6), sharex=True)
    for ax, (metric_key, cfg) in zip(axes.flat, metric_items):
        summaries = summarize_curve({m: data[m] for m in methods if m in data}, metric_key)
        plot_summary_lines(ax, summaries, methods)
        ax.set_ylim(cfg["ylim"])
        add_threshold(ax, metric_key)
        # axis style with reduced font sizes for compact subplots
        is_bottom = ax in axes[-1, :]
        if is_bottom:
            ax.set_xlabel("Environment steps", fontsize=8.5)
        ax.set_ylabel(cfg["ylabel"], fontsize=8.5)
        ax.tick_params(labelsize=7.5)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(100_000))
        ax.grid(True, color="0.88", linewidth=0.5, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # shared legend at top, one row for all 5 methods
    fig.legend(
        handles=handles_for([m for m in methods if m in data]),
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=len(methods), frameon=False,
        fontsize=7.2, handlelength=1.3, handletextpad=0.3, columnspacing=0.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), h_pad=1.0, w_pad=0.8)
    stem = "Fig4_eiffel_training_combined"
    save_figure(fig, stem)
    return [str(FIG_DIR / f"{stem}.pdf"), str(FIG_DIR / f"{stem}.png")]


def make_penalty_supplement(data: dict[str, pd.DataFrame], penalty_methods: list[str]) -> list[str]:
    """Supplementary 2×3 grid showing all PPO-penalty weight variants."""
    if len(penalty_methods) < 2:
        print("[skip] supplementary penalty figure: fewer than two variants available")
        return []
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2), sharex=True)
    for ax, (metric_key, cfg) in zip(axes.flat, METRICS.items()):
        summaries = summarize_curve(
            {m: data[m] for m in penalty_methods if m in data}, metric_key
        )
        plot_summary_lines(ax, summaries, penalty_methods)
        apply_axis_style(ax, cfg["ylabel"], xlabel=ax in axes[-1, :])
        ax.set_ylim(cfg["ylim"])
        add_threshold(ax, metric_key)
    fig.legend(
        handles=handles_for(penalty_methods),
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=min(3, len(penalty_methods)),
        frameon=False, fontsize=8.0, handlelength=2.0, columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90), h_pad=1.2, w_pad=1.0)
    save_figure(fig, "FigS_penalty_weight_variants")
    return [
        str(FIG_DIR / "FigS_penalty_weight_variants.pdf"),
        str(FIG_DIR / "FigS_penalty_weight_variants.png"),
    ]


def main() -> None:
    set_style()
    all_sources = {**BASE_SOURCES, **PENALTY_SOURCES}
    data = load_method_logs(all_sources)
    best_penalty, penalty_selection = choose_best_penalty()

    # Main figures: all 5 methods
    all_method_order = [
        "CRPO-PPO", "Lagrangian-PPO",
        "PPO-penalty-mu05-mu10", "PPO-penalty-mu10-mu10", "PPO-penalty-mu15-mu10",
    ]
    main_methods = [m for m in all_method_order if m in data]
    # Supplementary: all available PPO-penalty variants
    penalty_methods = [m for m in PENALTY_SOURCES if m in data]

    generated = []
    generated.extend(make_single_figures(data, main_methods))
    generated.extend(make_combined_figure(data, main_methods))
    generated.extend(make_penalty_supplement(data, penalty_methods))
    table_path = write_last20_summary(data, main_methods)

    print("\n========== Fig. 4 training diagnostics ==========")
    print(f"Output dir  : {FIG_DIR.relative_to(ROOT)}")
    print(f"Last-20 CSV : {table_path.relative_to(ROOT)}")
    print(f"Shaded band : {SHADE}, alpha={SHADE_ALPHA}")
    print(f"Best penalty: {best_penalty}")
    print("\nPPO-penalty selection table:")
    print(penalty_selection.to_string(index=False))
    print("\nMain-figure methods (3):")
    for m in main_methods:
        print(f"  {METHOD_STYLES[m]['label']}")
    print("\nGenerated files:")
    for p in generated:
        print(f"  {Path(p).relative_to(ROOT)}")


if __name__ == "__main__":
    main()
