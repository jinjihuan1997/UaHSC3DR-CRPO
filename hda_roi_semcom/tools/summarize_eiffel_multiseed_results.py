#!/usr/bin/env python3
"""Summarize Eiffel multi-seed CRPO/PPO evaluation results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "avg_reconstruction_surrogate_reward",
    "avg_q3d",
    "avg_q_rgb",
    "avg_r_depth",
    "J_C_R",
    "J_C_D",
    "rgb_violation_rate",
    "depth_violation_rate",
    "avg_kd",
    "avg_beta_d",
    "episode_return",
    "psnr",
    "ssim",
    "lpips",
    "chamfer",
    "fscore",
    "completeness",
    "Q_app",
    "Q_geo",
    "Q_gt",
]

METHOD_ORDER = [
    "CRPO-PPO",
    "Lagrangian-PPO",
    "PPO-penalty",
    "fixed-balanced",
    "RGB-priority",
    "depth-priority",
    "random",
]


def _method_order(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else 999


def _clean_seed_values(series: pd.Series) -> list:
    values = series.replace("", pd.NA).dropna().unique()
    return sorted(values)


def _format_seed_values(values: list) -> str:
    formatted = []
    for value in values:
        try:
            number = float(value)
            if number.is_integer():
                formatted.append(str(int(number)))
                continue
        except (TypeError, ValueError):
            pass
        formatted.append(str(value))
    return ",".join(formatted)


def _load_split(root: Path, split: str) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(list(root.glob("seed*")) + [root / "baselines"]):
        if not run_dir.is_dir():
            continue
        seed = ""
        if run_dir.name.startswith("seed"):
            seed_text = run_dir.name.removeprefix("seed")
            try:
                seed = int(seed_text)
            except ValueError:
                continue
        path = run_dir / f"per_traj_results_{split}.csv"
        if not path.exists():
            print(f"[missing] {path}")
            continue
        df = pd.read_csv(path)
        df["train_seed"] = seed
        df["split"] = split
        df["run_dir"] = run_dir.name
        rows.append(df)
    if not rows:
        raise SystemExit(f"no per_traj_results_{split}.csv files found under {root}")
    return pd.concat(rows, ignore_index=True)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [col for col in METRIC_COLUMNS if col in df.columns]
    grouped = df.groupby("method", sort=False)
    rows = []
    for method, sub in grouped:
        train_seeds = _clean_seed_values(sub["train_seed"]) if "train_seed" in sub else []
        channel_seeds = _clean_seed_values(sub["shadow_seed"]) if "shadow_seed" in sub else []
        action_seeds = _clean_seed_values(sub["action_seed"]) if "action_seed" in sub else []
        row = {
            "method": method,
            "n_runs": int(len(sub)),
            "n_training_seeds": len(train_seeds),
            "training_seeds": _format_seed_values(train_seeds),
            "n_channel_seeds": len(channel_seeds),
            "channel_seeds": _format_seed_values(channel_seeds),
            "n_action_seeds": len(action_seeds),
            "action_seeds": _format_seed_values(action_seeds),
        }
        for col in metric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            mean = values.mean()
            std = values.std(ddof=1) if values.notna().sum() > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_order"] = out["method"].map(_method_order).fillna(999)
    return out.sort_values(["_order", "method"]).drop(columns=["_order"])


def _write_bundle(summary: pd.DataFrame, raw: pd.DataFrame, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(prefix.with_name(prefix.name + "_raw.csv"), index=False)
    summary.to_csv(prefix.with_suffix(".csv"), index=False)
    _write_markdown_table(summary, prefix.with_suffix(".md"))
    with prefix.with_suffix(".tex").open("w") as f:
        f.write(summary.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4f}"))


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in df.iterrows():
        values = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                values.append(f"{val:.4f}")
            else:
                values.append(str(val))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs/eiffel15_surrogate_test/multiseed_q070_d035"))
    ap.add_argument("--out", type=Path, default=None,
                    help="Output prefix. Defaults to <root>/multiseed_summary.")
    args = ap.parse_args()

    out_prefix = args.out or (args.root / "multiseed_summary")
    all_rows = []
    for split in ["train", "test"]:
        raw = _load_split(args.root, split)
        summary = _summarize(raw)
        _write_bundle(summary, raw, out_prefix.with_name(f"{out_prefix.name}_{split}"))
        all_rows.append(raw)
        print(f"[wrote] {out_prefix.with_name(f'{out_prefix.name}_{split}').with_suffix('.csv')}")
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(out_prefix.with_name(f"{out_prefix.name}_all_raw.csv"), index=False)


if __name__ == "__main__":
    main()
