#!/usr/bin/env python3
"""Summarize real GSFusion 3D mapping metrics as multi-run mean ± std."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from summarize_mapping_quality_results import (
    _add_quality_labels,
    _compute_basis,
    _read_conditions,
    _order_methods,
)


METRIC_COLUMNS = [
    "psnr",
    "ssim",
    "lpips",
    "chamfer",
    "fscore",
    "completeness",
    "Q_app",
    "Q_geo",
    "Q_gt",
    "Avg. Q_rgb",
    "Avg. R_depth",
    "Avg. k_d",
    "Avg. rho_d",
]


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


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, sub in df.groupby("Method", sort=False):
        train_seeds = _clean_seed_values(sub["training_seed"]) if "training_seed" in sub else []
        channel_seeds = _clean_seed_values(sub["shadow_seed"]) if "shadow_seed" in sub else []
        action_seeds = _clean_seed_values(sub["action_seed"]) if "action_seed" in sub else []
        row = {
            "Method": method,
            "n_runs": int(len(sub)),
            "n_training_seeds": len(train_seeds),
            "training_seeds": _format_seed_values(train_seeds),
            "n_channel_seeds": len(channel_seeds),
            "channel_seeds": _format_seed_values(channel_seeds),
            "n_action_seeds": len(action_seeds),
            "action_seeds": _format_seed_values(action_seeds),
        }
        for col in [c for c in METRIC_COLUMNS if c in sub.columns]:
            values = pd.to_numeric(sub[col], errors="coerce")
            mean = values.mean()
            std = values.std(ddof=1) if values.notna().sum() > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
        rows.append(row)
    return _order_methods(pd.DataFrame(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conditions-dir", type=Path, required=True,
                    help="Directory containing conditions_index.json.")
    ap.add_argument("--metrics", type=Path, required=True,
                    help="CSV with real GSFusion metrics.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output prefix, without extension.")
    args = ap.parse_args()

    cond_path = args.conditions_dir / "conditions_index.json"
    cond_df = _read_conditions(cond_path)
    metric_df = pd.read_csv(args.metrics)
    basis = _compute_basis(metric_df)
    df = cond_df.merge(metric_df, on="condition", how="inner")
    if df.empty:
        raise SystemExit("no overlapping conditions between conditions_index.json and metrics CSV")
    labels = _add_quality_labels(df, basis)
    summary = summarize(labels)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.out.with_name(args.out.name + "_raw.csv"), index=False)
    summary.to_csv(args.out.with_suffix(".csv"), index=False)
    _write_markdown_table(summary, args.out.with_suffix(".md"))
    with args.out.with_suffix(".tex").open("w") as f:
        f.write(summary.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4f}"))
    print(f"[wrote] {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
