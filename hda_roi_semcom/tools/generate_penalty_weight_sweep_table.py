#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs/eiffel15_surrogate_test/penalty_weight_sweep_q070_d035_muR_sweep_muD1/penalty_weight_sweep_test_summary.csv"
OUT_DIR = ROOT / "paper_results/tables"


def write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if not SUMMARY.exists():
        raise FileNotFoundError(
            f"Missing penalty sweep summary: {SUMMARY}\n"
            "Run ./scripts/run_eiffel_penalty_weight_sweep.sh first."
        )

    df = pd.read_csv(SUMMARY)
    columns = {
        "penalty_weights": "Penalty weights",
        "avg_q3d": "Avg. Q3D",
        "avg_q_rgb": "Avg. Q_rgb",
        "avg_r_depth": "Avg. R_depth",
        "J_C_R": "RGB cost",
        "J_C_D": "Depth cost",
        "avg_kd": "Avg. k_d",
        "avg_beta_d": "Avg. rho_d",
        "episode_return": "Episode return",
        "overall_feasible": "Overall feasible",
    }
    table = df[list(columns)].rename(columns=columns)
    for col in [
        "Avg. Q3D",
        "Avg. Q_rgb",
        "Avg. R_depth",
        "RGB cost",
        "Depth cost",
        "Avg. k_d",
        "Avg. rho_d",
        "Episode return",
    ]:
        table[col] = table[col].map(lambda value: f"{float(value):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "ppo_penalty_weight_sweep.csv", index=False)
    write_markdown_table(table, OUT_DIR / "ppo_penalty_weight_sweep.md")
    (OUT_DIR / "ppo_penalty_weight_sweep.tex").write_text(
        table.to_latex(index=False, escape=False),
        encoding="utf-8",
    )
    (OUT_DIR / "ppo_penalty_weight_sweep_caption.md").write_text(
        "PPO-penalty weight sweep on the Eiffel q_req=0.70, depth_req=0.35 setting. "
        f"Source: `{SUMMARY.relative_to(ROOT)}`. The sweep fixes the depth penalty at "
        "mu_D=1 and varies the RGB penalty mu_R in {0.5, 1.0, 1.5}, showing the "
        "sensitivity of PPO-penalty to manual penalty-weight tuning.\n",
        encoding="utf-8",
    )
    print(f"Wrote table bundle to {OUT_DIR.relative_to(ROOT)}/ppo_penalty_weight_sweep.*")


if __name__ == "__main__":
    main()
