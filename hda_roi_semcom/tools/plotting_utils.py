from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


METHOD_ORDER = [
    "Fixed-balanced allocation",
    "RGB-priority allocation",
    "Depth-priority allocation",
    "Random allocation",
    "PPO-penalty",
    "Lagrangian-PPO",
    "CRPO-guided PPO",
]


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "font.weight": "bold",
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.linewidth": 1.2,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.2,
            "lines.markersize": 5,
            "figure.dpi": 160,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.tight_layout()
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=600)
    plt.close(fig)
    return {"pdf": pdf, "png": png}


def moving_average(values, window: int = 7):
    return values.rolling(window=window, min_periods=1).mean()
