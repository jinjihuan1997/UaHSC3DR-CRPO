from __future__ import annotations

from pathlib import Path

import pandas as pd


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in df.iterrows():
        values = []
        for value in row.tolist():
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def save_table_bundle(df: pd.DataFrame, out_dir: Path, name: str) -> dict[str, Path]:
    """Save one table as CSV, Markdown, and LaTeX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "csv": out_dir / f"{name}.csv",
        "md": out_dir / f"{name}.md",
        "tex": out_dir / f"{name}.tex",
    }
    df.to_csv(paths["csv"], index=False)
    paths["md"].write_text(_markdown_table(df) + "\n", encoding="utf-8")
    paths["tex"].write_text(
        df.to_latex(index=False, escape=False, float_format="%.4f"),
        encoding="utf-8",
    )
    return paths


def write_caption(path: Path, title: str, source_files: list[Path], conclusion: str) -> None:
    sources = "\n".join(f"- `{p}`" for p in source_files) if source_files else "- None"
    path.write_text(
        f"# {title}\n\n"
        f"**What it shows.** {title}.\n\n"
        f"**Source files.**\n{sources}\n\n"
        f"**Paper conclusion.** {conclusion}\n",
        encoding="utf-8",
    )
