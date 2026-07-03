from __future__ import annotations

from pathlib import Path


RESULT_EXTENSIONS = {
    ".csv",
    ".json",
    ".jsonl",
    ".npz",
    ".npy",
    ".pkl",
    ".pt",
    ".pth",
    ".txt",
    ".log",
    ".yaml",
    ".yml",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
}

LIKELY_ROOTS = [
    "outputs",
    "runs",
    "results",
    "eval",
    "paper_results",
    "logs",
    "checkpoints",
    "tensorboard",
    "wandb",
]


def discover_result_files(project_root: Path) -> list[Path]:
    discovered: list[Path] = []
    for root_name in LIKELY_ROOTS:
        root = project_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("events.out.tfevents") or path.suffix.lower() in RESULT_EXTENSIONS:
                discovered.append(path)
    return sorted(set(discovered))


def write_discovery_summary(files: list[Path], project_root: Path, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Discovered Result Files", ""]
    lines.append(f"Total files discovered: {len(files)}")
    lines.append("")
    for path in files:
        try:
            rel = path.relative_to(project_root)
        except ValueError:
            rel = path
        lines.append(f"- `{rel}`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
