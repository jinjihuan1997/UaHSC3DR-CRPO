# Open-Source Manifest

This manifest records what was kept in the lightweight GitHub-ready bundle and what was excluded.

## Included

### `hda_roi_semcom/`

Included:

- `src/`: channel model, dataset loader, semantic codec, CRPO/PPO resource-allocation environment, and utility modules.
- `scripts/`: experiment orchestration, surrogate fitting, GSFusion export, evaluation, and sanity-check scripts.
- `configs/`, `docs/`, `README.md`, `requirements.txt`, and root run scripts.

Excluded:

- `outputs/`, `paper_results/`, `Figures/`, `checkpoints/`, paper LaTeX/BibTeX sources, generated PDFs, build logs, temporary JSON/cache files, Python caches, and paper figure/table generation utilities.

### `GSFusion/`

Included:

- C++/CUDA source, headers, CMake files, configs, parameters, Dockerfile, README, license files, and lightweight third-party source dependencies.

Excluded:

- `.git/`, `build/`, `output/`, logs, large binary dependencies under `third_party/libtorch/` and `third_party/open3d/`.

### `MAGICIAN/`

Included:

- MAGICIAN/Macarons source, configuration files, trajectory/export scripts, tests, demos, environment file, README.
- RaDe-GS source components needed as lightweight code reference.

Excluded:

- `data/`, `weights/`, `results/`, `paper/`, large assets, logs, Python caches, and RaDe-GS heavy viewer/assets files.

## GitHub Upload Policy

Before pushing, check:

```bash
find . -type f -size +50M
git status --short
```

No file larger than 50 MB should be committed. Datasets, weights, generated outputs, and binary dependencies should remain external.
