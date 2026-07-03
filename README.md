# UaHSC3DR-CRPO

Lightweight research-code release for the paper project on UAV-assisted hybrid semantic communication for remote RGB-D 3D reconstruction with CRPO-guided PPO resource allocation.

This repository is organized as a lightweight bundle of three local projects:

- `hda_roi_semcom/`: reconstruction-aware RGB-D transmission, surrogate fitting, CMDP resource allocation, CRPO/PPO training, evaluation, and paper-result generation scripts.
- `GSFusion/`: GSFusion-style RGB-D reconstruction backend source used for final reconstruction-quality evaluation.
- `MAGICIAN/`: active mapping / trajectory-generation utilities and export scripts used to prepare RGB-D observation trajectories.

The release intentionally excludes large generated artifacts, datasets, checkpoints, pretrained weights, build products, and reconstruction outputs. See `DATA_AND_WEIGHTS.md` and `OPEN_SOURCE_MANIFEST.md`.

## Recommended Entry Points

Main resource-allocation project:

```bash
cd hda_roi_semcom
pip install -r requirements.txt
```

Representative scripts:

- `scripts/run_eiffel15_surrogate_test.sh`: builds surrogate-calibration data flow for the Eiffel scene.
- `scripts/run_eiffel_multiseed_training.sh`: multi-seed CRPO/PPO training and evaluation.
- `scripts/run_eiffel_penalty_multiseed_sweep.sh`: PPO-penalty weight sensitivity.
- `scripts/run_oracle_baseline_sweep.sh`: per-slot oracle and architecture ablation.
- `scripts/run_eiffel_mapping_quality_comparison.sh`: exports policy-selected sequences for GSFusion evaluation.

GSFusion backend:

```bash
cd GSFusion
```

The source tree is included, but heavy binary dependencies such as libtorch and Open3D are not vendored in this lightweight release.

MAGICIAN trajectory/data export:

```bash
cd MAGICIAN
conda env create -f environment.yml
```

The large Macarons++ data directory and pretrained weights are excluded.

## What Is Not Included

The following are excluded by design:

- RGB-D datasets and Macarons++ scene data.
- GSFusion reconstruction outputs and mesh caches.
- PPO checkpoints and neural network weights.
- Large third-party binary packages such as libtorch and Open3D.
- Paper build outputs, generated figures, logs, and temporary caches.

This keeps the GitHub repository suitable for code review and method reproduction while requiring users to download datasets and regenerate experiment outputs locally.
