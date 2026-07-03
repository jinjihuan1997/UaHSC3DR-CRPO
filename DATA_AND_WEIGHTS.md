# Data, Weights, and Generated Artifacts

This lightweight release does not include large local artifacts. The original working tree contained more than 100 GB of scene data and several GB of generated outputs. These files should be regenerated or downloaded separately.

## Excluded Data

- `MAGICIAN/data/`: Macarons++ scene data, including Eiffel/Fushimi scene assets and high-resolution trajectory memories.
- `hda_semcom/outputs/`: degraded RGB-D sequences, GSFusion run outputs, cached geometry samples, logs, and intermediate CSV files.
- `hda_semcom/paper_results/` and `hda_semcom/Figures/`: generated paper figures and tables.

## Excluded Weights and Checkpoints

- `MAGICIAN/weights/`: pretrained MAGICIAN/Macarons/SCONE/ResNet weights.
- `hda_semcom/checkpoints/`: CRPO/PPO/PPO-penalty trained policy checkpoints.
- GSFusion generated reconstruction states and mesh outputs.

## Excluded Third-Party Binaries

- `GSFusion/third_party/libtorch/`
- `GSFusion/third_party/open3d/`
- `GSFusion/build/`

Install these dependencies following the upstream GSFusion instructions or your local CUDA/PyTorch/Open3D environment requirements.

## Expected Local Layout for Full Experiments

For a full local reproduction, place or generate data outside the Git repository, for example:

```text
datasets/
MAGICIAN/data/
MAGICIAN/weights/
hda_semcom/outputs/
hda_semcom/checkpoints/
```

Keep these paths ignored by Git.

