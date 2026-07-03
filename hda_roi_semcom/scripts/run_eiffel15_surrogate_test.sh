#!/usr/bin/env bash
set -euo pipefail

# Test the saturation surrogate with 15 Eiffel trajectories.
#
# Split:
#   fit  : traj0..traj9
#   test : traj10..traj14
#
# Default run:
#   bash scripts/run_eiffel15_surrogate_test.sh
#
# Useful overrides:
#   FORCE=1 bash scripts/run_eiffel15_surrogate_test.sh
#   DEVICE=cpu bash scripts/run_eiffel15_surrogate_test.sh
#   GRID_MODE=full bash scripts/run_eiffel15_surrogate_test.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-/home/king/Downloads/Projects/TCOM/datasets/hda_roi_semcom_dataset}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/default.yaml}"
GSFUSION_ROOT="${GSFUSION_ROOT:-/home/king/Downloads/Projects/TCOM/GSFusion}"
MESH="${MESH:-/home/king/Downloads/Projects/TCOM/MAGICIAN/data/Macarons++/macarons++/eiffel/eiffel_rgb.obj}"
OUT_ROOT="${OUT_ROOT:-outputs/eiffel15_surrogate_test}"
SEMCOM_CKPT="${SEMCOM_CKPT:-checkpoints/stage3_final.pt}"
METRICS_GT_MESH="${METRICS_GT_MESH:-$MESH}"

SNRS="${SNRS:-0 2 4 6 8 10 12 14 16 18 20}"
KD_GRID="${KD_GRID:-10 12 14 16 18 20 22}"
MAX_FRAMES="${MAX_FRAMES:-102}"
GRID_MODE="${GRID_MODE:-full}"
REPRESENTATIVE_GRID="${REPRESENTATIVE_GRID:-0:10 0:22 4:14 8:10 8:22 12:18 16:14 20:22}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
GSFUSION_WIDTH="${GSFUSION_WIDTH:-912}"
GSFUSION_HEIGHT="${GSFUSION_HEIGHT:-512}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
CROP_GT_BBOX_MARGIN="${CROP_GT_BBOX_MARGIN:--1}"
RUN_COMPOSITE_SELECTION="${RUN_COMPOSITE_SELECTION:-0}"
GEOMETRY_SAMPLE_POINTS="${GEOMETRY_SAMPLE_POINTS:-0}"
GEOMETRY_SAMPLE_NAME="${GEOMETRY_SAMPLE_NAME:-sampled_surface_100k.ply}"
PRUNE_MESH_AFTER_CACHE="${PRUNE_MESH_AFTER_CACHE:-0}"
FORCE="${FORCE:-0}"

if [[ "$CROP_GT_BBOX_MARGIN" == -* ]]; then
  CROP_TAG="nocrop"
else
  CROP_TAG="crop${CROP_GT_BBOX_MARGIN//./p}"
fi

MANIFEST_REL_DIR="manifests/eiffel15_surrogate_test"
MANIFEST_DIR="$DATA_ROOT/$MANIFEST_REL_DIR"
CONFIG_OUT="$OUT_ROOT/config_eiffel15_surrogate.yaml"
LOOKUP_DIR="$OUT_ROOT/lookups"
SELECTED_LOOKUP_DIR="$OUT_ROOT/lookups_${GRID_MODE}"
DEGRADED_DIR="$OUT_ROOT/gsfusion_degraded_${GRID_MODE}"
METRICS_DIR="$OUT_ROOT/metrics_${GRID_MODE}_${CROP_TAG}"
PLOTS_DIR="$OUT_ROOT/plots"
FIT_METRICS="$OUT_ROOT/gsfusion_real_metrics_eiffel15_${GRID_MODE}_${CROP_TAG}_fit.csv"
TEST_METRICS="$OUT_ROOT/gsfusion_real_metrics_eiffel15_${GRID_MODE}_${CROP_TAG}_test.csv"
LOOKUP_NORM="$OUT_ROOT/lookup_qrgb_normalization_fit.csv"
STABLE_JSON="$OUT_ROOT/stable_scene_surrogate_selection_${GRID_MODE}.json"
STABLE_YAML="$OUT_ROOT/gsfusion_surrogate_weights_stable_scene_${GRID_MODE}.yaml"
JOINT_JSON="$OUT_ROOT/stable_scene_surrogate_selection_${GRID_MODE}_joint.json"
JOINT_YAML="$OUT_ROOT/gsfusion_surrogate_weights_stable_scene_${GRID_MODE}_joint.yaml"
GEOMETRY_JSON="$OUT_ROOT/eiffel15_geometry_saturation_${GRID_MODE}_${CROP_TAG}.json"
GEOMETRY_YAML="$OUT_ROOT/gsfusion_surrogate_weights_eiffel15_geometry_${GRID_MODE}_${CROP_TAG}.yaml"
GEOMETRY_PLOT="$PLOTS_DIR/eiffel15_geometry_saturation_${GRID_MODE}_${CROP_TAG}.png"
COMPOSITE_JSON="$OUT_ROOT/eiffel15_composite_surrogate_${GRID_MODE}_${CROP_TAG}.json"
COMPOSITE_YAML="$OUT_ROOT/gsfusion_surrogate_weights_eiffel15_composite_${GRID_MODE}_${CROP_TAG}.yaml"
COMPOSITE_PLOT="$PLOTS_DIR/eiffel15_composite_surrogate_${GRID_MODE}_${CROP_TAG}.png"

mkdir -p "$OUT_ROOT" "$MANIFEST_DIR" "$LOOKUP_DIR" "$SELECTED_LOOKUP_DIR" "$DEGRADED_DIR" "$METRICS_DIR" "$PLOTS_DIR"

echo "========== 0. Input paths =========="
test -d "$DATA_ROOT" || { echo "ERROR: data root not found: $DATA_ROOT"; exit 1; }
test -f "$CONFIG_TEMPLATE" || { echo "ERROR: config template not found: $CONFIG_TEMPLATE"; exit 1; }
test -f "$SEMCOM_CKPT" || { echo "ERROR: SemCom checkpoint not found: $SEMCOM_CKPT"; exit 1; }
test -d "$GSFUSION_ROOT" || { echo "ERROR: GSFusion root not found: $GSFUSION_ROOT"; exit 1; }
test -f "$MESH" || { echo "ERROR: Eiffel GT mesh not found: $MESH"; exit 1; }
test -f "$METRICS_GT_MESH" || { echo "ERROR: GT mesh for metrics not found: $METRICS_GT_MESH (set METRICS_GT_MESH or create the geometry cache first)"; exit 1; }

echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "SEMCOM_CKPT=$SEMCOM_CKPT"
echo "fit trajectories: traj0..traj9"
echo "test trajectories: traj10..traj14"
echo "lookup grid: SNRS=[$SNRS], KD_GRID=[$KD_GRID]"
echo "test grid mode: $GRID_MODE"
echo "geometry crop margin: $CROP_GT_BBOX_MARGIN"
if [[ "$GRID_MODE" == "representative" ]]; then
  echo "representative SNR:KD pairs: $REPRESENTATIVE_GRID"
fi

echo "========== 1. Build Eiffel manifests =========="
export DATA_ROOT MANIFEST_DIR
python - <<'PY'
import json
import os
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
manifest_dir = Path(os.environ["MANIFEST_DIR"])
traj_root = data_root / "scenes" / "eiffel" / "observations"

splits = {
    "fit": [f"traj{i}" for i in range(10)],
    "test": [f"traj{i}" for i in range(10, 15)],
}

def read_traj(traj):
    path = traj_root / traj / "frame_manifest.jsonl"
    if not path.exists():
        raise SystemExit(f"missing manifest: {path}")
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["scene_id"] = "eiffel"
            row["trajectory_id"] = traj
            base = f"scenes/eiffel/observations/{traj}"
            for key in ("clean_image", "depth", "valid_mask", "camera_pose"):
                value = row.get(key)
                if value and not str(value).startswith("scenes/"):
                    row[key] = f"{base}/{value}"
            rows.append(row)
    if not rows:
        raise SystemExit(f"empty manifest: {path}")
    return rows

manifest_dir.mkdir(parents=True, exist_ok=True)
summary = []
for split, trajs in splits.items():
    rows = []
    for traj in trajs:
        cur = read_traj(traj)
        rows.extend(cur)
        summary.append((split, traj, len(cur)))
    with (manifest_dir / f"{split}_eiffel.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

with (manifest_dir / "train.jsonl").open("w") as dst, (manifest_dir / "fit_eiffel.jsonl").open() as src:
    dst.write(src.read())
with (manifest_dir / "val.jsonl").open("w") as dst, (manifest_dir / "test_eiffel.jsonl").open() as src:
    dst.write(src.read())
with (manifest_dir / "test.jsonl").open("w") as dst, (manifest_dir / "test_eiffel.jsonl").open() as src:
    dst.write(src.read())

with (manifest_dir / "split_plan.csv").open("w") as f:
    f.write("split,scene,trajectory,frames\n")
    for split, traj, frames in summary:
        f.write(f"{split},eiffel,{traj},{frames}\n")

print(f"wrote manifests under {manifest_dir}")
for split, traj, frames in summary:
    print(f"{split},eiffel,{traj},{frames}")
PY

echo "========== 2. Write generated config =========="
export CONFIG_TEMPLATE CONFIG_OUT DATA_ROOT MANIFEST_REL_DIR DEVICE
python - <<'PY'
import os
from pathlib import Path

import yaml

with open(os.environ["CONFIG_TEMPLATE"]) as f:
    cfg = yaml.safe_load(f)
cfg["data"]["root"] = os.environ["DATA_ROOT"]
cfg["data"]["train_manifest"] = f'{os.environ["MANIFEST_REL_DIR"]}/train.jsonl'
cfg["data"]["val_manifest"] = f'{os.environ["MANIFEST_REL_DIR"]}/val.jsonl'
cfg["data"]["test_manifest"] = f'{os.environ["MANIFEST_REL_DIR"]}/test.jsonl'
cfg["data"]["train_scene_id"] = None
cfg["data"]["min_rgb_std"] = 0.0
cfg["train"]["device"] = os.environ["DEVICE"]
out = Path(os.environ["CONFIG_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"wrote {out}")
PY

build_lookup() {
  local split="$1"
  local manifest_rel="$MANIFEST_REL_DIR/${split}_eiffel.jsonl"
  local lookup="$LOOKUP_DIR/${split}_eiffel.csv"
  echo "========== 3. Lookup: $split =========="
  if [[ "$FORCE" == "1" || ! -s "$lookup" ]]; then
    python scripts/build_offline_lookup_table.py \
      --config "$CONFIG_OUT" \
      --ckpt "$SEMCOM_CKPT" \
      --split test \
      --manifest "$manifest_rel" \
      --out "$lookup" \
      --snrs $SNRS \
      --digital-subcarriers $KD_GRID \
      --seed "$SEED"
  else
    echo "reuse lookup: $lookup"
  fi

  LOOKUP_CHECK="$lookup" EXPECTED_FRAMES="$MAX_FRAMES" python - <<'PY'
import csv
import os
from collections import Counter, defaultdict

lookup = os.environ["LOOKUP_CHECK"]
expected = int(os.environ["EXPECTED_FRAMES"])
groups = Counter()
frames = defaultdict(set)
with open(lookup, newline="") as f:
    reader = csv.DictReader(f)
    required = {"trajectory_id", "frame_id", "snr_db", "k_d", "sample_idx"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise SystemExit(f"ERROR: {lookup} missing columns: {sorted(missing)}")
    for row in reader:
        traj = row["trajectory_id"]
        frame = int(float(row["frame_id"]))
        key = (traj, float(row["snr_db"]), int(float(row["k_d"])))
        groups[key] += 1
        frames[traj].add(frame)

bad_groups = [(key, count) for key, count in groups.items() if count != expected]
bad_trajs = {
    traj: sorted(set(range(expected)) - cur_frames)
    for traj, cur_frames in frames.items()
    if len(cur_frames) != expected
}
if bad_groups or bad_trajs:
    print(f"ERROR: invalid lookup frame coverage: {lookup}")
    print(f"  expected_frames_per_condition={expected}")
    for key, count in bad_groups[:20]:
        print(f"  bad_group {key}: rows={count}")
    for traj, missing_frames in list(bad_trajs.items())[:20]:
        print(f"  missing_frames {traj}: count={len(missing_frames)} first={missing_frames[:20]}")
    raise SystemExit(1)
print(f"lookup validation ok: {lookup} groups={len(groups)} frames_per_group={expected}")
PY
}

if [[ "$GEOMETRY_SAMPLE_POINTS" -gt 0 && "$METRICS_GT_MESH" != "$MESH" && ! -f "$METRICS_GT_MESH" ]]; then
  echo "========== 0b. Build GT geometry cache =========="
  mkdir -p "$(dirname "$METRICS_GT_MESH")"
  python scripts/cache_gsfusion_geometry_samples.py \
    --mesh "$MESH" \
    --out "$METRICS_GT_MESH" \
    --sample-points "$GEOMETRY_SAMPLE_POINTS" \
    --surface-sample-gt
fi

build_lookup fit
build_lookup test

echo "========== 4. Select condition grid =========="
export LOOKUP_DIR SELECTED_LOOKUP_DIR GRID_MODE REPRESENTATIVE_GRID
python - <<'PY'
import csv
import os
from pathlib import Path

lookup_dir = Path(os.environ["LOOKUP_DIR"])
selected_dir = Path(os.environ["SELECTED_LOOKUP_DIR"])
grid_mode = os.environ["GRID_MODE"]
selected_dir.mkdir(parents=True, exist_ok=True)

if grid_mode == "full":
    selected_pairs = None
elif grid_mode == "representative":
    selected_pairs = set()
    for item in os.environ["REPRESENTATIVE_GRID"].split():
        snr, kd = item.split(":", 1)
        selected_pairs.add((float(snr), int(kd)))
else:
    raise SystemExit(f"unsupported GRID_MODE={grid_mode}; use representative or full")

for split in ("fit", "test"):
    src = lookup_dir / f"{split}_eiffel.csv"
    dst = selected_dir / f"{split}_eiffel.csv"
    with src.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = []
        for row in reader:
            pair = (float(row["snr_db"]), int(float(row["k_d"])))
            if selected_pairs is None or pair in selected_pairs:
                rows.append(row)
    if not rows:
        raise SystemExit(f"no rows selected for {split} from {src}")
    with dst.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    groups = {
        (row["trajectory_id"], float(row["snr_db"]), int(float(row["k_d"])))
        for row in rows
    }
    print(f"wrote {dst} rows={len(rows)} condition_groups={len(groups)}")
PY

echo "========== 5. Build Q_rgb normalization lookup =========="
if [[ "$FORCE" == "1" || ! -s "$LOOKUP_NORM" ]]; then
  cp "$LOOKUP_DIR/fit_eiffel.csv" "$LOOKUP_NORM"
  echo "wrote $LOOKUP_NORM"
else
  echo "reuse normalization lookup: $LOOKUP_NORM"
fi

run_split() {
  local split="$1"
  local manifest_rel="$MANIFEST_REL_DIR/${split}_eiffel.jsonl"
  local lookup="$SELECTED_LOOKUP_DIR/${split}_eiffel.csv"
  local degraded="$DEGRADED_DIR/${split}_eiffel"
  local metrics="$METRICS_DIR/${split}_eiffel.csv"

  echo "========== 6. Export degraded RGB-D: $split =========="
  if [[ "$FORCE" == "1" || ! -s "$degraded/conditions_index.json" ]]; then
    python scripts/export_gsfusion_degraded_sequences.py \
      --lookup "$lookup" \
      --q-normalization-lookup "$LOOKUP_NORM" \
      --config "$CONFIG_OUT" \
      --semcom-ckpt "$SEMCOM_CKPT" \
      --split test \
      --manifest "$manifest_rel" \
      --out "$degraded" \
      --gsfusion-root "$GSFUSION_ROOT" \
      --num-conditions 0 \
      --max-frames "$MAX_FRAMES" \
      --seed "$SEED" \
      --device "$DEVICE"
  else
    echo "reuse degraded sequences: $degraded"
  fi

  echo "========== 6b. Validate GSFusion poses: $split =========="
  DEGRADED_CHECK_DIR="$degraded" python - <<'PY'
import os
from pathlib import Path

import numpy as np

root = Path(os.environ["DEGRADED_CHECK_DIR"])
bad = []
total = 0
for traj_path in root.rglob("sequence/traj.txt"):
    total += 1
    mats = np.loadtxt(traj_path).reshape(-1, 4, 4)
    if len(mats) < 2:
        bad.append((traj_path, "too_few_poses"))
        continue
    centers = mats[:, :3, 3]
    step_max = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).max())
    if step_max <= 1e-6:
        bad.append((traj_path, f"static_camera step_max={step_max:g}"))
        continue
    if not np.allclose(mats[0], np.eye(4), atol=1e-5):
        bad.append((traj_path, "first_pose_not_identity"))
        continue
    dets = np.linalg.det(mats[:, :3, :3])
    if float(dets.min()) < 0.9 or float(dets.max()) > 1.1:
        bad.append((traj_path, f"bad_rotation_det min={dets.min():g} max={dets.max():g}"))

if bad:
    print(f"ERROR: invalid GSFusion poses under {root}: bad={len(bad)} total={total}")
    for path, reason in bad[:20]:
        print(f"  {reason}: {path}")
    raise SystemExit(1)
print(f"pose validation ok: {total} trajectories under {root}")
PY

  echo "========== 7. Run GSFusion: $split =========="
  prune_mesh_args=()
  if [[ "$PRUNE_MESH_AFTER_CACHE" == "1" ]]; then
    prune_mesh_args+=(--prune-mesh-after-cache)
  fi
  python scripts/run_gsfusion_conditions.py \
    --gsfusion-root "$GSFUSION_ROOT" \
    --conditions-dir "$degraded" \
    --skip-existing \
    --continue-on-error \
    --gsfusion-width "$GSFUSION_WIDTH" \
    --gsfusion-height "$GSFUSION_HEIGHT" \
    --min-free-gb "$MIN_FREE_GB" \
    --cache-geometry-sample-points "$GEOMETRY_SAMPLE_POINTS" \
    --geometry-sample-name "$GEOMETRY_SAMPLE_NAME" \
    "${prune_mesh_args[@]}"

  echo "========== 8. Build real metrics: $split =========="
  if [[ "$FORCE" == "1" || ! -s "$metrics" ]]; then
    python scripts/build_gsfusion_real_metrics.py \
      --conditions-dir "$degraded" \
      --out "$metrics" \
      --gt-mesh "$METRICS_GT_MESH" \
      --pred-sampled-name "$GEOMETRY_SAMPLE_NAME" \
      --include-missing \
      --crop-gt-bbox-margin "$CROP_GT_BBOX_MARGIN"
  else
    echo "reuse metrics: $metrics"
  fi
}

run_split fit
run_split test

echo "========== 9. Merge metrics =========="
export METRICS_DIR FIT_METRICS TEST_METRICS
python - <<'PY'
import csv
import os
from pathlib import Path

metrics_dir = Path(os.environ["METRICS_DIR"])
targets = {
    "fit": Path(os.environ["FIT_METRICS"]),
    "test": Path(os.environ["TEST_METRICS"]),
}
for split, out_path in targets.items():
    src = metrics_dir / f"{split}_eiffel.csv"
    with src.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = ["source_split", "source_scene"] + list(reader.fieldnames or [])
        rows = [
            {"source_split": split, "source_scene": "eiffel", **row}
            for row in reader
        ]
    if not rows:
        raise SystemExit(f"empty metrics file: {src}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} rows={len(rows)}")
PY

echo "========== 10. Fit composite (PSNR + geometry) saturation surrogate =========="
python scripts/fit_gsfusion_surrogate.py "$FIT_METRICS" \
  --model saturation \
  --out "$COMPOSITE_YAML" \
  --json-out "$COMPOSITE_JSON" \
  --plot-out "$COMPOSITE_PLOT" \
  --app-weights 0.6 0.2 0.2 \
  --geo-weights 0.35 0.35 0.30 \
  --lambda-rgb-grid 1 2 3 4 6 8 \
  --lambda-depth-grid 1 2 3 4 6 8 10 12

if [[ "$RUN_COMPOSITE_SELECTION" == "1" ]]; then
  echo "========== 11. Optional composite render+geometry surrogate =========="
  python scripts/select_stable_scene_surrogate.py \
    --fit-metrics "$FIT_METRICS" \
    --test-metrics "$TEST_METRICS" \
    --out-json "$STABLE_JSON" \
    --out-yaml "$STABLE_YAML" \
    --bootstrap 200 \
    --seed "$SEED"

  python scripts/select_stable_scene_surrogate.py \
    --fit-metrics "$FIT_METRICS" \
    --test-metrics "$TEST_METRICS" \
    --out-json "$JOINT_JSON" \
    --out-yaml "$JOINT_YAML" \
    --bootstrap 200 \
    --seed "$SEED" \
    --allow-joint
fi

echo "========== Done =========="
echo "Fit metrics      : $FIT_METRICS"
echo "Test metrics     : $TEST_METRICS"
echo "Composite JSON   : $COMPOSITE_JSON"
echo "Composite YAML   : $COMPOSITE_YAML"
echo "Composite plot   : $COMPOSITE_PLOT"
if [[ "$RUN_COMPOSITE_SELECTION" == "1" ]]; then
  echo "Stable JSON      : $STABLE_JSON"
  echo "Stable YAML      : $STABLE_YAML"
  echo "Joint JSON       : $JOINT_JSON"
  echo "Joint YAML       : $JOINT_YAML"
fi
