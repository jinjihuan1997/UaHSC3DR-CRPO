#!/usr/bin/env bash
set -euo pipefail

# End-to-end fushimi geometry saturation surrogate test.
#
# Default split:
#   fit  : traj0..traj9
#   test : traj10
#
# Default run:
#   bash scripts/run_fushimi_surrogate_test.sh
#
# Smoke test:
#   OUT_ROOT=outputs/fushimi_surrogate_smoke NUM_CONDITIONS=10 bash scripts/run_fushimi_surrogate_test.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-/home/king/Downloads/Projects/TCOM/datasets/hda_semcom_dataset}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/default.yaml}"
GSFUSION_ROOT="${GSFUSION_ROOT:-/home/king/Downloads/Projects/TCOM/GSFusion}"
SEMCOM_CKPT="${SEMCOM_CKPT:-checkpoints/stage3_final.pt}"
MESH="${MESH:-/home/king/Downloads/Projects/TCOM/MAGICIAN/data/Macarons++/macarons++/fushimi/fushimi_rgb.obj}"
OUT_ROOT="${OUT_ROOT:-outputs/fushimi_surrogate_test}"

FIT_TRAJ_IDS="${FIT_TRAJ_IDS:-traj0,traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8,traj9}"
TEST_TRAJ_IDS="${TEST_TRAJ_IDS:-traj10}"
SNRS="${SNRS:-0 2 4 6 8 10 12 14 16 18 20}"
KD_GRID="${KD_GRID:-10 12 14 16 18 20 22}"
MAX_FRAMES="${MAX_FRAMES:-102}"
NUM_CONDITIONS="${NUM_CONDITIONS:-0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
GSFUSION_WIDTH="${GSFUSION_WIDTH:-912}"
GSFUSION_HEIGHT="${GSFUSION_HEIGHT:-512}"
MIN_FREE_GB="${MIN_FREE_GB:-100}"
CROP_GT_BBOX_MARGIN="${CROP_GT_BBOX_MARGIN:--1}"
GEOMETRY_SAMPLE_POINTS="${GEOMETRY_SAMPLE_POINTS:-100000}"
GEOMETRY_SAMPLE_NAME="${GEOMETRY_SAMPLE_NAME:-sampled_surface_100k.ply}"
PRUNE_MESH_AFTER_CACHE="${PRUNE_MESH_AFTER_CACHE:-1}"
FORCE="${FORCE:-0}"

MANIFEST_REL_DIR="${MANIFEST_REL_DIR:-manifests/fushimi_surrogate_test}"
MANIFEST_DIR="$DATA_ROOT/$MANIFEST_REL_DIR"
CONFIG_OUT="$OUT_ROOT/config_fushimi_surrogate.yaml"
GEOMETRY_CACHE="$OUT_ROOT/geometry_cache/fushimi_gt_surface_100k.ply"
METRICS_GT_MESH="${METRICS_GT_MESH:-$GEOMETRY_CACHE}"
LOOKUP_DIR="$OUT_ROOT/lookups"
DEGRADED_DIR="$OUT_ROOT/gsfusion_degraded"
METRICS_DIR="$OUT_ROOT/metrics"
PLOTS_DIR="$OUT_ROOT/plots"
FIT_LOOKUP="$LOOKUP_DIR/fit_fushimi.csv"
TEST_LOOKUP="$LOOKUP_DIR/test_fushimi.csv"
FIT_METRICS="$OUT_ROOT/gsfusion_real_metrics_fushimi_fit.csv"
TEST_METRICS="$OUT_ROOT/gsfusion_real_metrics_fushimi_test.csv"
GEOMETRY_JSON="$OUT_ROOT/fushimi_geometry_saturation.json"
GEOMETRY_YAML="$OUT_ROOT/gsfusion_surrogate_weights_fushimi_geometry.yaml"
GEOMETRY_PLOT="$PLOTS_DIR/fushimi_geometry_saturation.png"

mkdir -p "$OUT_ROOT" "$MANIFEST_DIR" "$LOOKUP_DIR" "$DEGRADED_DIR" "$METRICS_DIR" "$PLOTS_DIR" "$OUT_ROOT/geometry_cache"

echo "========== 0. Input paths =========="
test -d "$DATA_ROOT" || { echo "ERROR: data root not found: $DATA_ROOT"; exit 1; }
test -f "$CONFIG_TEMPLATE" || { echo "ERROR: config template not found: $CONFIG_TEMPLATE"; exit 1; }
test -f "$SEMCOM_CKPT" || { echo "ERROR: SemCom checkpoint not found: $SEMCOM_CKPT"; exit 1; }
test -d "$GSFUSION_ROOT" || { echo "ERROR: GSFusion root not found: $GSFUSION_ROOT"; exit 1; }
test -f "$MESH" || { echo "ERROR: fushimi GT mesh not found: $MESH"; exit 1; }

echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "FIT_TRAJ_IDS=$FIT_TRAJ_IDS"
echo "TEST_TRAJ_IDS=$TEST_TRAJ_IDS"
echo "SNRS=[$SNRS]"
echo "KD_GRID=[$KD_GRID]"
echo "NUM_CONDITIONS=$NUM_CONDITIONS"
echo "GEOMETRY_SAMPLE_POINTS=$GEOMETRY_SAMPLE_POINTS"
echo "PRUNE_MESH_AFTER_CACHE=$PRUNE_MESH_AFTER_CACHE"

echo "========== 1. Build fushimi manifests =========="
export DATA_ROOT MANIFEST_DIR FIT_TRAJ_IDS TEST_TRAJ_IDS MAX_FRAMES
python - <<'PY'
import json
import os
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
manifest_dir = Path(os.environ["MANIFEST_DIR"])
scene = "fushimi"
traj_root = data_root / "scenes" / scene / "observations"
max_frames = int(os.environ["MAX_FRAMES"])

splits = {
    "fit": [x for x in os.environ["FIT_TRAJ_IDS"].split(",") if x],
    "test": [x for x in os.environ["TEST_TRAJ_IDS"].split(",") if x],
}

def records_for_traj(traj):
    tdir = traj_root / traj
    if not tdir.is_dir():
        raise SystemExit(f"missing trajectory directory: {tdir}")
    manifest = tdir / "frame_manifest.jsonl"
    rows = []

    if manifest.exists():
        with manifest.open() as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["scene_id"] = scene
                row["trajectory_id"] = traj
                base = f"scenes/{scene}/observations/{traj}"
                for key in ["clean_image", "depth", "valid_mask", "camera_pose"]:
                    value = row.get(key)
                    if value and not str(value).startswith("scenes/"):
                        row[key] = f"{base}/{value}"
                rows.append(row)
    else:
        for image in sorted((tdir / "clean_images").glob("*.png")):
            stem = image.stem
            rows.append({
                "scene_id": scene,
                "trajectory_id": traj,
                "frame_id": int(stem) if stem.isdigit() else stem,
                "clean_image": f"scenes/{scene}/observations/{traj}/clean_images/{image.name}",
                "depth": f"scenes/{scene}/observations/{traj}/depth/{image.name}",
                "valid_mask": f"scenes/{scene}/observations/{traj}/valid_masks/{image.name}",
                "camera_pose": f"scenes/{scene}/observations/{traj}/poses/{stem}.json",
            })

    if len(rows) != max_frames:
        raise SystemExit(f"{traj} has {len(rows)} frames, expected {max_frames}")
    return rows

manifest_dir.mkdir(parents=True, exist_ok=True)
summary = []
for split, trajs in splits.items():
    rows = []
    for traj in trajs:
        cur = records_for_traj(traj)
        rows.extend(cur)
        summary.append((split, scene, traj, len(cur)))
    out = manifest_dir / f"{split}_fushimi.jsonl"
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {out} rows={len(rows)}")

aliases = {
    "train.jsonl": "fit_fushimi.jsonl",
    "val.jsonl": "test_fushimi.jsonl",
    "test.jsonl": "test_fushimi.jsonl",
}
for alias, src in aliases.items():
    (manifest_dir / alias).write_text((manifest_dir / src).read_text())

with (manifest_dir / "split_plan.csv").open("w") as f:
    f.write("split,scene,trajectory,frames\n")
    for split, scene_id, traj, frames in summary:
        f.write(f"{split},{scene_id},{traj},{frames}\n")
        print(f"{split},{scene_id},{traj},{frames}")
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
cfg["data"]["train_manifest"] = f'{os.environ["MANIFEST_REL_DIR"]}/fit_fushimi.jsonl'
cfg["data"]["val_manifest"] = f'{os.environ["MANIFEST_REL_DIR"]}/test_fushimi.jsonl'
cfg["data"]["test_manifest"] = f'{os.environ["MANIFEST_REL_DIR"]}/test_fushimi.jsonl'
cfg["data"]["train_scene_id"] = None
cfg["data"]["min_rgb_std"] = 0.0
if "train" in cfg:
    cfg["train"]["device"] = os.environ["DEVICE"]

out = Path(os.environ["CONFIG_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"wrote {out}")
PY

echo "========== 3. Cache GT geometry =========="
if [[ "$FORCE" == "1" || ! -s "$GEOMETRY_CACHE" ]]; then
  python scripts/cache_gsfusion_geometry_samples.py \
    --mesh "$MESH" \
    --out "$GEOMETRY_CACHE" \
    --sample-points "$GEOMETRY_SAMPLE_POINTS" \
    --surface-sample-gt
else
  echo "reuse GT geometry cache: $GEOMETRY_CACHE"
fi
test -f "$METRICS_GT_MESH" || { echo "ERROR: metrics GT mesh missing: $METRICS_GT_MESH"; exit 1; }

check_lookup_groups() {
  local lookup="$1"
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
        key = (row["trajectory_id"], row["snr_db"], row["k_d"])
        groups[key] += 1
        frames[key].add(int(float(row["frame_id"])))
bad = [(key, count, len(frames[key])) for key, count in groups.items() if count != expected or len(frames[key]) != expected]
if bad:
    print(f"ERROR: incomplete lookup groups in {lookup}: {len(bad)}")
    for key, count, unique_frames in bad[:20]:
        print(f"  {key}: rows={count} unique_frames={unique_frames}")
    raise SystemExit(1)
print(f"lookup validation ok: groups={len(groups)} frames_per_group={expected} file={lookup}")
PY
}

expected_conditions_from_lookup() {
  local lookup="$1"
  if [[ "$NUM_CONDITIONS" != "0" ]]; then
    echo "$NUM_CONDITIONS"
    return
  fi
  LOOKUP_CHECK="$lookup" python - <<'PY'
import csv
import os

groups = set()
with open(os.environ["LOOKUP_CHECK"], newline="") as f:
    for row in csv.DictReader(f):
        groups.add((row["trajectory_id"], row["snr_db"], row["k_d"]))
print(len(groups))
PY
}

current_conditions_count() {
  local degraded="$1"
  CONDITIONS_JSON="$degraded/conditions_index.json" python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["CONDITIONS_JSON"])
if not path.exists():
    print(0)
else:
    print(len(json.loads(path.read_text())))
PY
}

metrics_rows_count() {
  local metrics="$1"
  METRICS_CSV="$metrics" python - <<'PY'
import csv
import os
from pathlib import Path

path = Path(os.environ["METRICS_CSV"])
if not path.exists():
    print(0)
else:
    with path.open(newline="") as f:
        print(sum(1 for _ in csv.DictReader(f)))
PY
}

build_lookup() {
  local split="$1"
  local manifest_rel="$2"
  local traj_ids="$3"
  local lookup="$4"
  echo "========== 4. Build lookup: $split =========="
  if [[ "$FORCE" == "1" || ! -s "$lookup" ]]; then
    python scripts/build_offline_lookup_table.py \
      --config "$CONFIG_OUT" \
      --ckpt "$SEMCOM_CKPT" \
      --split test \
      --manifest "$manifest_rel" \
      --scene-id fushimi \
      --trajectory-ids "$traj_ids" \
      --snrs $SNRS \
      --digital-subcarriers $KD_GRID \
      --out "$lookup" \
      --seed "$SEED"
  else
    echo "reuse lookup: $lookup"
  fi
  check_lookup_groups "$lookup"
}

validate_gsfusion_poses() {
  local degraded="$1"
  VALIDATE_ROOT="$degraded" python - <<'PY'
import os
from pathlib import Path
import numpy as np

root = Path(os.environ["VALIDATE_ROOT"])
bad = []
total = 0
for traj_path in sorted(root.glob("*/sequence/traj.txt")):
    total += 1
    try:
        raw = np.loadtxt(traj_path)
        mats = raw.reshape((-1, 4, 4))
    except Exception as exc:
        bad.append((traj_path, f"unreadable: {exc}"))
        continue
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
}

run_split() {
  local split="$1"
  local manifest_rel="$2"
  local traj_ids="$3"
  local lookup="$4"
  local degraded="$DEGRADED_DIR/${split}_fushimi"
  local metrics="$5"
  local expected_conditions
  local current_conditions
  local metrics_rows
  local num_conditions_file="$degraded/num_conditions.txt"
  local old_num_conditions=""
  if [[ -s "$num_conditions_file" ]]; then
    old_num_conditions="$(<"$num_conditions_file")"
  fi

  build_lookup "$split" "$manifest_rel" "$traj_ids" "$lookup"
  expected_conditions="$(expected_conditions_from_lookup "$lookup")"
  current_conditions="$(current_conditions_count "$degraded")"

  echo "========== 5. Export degraded GSFusion sequences: $split =========="
  if [[ "$FORCE" == "1" || ! -s "$degraded/conditions_index.json" || "$old_num_conditions" != "$NUM_CONDITIONS" || "$current_conditions" != "$expected_conditions" ]]; then
    python scripts/export_gsfusion_degraded_sequences.py \
      --lookup "$lookup" \
      --q-normalization-lookup "$FIT_LOOKUP" \
      --config "$CONFIG_OUT" \
      --semcom-ckpt "$SEMCOM_CKPT" \
      --split test \
      --manifest "$manifest_rel" \
      --scene-id fushimi \
      --trajectory-ids "$traj_ids" \
      --out "$degraded" \
      --gsfusion-root "$GSFUSION_ROOT" \
      --num-conditions "$NUM_CONDITIONS" \
      --max-frames "$MAX_FRAMES" \
      --device "$DEVICE"
    echo "$NUM_CONDITIONS" > "$num_conditions_file"
  else
    echo "reuse degraded sequences: $degraded"
  fi
  validate_gsfusion_poses "$degraded"

  echo "========== 6. Run GSFusion: $split =========="
  prune_args=()
  if [[ "$PRUNE_MESH_AFTER_CACHE" == "1" ]]; then
    prune_args+=(--prune-mesh-after-cache)
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
    "${prune_args[@]}"

  echo "========== 7. Build real metrics: $split =========="
  metrics_rows="$(metrics_rows_count "$metrics")"
  if [[ "$FORCE" == "1" || ! -s "$metrics" || "$metrics_rows" != "$expected_conditions" ]]; then
    if [[ -s "$metrics" && "$metrics_rows" != "$expected_conditions" ]]; then
      echo "rebuild metrics: $metrics rows=$metrics_rows expected=$expected_conditions"
    fi
    python scripts/build_gsfusion_real_metrics.py \
      --conditions-dir "$degraded" \
      --out "$metrics" \
      --gt-mesh "$METRICS_GT_MESH" \
      --pred-sampled-name "$GEOMETRY_SAMPLE_NAME" \
      --max-points "$GEOMETRY_SAMPLE_POINTS" \
      --f-threshold 1.0 \
      --include-missing \
      --crop-gt-bbox-margin "$CROP_GT_BBOX_MARGIN"
  else
    echo "reuse metrics: $metrics rows=$metrics_rows"
  fi
}

run_split fit "$MANIFEST_REL_DIR/fit_fushimi.jsonl" "$FIT_TRAJ_IDS" "$FIT_LOOKUP" "$FIT_METRICS"
run_split test "$MANIFEST_REL_DIR/test_fushimi.jsonl" "$TEST_TRAJ_IDS" "$TEST_LOOKUP" "$TEST_METRICS"

echo "========== 8. Fit/test geometry saturation surrogate =========="
python scripts/evaluate_geometry_saturation_surrogate.py \
  --fit-metrics "$FIT_METRICS" \
  --test-metrics "$TEST_METRICS" \
  --out-json "$GEOMETRY_JSON" \
  --out-yaml "$GEOMETRY_YAML" \
  --plot-out "$GEOMETRY_PLOT" \
  --allow-joint \
  --lambda-rgb-grid 1 2 3 4 6 8 10 12 \
  --lambda-depth-grid 0.5 1 1.5 2 3 4 6 8 10 12

echo "========== 9. Summary =========="
export OUT_ROOT
python - <<'PY'
import json
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["OUT_ROOT"])
for name in ["fit", "test"]:
    path = root / f"gsfusion_real_metrics_fushimi_{name}.csv"
    df = pd.read_csv(path)
    print(f"{name}: rows={len(df)}")
    print(df["geometry_source"].value_counts(dropna=False).to_string())

result = json.loads((root / "fushimi_geometry_saturation.json").read_text())
print("fit:", result["fit"])
print("heldout_test:", result["heldout_test"])
print("outputs:")
for rel in [
    "fushimi_geometry_saturation.json",
    "gsfusion_surrogate_weights_fushimi_geometry.yaml",
    "plots/fushimi_geometry_saturation.png",
    "gsfusion_real_metrics_fushimi_fit.csv",
    "gsfusion_real_metrics_fushimi_test.csv",
]:
    print(root / rel)
PY
