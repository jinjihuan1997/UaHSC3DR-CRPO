#!/usr/bin/env bash
set -euo pipefail

# One-command multi-scene surrogate fitting pipeline.
#
# Default command:
#   bash scripts/run_multiscene_surrogate_pipeline.sh
#
# Useful overrides:
#   STRICT_REQUIRED_SCENES=0 bash scripts/run_multiscene_surrogate_pipeline.sh
#   FORCE=1 bash scripts/run_multiscene_surrogate_pipeline.sh
#   NUM_CONDITIONS_PER_SCENE=24 bash scripts/run_multiscene_surrogate_pipeline.sh
#   FORCE_SEMCOM_TRAIN=1 bash scripts/run_multiscene_surrogate_pipeline.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-/home/king/Downloads/Projects/TCOM/datasets/hda_roi_semcom_dataset}"
CONFIG_TEMPLATE="${CONFIG_TEMPLATE:-configs/default.yaml}"
GSFUSION_ROOT="${GSFUSION_ROOT:-/home/king/Downloads/Projects/TCOM/GSFusion}"
MESH_ROOT="${MESH_ROOT:-/home/king/Downloads/Projects/TCOM/MAGICIAN/data/Macarons++/macarons++}"
OUT_ROOT="${OUT_ROOT:-outputs/multiscene_surrogate}"

REQUIRED_SCENES="${REQUIRED_SCENES:-alhambra bannerman barts bridge colosseum dunnottar eiffel}"
FUSHIMI_FIT_TRAJS="${FUSHIMI_FIT_TRAJS:-traj0 traj1 traj2 traj3 traj4 traj5 traj6 traj7 traj8 traj9}"
FUSHIMI_TEST_TRAJ="${FUSHIMI_TEST_TRAJ:-traj10}"
RANDOM_TEST_SCENE="${RANDOM_TEST_SCENE:-bannerman}"
RANDOM_TEST_TRAJ="${RANDOM_TEST_TRAJ:-traj2}"
TRAJS_PER_REQUIRED_SCENE="${TRAJS_PER_REQUIRED_SCENE:-5}"

SNRS="${SNRS:-0 4 8 12 16 20}"
KD_GRID="${KD_GRID:-10 14 18 22}"
MAX_FRAMES="${MAX_FRAMES:-102}"
NUM_CONDITIONS_PER_SCENE="${NUM_CONDITIONS_PER_SCENE:-0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
GSFUSION_WIDTH="${GSFUSION_WIDTH:-912}"
GSFUSION_HEIGHT="${GSFUSION_HEIGHT:-512}"

STRICT_REQUIRED_SCENES="${STRICT_REQUIRED_SCENES:-1}"
FORCE="${FORCE:-0}"
FORCE_SEMCOM_TRAIN="${FORCE_SEMCOM_TRAIN:-0}"
SKIP_SEMCOM_TRAIN="${SKIP_SEMCOM_TRAIN:-0}"

MANIFEST_REL_DIR="manifests/multiscene_surrogate"
MANIFEST_DIR="$DATA_ROOT/$MANIFEST_REL_DIR"
CONFIG_OUT="$OUT_ROOT/config_multiscene_surrogate.yaml"
CKPT_DIR="$OUT_ROOT/checkpoints"
SEMCOM_CKPT="${SEMCOM_CKPT:-$CKPT_DIR/stage3_final.pt}"

LOOKUP_DIR="$OUT_ROOT/lookups"
DEGRADED_DIR="$OUT_ROOT/gsfusion_degraded"
METRICS_DIR="$OUT_ROOT/metrics"
PLOTS_DIR="$OUT_ROOT/plots"
FIT_METRICS="$OUT_ROOT/gsfusion_real_metrics_multiscene_fit.csv"
TEST_METRICS="$OUT_ROOT/gsfusion_real_metrics_multiscene_test.csv"
LOOKUP_NORM="$OUT_ROOT/lookup_qrgb_normalization_fit.csv"
WEIGHTS_OUT="$OUT_ROOT/gsfusion_surrogate_weights_multiscene_saturation.yaml"
FIT_JSON="$OUT_ROOT/gsfusion_surrogate_fit_multiscene_saturation.json"
TEST_JSON="$OUT_ROOT/gsfusion_surrogate_test_diagnostics.json"
FIT_PLOT="$PLOTS_DIR/q3d_surrogate_fit_multiscene_saturation.png"

mkdir -p "$OUT_ROOT" "$MANIFEST_DIR" "$LOOKUP_DIR" "$DEGRADED_DIR" "$METRICS_DIR" "$PLOTS_DIR" "$CKPT_DIR"

echo "========== 0. Input paths =========="
test -d "$DATA_ROOT" || { echo "ERROR: data root not found: $DATA_ROOT"; exit 1; }
test -f "$CONFIG_TEMPLATE" || { echo "ERROR: config template not found: $CONFIG_TEMPLATE"; exit 1; }
test -d "$GSFUSION_ROOT" || { echo "ERROR: GSFusion root not found: $GSFUSION_ROOT"; exit 1; }
test -d "$MESH_ROOT" || { echo "ERROR: mesh root not found: $MESH_ROOT"; exit 1; }

echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "REQUIRED_SCENES=$REQUIRED_SCENES"
echo "held-out test trajectories: fushimi/$FUSHIMI_TEST_TRAJ and $RANDOM_TEST_SCENE/$RANDOM_TEST_TRAJ"

echo "========== 1. Build multi-scene manifests =========="
export DATA_ROOT MANIFEST_DIR REQUIRED_SCENES FUSHIMI_FIT_TRAJS FUSHIMI_TEST_TRAJ
export RANDOM_TEST_SCENE RANDOM_TEST_TRAJ TRAJS_PER_REQUIRED_SCENE STRICT_REQUIRED_SCENES
python - <<'PY'
import json
import os
import sys
from pathlib import Path

data_root = Path(os.environ["DATA_ROOT"])
manifest_dir = Path(os.environ["MANIFEST_DIR"])
required_scenes = os.environ["REQUIRED_SCENES"].split()
fushimi_fit = os.environ["FUSHIMI_FIT_TRAJS"].split()
fushimi_test = ("fushimi", os.environ["FUSHIMI_TEST_TRAJ"])
random_test = (os.environ["RANDOM_TEST_SCENE"], os.environ["RANDOM_TEST_TRAJ"])
trajs_per_scene = int(os.environ["TRAJS_PER_REQUIRED_SCENE"])
strict = os.environ.get("STRICT_REQUIRED_SCENES", "1") == "1"

manifest_dir.mkdir(parents=True, exist_ok=True)

def rel(path):
    return str(Path(path).resolve().relative_to(data_root.resolve()))

def find_traj(scene, traj):
    scene_dir = data_root / "scenes" / scene
    if not scene_dir.exists():
        return None
    matches = sorted(p for p in scene_dir.glob(f"*/{traj}") if p.is_dir())
    complete = []
    for path in matches:
        if (
            (path / "clean_images").exists()
            and any((path / "clean_images").glob("*.png"))
            and (path / "depth").exists()
            and any((path / "depth").glob("*.png"))
            and (path / "poses").exists()
            and any((path / "poses").glob("*.json"))
        ):
            complete.append(path)
    if not complete:
        return None
    preferred = [path for path in complete if path.parent.name == "observations"]
    return (preferred or complete)[0]

def normalize_record(scene, traj_dir, raw):
    out = dict(raw)
    out["scene_id"] = scene
    out["trajectory_id"] = traj_dir.name
    if "frame_id" not in out:
        stem = Path(out.get("image_name", out.get("clean_image", "0"))).stem
        try:
            out["frame_id"] = int(stem)
        except ValueError:
            out["frame_id"] = stem
    for key in ("clean_image", "roi_mask", "roi_priority_map", "overlay", "depth", "valid_mask", "camera_pose"):
        value = out.get(key)
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute() and not str(p).startswith("scenes/"):
            out[key] = rel(traj_dir / p)
        elif p.is_absolute():
            out[key] = rel(p)
    return out

def records_from_traj(scene, traj):
    traj_dir = find_traj(scene, traj)
    if traj_dir is None:
        return None
    manifest = traj_dir / "frame_manifest.jsonl"
    rows = []
    if manifest.exists():
        with manifest.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(normalize_record(scene, traj_dir, json.loads(line)))
    else:
        image_dir = traj_dir / "clean_images"
        for image_path in sorted(image_dir.glob("*.png")):
            stem = image_path.stem
            raw = {
                "scene_id": scene,
                "trajectory_id": traj,
                "frame_id": int(stem) if stem.isdigit() else stem,
                "image_name": image_path.name,
                "clean_image": f"clean_images/{image_path.name}",
                "depth": f"depth/{image_path.name}",
                "valid_mask": f"valid_masks/{image_path.name}",
                "roi_mask": f"roi_masks/{image_path.name}",
                "camera_pose": f"poses/{stem}.json",
            }
            rows.append(normalize_record(scene, traj_dir, raw))
    required = ("clean_image", "depth", "camera_pose")
    good = []
    for row in rows:
        if all((data_root / row[key]).exists() for key in required):
            good.append(row)
    return good

required_pairs = []
for scene in required_scenes:
    for idx in range(trajs_per_scene):
        required_pairs.append((scene, f"traj{idx}"))
fit_pairs = required_pairs + [("fushimi", traj) for traj in fushimi_fit]
test_pairs = [fushimi_test, random_test]
test_set = set(test_pairs)
fit_pairs = [pair for pair in fit_pairs if pair not in test_set]

missing = []
records_by_pair = {}
for pair in sorted(set(fit_pairs + test_pairs)):
    rows = records_from_traj(*pair)
    if not rows:
        missing.append(pair)
    else:
        records_by_pair[pair] = rows

if missing and strict:
    print("ERROR: required RGB-D trajectories are missing or empty:", file=sys.stderr)
    for scene, traj in missing:
        print(f"  - {scene}/{traj}", file=sys.stderr)
    print("Set STRICT_REQUIRED_SCENES=0 for a partial local run.", file=sys.stderr)
    sys.exit(2)

if missing:
    missing_set = set(missing)
    fit_pairs = [pair for pair in fit_pairs if pair not in missing_set]
    test_pairs = [pair for pair in test_pairs if pair not in missing_set]
    if len(test_pairs) < 2:
        available_non_fushimi = sorted(
            pair for pair in records_by_pair
            if pair[0] != "fushimi" and pair not in set(fit_pairs)
        )
        if not available_non_fushimi:
            available_non_fushimi = sorted(pair for pair in records_by_pair if pair[0] != "fushimi")
        if available_non_fushimi:
            replacement = available_non_fushimi[-1]
            test_pairs = [pair for pair in test_pairs if pair[0] == "fushimi"] + [replacement]
            fit_pairs = [pair for pair in fit_pairs if pair != replacement]

def flatten(pairs):
    rows = []
    for pair in pairs:
        rows.extend(records_by_pair.get(pair, []))
    return rows

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

fit_rows = flatten(fit_pairs)
test_rows = flatten(test_pairs)
if not fit_rows:
    raise SystemExit("no fitting records available")
if not test_rows:
    raise SystemExit("no test records available")

write_jsonl(manifest_dir / "train.jsonl", fit_rows)
write_jsonl(manifest_dir / "val.jsonl", test_rows)
write_jsonl(manifest_dir / "test.jsonl", test_rows)
write_jsonl(manifest_dir / "surrogate_fit.jsonl", fit_rows)
write_jsonl(manifest_dir / "surrogate_test.jsonl", test_rows)

fit_scene_names = sorted({scene for scene, _ in fit_pairs})
test_scene_names = sorted({scene for scene, _ in test_pairs})
for scene in fit_scene_names:
    write_jsonl(manifest_dir / f"fit_{scene}.jsonl", flatten(pair for pair in fit_pairs if pair[0] == scene))
for scene in test_scene_names:
    write_jsonl(manifest_dir / f"test_{scene}.jsonl", flatten(pair for pair in test_pairs if pair[0] == scene))

(manifest_dir / "fit_scenes.txt").write_text("\n".join(fit_scene_names) + "\n")
(manifest_dir / "test_scenes.txt").write_text("\n".join(test_scene_names) + "\n")
with (manifest_dir / "split_plan.csv").open("w") as f:
    f.write("split,scene,trajectory,frames\n")
    for split, pairs in (("fit", fit_pairs), ("test", test_pairs)):
        for scene, traj in pairs:
            f.write(f"{split},{scene},{traj},{len(records_by_pair.get((scene, traj), []))}\n")

print(f"fit trajectories: {len(fit_pairs)} frames={len(fit_rows)}")
print(f"test trajectories: {len(test_pairs)} frames={len(test_rows)}")
print(f"wrote manifests under {manifest_dir}")
PY

echo "========== 2. Write generated config =========="
export CONFIG_TEMPLATE CONFIG_OUT DATA_ROOT MANIFEST_REL_DIR CKPT_DIR DEVICE
python - <<'PY'
import os
import yaml
from pathlib import Path

with open(os.environ["CONFIG_TEMPLATE"]) as f:
    cfg = yaml.safe_load(f)
cfg["data"]["root"] = os.environ["DATA_ROOT"]
cfg["data"]["train_manifest"] = f'{os.environ["MANIFEST_REL_DIR"]}/train.jsonl'
cfg["data"]["val_manifest"] = f'{os.environ["MANIFEST_REL_DIR"]}/val.jsonl'
cfg["data"]["test_manifest"] = f'{os.environ["MANIFEST_REL_DIR"]}/test.jsonl'
cfg["data"]["train_scene_id"] = None
cfg["data"]["min_rgb_std"] = 0.0
cfg["train"]["save_dir"] = os.environ["CKPT_DIR"]
cfg["train"]["device"] = os.environ["DEVICE"]
out = Path(os.environ["CONFIG_OUT"])
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"wrote {out}")
PY

echo "========== 3. Train or reuse SemCom checkpoint =========="
if [[ "$SKIP_SEMCOM_TRAIN" == "1" ]]; then
  test -f "$SEMCOM_CKPT" || { echo "ERROR: SKIP_SEMCOM_TRAIN=1 but checkpoint not found: $SEMCOM_CKPT"; exit 1; }
  echo "reusing checkpoint: $SEMCOM_CKPT"
elif [[ "$FORCE_SEMCOM_TRAIN" == "1" || ! -f "$SEMCOM_CKPT" ]]; then
  python scripts/train.py --config "$CONFIG_OUT"
  test -f "$SEMCOM_CKPT" || { echo "ERROR: expected checkpoint not found after training: $SEMCOM_CKPT"; exit 1; }
else
  echo "reusing existing checkpoint: $SEMCOM_CKPT"
fi

mesh_for_scene() {
  local scene="$1"
  local mesh_dir="$MESH_ROOT/$scene"
  find "$mesh_dir" -maxdepth 1 -type f \( -name '*.obj' -o -name '*.ply' \) | sort | head -n 1
}

build_lookup_only() {
  local split="$1"
  local scene="$2"
  local manifest_rel="$MANIFEST_REL_DIR/${split}_${scene}.jsonl"
  local lookup="$LOOKUP_DIR/${split}_${scene}.csv"
  test -f "$DATA_ROOT/$manifest_rel" || { echo "ERROR: manifest missing: $DATA_ROOT/$manifest_rel"; exit 1; }

  echo "----- [$split/$scene] lookup -----"
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
}

run_scene_after_lookup() {
  local split="$1"
  local scene="$2"
  local manifest_rel="$MANIFEST_REL_DIR/${split}_${scene}.jsonl"
  local lookup="$LOOKUP_DIR/${split}_${scene}.csv"
  local degraded="$DEGRADED_DIR/${split}_${scene}"
  local metrics="$METRICS_DIR/${split}_${scene}.csv"
  local mesh
  mesh="$(mesh_for_scene "$scene")"
  test -f "$DATA_ROOT/$manifest_rel" || { echo "ERROR: manifest missing: $DATA_ROOT/$manifest_rel"; exit 1; }
  test -s "$lookup" || { echo "ERROR: lookup missing: $lookup"; exit 1; }
  test -s "$LOOKUP_NORM" || { echo "ERROR: global Q normalization lookup missing: $LOOKUP_NORM"; exit 1; }
  test -n "$mesh" || { echo "ERROR: GT mesh missing for scene=$scene under $MESH_ROOT/$scene"; exit 1; }

  if [[ "$FORCE" != "1" && -s "$metrics" ]]; then
    echo "reuse metrics and skip GSFusion: $metrics"
    return
  fi

  echo "----- [$split/$scene] export degraded RGB-D -----"
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
      --num-conditions "$NUM_CONDITIONS_PER_SCENE" \
      --max-frames "$MAX_FRAMES" \
      --seed "$SEED" \
      --device "$DEVICE"
  else
    echo "reuse degraded sequences: $degraded"
  fi

  echo "----- [$split/$scene] run GSFusion -----"
  python scripts/run_gsfusion_conditions.py \
    --gsfusion-root "$GSFUSION_ROOT" \
    --conditions-dir "$degraded" \
    --skip-existing \
    --continue-on-error \
    --gsfusion-width "$GSFUSION_WIDTH" \
    --gsfusion-height "$GSFUSION_HEIGHT"

  echo "----- [$split/$scene] build real metrics -----"
  if [[ "$FORCE" == "1" || ! -s "$metrics" ]]; then
    python scripts/build_gsfusion_real_metrics.py \
      --conditions-dir "$degraded" \
      --out "$metrics" \
      --gt-mesh "$mesh"
  else
    echo "reuse metrics: $metrics"
  fi
}

echo "========== 4. Build offline lookup tables =========="
while read -r scene; do
  [[ -z "$scene" ]] && continue
  build_lookup_only fit "$scene"
done < "$MANIFEST_DIR/fit_scenes.txt"

while read -r scene; do
  [[ -z "$scene" ]] && continue
  build_lookup_only test "$scene"
done < "$MANIFEST_DIR/test_scenes.txt"

echo "========== 5. Build global Q_rgb normalization lookup from fit split =========="
if [[ "$FORCE" == "1" || ! -s "$LOOKUP_NORM" ]]; then
  export LOOKUP_DIR LOOKUP_NORM
  python - <<'PY'
import csv
import os
from pathlib import Path

lookup_dir = Path(os.environ["LOOKUP_DIR"])
out = Path(os.environ["LOOKUP_NORM"])
paths = sorted(lookup_dir.glob("fit_*.csv"))
if not paths:
    raise SystemExit(f"no fit lookup files found in {lookup_dir}")
fieldnames = None
rows = []
for path in paths:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if fieldnames is None:
            fieldnames = reader.fieldnames
        rows.extend(reader)
if not rows:
    raise SystemExit("fit lookup files contain no rows")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out} rows={len(rows)}")
PY
else
  echo "reuse global normalization lookup: $LOOKUP_NORM"
fi

echo "========== 6. Run GSFusion and compute metrics =========="
while read -r scene; do
  [[ -z "$scene" ]] && continue
  run_scene_after_lookup fit "$scene"
done < "$MANIFEST_DIR/fit_scenes.txt"

while read -r scene; do
  [[ -z "$scene" ]] && continue
  run_scene_after_lookup test "$scene"
done < "$MANIFEST_DIR/test_scenes.txt"

echo "========== 7. Merge metrics =========="
export METRICS_DIR FIT_METRICS TEST_METRICS
python - <<'PY'
import csv
import os
from pathlib import Path

metrics_dir = Path(os.environ["METRICS_DIR"])
out_paths = {
    "fit": Path(os.environ["FIT_METRICS"]),
    "test": Path(os.environ["TEST_METRICS"]),
}

for split, out_path in out_paths.items():
    paths = sorted(metrics_dir.glob(f"{split}_*.csv"))
    rows = []
    fieldnames = None
    for path in paths:
        scene = path.stem[len(split) + 1:]
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = ["source_split", "source_scene"] + list(reader.fieldnames or [])
            for row in reader:
                row = {"source_split": split, "source_scene": scene, **row}
                rows.append(row)
    if not rows:
        raise SystemExit(f"no {split} metrics found in {metrics_dir}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} rows={len(rows)}")
PY

echo "========== 8. Fit saturation surrogate =========="
python scripts/fit_gsfusion_surrogate.py "$FIT_METRICS" \
  --model saturation \
  --out "$WEIGHTS_OUT" \
  --json-out "$FIT_JSON" \
  --plot-out "$FIT_PLOT" \
  --lambda-rgb-grid 1 2 3 4 \
  --lambda-depth-grid 0.5 1 1.5 2 3 \
  --min-w-depth 0.10

echo "========== 9. Held-out test diagnostics and sensitivity plots =========="
export WEIGHTS_OUT TEST_METRICS TEST_JSON PLOTS_DIR
python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import yaml

weights_path = Path(os.environ["WEIGHTS_OUT"])
test_metrics = Path(os.environ["TEST_METRICS"])
test_json = Path(os.environ["TEST_JSON"])
plots_dir = Path(os.environ["PLOTS_DIR"])
plots_dir.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load(weights_path.read_text())["mapping_quality"]["gsfusion_weights"]

def col(rows, name):
    return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

def filter_finite_rows(rows, required_columns):
    invalid_counts = {name: 0 for name in required_columns}
    valid = []
    for row in rows:
        keep = True
        for name in required_columns:
            try:
                value = float(row[name])
            except (KeyError, ValueError):
                invalid_counts[name] += 1
                keep = False
                continue
            if not np.isfinite(value):
                invalid_counts[name] += 1
                keep = False
        if keep:
            valid.append(row)
    invalid_counts = {name: count for name, count in invalid_counts.items() if count}
    return valid, invalid_counts

def norm_high(x):
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    return np.clip((x - lo) / max(hi - lo, 1e-12), 0.0, 1.0)

def norm_low(x):
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    return np.clip((hi - x) / max(hi - lo, 1e-12), 0.0, 1.0)

def rankdata(x):
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks

def spearman(x, y):
    if len(x) < 2:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    sx, sy = rx.std(), ry.std()
    if sx <= 1e-12 or sy <= 1e-12:
        return float("nan")
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))

def q3d(q, r):
    q = np.clip(q, 0.0, 1.0)
    r = np.clip(r, 0.0, 1.0)
    if cfg["model"] == "saturation":
        gr = 1.0 - np.exp(-float(cfg["lambda_rgb"]) * q)
        gd = 1.0 - np.exp(-float(cfg["lambda_depth"]) * r)
    else:
        gr, gd = q, r
    pred = (
        float(cfg["bias"])
        + float(cfg["w_rgb"]) * gr
        + float(cfg["w_depth"]) * gd
        + float(cfg["w_joint"]) * gr * gd
    )
    return np.clip(pred, 0.0, 1.0)

with test_metrics.open(newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    raise SystemExit(f"empty test metrics: {test_metrics}")
input_samples = len(rows)
rows, invalid_counts = filter_finite_rows(
    rows,
    ["q_rgb", "r_depth", "psnr", "ssim", "lpips", "chamfer", "fscore", "completeness"],
)
if not rows:
    raise SystemExit(f"no finite test samples in {test_metrics}; dropped {input_samples} rows")

q_render = 0.4 * norm_high(col(rows, "psnr")) + 0.4 * norm_high(col(rows, "ssim")) + 0.2 * norm_low(col(rows, "lpips"))
q_geometry = 0.35 * norm_high(col(rows, "fscore")) + 0.35 * norm_high(col(rows, "completeness")) + 0.30 * norm_low(col(rows, "chamfer"))
y = np.clip(0.45 * q_render + 0.45 * q_geometry + 0.10 * q_render * q_geometry, 0.0, 1.0)
pred = q3d(col(rows, "q_rgb"), col(rows, "r_depth"))
rmse = float(np.sqrt(np.square(pred - y).mean()))
mae = float(np.abs(pred - y).mean())
denom = float(np.square(y - y.mean()).sum())
r2 = 1.0 - float(np.square(pred - y).sum()) / max(denom, 1e-12)
diag = {
    "input_samples": input_samples,
    "test_samples": len(rows),
    "dropped_samples": input_samples - len(rows),
    "invalid_counts": invalid_counts,
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "spearman": spearman(y, pred),
}
test_json.write_text(json.dumps(diag, indent=2, allow_nan=False))
print(f"wrote {test_json}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(5.4, 4.4), dpi=180)
for q in [0.2, 0.4, 0.6, 0.8]:
    r = np.linspace(0, 1, 200)
    ax.plot(r, q3d(np.full_like(r, q), r), label=f"$Q_{{rgb}}={q:.1f}$")
ax.set_xlabel("$R_{dep}$")
ax.set_ylabel("$K_{sur}$")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.25)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(plots_dir / "q3d_surrogate_depth_sensitivity.png")
plt.close(fig)

q = np.linspace(0, 1, 101)
r = np.linspace(0, 1, 101)
qq, rr = np.meshgrid(q, r)
zz = q3d(qq, rr)
fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=180)
im = ax.imshow(zz, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xlabel("$Q_{rgb}$")
ax.set_ylabel("$R_{dep}$")
fig.colorbar(im, ax=ax, label="$K_{sur}$")
fig.tight_layout()
fig.savefig(plots_dir / "q3d_surrogate_surface_multiscene.png")
plt.close(fig)
print(f"wrote sensitivity plots under {plots_dir}")
PY

echo "========== Done =========="
echo "Generated config : $CONFIG_OUT"
echo "SemCom ckpt      : $SEMCOM_CKPT"
echo "Fit metrics      : $FIT_METRICS"
echo "Test metrics     : $TEST_METRICS"
echo "Weights          : $WEIGHTS_OUT"
echo "Fit diagnostics  : $FIT_JSON"
echo "Test diagnostics : $TEST_JSON"
echo "Plots            : $PLOTS_DIR"
