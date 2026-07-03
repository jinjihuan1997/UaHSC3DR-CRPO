#!/usr/bin/env bash
set -euo pipefail

LOOKUP="outputs/offline_lookup_fushimi_traj10_stage3_grid.csv"
CONFIG="configs/default.yaml"
SEMCOM_CKPT="checkpoints/stage3_final.pt"
GSFUSION_ROOT="/home/king/Downloads/Projects/TCOM/GSFusion"
GT_MESH="/home/king/Downloads/Projects/TCOM/MAGICIAN/data/Macarons++/macarons++/fushimi/fushimi_rgb.obj"

DEGRADED="outputs/gsfusion_degraded_fushimi_traj10_allframes"
METRICS="outputs/gsfusion_real_metrics_fushimi_traj10_allframes.csv"
WEIGHTS="outputs/gsfusion_surrogate_weights_fushimi_traj10_allframes.yaml"
FIT_JSON="outputs/gsfusion_surrogate_fit_fushimi_traj10_allframes.json"
PLOT_OUT="outputs/plots/q3d_surrogate_fit_fushimi_traj10_allframes.png"

echo "========== 0. Check input paths =========="
test -f "$LOOKUP"       || { echo "ERROR: lookup not found: $LOOKUP";       exit 1; }
test -f "$CONFIG"       || { echo "ERROR: config not found: $CONFIG";       exit 1; }
test -f "$SEMCOM_CKPT"  || { echo "ERROR: checkpoint not found: $SEMCOM_CKPT"; exit 1; }
test -d "$GSFUSION_ROOT"|| { echo "ERROR: GSFusion root not found: $GSFUSION_ROOT"; exit 1; }
test -f "$GT_MESH"      || { echo "ERROR: GT mesh not found: $GT_MESH";     exit 1; }
echo "All inputs OK."

mkdir -p outputs outputs/plots

echo "========== 1. Clean old degraded outputs =========="
rm -rf "$DEGRADED"
echo "Cleaned: $DEGRADED"

echo "========== 2. Export degraded RGB-D sequences =========="
python scripts/export_gsfusion_degraded_sequences.py \
  --lookup "$LOOKUP" \
  --config "$CONFIG" \
  --semcom-ckpt "$SEMCOM_CKPT" \
  --split test \
  --out "$DEGRADED" \
  --gsfusion-root "$GSFUSION_ROOT" \
  --num-conditions 77 \
  --max-frames 102 \
  --seed 42 \
  --device cuda

echo "========== 3. Run GSFusion conditions =========="
python scripts/run_gsfusion_conditions.py \
  --gsfusion-root "$GSFUSION_ROOT" \
  --conditions-dir "$DEGRADED" \
  --skip-existing \
  --continue-on-error \
  --gsfusion-width 912 \
  --gsfusion-height 512

echo "========== 4. Build real GSFusion metrics =========="
python scripts/build_gsfusion_real_metrics.py \
  --conditions-dir "$DEGRADED" \
  --out "$METRICS" \
  --gt-mesh "$GT_MESH"

echo "========== 5. Fit GSFusion surrogate =========="
python scripts/fit_gsfusion_surrogate.py "$METRICS" \
  --out "$WEIGHTS" \
  --json-out "$FIT_JSON" \
  --plot-out "$PLOT_OUT"

echo "========== Done =========="
echo "Degraded sequences : $DEGRADED"
echo "Metrics CSV        : $METRICS"
echo "Weights YAML       : $WEIGHTS"
echo "Fit JSON           : $FIT_JSON"
echo "Fit plot           : $PLOT_OUT"
