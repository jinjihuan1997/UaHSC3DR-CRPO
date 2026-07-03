#!/usr/bin/env bash
set -euo pipefail

# Multi-run real GSFusion evaluation. This is intentionally separate from
# RL surrogate evaluation because running GSFusion for every seed combination
# is expensive.

cd /home/king/Downloads/Projects/TCOM/hda_roi_semcom

OUT_ROOT="${OUT_ROOT:-outputs/eiffel15_mapping_quality_multiseed_q070_d035}"
CONDITIONS_DIR="${CONDITIONS_DIR:-$OUT_ROOT/conditions}"
METRICS_CSV="${METRICS_CSV:-$OUT_ROOT/gsfusion_real_metrics.csv}"
SUMMARY_PREFIX="${SUMMARY_PREFIX:-$OUT_ROOT/mapping_quality_multiseed_summary}"

TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4}"
CHANNEL_SEEDS="${CHANNEL_SEEDS:-100 101 102 103 104}"
ACTION_SEEDS="${ACTION_SEEDS:-0 1 2 3 4}"
TEST_TRAJS="${TEST_TRAJS:-traj10,traj11,traj12,traj13,traj14}"

MULTISEED_CKPT_ROOT="${MULTISEED_CKPT_ROOT:-checkpoints/eiffel_multiseed_q070_d035}"
SEMCOM_CKPT="${SEMCOM_CKPT:-checkpoints/stage3_final.pt}"
TABLE="${TABLE:-outputs/eiffel15_surrogate_test/lookups/test_eiffel.csv}"
TRAJECTORY_CSV="${TRAJECTORY_CSV:-outputs/eiffel15_surrogate_test/trajectory_eiffel_test.csv}"
SURROGATE_YAML="${SURROGATE_YAML:-outputs/eiffel15_surrogate_test/per_trajectory/eiffel_per_trajectory_composite_app60202_rerun_surrogates.yaml}"
CONFIG="${CONFIG:-configs/default.yaml}"
MANIFEST="${MANIFEST:-manifests/eiffel15_surrogate_test/test_eiffel.jsonl}"
GSFUSION_ROOT="${GSFUSION_ROOT:-/home/king/Downloads/Projects/TCOM/GSFusion}"
GT_MESH="${GT_MESH:-/home/king/Downloads/Projects/TCOM/MAGICIAN/data/Macarons++/macarons++/eiffel/eiffel_rgb.obj}"

RUN_GSFUSION="${RUN_GSFUSION:-0}"
FORCE="${FORCE:-0}"
DEVICE="${DEVICE:-cuda}"

if [[ -e "$CONDITIONS_DIR/conditions_index.json" && "$FORCE" != "1" ]]; then
  echo "[exists] $CONDITIONS_DIR/conditions_index.json"
  echo "Set FORCE=1 to rebuild the export index, or choose a different OUT_ROOT."
  exit 1
fi
if [[ "$FORCE" == "1" ]]; then
  rm -rf "$CONDITIONS_DIR"
fi
mkdir -p "$CONDITIONS_DIR"

EXPORT_COMMON=(
  --table "$TABLE"
  --trajectory-csv "$TRAJECTORY_CSV"
  --test-trajs "$TEST_TRAJS"
  --config "$CONFIG"
  --semcom-ckpt "$SEMCOM_CKPT"
  --out "$CONDITIONS_DIR"
  --gsfusion-root "$GSFUSION_ROOT"
  --manifest "$MANIFEST"
  --episode-len 102
  --device "$DEVICE"
  --q-req 0.70
  --depth-req 0.35
  --shadow-sigma-db 4.0
  --per-trajectory-surrogate-yaml "$SURROGATE_YAML"
  --psnr-ref-min 8.9171
  --psnr-ref-max 22.5791
  --payload-ref-min 78344.0
  --payload-ref-max 644408.0
  --append-index
)

for train_seed in $TRAIN_SEEDS; do
  ckpt_dir="$MULTISEED_CKPT_ROOT/seed${train_seed}"
  crpo_ckpt="$ckpt_dir/crpo.pt"
  penalty_ckpt="$ckpt_dir/penalty.pt"
  lagrangian_ckpt="$ckpt_dir/lagrangian.pt"
  for required in "$crpo_ckpt" "$penalty_ckpt" "$lagrangian_ckpt"; do
    if [[ ! -f "$required" ]]; then
      echo "[missing] $required" >&2
      exit 1
    fi
  done
  for channel_seed in $CHANNEL_SEEDS; do
    echo "========== Export learned policies train_seed=$train_seed channel_seed=$channel_seed =========="
    python scripts/export_policy_gsfusion_sequences.py \
      "${EXPORT_COMMON[@]}" \
      --seed "$channel_seed" \
      --condition-tag "trainseed${train_seed}" \
      --model "$crpo_ckpt" \
      --ppo-penalty-model "$penalty_ckpt" \
      --lagrangian-model "$lagrangian_ckpt" \
      --methods CRPO-PPO,PPO-penalty,Lagrangian-PPO
  done
done

first_seed="${TRAIN_SEEDS%% *}"
reference_ckpt="$MULTISEED_CKPT_ROOT/seed${first_seed}/crpo.pt"
for channel_seed in $CHANNEL_SEEDS; do
  echo "========== Export fixed baselines channel_seed=$channel_seed =========="
  python scripts/export_policy_gsfusion_sequences.py \
    "${EXPORT_COMMON[@]}" \
    --seed "$channel_seed" \
    --condition-tag baseline \
    --model "$reference_ckpt" \
    --methods fixed-balanced,RGB-priority,depth-priority

  for action_seed in $ACTION_SEEDS; do
    echo "========== Export random baseline channel_seed=$channel_seed action_seed=$action_seed =========="
    python scripts/export_policy_gsfusion_sequences.py \
      "${EXPORT_COMMON[@]}" \
      --seed "$channel_seed" \
      --random-action-seed "$action_seed" \
      --condition-tag baseline \
      --model "$reference_ckpt" \
      --methods random
  done
done

echo "[export done] $CONDITIONS_DIR/conditions_index.json"

if [[ "$RUN_GSFUSION" == "1" ]]; then
  python scripts/run_gsfusion_conditions.py \
    --gsfusion-root "$GSFUSION_ROOT" \
    --conditions-dir "$CONDITIONS_DIR" \
    --skip-existing \
    --continue-on-error \
    --gsfusion-width 912 \
    --gsfusion-height 512 \
    --min-free-gb 10 \
    --cache-geometry-sample-points 100000 \
    --geometry-sample-name sampled_surface_100k.ply \
    --prune-mesh-after-cache

  python scripts/build_gsfusion_real_metrics.py \
    --conditions-dir "$CONDITIONS_DIR" \
    --out "$METRICS_CSV" \
    --gt-mesh "$GT_MESH" \
    --pred-sampled-name sampled_surface_100k.ply \
    --max-points 100000 \
    --f-threshold 1.0 \
    --crop-gt-bbox-margin 3.0 \
    --include-missing

  python tools/summarize_mapping_quality_multiseed.py \
    --conditions-dir "$CONDITIONS_DIR" \
    --metrics "$METRICS_CSV" \
    --out "$SUMMARY_PREFIX"
fi

echo "========== Done =========="
echo "Conditions: $CONDITIONS_DIR/conditions_index.json"
echo "Metrics:    $METRICS_CSV"
echo "Summary:    ${SUMMARY_PREFIX}.csv"
