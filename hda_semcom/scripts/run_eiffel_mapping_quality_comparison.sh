#!/usr/bin/env bash
set -euo pipefail

# End-to-end real GSFusion evaluation for paper Section 6.4.
# The defaults target the current Eiffel composite-surrogate experiment.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${OUT_ROOT:-outputs/eiffel15_mapping_quality}"
CONDITIONS_DIR="${CONDITIONS_DIR:-$OUT_ROOT/conditions}"
METRICS_CSV="${METRICS_CSV:-$OUT_ROOT/gsfusion_real_metrics.csv}"
PAPER_OUT_DIR="${PAPER_OUT_DIR:-Figures}"

GSFUSION_ROOT="${GSFUSION_ROOT:-/home/king/Downloads/Projects/TCOM/GSFusion}"
GT_MESH="${GT_MESH:-/home/king/Downloads/Projects/TCOM/MAGICIAN/data/Macarons++/macarons++/eiffel/eiffel_rgb.obj}"
DATASET_ROOT="${DATASET_ROOT:-/home/king/Downloads/Projects/TCOM/datasets/hda_semcom_dataset}"

TABLE="${TABLE:-outputs/eiffel15_surrogate_test/lookups/test_eiffel.csv}"
TRAJECTORY_CSV="${TRAJECTORY_CSV:-outputs/eiffel15_surrogate_test/trajectory_eiffel_test.csv}"
MANIFEST="${MANIFEST:-manifests/eiffel15_surrogate_test/test_eiffel.jsonl}"
TEST_TRAJS="${TEST_TRAJS:-traj10,traj11,traj12,traj13,traj14}"
SURROGATE_YAML="${SURROGATE_YAML:-outputs/eiffel15_surrogate_test/per_trajectory/eiffel_per_trajectory_composite_app60202_rerun_surrogates.yaml}"

SEMCOM_CKPT="${SEMCOM_CKPT:-checkpoints/stage3_final.pt}"
CRPO_CKPT="${CRPO_CKPT:-checkpoints/crpo_eiffel_composite_app60202_q070_d035.pt}"
PENALTY_CKPT="${PENALTY_CKPT:-checkpoints/penalty_eiffel_composite_app60202_q070_d035.pt}"
LAGRANGIAN_CKPT="${LAGRANGIAN_CKPT:-checkpoints/lagrangian_eiffel_composite_app60202_q070_d035.pt}"

METHODS="${METHODS:-fixed-balanced,RGB-priority,depth-priority,random,PPO-penalty,Lagrangian-PPO,CRPO-PPO}"
DEVICE="${DEVICE:-cuda}"
Q_REQ="${Q_REQ:-0.70}"
DEPTH_REQ="${DEPTH_REQ:-0.35}"
EPISODE_LEN="${EPISODE_LEN:-102}"
SEED="${SEED:-123}"

GSFUSION_WIDTH="${GSFUSION_WIDTH:-912}"
GSFUSION_HEIGHT="${GSFUSION_HEIGHT:-512}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
GEOMETRY_SAMPLE_POINTS="${GEOMETRY_SAMPLE_POINTS:-100000}"
GEOMETRY_SAMPLE_NAME="${GEOMETRY_SAMPLE_NAME:-sampled_surface_100k.ply}"
F_THRESHOLD="${F_THRESHOLD:-1.0}"
GT_BBOX_MARGIN="${GT_BBOX_MARGIN:-3.0}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

method_enabled() {
  local needle="$1"
  IFS=',' read -ra method_list <<< "$METHODS"
  for method in "${method_list[@]}"; do
    method="${method//[[:space:]]/}"
    if [[ "$method" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

required_files=(
  "$TABLE"
  "$TRAJECTORY_CSV"
  "$DATASET_ROOT/$MANIFEST"
  "$SURROGATE_YAML"
  "$SEMCOM_CKPT"
  "$GT_MESH"
)
if method_enabled "CRPO-PPO"; then
  required_files+=("$CRPO_CKPT")
fi
if method_enabled "PPO-penalty"; then
  required_files+=("$PENALTY_CKPT")
fi
if method_enabled "Lagrangian-PPO"; then
  required_files+=("$LAGRANGIAN_CKPT")
fi
for path in "${required_files[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "[missing] $path" >&2
    exit 1
  fi
done
if [[ ! -d "$GSFUSION_ROOT" ]]; then
  echo "[missing] GSFusion root: $GSFUSION_ROOT" >&2
  exit 1
fi

echo "========== 1. Export policy-generated degraded RGB-D sequences =========="
python scripts/export_policy_gsfusion_sequences.py \
  --table "$TABLE" \
  --trajectory-csv "$TRAJECTORY_CSV" \
  --manifest "$MANIFEST" \
  --test-trajs "$TEST_TRAJS" \
  --semcom-ckpt "$SEMCOM_CKPT" \
  --model "$CRPO_CKPT" \
  --ppo-penalty-model "$PENALTY_CKPT" \
  --lagrangian-model "$LAGRANGIAN_CKPT" \
  --out "$CONDITIONS_DIR" \
  --gsfusion-root "$GSFUSION_ROOT" \
  --methods "$METHODS" \
  --device "$DEVICE" \
  --q-req "$Q_REQ" \
  --depth-req "$DEPTH_REQ" \
  --episode-len "$EPISODE_LEN" \
  --seed "$SEED" \
  --per-trajectory-surrogate-yaml "$SURROGATE_YAML"

echo "========== 2. Run GSFusion on exported method sequences =========="
python scripts/run_gsfusion_conditions.py \
  --gsfusion-root "$GSFUSION_ROOT" \
  --conditions-dir "$CONDITIONS_DIR" \
  --skip-existing \
  --continue-on-error \
  --gsfusion-width "$GSFUSION_WIDTH" \
  --gsfusion-height "$GSFUSION_HEIGHT" \
  --min-free-gb "$MIN_FREE_GB" \
  --cache-geometry-sample-points "$GEOMETRY_SAMPLE_POINTS" \
  --geometry-sample-name "$GEOMETRY_SAMPLE_NAME" \
  --prune-mesh-after-cache \
  "${LIMIT_ARGS[@]}"

echo "========== 3. Build real GSFusion reconstruction metrics =========="
python scripts/build_gsfusion_real_metrics.py \
  --conditions-dir "$CONDITIONS_DIR" \
  --out "$METRICS_CSV" \
  --gt-mesh "$GT_MESH" \
  --pred-sampled-name "$GEOMETRY_SAMPLE_NAME" \
  --max-points "$GEOMETRY_SAMPLE_POINTS" \
  --f-threshold "$F_THRESHOLD" \
  --crop-gt-bbox-margin "$GT_BBOX_MARGIN" \
  --include-missing

echo "========== 4. Generate Section 6.4 paper table and figures =========="
python tools/summarize_mapping_quality_results.py \
  --conditions-dir "$CONDITIONS_DIR" \
  --metrics "$METRICS_CSV" \
  --out-dir "$PAPER_OUT_DIR"

echo "[done] Section 6.4 outputs:"
echo "  metrics: $METRICS_CSV"
echo "  table:   $PAPER_OUT_DIR/tables/main_3d_mapping_quality_comparison.csv"
echo "  figures: $PAPER_OUT_DIR/Fig5a_quality_q_gt.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig5b_quality_q_app.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig5c_quality_q_geo.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig6a_metric_psnr.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig6b_metric_ssim.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig6c_metric_lpips.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig6d_metric_chamfer.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig6e_metric_fscore.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig6f_metric_completeness.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig7a_crpo_guided_ppo.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig7b_lagrangian_ppo.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig7c_ppo_penalty.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig7d_random_allocation.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig7e_depth_priority_allocation.{pdf,png}"
echo "           $PAPER_OUT_DIR/Fig7f_reference.{pdf,png}"
