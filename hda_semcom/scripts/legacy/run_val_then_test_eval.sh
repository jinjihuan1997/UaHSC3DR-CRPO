#!/usr/bin/env bash
set -euo pipefail

cd /home/king/Downloads/Projects/TCOM/hda_semcom

mkdir -p outputs/eval_cmdp

TABLE="outputs/offline_lookup_fushimi_traj0_10_stage3_grid.csv"
CONFIG="configs/default.yaml"
TRAJ_CSV="outputs/trajectory_fushimi_a2g.csv"
CRPO_MODEL="${CRPO_MODEL:-checkpoints/crpo_qreq065_epsR020_epsD005.pt}"
PENALTY_MODEL="${PENALTY_MODEL:-checkpoints/ppo_penalty_qreq065_lam50.pt}"

if [[ ! -f "$CRPO_MODEL" ]]; then
  echo "ERROR: CRPO model not found: $CRPO_MODEL"
  echo "Train it first or run with CRPO_MODEL=/path/to/model.pt"
  exit 1
fi

if [[ ! -f "$PENALTY_MODEL" ]]; then
  echo "ERROR: PPO-penalty model not found: $PENALTY_MODEL"
  echo "Train it first or run with PENALTY_MODEL=/path/to/model.pt"
  exit 1
fi

COMMON_EVAL_ARGS=(
  --table "$TABLE"
  --config "$CONFIG"
  --trajectory-csv "$TRAJ_CSV"
  --multi-trajectory
  --shadow-seeds 101,102,103,104,105
  --episodes 20
  --q-req 0.65
  --depth-req 0.35
  --model "$CRPO_MODEL"
  --ppo-penalty-model "$PENALTY_MODEL"
  --device cuda
)

echo "========== Validation eval: traj7,traj8 =========="
python scripts/eval_crpo_ppo.py \
  "${COMMON_EVAL_ARGS[@]}" \
  --eval-split val \
  --per-traj-csv outputs/eval_cmdp/val_selected_per_traj.csv \
  --aggregate-csv outputs/eval_cmdp/val_selected_aggregate.csv

echo
echo "========== Test eval: traj9,traj10 =========="
python scripts/eval_crpo_ppo.py \
  "${COMMON_EVAL_ARGS[@]}" \
  --eval-split test \
  --per-traj-csv outputs/eval_cmdp/test_selected_per_traj.csv \
  --aggregate-csv outputs/eval_cmdp/test_selected_aggregate.csv

echo
echo "========== Validation aggregate =========="
cat outputs/eval_cmdp/val_selected_aggregate.csv

echo
echo "========== Test aggregate =========="
cat outputs/eval_cmdp/test_selected_aggregate.csv

echo
echo "Saved:"
echo "  outputs/eval_cmdp/val_selected_aggregate.csv"
echo "  outputs/eval_cmdp/val_selected_per_traj.csv"
echo "  outputs/eval_cmdp/test_selected_aggregate.csv"
echo "  outputs/eval_cmdp/test_selected_per_traj.csv"
