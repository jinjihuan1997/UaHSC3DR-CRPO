#!/usr/bin/env bash
set -euo pipefail

cd /home/king/Downloads/Projects/TCOM/hda_semcom

mkdir -p checkpoints outputs/ppo_logs outputs/eval_cmdp

TABLE="outputs/offline_lookup_fushimi_traj0_10_stage3_grid.csv"
CONFIG="configs/default.yaml"
TRAJ_CSV="outputs/trajectory_fushimi_a2g.csv"

COMMON_TRAIN_ARGS=(
  --table "$TABLE"
  --config "$CONFIG"
  --trajectory-csv "$TRAJ_CSV"
  --multi-trajectory
  --q-req 0.65
  --depth-req 0.35
  --epsilon-rgb 0.20
  --epsilon-depth 0.05
  --timesteps 1000000
  --n-steps 4096
  --batch-size 256
  --update-epochs 10
  --lr 3e-4
  --seed 42
  --device cuda
)

COMMON_EVAL_ARGS=(
  --table "$TABLE"
  --config "$CONFIG"
  --trajectory-csv "$TRAJ_CSV"
  --multi-trajectory
  --shadow-seeds 101,102,103,104,105
  --episodes 20
  --q-req 0.65
  --depth-req 0.35
  --model checkpoints/crpo_standard.pt
  --ppo-penalty-model checkpoints/ppo_penalty_standard.pt
  --device cuda
)

echo "========== 1. Train CRPO-PPO on train trajectories traj0-traj6 =========="
python scripts/train_crpo_ppo.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --objective crpo \
  --out checkpoints/crpo_standard.pt \
  --log-csv outputs/ppo_logs/crpo_standard.csv

echo "========== 2. Train PPO-penalty on train trajectories traj0-traj6 =========="
python scripts/train_crpo_ppo.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --objective penalty \
  --penalty-rgb 5.0 \
  --penalty-depth 1.0 \
  --out checkpoints/ppo_penalty_standard.pt \
  --log-csv outputs/ppo_logs/ppo_penalty_standard.csv

echo "========== 3. Validation eval on traj7-traj8 =========="
python scripts/eval_crpo_ppo.py \
  "${COMMON_EVAL_ARGS[@]}" \
  --eval-split val \
  --per-traj-csv outputs/eval_cmdp/val_standard_per_traj.csv \
  --aggregate-csv outputs/eval_cmdp/val_standard_aggregate.csv

echo "========== 4. Final test eval on traj9-traj10 =========="
python scripts/eval_crpo_ppo.py \
  "${COMMON_EVAL_ARGS[@]}" \
  --eval-split test \
  --per-traj-csv outputs/eval_cmdp/test_standard_per_traj.csv \
  --aggregate-csv outputs/eval_cmdp/test_standard_aggregate.csv

echo
echo "========== Validation aggregate =========="
cat outputs/eval_cmdp/val_standard_aggregate.csv

echo
echo "========== Test aggregate =========="
cat outputs/eval_cmdp/test_standard_aggregate.csv

echo
echo "Saved:"
echo "  checkpoints/crpo_standard.pt"
echo "  checkpoints/ppo_penalty_standard.pt"
echo "  outputs/ppo_logs/crpo_standard.csv"
echo "  outputs/ppo_logs/ppo_penalty_standard.csv"
echo "  outputs/eval_cmdp/val_standard_aggregate.csv"
echo "  outputs/eval_cmdp/test_standard_aggregate.csv"
