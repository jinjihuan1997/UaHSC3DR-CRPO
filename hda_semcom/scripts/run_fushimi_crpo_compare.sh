#!/usr/bin/env bash
# Train and evaluate CRPO-PPO / Lagrangian-PPO / Penalty-PPO on fushimi scene.
# Training: traj0-traj9 (10 trajs), Evaluation: traj0-9 (in-dist) + traj10 (out-of-dist).
#
# Normalization: p5/p95 from all 11 fushimi trajectories.
#   PSNR: p5=13.65 dB, p95=23.00 dB
#   Depth: p5=176864, p95=797504 bits
# Constraint thresholds calibrated from the empirical train-time
# expected-shortfall distribution, with epsilon_rgb=epsilon_depth=0.05:
#   q_req=0.60, depth_req=0.27
# Fushimi has much heavier depth payloads than Eiffel, so the feasible depth
# threshold is substantially lower than in the Eiffel setting.
set -euo pipefail

OUT_ROOT="outputs/fushimi_surrogate_test"
TRAIN_TABLE="$OUT_ROOT/lookups/fit_fushimi.csv"
TEST_TABLE="$OUT_ROOT/lookups/test_fushimi.csv"
SURROGATE_YAML="$OUT_ROOT/per_trajectory/fushimi_per_trajectory_composite_app60202_rerun_surrogates.yaml"
TRAIN_TRAJ_CSV="$OUT_ROOT/trajectory_fushimi_train.csv"
TEST_TRAJ_CSV="$OUT_ROOT/trajectory_fushimi_test.csv"
CONFIG="configs/default.yaml"

CKPT_CRPO="checkpoints/crpo_fushimi_composite_app60202_q060_d027.pt"
CKPT_LAG="checkpoints/lagrangian_fushimi_composite_app60202_q060_d027.pt"
CKPT_PEN="checkpoints/penalty_fushimi_composite_app60202_q060_d027.pt"

EVAL_OUT="$OUT_ROOT/eval_composite_app60202_q060_d027"
mkdir -p "$EVAL_OUT" checkpoints

# Unified normalization reference: p5/p95 from all 11 fushimi trajectories.
REF_NORM="--psnr-ref-min 13.6451 --psnr-ref-max 23.0013
  --payload-ref-min 176864.0 --payload-ref-max 797504.0"

COMMON_TRAIN="--table $TRAIN_TABLE
  --config $CONFIG
  --trajectory-csv $TRAIN_TRAJ_CSV
  --per-trajectory-surrogate-yaml $SURROGATE_YAML
  --trajectory-reset trajectory
  --episode-len 102
  --timesteps 300000
  --n-steps 2048
  --batch-size 64
  --update-epochs 10
  --lr 3e-4
  --q-req 0.60
  --depth-req 0.27
  --k-d-choices 10,12,14,16,18,20,22
  --mapping-quality-mode gsfusion_surrogate
  --shadow-sigma-db 4.0
  $REF_NORM
  --seed 42
  --device cuda"

COMMON_EVAL_ARGS="--config $CONFIG
  --per-trajectory-surrogate-yaml $SURROGATE_YAML
  --trajectory-reset trajectory
  --multi-trajectory
  --episodes 20
  --episode-len 102
  --q-req 0.60
  --depth-req 0.27
  --mapping-quality-mode gsfusion_surrogate
  --shadow-sigma-db 4.0
  $REF_NORM"

echo "========== 1. Train CRPO-PPO (fushimi, traj0-9) =========="
python scripts/train_crpo_ppo.py \
  $COMMON_TRAIN \
  --objective crpo \
  --epsilon-rgb 0.05 \
  --epsilon-depth 0.05 \
  --run-name crpo_fushimi_per_traj \
  --out "$CKPT_CRPO" \
  --log-csv "$EVAL_OUT/crpo_fushimi_train_log.csv"

echo "========== 2. Train Lagrangian-PPO (fushimi, traj0-9) =========="
python scripts/train_crpo_ppo.py \
  $COMMON_TRAIN \
  --objective lagrangian \
  --lr-dual 0.005 \
  --run-name lagrangian_fushimi_per_traj \
  --out "$CKPT_LAG" \
  --log-csv "$EVAL_OUT/lagrangian_fushimi_train_log.csv"

echo "========== 3. Train Penalty-PPO (fixed lambda=1.0, fushimi, traj0-9) =========="
python scripts/train_crpo_ppo.py \
  $COMMON_TRAIN \
  --objective penalty \
  --penalty-rgb 1.0 \
  --penalty-depth 1.0 \
  --run-name penalty_fushimi_per_traj \
  --out "$CKPT_PEN" \
  --log-csv "$EVAL_OUT/penalty_fushimi_train_log.csv"

echo "========== 4. Evaluate on TRAIN trajectories (traj0-9, in-distribution) =========="
python scripts/eval_crpo_ppo.py \
  $COMMON_EVAL_ARGS \
  --table $TRAIN_TABLE \
  --trajectory-csv $TRAIN_TRAJ_CSV \
  --test-trajs traj0,traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8,traj9 \
  --seed 42 \
  --model "$CKPT_CRPO" \
  --ppo-penalty-model "$CKPT_PEN" \
  --lagrangian-model "$CKPT_LAG" \
  --per-traj-csv "$EVAL_OUT/per_traj_results_fushimi_train.csv" \
  --aggregate-csv "$EVAL_OUT/aggregate_results_fushimi_train.csv" \
  2>&1 | tee "$EVAL_OUT/eval_fushimi_train_result.txt"

echo "========== 5. Evaluate on TEST trajectory (traj10, out-of-distribution) =========="
python scripts/eval_crpo_ppo.py \
  $COMMON_EVAL_ARGS \
  --table $TEST_TABLE \
  --trajectory-csv $TEST_TRAJ_CSV \
  --test-trajs traj10 \
  --seed 123 \
  --model "$CKPT_CRPO" \
  --ppo-penalty-model "$CKPT_PEN" \
  --lagrangian-model "$CKPT_LAG" \
  --per-traj-csv "$EVAL_OUT/per_traj_results_fushimi_test.csv" \
  --aggregate-csv "$EVAL_OUT/aggregate_results_fushimi_test.csv" \
  2>&1 | tee "$EVAL_OUT/eval_fushimi_test_result.txt"

echo "========== Done =========="
echo "Train eval: $EVAL_OUT/aggregate_results_fushimi_train.csv"
echo "Test eval:  $EVAL_OUT/aggregate_results_fushimi_test.csv"
