#!/usr/bin/env bash
# Train and evaluate CRPO-PPO / Lagrangian-PPO / Penalty-PPO on eiffel scene.
# Training: traj0-traj9, Evaluation: traj0-9 (in-dist) + traj10-14 (out-of-dist).
#
# Key fixes vs previous runs:
#   1. Unified p5/p95 normalization from all 15 trajectories: same scale for
#      train and test evaluation (no longer 65 dB vs 34 dB psnr_max mismatch).
#   2. q_req and depth_req calibrated from the empirical train-time
#      expected-shortfall distribution, with epsilon_rgb=epsilon_depth=0.05:
#        q_req=0.70, depth_req=0.35
#      This keeps the RGB constraint meaningful while avoiding the infeasible
#      q_req=0.88/depth_req=0.63 stress-test setting.
set -euo pipefail

OUT_ROOT="outputs/eiffel15_surrogate_test"
TRAIN_TABLE="$OUT_ROOT/lookups/fit_eiffel.csv"
TEST_TABLE="$OUT_ROOT/lookups/test_eiffel.csv"
SURROGATE_YAML="$OUT_ROOT/per_trajectory/eiffel_per_trajectory_composite_app60202_rerun_surrogates.yaml"
TRAIN_TRAJ_CSV="$OUT_ROOT/trajectory_eiffel_train.csv"
TEST_TRAJ_CSV="$OUT_ROOT/trajectory_eiffel_test.csv"
CONFIG="configs/default.yaml"

CKPT_CRPO="checkpoints/crpo_eiffel_composite_app60202_q070_d035.pt"
CKPT_LAG="checkpoints/lagrangian_eiffel_composite_app60202_q070_d035.pt"
CKPT_PEN="checkpoints/penalty_eiffel_composite_app60202_q070_d035.pt"

EVAL_OUT="$OUT_ROOT/eval_composite_app60202_q070_d035"
mkdir -p "$EVAL_OUT" checkpoints

# Unified normalization reference: p5/p95 computed from all 15 trajectories.
# PSNR: p5=8.92 dB, p95=22.58 dB; Depth payload: p5=78344, p95=644408 bits.
REF_NORM="--psnr-ref-min 8.9171 --psnr-ref-max 22.5791
  --payload-ref-min 78344.0 --payload-ref-max 644408.0"

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
  --q-req 0.70
  --depth-req 0.35
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
  --q-req 0.70
  --depth-req 0.35
  --mapping-quality-mode gsfusion_surrogate
  --shadow-sigma-db 4.0
  $REF_NORM"

echo "========== 1. Train CRPO-PPO (eiffel, traj0-9) =========="
python scripts/train_crpo_ppo.py \
  $COMMON_TRAIN \
  --objective crpo \
  --epsilon-rgb 0.05 \
  --epsilon-depth 0.05 \
  --run-name crpo_eiffel_per_traj \
  --out "$CKPT_CRPO" \
  --log-csv "$EVAL_OUT/crpo_train_log.csv"

echo "========== 2. Train Lagrangian-PPO (eiffel, traj0-9) =========="
python scripts/train_crpo_ppo.py \
  $COMMON_TRAIN \
  --objective lagrangian \
  --lr-dual 0.005 \
  --run-name lagrangian_eiffel_per_traj \
  --out "$CKPT_LAG" \
  --log-csv "$EVAL_OUT/lagrangian_train_log.csv"

echo "========== 3. Train Penalty-PPO (fixed lambda=1.0, eiffel, traj0-9) =========="
python scripts/train_crpo_ppo.py \
  $COMMON_TRAIN \
  --objective penalty \
  --penalty-rgb 1.0 \
  --penalty-depth 1.0 \
  --run-name penalty_eiffel_per_traj \
  --out "$CKPT_PEN" \
  --log-csv "$EVAL_OUT/penalty_train_log.csv"

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
  --per-traj-csv "$EVAL_OUT/per_traj_results_train.csv" \
  --aggregate-csv "$EVAL_OUT/aggregate_results_train.csv" \
  2>&1 | tee "$EVAL_OUT/eval_train_result.txt"

echo "========== 5. Evaluate on TEST trajectories (traj10-14, out-of-distribution) =========="
python scripts/eval_crpo_ppo.py \
  $COMMON_EVAL_ARGS \
  --table $TEST_TABLE \
  --trajectory-csv $TEST_TRAJ_CSV \
  --test-trajs traj10,traj11,traj12,traj13,traj14 \
  --seed 123 \
  --model "$CKPT_CRPO" \
  --ppo-penalty-model "$CKPT_PEN" \
  --lagrangian-model "$CKPT_LAG" \
  --per-traj-csv "$EVAL_OUT/per_traj_results_test.csv" \
  --aggregate-csv "$EVAL_OUT/aggregate_results_test.csv" \
  2>&1 | tee "$EVAL_OUT/eval_test_result.txt"

echo "========== Done =========="
echo "Train eval: $EVAL_OUT/aggregate_results_train.csv"
echo "Test eval:  $EVAL_OUT/aggregate_results_test.csv"
