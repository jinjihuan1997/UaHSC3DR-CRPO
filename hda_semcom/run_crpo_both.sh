#!/usr/bin/env bash
set -euo pipefail

TABLE="outputs/offline_lookup_fushimi_traj10_stage3_grid.csv"
CKPT_LINEAR="checkpoints/crpo_fushimi_traj10_linear_v4.pt"
CKPT_SAT="checkpoints/crpo_fushimi_traj10_saturation_v4.pt"
TRAJ_CSV="outputs/trajectory_fushimi_a2g.csv"
EVAL_CONFIG="configs/crpo_saturation.yaml"   # 评估统一用更准确的 saturation 代理

# depth_req 从 0.8 降至 0.35：查表验证 0.8 在 SNR≈10dB 时物理不可达，
# 0.35 在 62% 的状态下可以通过最优动作满足，约束切换机制才能正常工作。
COMMON_TRAIN="--table $TABLE
  --trajectory-csv $TRAJ_CSV
  --trajectory-reset trajectory
  --shadow-sigma-db 4.0
  --episode-len 102
  --timesteps 200000
  --n-steps 2048
  --batch-size 64
  --update-epochs 10
  --lr 3e-4
  --q-req 0.6
  --depth-req 0.35
  --k-d-choices 10,12,14,16,18,20,22
  --mapping-quality-mode gsfusion_surrogate
  --seed 42
  --device cuda"

COMMON_EVAL="--table $TABLE
  --trajectory-csv $TRAJ_CSV
  --trajectory-reset trajectory
  --shadow-sigma-db 4.0
  --config $EVAL_CONFIG
  --episodes 50
  --episode-len 102
  --q-req 0.6
  --depth-req 0.35
  --mapping-quality-mode gsfusion_surrogate
  --seed 123"

echo "========== 1. Train CRPO (linear surrogate) =========="
python scripts/train_crpo_ppo.py \
  $COMMON_TRAIN \
  --config configs/crpo_linear.yaml \
  --run-name crpo_fushimi_traj10_linear_v4 \
  --out "$CKPT_LINEAR" \
  --log-csv outputs/ppo_logs/crpo_linear_v4/crpo_ppo_log.csv

echo "========== 2. Train CRPO (saturation surrogate) =========="
python scripts/train_crpo_ppo.py \
  $COMMON_TRAIN \
  --config configs/crpo_saturation.yaml \
  --run-name crpo_fushimi_traj10_saturation_v4 \
  --out "$CKPT_SAT" \
  --log-csv outputs/ppo_logs/crpo_saturation_v4/crpo_ppo_log.csv

echo "========== 3. Eval — linear policy (vs saturation surrogate) =========="
python scripts/eval_crpo_ppo.py \
  $COMMON_EVAL \
  --model "$CKPT_LINEAR" \
  2>&1 | tee outputs/ppo_logs/crpo_linear_v4/eval_result.txt

echo "========== 4. Eval — saturation policy (vs saturation surrogate) =========="
python scripts/eval_crpo_ppo.py \
  $COMMON_EVAL \
  --model "$CKPT_SAT" \
  2>&1 | tee outputs/ppo_logs/crpo_saturation_v4/eval_result.txt

echo "========== Done =========="
echo "Linear    ckpt : $CKPT_LINEAR"
echo "Saturation ckpt: $CKPT_SAT"
echo "Eval logs: outputs/ppo_logs/"
