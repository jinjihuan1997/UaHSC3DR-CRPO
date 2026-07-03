#!/usr/bin/env bash
set -euo pipefail

cd /home/king/Downloads/Projects/TCOM/hda_roi_semcom

TRAIN_TABLE="outputs/eiffel15_surrogate_test/lookups/fit_eiffel.csv"
TEST_TABLE="outputs/eiffel15_surrogate_test/lookups/test_eiffel.csv"
CONFIG="configs/default.yaml"
TRAIN_TRAJ_CSV="outputs/eiffel15_surrogate_test/trajectory_eiffel_train.csv"
TEST_TRAJ_CSV="outputs/eiffel15_surrogate_test/trajectory_eiffel_test.csv"
SURROGATE_YAML="outputs/eiffel15_surrogate_test/per_trajectory/eiffel_per_trajectory_composite_app60202_rerun_surrogates.yaml"
CKPT_CRPO="checkpoints/crpo_eiffel_composite_app60202_q070_d035.pt"

OUT_SWEEP="outputs/eiffel15_surrogate_test/penalty_weight_sweep_q070_d035_muR_sweep_muD1"
mkdir -p "$OUT_SWEEP" checkpoints

for required in "$TRAIN_TABLE" "$TEST_TABLE" "$CONFIG" "$TRAIN_TRAJ_CSV" "$TEST_TRAJ_CSV" "$SURROGATE_YAML" "$CKPT_CRPO"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: required file not found: $required" >&2
    exit 1
  fi
done

COMMON_TRAIN=(
  --table "$TRAIN_TABLE"
  --config "$CONFIG"
  --trajectory-csv "$TRAIN_TRAJ_CSV"
  --per-trajectory-surrogate-yaml "$SURROGATE_YAML"
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
  --psnr-ref-min 8.9171
  --psnr-ref-max 22.5791
  --payload-ref-min 78344.0
  --payload-ref-max 644408.0
  --seed 42
  --device cuda
)

COMMON_EVAL=(
  --config "$CONFIG"
  --per-trajectory-surrogate-yaml "$SURROGATE_YAML"
  --trajectory-reset trajectory
  --multi-trajectory
  --episodes 20
  --episode-len 102
  --q-req 0.70
  --depth-req 0.35
  --mapping-quality-mode gsfusion_surrogate
  --shadow-sigma-db 4.0
  --psnr-ref-min 8.9171
  --psnr-ref-max 22.5791
  --payload-ref-min 78344.0
  --payload-ref-max 644408.0
)

run_one() {
  local tag="$1"
  local mu_rgb="$2"
  local mu_depth="$3"
  local ckpt="checkpoints/penalty_eiffel_composite_app60202_q070_d035_${tag}.pt"

  echo "========== Train PPO-penalty ${tag} (mu_R=${mu_rgb}, mu_D=${mu_depth}) =========="
  python scripts/train_crpo_ppo.py \
    "${COMMON_TRAIN[@]}" \
    --objective penalty \
    --penalty-rgb "$mu_rgb" \
    --penalty-depth "$mu_depth" \
    --run-name "penalty_eiffel_${tag}" \
    --out "$ckpt" \
    --log-csv "$OUT_SWEEP/penalty_${tag}_train_log.csv"

  echo "========== Evaluate TRAIN ${tag} =========="
  python scripts/eval_crpo_ppo.py \
    "${COMMON_EVAL[@]}" \
    --table "$TRAIN_TABLE" \
    --trajectory-csv "$TRAIN_TRAJ_CSV" \
    --test-trajs traj0,traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8,traj9 \
    --seed 42 \
    --model "$CKPT_CRPO" \
    --ppo-penalty-model "$ckpt" \
    --ppo-penalty-label "PPO-penalty-${tag}" \
    --per-traj-csv "$OUT_SWEEP/per_traj_train_${tag}.csv" \
    --aggregate-csv "$OUT_SWEEP/aggregate_train_${tag}.csv" \
    2>&1 | tee "$OUT_SWEEP/eval_train_${tag}.txt"

  echo "========== Evaluate TEST ${tag} =========="
  python scripts/eval_crpo_ppo.py \
    "${COMMON_EVAL[@]}" \
    --table "$TEST_TABLE" \
    --trajectory-csv "$TEST_TRAJ_CSV" \
    --test-trajs traj10,traj11,traj12,traj13,traj14 \
    --seed 123 \
    --model "$CKPT_CRPO" \
    --ppo-penalty-model "$ckpt" \
    --ppo-penalty-label "PPO-penalty-${tag}" \
    --per-traj-csv "$OUT_SWEEP/per_traj_test_${tag}.csv" \
    --aggregate-csv "$OUT_SWEEP/aggregate_test_${tag}.csv" \
    2>&1 | tee "$OUT_SWEEP/eval_test_${tag}.txt"
}

run_one "mu05_mu10" "0.5" "1.0"
run_one "mu10_mu10" "1.0" "1.0"
run_one "mu15_mu10" "1.5" "1.0"

echo "========== Summarize TEST penalty sweep =========="
python - <<'PY'
from pathlib import Path
import pandas as pd

out = Path("outputs/eiffel15_surrogate_test/penalty_weight_sweep_q070_d035_muR_sweep_muD1")
rows = []
for tag, weights in [
    ("mu05_mu10", "0.5/1.0"),
    ("mu10_mu10", "1.0/1.0"),
    ("mu15_mu10", "1.5/1.0"),
]:
    path = out / f"aggregate_test_{tag}.csv"
    df = pd.read_csv(path)
    row = df[df["method"].str.contains("PPO-penalty", na=False)].iloc[0].to_dict()
    row["penalty_weights"] = weights
    rows.append(row)

summary = pd.DataFrame(rows)
cols = [
    "penalty_weights", "method", "avg_q3d", "avg_q_rgb", "avg_r_depth",
    "J_C_R", "J_C_D", "rgb_violation_rate", "depth_violation_rate",
    "avg_kd", "avg_beta_d", "episode_return",
]
summary = summary[cols]
summary["rgb_feasible"] = summary["J_C_R"] <= 0.05
summary["depth_feasible"] = summary["J_C_D"] <= 0.05
summary["overall_feasible"] = summary["rgb_feasible"] & summary["depth_feasible"]
summary.to_csv(out / "penalty_weight_sweep_test_summary.csv", index=False)
print(summary.to_string(index=False))
print(f"Saved: {out / 'penalty_weight_sweep_test_summary.csv'}")
PY

echo "========== Done =========="
echo "Summary: $OUT_SWEEP/penalty_weight_sweep_test_summary.csv"
