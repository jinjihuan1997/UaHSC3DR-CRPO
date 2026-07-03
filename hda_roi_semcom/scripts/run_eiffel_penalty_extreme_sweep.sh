#!/usr/bin/env bash
set -euo pipefail

# Extreme PPO-penalty weight sweep (review fix M1): mu_R in {0.1, 5.0}.
# Mirrors run_eiffel_penalty_multiseed_sweep.sh (same seeds, tables,
# normalization, and evaluation protocol) and writes into the same sweep root
# so the combined summary covers mu_R in {0.1, 0.5, 1.0, 1.5, 5.0}.

cd /home/king/Downloads/Projects/TCOM/hda_roi_semcom

OUT_ROOT="${OUT_ROOT:-outputs/eiffel15_surrogate_test}"
TRAIN_TABLE="${TRAIN_TABLE:-$OUT_ROOT/lookups/fit_eiffel.csv}"
TEST_TABLE="${TEST_TABLE:-$OUT_ROOT/lookups/test_eiffel.csv}"
TRAIN_TRAJ_CSV="${TRAIN_TRAJ_CSV:-$OUT_ROOT/trajectory_eiffel_train.csv}"
TEST_TRAJ_CSV="${TEST_TRAJ_CSV:-$OUT_ROOT/trajectory_eiffel_test.csv}"
SURROGATE_YAML="${SURROGATE_YAML:-$OUT_ROOT/gsfusion_surrogate_weights_eiffel15_per_traj_composite_testeval.yaml}"
CONFIG="${CONFIG:-configs/default.yaml}"

MAIN_CKPT_ROOT="${MAIN_CKPT_ROOT:-checkpoints/eiffel_multiseed_q070_d035}"
SWEEP_ROOT="${SWEEP_ROOT:-$OUT_ROOT/penalty_multiseed_sweep_q070_d035}"
SWEEP_CKPT_ROOT="${SWEEP_CKPT_ROOT:-checkpoints/eiffel_penalty_multiseed_sweep_q070_d035}"

TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4}"
CHANNEL_SEEDS="${CHANNEL_SEEDS:-100 101 102 103 104}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"

mkdir -p "$SWEEP_ROOT" "$SWEEP_CKPT_ROOT"

REF_NORM=(
  --psnr-ref-min 8.9171
  --psnr-ref-max 22.5791
  --payload-ref-min 78344.0
  --payload-ref-max 644408.0
)

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
  "${REF_NORM[@]}"
  --device "$DEVICE"
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
  "${REF_NORM[@]}"
)

run_penalty_train() {
  local seed="$1" tag="$2" mu_r="$3" mu_d="$4"
  local seed_dir="$SWEEP_ROOT/$tag/seed${seed}"
  local ckpt_dir="$SWEEP_CKPT_ROOT/$tag/seed${seed}"
  local out="$ckpt_dir/penalty.pt"
  local log_csv="$seed_dir/penalty_train_log.csv"
  mkdir -p "$seed_dir" "$ckpt_dir"
  if [[ "$FORCE" != "1" && -f "$out" && -f "$log_csv" ]]; then
    echo "[skip] train PPO-penalty $tag seed=$seed already exists"
    return
  fi
  echo "========== Train PPO-penalty $tag seed=$seed (mu_R=$mu_r, mu_D=$mu_d) =========="
  python scripts/train_crpo_ppo.py \
    "${COMMON_TRAIN[@]}" \
    --seed "$seed" \
    --objective penalty \
    --penalty-rgb "$mu_r" \
    --penalty-depth "$mu_d" \
    --run-name "penalty_eiffel_${tag}_seed${seed}" \
    --out "$out" \
    --log-csv "$log_csv"
}

run_penalty_eval() {
  local seed="$1" tag="$2" split="$3" table="$4" traj_csv="$5" test_trajs="$6"
  local seed_dir="$SWEEP_ROOT/$tag/seed${seed}"
  local ckpt="$SWEEP_CKPT_ROOT/$tag/seed${seed}/penalty.pt"
  local crpo_ckpt="$MAIN_CKPT_ROOT/seed${seed}/crpo.pt"
  local per_traj_csv="$seed_dir/per_traj_results_${split}.csv"
  local aggregate_csv="$seed_dir/aggregate_results_${split}.csv"
  if [[ "$FORCE" != "1" && -f "$per_traj_csv" && -f "$aggregate_csv" ]]; then
    echo "[skip] eval $tag $split seed=$seed already exists"
    return
  fi
  echo "========== Evaluate PPO-penalty $tag $split seed=$seed =========="
  python scripts/eval_crpo_ppo.py \
    "${COMMON_EVAL[@]}" \
    --table "$table" \
    --trajectory-csv "$traj_csv" \
    --test-trajs "$test_trajs" \
    --seed "${CHANNEL_SEEDS%% *}" \
    --shadow-seeds "${CHANNEL_SEEDS// /,}" \
    --methods PPO-penalty \
    --model "$crpo_ckpt" \
    --ppo-penalty-model "$ckpt" \
    --ppo-penalty-label "PPO-penalty-${tag}" \
    --per-traj-csv "$per_traj_csv" \
    --aggregate-csv "$aggregate_csv" \
    2>&1 | tee "$seed_dir/eval_${split}_result.txt"
}

run_tag() {
  local tag="$1" mu_r="$2" mu_d="$3"
  for seed in $TRAIN_SEEDS; do
    run_penalty_train "$seed" "$tag" "$mu_r" "$mu_d"
    run_penalty_eval "$seed" "$tag" train "$TRAIN_TABLE" "$TRAIN_TRAJ_CSV" \
      traj0,traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8,traj9
    run_penalty_eval "$seed" "$tag" test "$TEST_TABLE" "$TEST_TRAJ_CSV" \
      traj10,traj11,traj12,traj13,traj14
  done
}

run_tag "mu01_mu10" "0.1" "1.0"
run_tag "mu50_mu10" "5.0" "1.0"

SWEEP_ROOT="$SWEEP_ROOT" python - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["SWEEP_ROOT"])
weights = {
    "mu01_mu10": (0.1, 1.0),
    "mu05_mu10": (0.5, 1.0),
    "mu10_mu10": (1.0, 1.0),
    "mu15_mu10": (1.5, 1.0),
    "mu50_mu10": (5.0, 1.0),
}
metric_cols = [
    "avg_q3d", "avg_q_rgb", "avg_r_depth", "J_C_R", "J_C_D",
    "rgb_violation_rate", "depth_violation_rate", "avg_kd", "avg_beta_d",
    "episode_return",
]
for split in ["train", "test"]:
    raw_parts = []
    for tag in weights:
        tag_dir = root / tag
        if not tag_dir.exists():
            continue
        for seed_dir in sorted(tag_dir.glob("seed*")):
            path = seed_dir / f"per_traj_results_{split}.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["penalty_tag"] = tag
            df["mu_R"], df["mu_D"] = weights[tag]
            df["train_seed"] = int(seed_dir.name.removeprefix("seed"))
            raw_parts.append(df)
    raw = pd.concat(raw_parts, ignore_index=True)
    rows = []
    for tag, sub in raw.groupby("penalty_tag", sort=False):
        row = {
            "penalty_tag": tag,
            "mu_R": weights[tag][0],
            "mu_D": weights[tag][1],
            "method": f"PPO-penalty-{tag}",
            "n_runs": len(sub),
        }
        for col in metric_cols:
            values = pd.to_numeric(sub[col], errors="coerce")
            mean = values.mean()
            std = values.std(ddof=1) if values.notna().sum() > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_mean_std"] = f"{mean:.4f} ± {std:.4f}"
        row["rgb_feasible"] = row["J_C_R_mean"] <= 0.05
        row["depth_feasible"] = row["J_C_D_mean"] <= 0.05
        row["overall_feasible"] = row["rgb_feasible"] and row["depth_feasible"]
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("mu_R")
    raw.to_csv(root / f"penalty_extended_sweep_{split}_raw.csv", index=False)
    summary.to_csv(root / f"penalty_extended_sweep_{split}_summary.csv", index=False)
    if split == "test":
        cols = ["method", "mu_R", "mu_D", "avg_q3d_mean_std", "avg_q_rgb_mean_std",
                "avg_r_depth_mean_std", "J_C_R_mean_std", "J_C_D_mean_std", "overall_feasible"]
        print(summary[cols].to_string(index=False))
PY

echo "Done: $SWEEP_ROOT/penalty_extended_sweep_test_summary.csv"
