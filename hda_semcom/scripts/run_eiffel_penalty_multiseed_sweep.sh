#!/usr/bin/env bash
set -euo pipefail

# Multi-seed PPO-penalty weight sweep for Section 6.3.
# This script complements run_eiffel_multiseed_training.sh by training/evaluating
# multiple fixed penalty weights under the same seeds and evaluation protocol.

cd /home/king/Downloads/Projects/TCOM/hda_semcom

OUT_ROOT="${OUT_ROOT:-outputs/eiffel15_surrogate_test}"
TRAIN_TABLE="${TRAIN_TABLE:-$OUT_ROOT/lookups/fit_eiffel.csv}"
TEST_TABLE="${TEST_TABLE:-$OUT_ROOT/lookups/test_eiffel.csv}"
TRAIN_TRAJ_CSV="${TRAIN_TRAJ_CSV:-$OUT_ROOT/trajectory_eiffel_train.csv}"
TEST_TRAJ_CSV="${TEST_TRAJ_CSV:-$OUT_ROOT/trajectory_eiffel_test.csv}"
SURROGATE_YAML="${SURROGATE_YAML:-$OUT_ROOT/gsfusion_surrogate_weights_eiffel15_per_traj_composite_testeval.yaml}"
CONFIG="${CONFIG:-configs/default.yaml}"

MAIN_MULTI_ROOT="${MAIN_MULTI_ROOT:-$OUT_ROOT/multiseed_q070_d035}"
MAIN_CKPT_ROOT="${MAIN_CKPT_ROOT:-checkpoints/eiffel_multiseed_q070_d035}"
SWEEP_ROOT="${SWEEP_ROOT:-$OUT_ROOT/penalty_multiseed_sweep_q070_d035}"
SWEEP_CKPT_ROOT="${SWEEP_CKPT_ROOT:-checkpoints/eiffel_penalty_multiseed_sweep_q070_d035}"

TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4}"
CHANNEL_SEEDS="${CHANNEL_SEEDS:-100 101 102 103 104}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"
REUSE_DEFAULT_MU10="${REUSE_DEFAULT_MU10:-1}"

mkdir -p "$SWEEP_ROOT" "$SWEEP_CKPT_ROOT"

required_files=(
  "$TRAIN_TABLE"
  "$TEST_TABLE"
  "$TRAIN_TRAJ_CSV"
  "$TEST_TRAJ_CSV"
  "$SURROGATE_YAML"
  "$CONFIG"
)
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "[missing] $path" >&2
    exit 1
  fi
done

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
  local seed="$1"
  local tag="$2"
  local mu_r="$3"
  local mu_d="$4"
  local seed_dir="$SWEEP_ROOT/$tag/seed${seed}"
  local ckpt_dir="$SWEEP_CKPT_ROOT/$tag/seed${seed}"
  local out="$ckpt_dir/penalty.pt"
  local log_csv="$seed_dir/penalty_train_log.csv"
  mkdir -p "$seed_dir" "$ckpt_dir"

  if [[ "$tag" == "mu10_mu10" && "$REUSE_DEFAULT_MU10" == "1" ]]; then
    local src_ckpt="$MAIN_CKPT_ROOT/seed${seed}/penalty.pt"
    local src_log="$MAIN_MULTI_ROOT/seed${seed}/penalty_train_log.csv"
    if [[ ! -f "$src_ckpt" || ! -f "$src_log" ]]; then
      echo "[missing] default mu10/mu10 source for seed=$seed; run ./scripts/run_eiffel_multiseed_training.sh first" >&2
      exit 1
    fi
    cp "$src_ckpt" "$out"
    cp "$src_log" "$log_csv"
    echo "[reuse] PPO-penalty $tag seed=$seed from $MAIN_MULTI_ROOT/seed${seed}"
    return
  fi

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
  local seed="$1"
  local tag="$2"
  local split="$3"
  local table="$4"
  local traj_csv="$5"
  local test_trajs="$6"
  local seed_dir="$SWEEP_ROOT/$tag/seed${seed}"
  local ckpt="$SWEEP_CKPT_ROOT/$tag/seed${seed}/penalty.pt"
  local crpo_ckpt="$MAIN_CKPT_ROOT/seed${seed}/crpo.pt"
  local per_traj_csv="$seed_dir/per_traj_results_${split}.csv"
  local aggregate_csv="$seed_dir/aggregate_results_${split}.csv"
  if [[ ! -f "$crpo_ckpt" ]]; then
    echo "[missing] reference CRPO checkpoint for seed=$seed: $crpo_ckpt" >&2
    exit 1
  fi
  if [[ "$FORCE" != "1" && -f "$per_traj_csv" && -f "$aggregate_csv" ]]; then
    echo "[skip] eval $tag $split seed=$seed already exists"
    return
  fi
  echo "========== Evaluate PPO-penalty $tag $split seed=$seed channel_seeds=${CHANNEL_SEEDS} =========="
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
  local tag="$1"
  local mu_r="$2"
  local mu_d="$3"
  for seed in $TRAIN_SEEDS; do
    run_penalty_train "$seed" "$tag" "$mu_r" "$mu_d"
    run_penalty_eval "$seed" "$tag" train "$TRAIN_TABLE" "$TRAIN_TRAJ_CSV" \
      traj0,traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8,traj9
    run_penalty_eval "$seed" "$tag" test "$TEST_TABLE" "$TEST_TRAJ_CSV" \
      traj10,traj11,traj12,traj13,traj14
  done
}

run_tag "mu05_mu10" "0.5" "1.0"
run_tag "mu10_mu10" "1.0" "1.0"
run_tag "mu15_mu10" "1.5" "1.0"

SWEEP_ROOT="$SWEEP_ROOT" python - <<'PY'
from pathlib import Path
import os
import pandas as pd

root = Path(os.environ["SWEEP_ROOT"])
weights = {
    "mu05_mu10": (0.5, 1.0),
    "mu10_mu10": (1.0, 1.0),
    "mu15_mu10": (1.5, 1.0),
}
metric_cols = [
    "avg_q3d", "avg_q_rgb", "avg_r_depth", "J_C_R", "J_C_D",
    "rgb_violation_rate", "depth_violation_rate", "avg_kd", "avg_beta_d",
    "episode_return",
]
for split in ["train", "test"]:
    raw_parts = []
    for tag in weights:
        for seed_dir in sorted((root / tag).glob("seed*")):
            path = seed_dir / f"per_traj_results_{split}.csv"
            if not path.exists():
                raise SystemExit(f"missing {path}")
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
            "n_training_seeds": sub["train_seed"].nunique(),
            "training_seeds": ",".join(str(int(x)) for x in sorted(sub["train_seed"].unique())),
            "n_channel_seeds": sub["shadow_seed"].nunique(),
            "channel_seeds": ",".join(str(int(x)) for x in sorted(sub["shadow_seed"].unique())),
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
    summary = pd.DataFrame(rows)
    raw.to_csv(root / f"penalty_multiseed_sweep_{split}_raw.csv", index=False)
    summary.to_csv(root / f"penalty_multiseed_sweep_{split}_summary.csv", index=False)
    print(f"[wrote] {root / f'penalty_multiseed_sweep_{split}_summary.csv'}")
    if split == "test":
        cols = ["method", "mu_R", "mu_D", "avg_q3d_mean_std", "avg_q_rgb_mean_std", "avg_r_depth_mean_std",
                "J_C_R_mean_std", "J_C_D_mean_std", "overall_feasible"]
        print(summary[cols].to_string(index=False))
PY

echo "========== Done =========="
echo "Sweep root: $SWEEP_ROOT"
echo "Test summary: $SWEEP_ROOT/penalty_multiseed_sweep_test_summary.csv"
