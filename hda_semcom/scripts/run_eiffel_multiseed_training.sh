#!/usr/bin/env bash
set -euo pipefail

# Five-seed Eiffel training/evaluation for CRPO-PPO, PPO-penalty, and
# Lagrangian-PPO. Evaluation expands independent channel/shadowing seeds.
# Random allocation additionally expands independent action-sampling seeds.

cd /home/king/Downloads/Projects/TCOM/hda_semcom

OUT_ROOT="${OUT_ROOT:-outputs/eiffel15_surrogate_test}"
TRAIN_TABLE="${TRAIN_TABLE:-$OUT_ROOT/lookups/fit_eiffel.csv}"
TEST_TABLE="${TEST_TABLE:-$OUT_ROOT/lookups/test_eiffel.csv}"
TRAIN_TRAJ_CSV="${TRAIN_TRAJ_CSV:-$OUT_ROOT/trajectory_eiffel_train.csv}"
TEST_TRAJ_CSV="${TEST_TRAJ_CSV:-$OUT_ROOT/trajectory_eiffel_test.csv}"
SURROGATE_YAML="${SURROGATE_YAML:-$OUT_ROOT/gsfusion_surrogate_weights_eiffel15_per_traj_composite_testeval.yaml}"
CONFIG="${CONFIG:-configs/default.yaml}"

MULTISEED_ROOT="${MULTISEED_ROOT:-$OUT_ROOT/multiseed_q070_d035}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints/eiffel_multiseed_q070_d035}"
TRAIN_SEEDS="${TRAIN_SEEDS:-0 1 2 3 4}"
CHANNEL_SEEDS="${CHANNEL_SEEDS:-100 101 102 103 104}"
ACTION_SEEDS="${ACTION_SEEDS:-0 1 2 3 4}"
DEVICE="${DEVICE:-cuda}"
FORCE="${FORCE:-0}"

mkdir -p "$MULTISEED_ROOT" "$CKPT_ROOT"

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

run_train() {
  local seed="$1"
  local objective="$2"
  local out="$3"
  local log_csv="$4"
  shift 4
  if [[ "$FORCE" != "1" && -f "$out" && -f "$log_csv" ]]; then
    echo "[skip] train $objective seed=$seed already exists"
    return
  fi
  echo "========== Train $objective seed=$seed =========="
  python scripts/train_crpo_ppo.py \
    "${COMMON_TRAIN[@]}" \
    --seed "$seed" \
    --objective "$objective" \
    "$@" \
    --out "$out" \
    --log-csv "$log_csv"
}

run_eval() {
  local seed="$1"
  local split="$2"
  local table="$3"
  local traj_csv="$4"
  local test_trajs="$5"
  local out_dir="$6"
  local crpo_ckpt="$7"
  local penalty_ckpt="$8"
  local lag_ckpt="$9"
  local per_traj_csv="$out_dir/per_traj_results_${split}.csv"
  local aggregate_csv="$out_dir/aggregate_results_${split}.csv"
  if [[ "$FORCE" != "1" && -f "$per_traj_csv" && -f "$aggregate_csv" ]]; then
    echo "[skip] eval $split seed=$seed already exists"
    return
  fi
  echo "========== Evaluate $split train_seed=$seed channel_seeds=${CHANNEL_SEEDS} =========="
  python scripts/eval_crpo_ppo.py \
    "${COMMON_EVAL[@]}" \
    --table "$table" \
    --trajectory-csv "$traj_csv" \
    --test-trajs "$test_trajs" \
    --seed "${CHANNEL_SEEDS%% *}" \
    --shadow-seeds "${CHANNEL_SEEDS// /,}" \
    --methods CRPO-PPO,PPO-penalty,Lagrangian-PPO \
    --model "$crpo_ckpt" \
    --ppo-penalty-model "$penalty_ckpt" \
    --lagrangian-model "$lag_ckpt" \
    --per-traj-csv "$per_traj_csv" \
    --aggregate-csv "$aggregate_csv" \
    2>&1 | tee "$out_dir/eval_${split}_result.txt"
}

run_baseline_eval() {
  local split="$1"
  local table="$2"
  local traj_csv="$3"
  local test_trajs="$4"
  local out_dir="$5"
  local reference_ckpt="$6"
  local per_traj_csv="$out_dir/per_traj_results_${split}.csv"
  local aggregate_csv="$out_dir/aggregate_results_${split}.csv"
  if [[ "$FORCE" != "1" && -f "$per_traj_csv" && -f "$aggregate_csv" ]]; then
    echo "[skip] baseline eval $split already exists"
    return
  fi
  echo "========== Evaluate baselines $split =========="
  python scripts/eval_crpo_ppo.py \
    "${COMMON_EVAL[@]}" \
    --table "$table" \
    --trajectory-csv "$traj_csv" \
    --test-trajs "$test_trajs" \
    --seed "${CHANNEL_SEEDS%% *}" \
    --shadow-seeds "${CHANNEL_SEEDS// /,}" \
    --random-action-seeds "${ACTION_SEEDS// /,}" \
    --methods fixed-balanced,RGB-priority,depth-priority,random \
    --model "$reference_ckpt" \
    --per-traj-csv "$per_traj_csv" \
    --aggregate-csv "$aggregate_csv" \
    2>&1 | tee "$out_dir/eval_${split}_result.txt"
}

for seed in $TRAIN_SEEDS; do
  seed_dir="$MULTISEED_ROOT/seed${seed}"
  ckpt_dir="$CKPT_ROOT/seed${seed}"
  mkdir -p "$seed_dir" "$ckpt_dir"

  crpo_ckpt="$ckpt_dir/crpo.pt"
  penalty_ckpt="$ckpt_dir/penalty.pt"
  lag_ckpt="$ckpt_dir/lagrangian.pt"

  run_train "$seed" crpo "$crpo_ckpt" "$seed_dir/crpo_train_log.csv" \
    --epsilon-rgb 0.05 \
    --epsilon-depth 0.05 \
    --run-name "crpo_eiffel_seed${seed}"

  run_train "$seed" penalty "$penalty_ckpt" "$seed_dir/penalty_train_log.csv" \
    --penalty-rgb 1.0 \
    --penalty-depth 1.0 \
    --run-name "penalty_eiffel_seed${seed}"

  run_train "$seed" lagrangian "$lag_ckpt" "$seed_dir/lagrangian_train_log.csv" \
    --lr-dual 0.005 \
    --run-name "lagrangian_eiffel_seed${seed}"

  run_eval "$seed" train "$TRAIN_TABLE" "$TRAIN_TRAJ_CSV" \
    traj0,traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8,traj9 \
    "$seed_dir" "$crpo_ckpt" "$penalty_ckpt" "$lag_ckpt"

  run_eval "$seed" test "$TEST_TABLE" "$TEST_TRAJ_CSV" \
    traj10,traj11,traj12,traj13,traj14 \
    "$seed_dir" "$crpo_ckpt" "$penalty_ckpt" "$lag_ckpt"
done

baseline_dir="$MULTISEED_ROOT/baselines"
mkdir -p "$baseline_dir"
first_seed="${TRAIN_SEEDS%% *}"
reference_ckpt="$CKPT_ROOT/seed${first_seed}/crpo.pt"
if [[ ! -f "$reference_ckpt" ]]; then
  echo "[missing] reference checkpoint for baseline action space: $reference_ckpt" >&2
  exit 1
fi
run_baseline_eval train "$TRAIN_TABLE" "$TRAIN_TRAJ_CSV" \
  traj0,traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8,traj9 \
  "$baseline_dir" "$reference_ckpt"
run_baseline_eval test "$TEST_TABLE" "$TEST_TRAJ_CSV" \
  traj10,traj11,traj12,traj13,traj14 \
  "$baseline_dir" "$reference_ckpt"

python tools/summarize_eiffel_multiseed_results.py \
  --root "$MULTISEED_ROOT" \
  --out "$MULTISEED_ROOT/multiseed_summary"

echo "========== Done =========="
echo "Per-seed outputs: $MULTISEED_ROOT/seed{0,1,2,3,4}/"
echo "Baseline outputs: $MULTISEED_ROOT/baselines/"
echo "Summary: $MULTISEED_ROOT/multiseed_summary_test.csv"
