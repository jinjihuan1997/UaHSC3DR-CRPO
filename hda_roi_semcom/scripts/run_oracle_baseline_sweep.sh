#!/usr/bin/env bash
set -euo pipefail

# Per-slot oracle baselines (C1 review fix):
# 1) Tune per-slot Lagrangian multipliers on the train split (grid search).
# 2) Evaluate greedy-oracle and the tuned per-slot Lagrangian on the test
#    split with the same protocol as the paper (5 channel seeds, 20 episodes).

cd /home/king/Downloads/Projects/TCOM/hda_roi_semcom

OUT_ROOT="outputs/eiffel15_surrogate_test"
TRAIN_TABLE="$OUT_ROOT/lookups/fit_eiffel.csv"
TEST_TABLE="$OUT_ROOT/lookups/test_eiffel.csv"
TRAIN_TRAJ_CSV="$OUT_ROOT/trajectory_eiffel_train.csv"
TEST_TRAJ_CSV="$OUT_ROOT/trajectory_eiffel_test.csv"
SURROGATE_YAML="$OUT_ROOT/gsfusion_surrogate_weights_eiffel15_per_traj_composite_testeval.yaml"
REF_CKPT="checkpoints/eiffel_multiseed_q070_d035/seed0/crpo.pt"
SWEEP_DIR="$OUT_ROOT/oracle_baseline_q070_d035"
mkdir -p "$SWEEP_DIR"

COMMON=(
  --config configs/default.yaml
  --per-trajectory-surrogate-yaml "$SURROGATE_YAML"
  --trajectory-reset trajectory
  --multi-trajectory
  --episode-len 102
  --q-req 0.70
  --depth-req 0.35
  --mapping-quality-mode gsfusion_surrogate
  --shadow-sigma-db 4.0
  --psnr-ref-min 8.9171
  --psnr-ref-max 22.5791
  --payload-ref-min 78344.0
  --payload-ref-max 644408.0
  --model "$REF_CKPT"
)

# ---- Stage 1: multiplier grid search on the train split ----
for LR in 0.5 1.0 2.0 4.0; do
  for LD in 0.5 1.0 2.0 4.0; do
    tag="lr${LR}_ld${LD}"
    out_csv="$SWEEP_DIR/train_grid_${tag}.csv"
    if [[ -f "$out_csv" ]]; then
      echo "[skip] $tag"
      continue
    fi
    echo "========== train grid $tag =========="
    python scripts/eval_crpo_ppo.py \
      "${COMMON[@]}" \
      --table "$TRAIN_TABLE" \
      --trajectory-csv "$TRAIN_TRAJ_CSV" \
      --test-trajs traj0,traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8,traj9 \
      --episodes 5 \
      --seed 100 --shadow-seeds 100 \
      --methods per-slot-lagrangian \
      --oracle-lambda-r "$LR" --oracle-lambda-d "$LD" \
      --aggregate-csv "$out_csv" \
      > "$SWEEP_DIR/train_grid_${tag}.log" 2>&1
  done
done

# Greedy oracle on the train split for reference.
if [[ ! -f "$SWEEP_DIR/train_greedy.csv" ]]; then
  python scripts/eval_crpo_ppo.py \
    "${COMMON[@]}" \
    --table "$TRAIN_TABLE" \
    --trajectory-csv "$TRAIN_TRAJ_CSV" \
    --test-trajs traj0,traj1,traj2,traj3,traj4,traj5,traj6,traj7,traj8,traj9 \
    --episodes 5 \
    --seed 100 --shadow-seeds 100 \
    --methods greedy-oracle \
    --aggregate-csv "$SWEEP_DIR/train_greedy.csv" \
    > "$SWEEP_DIR/train_greedy.log" 2>&1
fi

echo "Grid search finished. Summary (feasible = J_C_R<=0.05 and J_C_D<=0.05):"
python - "$SWEEP_DIR" <<'EOF'
import csv, glob, os, sys
rows = []
for path in sorted(glob.glob(os.path.join(sys.argv[1], "train_grid_*.csv"))):
    tag = os.path.basename(path)[len("train_grid_"):-len(".csv")]
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append((tag, float(r["avg_q3d"]), float(r["J_C_R"]), float(r["J_C_D"])))
rows.sort(key=lambda x: -x[1])
best = None
for tag, q, jr, jd in rows:
    feas = jr <= 0.05 and jd <= 0.05
    if feas and best is None:
        best = tag
    print(f"{tag:16s} q3d={q:.4f} J_C_R={jr:.4f} J_C_D={jd:.4f} feasible={feas}")
print(f"\nBEST_FEASIBLE={best}")
with open(os.path.join(sys.argv[1], "best_lambda.txt"), "w") as f:
    f.write(best or "none")
EOF

BEST=$(cat "$SWEEP_DIR/best_lambda.txt")
if [[ "$BEST" == "none" ]]; then
  echo "No feasible multiplier pair found on train split." >&2
  exit 1
fi
LR="${BEST#lr}"; LR="${LR%%_*}"
LD="${BEST##*_ld}"
echo "Selected multipliers: lambda_r=$LR lambda_d=$LD"

# ---- Stage 2: test-split evaluation with the paper protocol ----
python scripts/eval_crpo_ppo.py \
  "${COMMON[@]}" \
  --table "$TEST_TABLE" \
  --trajectory-csv "$TEST_TRAJ_CSV" \
  --test-trajs traj10,traj11,traj12,traj13,traj14 \
  --episodes 20 \
  --seed 100 --shadow-seeds 100,101,102,103,104 \
  --methods greedy-oracle,per-slot-lagrangian \
  --oracle-lambda-r "$LR" --oracle-lambda-d "$LD" \
  --per-traj-csv "$SWEEP_DIR/per_traj_results_test.csv" \
  --aggregate-csv "$SWEEP_DIR/aggregate_results_test.csv" \
  2>&1 | tee "$SWEEP_DIR/eval_test_result.txt"

echo "Done: $SWEEP_DIR/aggregate_results_test.csv"
