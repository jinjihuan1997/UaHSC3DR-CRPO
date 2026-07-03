#!/usr/bin/env bash
# Standard protocol:
#   1. Train on traj0-traj6.
#   2. Select PPO-penalty lambda on validation traj7-traj8.
#   3. Run final test once on traj9-traj10 with the selected PPO model.
#
# Option B parameters: q_req=0.60 (relaxed RGB, non-binding for most policies),
#   depth_req=0.40 (tighter depth, becomes the binding constraint).
#   CRPO eps_rgb=0.20 stays high so rgb_score≈negative → stays in reward/depth modes.
#   CRPO eps_depth=0.14 just above natural training c_depth≈0.14-0.18.

set -euo pipefail

cd "$(dirname "$(dirname "$(realpath "$0")")")"
mkdir -p checkpoints outputs/ppo_logs outputs/eval_cmdp

TABLE="outputs/offline_lookup_fushimi_traj0_10_stage3_grid.csv"
TRAJ="outputs/trajectory_fushimi_a2g.csv"
CRPO_CKPT="checkpoints/crpo_dual.pt"
LAG_CKPT="checkpoints/lagrangian_ppo.pt"

COMMON_TRAIN=(
  --table "$TABLE"
  --trajectory-csv "$TRAJ"
  --multi-trajectory
  --q-req 0.60
  --depth-req 0.40
  --timesteps 1000000
  --n-steps 4096
  --batch-size 256
  --seed 42
  --device cuda
)

COMMON_EVAL=(
  --table "$TABLE"
  --trajectory-csv "$TRAJ"
  --multi-trajectory
  --q-req 0.60
  --depth-req 0.40
  --shadow-seeds 101,102,103,104,105
  --episodes 20
  --episode-len 102
  --device cuda
)

echo "=== Train CRPO-PPO: eps_rgb=0.20, eps_depth=0.14 ==="
rm -f "$CRPO_CKPT" outputs/ppo_logs/crpo_dual.csv
python scripts/train_crpo_ppo.py \
  "${COMMON_TRAIN[@]}" \
  --objective crpo \
  --epsilon-rgb 0.20 \
  --epsilon-depth 0.14 \
  --out "$CRPO_CKPT" \
  --log-csv outputs/ppo_logs/crpo_dual.csv

declare -A PPO_CKPT=(
  [A]=checkpoints/ppo_A_r10d10.pt
  [B]=checkpoints/ppo_B_r50d10.pt
  [C]=checkpoints/ppo_C_r10d25.pt
  [D]=checkpoints/ppo_D_r50d25.pt
)
declare -A LRGB=([A]=1.0 [B]=5.0 [C]=1.0 [D]=5.0)
declare -A LDEP=([A]=1.0 [B]=1.0 [C]=2.5 [D]=2.5)
declare -A LTAG=(
  [A]="lambda_rgb=1.0,lambda_depth=1.0"
  [B]="lambda_rgb=5.0,lambda_depth=1.0"
  [C]="lambda_rgb=1.0,lambda_depth=2.5"
  [D]="lambda_rgb=5.0,lambda_depth=2.5"
)

for KEY in A B C D; do
  CKPT="${PPO_CKPT[$KEY]}"
  if [ -f "$CKPT" ]; then
    echo "=== PPO-penalty (${LTAG[$KEY]}) checkpoint exists, skipping ==="
    continue
  fi
  CSV="outputs/ppo_logs/ppo_${KEY}.csv"
  rm -f "$CSV"
  echo "=== Train PPO-penalty (${LTAG[$KEY]}) ==="
  python scripts/train_crpo_ppo.py \
    "${COMMON_TRAIN[@]}" \
    --objective penalty \
    --penalty-rgb "${LRGB[$KEY]}" \
    --penalty-depth "${LDEP[$KEY]}" \
    --out "$CKPT" \
    --log-csv "$CSV"
done

# ── Lagrangian PPO（自适应 λ，ε 与 CRPO 对齐) ────────────────────────
echo "=== Train Lagrangian PPO (eps_rgb=0.20, eps_depth=0.14, lr_dual=0.005) ==="
rm -f "$LAG_CKPT" outputs/ppo_logs/lagrangian_ppo.csv
python scripts/train_crpo_ppo.py \
  "${COMMON_TRAIN[@]}" \
  --objective lagrangian \
  --epsilon-rgb 0.20 --epsilon-depth 0.14 \
  --lr-dual 0.005 \
  --out "$LAG_CKPT" \
  --log-csv outputs/ppo_logs/lagrangian_ppo.csv

for KEY in A B C D; do
  echo "=== Validation: CRPO vs PPO (${LTAG[$KEY]}) on traj7-traj8 ==="
  python scripts/eval_crpo_ppo.py \
    "${COMMON_EVAL[@]}" \
    --eval-split val \
    --model "$CRPO_CKPT" \
    --ppo-penalty-model "${PPO_CKPT[$KEY]}" \
    --ppo-penalty-label "PPO-${KEY}(${LTAG[$KEY]})" \
    --aggregate-csv "outputs/eval_cmdp/val_dual_${KEY}_aggregate.csv" \
    --per-traj-csv "outputs/eval_cmdp/val_dual_${KEY}_per_traj.csv"
done

# Lagrangian 单独 val（不参与 λ 选择，只用于展示对比）
echo "=== Validation: Lagrangian-PPO on traj7-traj8 ==="
python scripts/eval_crpo_ppo.py \
  "${COMMON_EVAL[@]}" \
  --eval-split val \
  --model "$CRPO_CKPT" \
  --lagrangian-model "$LAG_CKPT" \
  --aggregate-csv "outputs/eval_cmdp/val_lagrangian_aggregate.csv" \
  --per-traj-csv  "outputs/eval_cmdp/val_lagrangian_per_traj.csv"

echo
echo "===== Validation summary and PPO selection (J_C_R<=0.05, J_C_D<=0.14) ====="
SELECTED_KEY="$(
python3 - <<'PY'
import csv

configs = [("A", "1.0", "1.0"), ("B", "5.0", "1.0"), ("C", "1.0", "2.5"), ("D", "5.0", "2.5")]
eps_r, eps_d = 0.05, 0.14
rows = {}

for key, lr, ld in configs:
    path = f"outputs/eval_cmdp/val_dual_{key}_aggregate.csv"
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            m = row["method"]
            if m == "CRPO-PPO":
                rows.setdefault("CRPO", ("", "", row))
            elif m.startswith(f"PPO-{key}("):   # 匹配新标签格式 PPO-A(...), PPO-B(...) 等
                rows[key] = (lr, ld, row)
            elif m == "PPO-penalty":             # 兼容旧标签（复用旧 CSV 时）
                rows.setdefault(key, (lr, ld, row))

print(f"{'method':<24} {'q3d':>6} {'q_rgb':>6} {'r_dep':>6} {'J_C_R':>7} {'J_C_D':>7} {'kd':>5} {'beta':>6} feasible")
print("-" * 90)
crpo = rows.get("CRPO")
if crpo:
    _, _, row = crpo
    jcr, jcd = float(row["J_C_R"]), float(row["J_C_D"])
    print(f"{'CRPO-PPO':<24} {float(row['avg_q3d']):>6.3f} {float(row['avg_q_rgb']):>6.3f} "
          f"{float(row['avg_r_depth']):>6.3f} {jcr:>7.3f} {jcd:>7.3f} "
          f"{float(row['avg_kd']):>5.1f} {float(row['avg_beta_d']):>6.3f} {jcr <= eps_r and jcd <= eps_d}")

candidates = []
for key, lr, ld in configs:
    _, _, row = rows[key]
    jcr, jcd = float(row["J_C_R"]), float(row["J_C_D"])
    excess = max(jcr - eps_r, 0.0) + max(jcd - eps_d, 0.0)
    feasible = excess <= 1e-12
    q3d = float(row["avg_q3d"])
    candidates.append((int(feasible), -excess, q3d, key))
    name = f"PPO {key} ({lr},{ld})"
    print(f"{name:<24} {q3d:>6.3f} {float(row['avg_q_rgb']):>6.3f} "
          f"{float(row['avg_r_depth']):>6.3f} {jcr:>7.3f} {jcd:>7.3f} "
          f"{float(row['avg_kd']):>5.1f} {float(row['avg_beta_d']):>6.3f} {feasible}")

candidates.sort(reverse=True)
selected = candidates[0][3]
print(f"Selected PPO key on validation: {selected}")
print(selected)
PY
)"
SELECTED_KEY="$(echo "$SELECTED_KEY" | tail -n 1)"
SELECTED_MODEL="${PPO_CKPT[$SELECTED_KEY]}"

echo "$SELECTED_KEY" > outputs/eval_cmdp/selected_ppo_key.txt
echo "$SELECTED_MODEL" > outputs/eval_cmdp/selected_ppo_model.txt

echo
echo "=== Final test: all methods on traj9-traj10 ==="
# CRPO + 选出的 PPO + Lagrangian（含 random 和固定基线）
python scripts/eval_crpo_ppo.py \
  "${COMMON_EVAL[@]}" \
  --eval-split test \
  --model "$CRPO_CKPT" \
  --ppo-penalty-model "$SELECTED_MODEL" \
  --ppo-penalty-label "PPO-penalty(selected-${SELECTED_KEY})" \
  --lagrangian-model "$LAG_CKPT" \
  --aggregate-csv outputs/eval_cmdp/test_main_aggregate.csv \
  --per-traj-csv  outputs/eval_cmdp/test_main_per_traj.csv

echo
echo "===== Final test summary (J_C_R<=0.05, J_C_D<=0.14) ====="
python3 - <<'EOF'
import csv

EPS_R, EPS_D = 0.05, 0.14
results = {}
for path in [
    "outputs/eval_cmdp/test_main_aggregate.csv",
]:
    with open(path) as f:
        for row in csv.DictReader(f):
            m = row["method"]
            if m not in results:
                results[m] = row

display_order = [
    "CRPO-PPO", "Lagrangian-PPO",
    lambda k: k.startswith("PPO-penalty(selected"),
    "fixed-balanced", "RGB-priority", "depth-priority", "random",
]

ordered = []
for key in display_order:
    if callable(key):
        ordered += [k for k in results if key(k) and k not in ordered]
    elif key in results and key not in ordered:
        ordered.append(key)
ordered += [k for k in results if k not in ordered]

hdr = f"{'方法':<34} {'q3d':>6} {'q_rgb':>6} {'r_dep':>6} {'J_C_R':>7} {'J_C_D':>7} {'beta':>6} {'kd':>4}  RGB  Dep  双✓"
print(hdr); print("-" * len(hdr))
for m in ordered:
    r = results[m]
    jcr, jcd = float(r["J_C_R"]), float(r["J_C_D"])
    ok_r = "✓" if jcr <= EPS_R else "✗"
    ok_d = "✓" if jcd <= EPS_D else "✗"
    ok   = "✓" if (jcr <= EPS_R and jcd <= EPS_D) else "✗"
    print(f"{m:<34} {float(r['avg_q3d']):>6.3f} {float(r['avg_q_rgb']):>6.3f} "
          f"{float(r['avg_r_depth']):>6.3f} {jcr:>7.3f} {jcd:>7.3f} "
          f"{float(r['avg_beta_d']):>6.3f} {float(r['avg_kd']):>4.1f}  {ok_r:<4} {ok_d:<4} {ok}")
EOF

echo
echo "Saved:"
echo "  outputs/eval_cmdp/val_dual_*_aggregate.csv"
echo "  outputs/eval_cmdp/val_lagrangian_aggregate.csv"
echo "  outputs/eval_cmdp/selected_ppo_key.txt"
echo "  outputs/eval_cmdp/test_main_aggregate.csv"
