#!/usr/bin/env bash
# 方案B：PPO-penalty lambda 扫描 vs CRPO，证明 CRPO 无需调参即满足约束
set -euo pipefail

cd "$(dirname "$(dirname "$(realpath "$0")")")"
mkdir -p checkpoints outputs/ppo_logs outputs/eval_cmdp

TABLE="outputs/offline_lookup_fushimi_traj0_10_stage3_grid.csv"
TRAJ="outputs/trajectory_fushimi_a2g.csv"
CRPO_CKPT="checkpoints/crpo_fushimi_qreq065.pt"

COMMON_TRAIN=(
  --table "$TABLE"
  --trajectory-csv "$TRAJ"
  --multi-trajectory
  --q-req 0.65
  --depth-req 0.35
  --timesteps 1000000
  --n-steps 4096 --batch-size 256
  --seed 42 --device cuda
)

COMMON_EVAL=(
  --table "$TABLE"
  --trajectory-csv "$TRAJ"
  --multi-trajectory
  --q-req 0.65 --depth-req 0.35
  --shadow-seeds 1,2,3,4,5
  --episodes 3 --episode-len 102
  --device cpu
)

# ── 1. CRPO（双约束：epsilon_depth 从 0.40 收紧至 0.11）────────────
if true; then   # 强制重训以覆盖旧 checkpoint
  echo "=== Train CRPO (eps_rgb=0.20, eps_depth=0.11) ==="
  python scripts/train_crpo_ppo.py \
    "${COMMON_TRAIN[@]}" \
    --objective crpo \
    --epsilon-rgb 0.20 --epsilon-depth 0.11 \
    --out "$CRPO_CKPT" \
    --log-csv outputs/ppo_logs/crpo_fushimi_qreq065.csv
fi

# ── 2. PPO-penalty lambda 扫描 ─────────────────────────────────────
declare -A LAM_CKPT=(
  ["03"]="checkpoints/ppo_penalty_qreq065_lam03.pt"
  ["10"]="checkpoints/ppo_penalty_fushimi_qreq065.pt"   # 已训练，复用
  ["20"]="checkpoints/ppo_penalty_qreq065_lam20.pt"
  ["50"]="checkpoints/ppo_penalty_qreq065_lam50.pt"
)
declare -A LAM_VAL=(["03"]="0.3" ["10"]="1.0" ["20"]="2.0" ["50"]="5.0")

for KEY in 03 10 20 50; do   # 已存在则跳过，只新训 lam=5.0
  CKPT="${LAM_CKPT[$KEY]}"
  LAM="${LAM_VAL[$KEY]}"
  if [ -f "$CKPT" ]; then
    echo "=== PPO-penalty λ=$LAM already trained, skipping ==="
    continue
  fi
  echo "=== Train PPO-penalty λ_rgb=$LAM ==="
  python scripts/train_crpo_ppo.py \
    "${COMMON_TRAIN[@]}" \
    --objective penalty \
    --penalty-rgb "$LAM" --penalty-depth 1.0 \
    --out "$CKPT" \
    --log-csv "outputs/ppo_logs/ppo_penalty_qreq065_lam${KEY}.csv"
done

# ── 3. Eval：CRPO vs 各 lambda ─────────────────────────────────────
for KEY in 03 10 20 50; do
  LAM="${LAM_VAL[$KEY]}"
  CKPT="${LAM_CKPT[$KEY]}"
  echo "=== Eval: CRPO vs PPO-penalty λ=$LAM ==="
  python scripts/eval_crpo_ppo.py \
    "${COMMON_EVAL[@]}" \
    --model "$CRPO_CKPT" \
    --ppo-penalty-model "$CKPT" \
    --aggregate-csv "outputs/eval_cmdp/compare_lam${KEY}_aggregate.csv" \
    --per-traj-csv  "outputs/eval_cmdp/compare_lam${KEY}_per_traj.csv"
done

# ── 4. 汇总打印 ────────────────────────────────────────────────────
echo ""
echo "===== 汇总对比（q_req=0.65, ε_rgb=0.07 约束线）====="
python3 - <<'EOF'
import csv, glob

results = {}
for key, lam in [("03","0.3"),("10","1.0"),("20","2.0"),("50","5.0")]:
    path = f"outputs/eval_cmdp/compare_lam{key}_aggregate.csv"
    with open(path) as f:
        for row in csv.DictReader(f):
            m = row["method"]
            tag = f"PPO-pen(λ={lam})" if m == "PPO-penalty" else m
            results[tag] = row

eps = 0.07
print(f"{'方法':<24} {'q3d':>7} {'q_rgb':>7} {'J_C_R':>7}  ε=0.07约束")
for tag in ["CRPO-PPO","PPO-pen(λ=0.3)","PPO-pen(λ=1.0)","PPO-pen(λ=2.0)","PPO-pen(λ=5.0)","fixed-balanced"]:
    if tag not in results: continue
    r = results[tag]
    jcr = float(r["J_C_R"])
    ok = "✓满足" if jcr <= eps else f"✗违约(+{jcr-eps:.3f})"
    print(f"{tag:<24} {float(r['avg_q3d']):>7.3f} {float(r['avg_q_rgb']):>7.3f} {jcr:>7.3f}  {ok}")
EOF

echo ""
echo "完整结果: outputs/eval_cmdp/"
