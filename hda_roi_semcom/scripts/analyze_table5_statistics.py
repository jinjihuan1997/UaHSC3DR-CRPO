"""Per-trajectory stats, paired tests, and Q_gt sensitivity for Table V (C3/C4)."""
import itertools
import numpy as np
import pandas as pd
from scipy import stats

CSV = "/home/king/Downloads/Projects/TCOM/hda_roi_semcom/outputs/eiffel15_mapping_quality/gsfusion_real_metrics.csv"
df = pd.read_csv(CSV)
df["method"] = df["condition"].str.replace(r"_traj\d+_seed\d+$", "", regex=True)
df["traj"] = df["condition"].str.extract(r"_(traj\d+)_seed")[0]

METRICS = ["psnr", "ssim", "lpips", "chamfer", "fscore", "completeness"]
HIB = {"psnr": True, "ssim": True, "lpips": False, "chamfer": False,
       "fscore": True, "completeness": True}

# p5/p95 basis over all 35 rows (reproducing summarize_mapping_quality_results.py)
basis = {m: (df[m].quantile(0.05), df[m].quantile(0.95)) for m in METRICS}
print("=== p5/p95 normalization bounds (raw metric ranges) ===")
for m in METRICS:
    lo, hi = basis[m]
    print(f"{m:14s} lo={lo:.4f} hi={hi:.4f} span={hi-lo:.4f} "
          f"full-range=[{df[m].min():.4f},{df[m].max():.4f}]")

def qbar(row, m):
    lo, hi = basis[m]
    s = np.clip((row[m] - lo) / (hi - lo), 0, 1)
    return s if HIB[m] else 1 - s

for m in METRICS:
    df[f"Qbar_{m}"] = df.apply(lambda r: qbar(r, m), axis=1)
df["Q_app"] = 0.6*df["Qbar_psnr"] + 0.2*df["Qbar_ssim"] + 0.2*df["Qbar_lpips"]
df["Q_geo"] = 0.35*df["Qbar_fscore"] + 0.35*df["Qbar_completeness"] + 0.30*df["Qbar_chamfer"]
df["Q_gt"] = (0.45*df["Q_app"] + 0.45*df["Q_geo"] + 0.10*df["Q_app"]*df["Q_geo"]).clip(0, 1)

ORDER = ["fixed_balanced_allocation", "rgb_priority_allocation",
         "depth_priority_allocation", "random_allocation",
         "ppo_penalty", "lagrangian_ppo", "crpo_guided_ppo"]

print("\n=== Reproduction check: method means (compare with Table V) ===")
agg = df.groupby("method")[METRICS + ["Q_app", "Q_geo", "Q_gt"]].mean().loc[ORDER]
print(agg.round(4).to_string())

print("\n=== Per-method mean ± std over 5 trajectories ===")
g = df.groupby("method")[METRICS + ["Q_app", "Q_geo", "Q_gt"]]
mean, std = g.mean().loc[ORDER], g.std(ddof=1).loc[ORDER]
for m in ORDER:
    cells = [f"{mean.loc[m,c]:.4f}±{std.loc[m,c]:.4f}" for c in METRICS + ["Q_app","Q_geo","Q_gt"]]
    print(f"{m:28s} " + "  ".join(cells))

print("\n=== Paired tests: CRPO vs each learning baseline (n=5 trajectories) ===")
piv = {c: df.pivot(index="traj", columns="method", values=c) for c in METRICS + ["Q_app","Q_geo","Q_gt"]}
for rival in ["lagrangian_ppo", "ppo_penalty"]:
    print(f"\n--- crpo_guided_ppo vs {rival} ---")
    for c in METRICS + ["Q_app", "Q_geo", "Q_gt"]:
        a = piv[c]["crpo_guided_ppo"]; b = piv[c][rival]
        d = a - b
        better = (d > 0).sum() if HIB.get(c, True) else (d < 0).sum()
        try:
            w = stats.wilcoxon(a, b, alternative="two-sided", method="exact")
            wp = w.pvalue
        except ValueError:
            wp = float("nan")
        t = stats.ttest_rel(a, b)
        print(f"{c:14s} mean diff={d.mean():+.4f}  better-on {better}/5  "
              f"wilcoxon p={wp:.4f}  paired-t p={t.pvalue:.4f}")

print("\n=== Q_gt weight sensitivity: method ranking under alternative weights ===")
for (wa, wg, wj) in [(0.45,0.45,0.10),(0.30,0.60,0.10),(0.60,0.30,0.10),
                     (0.50,0.50,0.00),(0.25,0.65,0.10),(0.65,0.25,0.10)]:
    q = (wa*df["Q_app"] + wg*df["Q_geo"] + wj*df["Q_app"]*df["Q_geo"]).clip(0,1)
    means = q.groupby(df["method"]).mean().loc[ORDER]
    rank = means.sort_values(ascending=False)
    tag = " > ".join(f"{i.replace('_allocation','').replace('_',' ')}({v:.3f})"
                     for i, v in rank.items())
    print(f"w=({wa},{wg},{wj}): {tag}")

print("\n=== Depth marginal effect: raw geometry deltas (rgb-priority vs depth-priority) ===")
for c in ["chamfer", "fscore", "completeness", "r_depth"]:
    a = df.pivot(index="traj", columns="method", values=c)
    print(f"{c:14s} rgb-pri={a['rgb_priority_allocation'].mean():.4f}  "
          f"depth-pri={a['depth_priority_allocation'].mean():.4f}  "
          f"delta={a['depth_priority_allocation'].mean()-a['rgb_priority_allocation'].mean():+.4f}")
