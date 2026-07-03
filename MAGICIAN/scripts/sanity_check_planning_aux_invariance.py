#!/usr/bin/env python3
"""Sanity check for the planning-level compression trajectory-invariance experiment.

Runs two sets of checks:

  A. Functional tests of src/planning_aux/compression.py (no MAGICIAN needed)
     - block_dropout rho=1.0  : output == input
     - block_dropout rho=0.50 : actual keep ratio approx 0.50
     - block_dropout rho=0.25 : actual keep ratio approx 0.25
     - downsample_upsample scale=1.0  : output == input
     - downsample_upsample scale=0.25 : output shape unchanged
     - mask stays bool
     - invalid depth not made valid
     - valid depth not set to zero when rho=1.0

  B. LMDB pre-flight checks (requires completed MAGICIAN runs)
     - Which LMDBs exist for the configured rho / scale values
     - Starting pose consistency (all trajectories should start at same position)
     - First-few-step pose-error table for a quick divergence preview
     - MAGICIAN config consistency across runs

Usage
-----
  # Full check (shows functional tests + LMDB status for default rho/scale values):
  python scripts/sanity_check_planning_aux_invariance.py

  # Skip LMDB checks if you only care about compression function tests:
  python scripts/sanity_check_planning_aux_invariance.py --no_lmdb

  # Only check specific rho/scale values:
  python scripts/sanity_check_planning_aux_invariance.py --rho_plan_list 1.0 0.5 --downsample_scales 0.25
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_raux_block_sweep_fushimi as raux  # noqa: E402

try:
    import torch
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False

LMDB_KEY = "fushimi/0"
PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"
WARN = "[WARN]"


# ---------------------------------------------------------------------------
# A. Functional compression tests
# ---------------------------------------------------------------------------

def _check_torch() -> bool:
    if not _TORCH_OK:
        print(f"{SKIP} torch not available; skipping functional compression tests.")
        return False
    return True


def _make_depth_mask(H: int = 64, W: int = 64, valid_frac: float = 0.8):
    """Create synthetic depth [1, H, W] and mask [1, H, W]."""
    import torch
    rng = torch.Generator()
    rng.manual_seed(12345)
    depth = torch.rand(1, H, W, generator=rng) * 10.0 + 0.5
    mask = (torch.rand(1, H, W, generator=rng) < valid_frac).bool()
    depth[~mask] = 0.0
    return depth, mask


def _valid_count(depth, mask) -> int:
    import torch
    return int((mask.bool() & torch.isfinite(depth) & (depth > 0)).sum().item())


def _run_functional_tests(verbose: bool) -> int:
    """Run all functional tests. Returns number of failures."""
    if not _check_torch():
        return 0

    import torch
    try:
        from planning_aux.compression import (
            block_dropout, pixel_dropout, downsample_upsample, compress_depth_mask
        )
    except ImportError as exc:
        print(f"{FAIL} Cannot import planning_aux.compression: {exc}")
        print(f"       Make sure src/ is in sys.path and src/planning_aux/compression.py exists.")
        return 1

    fails = 0

    def check(condition: bool, name: str, detail: str = "") -> None:
        nonlocal fails
        tag = PASS if condition else FAIL
        if not condition:
            fails += 1
        msg = f"  {tag} {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)

    depth, mask = _make_depth_mask(H=128, W=128, valid_frac=0.8)
    n_valid_orig = _valid_count(depth, mask)

    print("\n[A] block_dropout tests")

    # 1. rho=1.0 -> identity
    d1, m1 = block_dropout(depth, mask, r=1.0, seed=42)
    check(torch.allclose(d1, depth), "rho=1.0: depth unchanged")
    check((m1 == mask).all(), "rho=1.0: mask unchanged")
    check(_valid_count(d1, m1) == n_valid_orig, "rho=1.0: valid count unchanged")

    # 2. rho=0.50: actual keep ratio in [0.40, 0.60]
    d2, m2 = block_dropout(depth, mask, r=0.50, seed=42, block_size=16)
    n_kept = _valid_count(d2, m2)
    ratio = n_kept / max(n_valid_orig, 1)
    check(0.40 <= ratio <= 0.60, f"rho=0.50: actual keep ratio={ratio:.3f}", f"expected ~0.50")
    check(m2.dtype == torch.bool, "rho=0.50: mask is bool")
    # Invalid pixels must not become valid
    was_invalid = ~(mask.bool() & (depth > 0) & torch.isfinite(depth))
    newly_valid = m2.bool() & ~mask.bool()
    check(not newly_valid.any(), "rho=0.50: no invalid-to-valid promotion")

    # 3. rho=0.25: actual keep ratio in [0.15, 0.35]
    d3, m3 = block_dropout(depth, mask, r=0.25, seed=42, block_size=16)
    n_kept3 = _valid_count(d3, m3)
    ratio3 = n_kept3 / max(n_valid_orig, 1)
    check(0.15 <= ratio3 <= 0.35, f"rho=0.25: actual keep ratio={ratio3:.3f}", "expected ~0.25")
    check(m3.dtype == torch.bool, "rho=0.25: mask is bool")

    # 4. rho=0.25 keeps fewer pixels than rho=0.50
    check(n_kept3 <= n_kept, f"rho=0.25 keeps fewer pixels than rho=0.50 ({n_kept3} <= {n_kept})")

    print("\n[A] pixel_dropout tests")

    d4, m4 = pixel_dropout(depth, mask, r=1.0, seed=42)
    check(torch.allclose(d4, depth), "rho=1.0: depth unchanged")
    check((m4 == mask).all(), "rho=1.0: mask unchanged")

    d5, m5 = pixel_dropout(depth, mask, r=0.50, seed=42)
    ratio5 = _valid_count(d5, m5) / max(n_valid_orig, 1)
    check(0.40 <= ratio5 <= 0.60, f"rho=0.50: actual keep ratio={ratio5:.3f}", "expected ~0.50")
    check(m5.dtype == torch.bool, "rho=0.50: mask is bool")

    print("\n[A] downsample_upsample tests")

    # 5. scale=1.0 -> identity
    d6, m6 = downsample_upsample(depth, mask, scale=1.0)
    check(torch.allclose(d6.float(), depth.float()), "scale=1.0: depth unchanged")
    check((m6 == mask).all(), "scale=1.0: mask unchanged")
    check(d6.shape == depth.shape, f"scale=1.0: shape preserved {d6.shape}")

    # 6. scale=0.25: output shape same as input
    d7, m7 = downsample_upsample(depth, mask, scale=0.25)
    check(d7.shape == depth.shape, f"scale=0.25: shape preserved {d7.shape} == {depth.shape}")
    check(m7.dtype == torch.bool, "scale=0.25: mask is bool")
    # scale=0.25: mask should be 0/1 only (nearest-neighbour)
    unique_vals = m7.unique()
    check(set(unique_vals.tolist()).issubset({False, True}), "scale=0.25: mask is binary")

    # 7. After the AND-with-original-mask fix, no invalid-to-valid promotion should occur.
    newly_valid_px = m7.bool() & ~mask.bool()
    check(
        not newly_valid_px.any(),
        "scale=0.25: no invalid-to-valid promotion (AND-with-original-mask fix)",
        f"newly_valid={int(newly_valid_px.sum())}",
    )

    # 8. scale < 1.0 retains fewer or equal valid pixels than clean input.
    #    (Strict monotonicity between scale values is not guaranteed for random masks;
    #     only the property vs. the clean original is checked here.)
    d8, m8 = downsample_upsample(depth, mask, scale=0.5)
    n8 = _valid_count(d8, m8)
    n7 = _valid_count(d7, m7)
    check(n8 <= n_valid_orig, f"scale=0.5 keeps <= n_valid_orig valid pixels ({n8} <= {n_valid_orig})")
    check(n7 <= n_valid_orig, f"scale=0.25 keeps <= n_valid_orig valid pixels ({n7} <= {n_valid_orig})")

    print("\n[A] compress_depth_mask unified dispatch tests")

    db, mb = compress_depth_mask(depth, mask, mode="block_dropout", r=0.50, seed=42)
    check(torch.allclose(db.float(), d2.float()), "block_dropout dispatch matches direct call")

    dd, md = compress_depth_mask(depth, mask, mode="downsample_upsample", scale=0.25)
    check(torch.allclose(dd.float(), d7.float()), "downsample_upsample dispatch with scale=0.25")

    print(f"\n[A] Functional tests complete. Failures: {fails}")
    return fails


# ---------------------------------------------------------------------------
# B. LMDB pre-flight checks
# ---------------------------------------------------------------------------

def _auc(cov: list[float]) -> float:
    n = len(cov)
    if n < 2:
        return float(cov[0]) if n == 1 else 0.0
    return float(np.trapz(cov, np.arange(n)) / (n - 1))


def _dsup_run_id(scale: float, seed: int) -> str:
    stag = f"{scale:.3f}".replace(".", "p")
    return f"dsup_fushimi_scale{stag}_seed{seed}"


def check_lmdb(rho_values: list[float], downsample_scales: list[float], seed: int, n_show: int) -> None:
    print("\n[B] LMDB pre-flight checks")
    records: dict[str, dict] = {}

    # block_dropout LMDBs
    print(f"\n  block_dropout (rho x seed={seed}):")
    for r in rho_values:
        rid = raux.run_id(r, seed)
        lmdb_dir = raux.SCENE_RESULTS_DIR / f"{rid}_lmdb"
        if not lmdb_dir.exists():
            print(f"  [MISS] rho={r:.2f}: {lmdb_dir}")
            continue
        try:
            record = raux.load_lmdb_record(lmdb_dir, key=LMDB_KEY)
        except Exception as exc:
            print(f"  [ERR]  rho={r:.2f}: {exc}")
            continue
        xh = np.asarray(record.get("X_cam_history", []), dtype=np.float64)
        cov = [float(v) for v in record.get("coverage", [])]
        timing = record.get("timing", [])
        ret = float(np.mean([t.get("depth_retention_ratio", 0.0) for t in timing])) if timing else math.nan
        auc_val = _auc(cov)
        print(
            f"  [OK]   rho={r:.2f}: steps={len(cov)}  "
            f"start={xh[0].tolist() if len(xh) > 0 else 'N/A'}  "
            f"final_cov={cov[-1]:.6f}  auc={auc_val:.4f}  retention={ret:.4f}"
        )
        records[f"block_r{r:.2f}"] = record

        # Check that r_aux in timing matches expected value
        if timing:
            r_in_timing = float(timing[0].get("r_aux", 1.0))
            if abs(r_in_timing - r) > 0.001:
                print(f"  {WARN} timing[0].r_aux={r_in_timing} != expected rho={r}")

    # downsample_upsample LMDBs
    print(f"\n  downsample_upsample (scale x seed={seed}):")
    for scale in downsample_scales:
        if scale >= 1.0:
            print(f"  [--]   scale={scale:.3f}: identity (no LMDB needed, uses baseline)")
            continue
        rid = _dsup_run_id(scale, seed)
        lmdb_dir = raux.SCENE_RESULTS_DIR / f"{rid}_lmdb"
        if not lmdb_dir.exists():
            print(f"  [MISS] scale={scale:.3f}: {lmdb_dir}")
            continue
        try:
            record = raux.load_lmdb_record(lmdb_dir, key=LMDB_KEY)
        except Exception as exc:
            print(f"  [ERR]  scale={scale:.3f}: {exc}")
            continue
        xh = np.asarray(record.get("X_cam_history", []), dtype=np.float64)
        cov = [float(v) for v in record.get("coverage", [])]
        timing = record.get("timing", [])
        ret = float(np.mean([t.get("depth_retention_ratio", 0.0) for t in timing])) if timing else math.nan
        auc_val = _auc(cov)
        print(
            f"  [OK]   scale={scale:.3f}: steps={len(cov)}  "
            f"start={xh[0].tolist() if len(xh) > 0 else 'N/A'}  "
            f"final_cov={cov[-1]:.6f}  auc={auc_val:.4f}  retention={ret:.4f}"
        )
        records[f"dsup_scale{scale:.3f}"] = record

        # Check degrade_mode in timing
        if timing:
            mode_in_timing = timing[0].get("degrade_mode", "")
            if mode_in_timing != "downsample_upsample":
                print(f"  {WARN} timing[0].degrade_mode={mode_in_timing!r} != 'downsample_upsample'")
            r_aux_in_timing = float(timing[0].get("r_aux", 1.0))
            if abs(r_aux_in_timing - scale) > 0.001:
                print(f"  {WARN} timing[0].r_aux={r_aux_in_timing} != expected scale={scale}")

    # Starting pose consistency
    _check_start_poses(records)

    # Early divergence preview
    base_key = f"block_r{1.0:.2f}"
    if base_key in records:
        _early_divergence(records, base_key, n_show)

    # Config consistency
    _check_config_consistency(rho_values, downsample_scales, seed)


def _check_start_poses(records: dict[str, dict]) -> None:
    if not records:
        return
    base_key = f"block_r{1.0:.2f}"
    base = records.get(base_key)
    if base is None:
        print(f"\n  {WARN} Baseline (block_r1.00) not loaded; skipping start-pose check.")
        return
    xb = np.asarray(base["X_cam_history"], dtype=np.float64)
    vb = np.asarray(base["V_cam_history"], dtype=np.float64)
    print(f"\n  Starting pose consistency vs baseline:")
    for key, record in records.items():
        if key == base_key:
            continue
        xc = np.asarray(record.get("X_cam_history", []), dtype=np.float64)
        vc = np.asarray(record.get("V_cam_history", []), dtype=np.float64)
        pos_err = float(np.linalg.norm(xb[0] - xc[0])) if len(xb) > 0 and len(xc) > 0 else math.nan
        ang_err = float(np.linalg.norm(vb[0] - vc[0])) if len(vb) > 0 and len(vc) > 0 else math.nan
        same = pos_err < 0.01
        tag = PASS if same else FAIL
        print(f"  {tag} {key}: pos_err={pos_err:.6f}m  ang_err={ang_err:.4f}deg  same_start={'YES' if same else 'NO'}")


def _early_divergence(records: dict[str, dict], base_key: str, n_show: int) -> None:
    base = records[base_key]
    xb = np.asarray(base["X_cam_history"], dtype=np.float64)
    print(f"\n  First {n_show} step pose-errors vs baseline:")
    for key, record in records.items():
        if key == base_key:
            continue
        xc = np.asarray(record.get("X_cam_history", []), dtype=np.float64)
        n = min(len(xb), len(xc), n_show + 1)
        errors = [float(np.linalg.norm(xb[i] - xc[i])) for i in range(1, n)]
        n_same = sum(e < 1.0 for e in errors)
        print(f"  {key}: errors={[f'{e:.2f}' for e in errors]}  same_poses={n_same}/{len(errors)}")


def _check_config_consistency(rho_values: list[float], downsample_scales: list[float], seed: int) -> None:
    """Verify that all generated config JSONs reference the same base scene and beam parameters."""
    print(f"\n  Config consistency check:")
    configs: list[dict] = []
    keys: list[str] = []

    for r in rho_values:
        rid = raux.run_id(r, seed)
        config_path = raux.CONFIG_DIR / f"_{rid}.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            configs.append(cfg)
            keys.append(f"block_r{r:.2f}")

    for scale in downsample_scales:
        if scale >= 1.0:
            continue
        stag = f"{scale:.3f}".replace(".", "p")
        rid = f"dsup_fushimi_scale{stag}_seed{seed}"
        config_path = raux.CONFIG_DIR / f"_{rid}.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text())
            configs.append(cfg)
            keys.append(f"dsup_scale{scale:.3f}")

    if not configs:
        print(f"  {SKIP} No config JSON files found (run experiment first to generate them).")
        return

    ref = configs[0]
    ref_key = keys[0]
    consistent = True
    for cfg, key in zip(configs[1:], keys[1:]):
        for field in ("beam_width", "beam_steps", "test_scenes", "random_seed", "torch_seed"):
            if cfg.get(field) != ref.get(field):
                print(f"  {WARN} Config mismatch: {key}.{field}={cfg.get(field)} "
                      f"!= {ref_key}.{field}={ref.get(field)}")
                consistent = False
    tag = PASS if consistent else WARN
    n = len(configs)
    print(f"  {tag} {n} config(s) checked for beam_width/beam_steps/test_scenes/seed consistency.")

    # Verify that 'test_scenes' contains only fushimi for these runs
    for cfg, key in zip(configs, keys):
        scenes = cfg.get("test_scenes", [])
        if scenes != ["fushimi"]:
            print(f"  {WARN} {key}: test_scenes={scenes} (expected ['fushimi'])")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity check for planning-level compression experiment.")
    p.add_argument("--rho_plan_list", nargs="+", type=float,
                   default=[1.0, 0.75, 0.50, 0.25, 0.10])
    p.add_argument("--downsample_scales", nargs="+", type=float,
                   default=[1.0, 0.5, 0.25, 0.125])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_show", type=int, default=10,
                   help="Number of early steps to print in divergence preview.")
    p.add_argument("--no_functional", action="store_true",
                   help="Skip functional compression tests (Part A).")
    p.add_argument("--no_lmdb", action="store_true",
                   help="Skip LMDB pre-flight checks (Part B).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    total_fails = 0

    if not args.no_functional:
        print("=" * 70)
        print("PART A: Functional compression tests")
        print("=" * 70)
        total_fails += _run_functional_tests(verbose=True)

    if not args.no_lmdb:
        print("\n" + "=" * 70)
        print("PART B: LMDB pre-flight checks")
        print("=" * 70)
        check_lmdb(args.rho_plan_list, args.downsample_scales, args.seed, args.n_show)

        # Report which MAGICIAN runs are still missing
        missing_block = []
        for r in args.rho_plan_list:
            rid = raux.run_id(r, args.seed)
            lmdb_dir = raux.SCENE_RESULTS_DIR / f"{rid}_lmdb"
            if not lmdb_dir.exists():
                missing_block.append(r)
        missing_dsup = []
        for s in args.downsample_scales:
            if s >= 1.0:
                continue
            stag = f"{s:.3f}".replace(".", "p")
            rid = f"dsup_fushimi_scale{stag}_seed{args.seed}"
            lmdb_dir = raux.SCENE_RESULTS_DIR / f"{rid}_lmdb"
            if not lmdb_dir.exists():
                missing_dsup.append(s)

        if missing_block or missing_dsup:
            print(f"\n{WARN} Missing LMDB runs:")
            if missing_block:
                print(f"       block_dropout rho: {missing_block}")
            if missing_dsup:
                print(f"       downsample_upsample scale: {missing_dsup}")
            print("       Run the experiment to generate them:")
            print("       python scripts/experiments/test_magician_planning_aux_invariance.py")
        else:
            print(f"\n{PASS} All required LMDB trajectories are present.")

    print("\n" + "=" * 70)
    if total_fails == 0:
        print(f"[DONE] All functional tests passed.")
    else:
        print(f"[DONE] {total_fails} functional test(s) FAILED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
