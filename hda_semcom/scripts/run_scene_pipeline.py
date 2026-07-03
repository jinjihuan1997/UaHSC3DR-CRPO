"""Run the scene-level SemCom preparation pipeline.

Pipeline:
  1. rebuild dataset manifests for one scene
  2. train the RGB JSCC + digital auxiliary communication model for that scene
  3. build train/test sample-level offline lookup tables
  4. convert MAGICIAN trajectory CSVs to global meter coordinates for A2G/PPO

Use --dry-run first to print commands without executing them.
"""
import argparse
import os
import shlex
import subprocess
import sys

PYTHON = sys.executable


def run(cmd, dry_run=False):
    print("\n$ " + " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    if not dry_run:
        subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-id", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--dataset-root", default="/home/king/Downloads/Projects/TCOM/datasets/hda_semcom_dataset")
    ap.add_argument("--ckpt", default="checkpoints/stage3_final.pt")
    ap.add_argument("--out-prefix", default=None, help="Default: outputs/offline_lookup_<scene>_slot05_k10_22_snr0_20_step2")
    ap.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "val", "test"])
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    ap.add_argument("--digital-subcarriers", type=int, nargs="+", default=[10, 12, 14, 16, 18, 20, 22])
    ap.add_argument("--trajectory-inputs", nargs="+", default=None)
    ap.add_argument("--trajectory-input-dir", default=None)
    ap.add_argument("--trajectory-glob", default="*.csv")
    ap.add_argument("--trajectory-out", default=None)
    ap.add_argument("--scene-unit-m", type=float, default=2.0)
    ap.add_argument("--uav-min-altitude-m", type=float, default=20.0)
    ap.add_argument("--building-distance-m", type=float, default=2000.0)
    ap.add_argument("--vehicle-z-m", type=float, default=1.5)
    ap.add_argument("--skip-manifest", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-lookup", action="store_true")
    ap.add_argument("--skip-trajectory", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out_prefix = args.out_prefix or f"outputs/offline_lookup_{args.scene_id}_slot05_k10_22_snr0_20_step2"
    trajectory_out = args.trajectory_out or (
        f"outputs/magician_uav_trajectory/{args.scene_id}_trajectories_global_vehicle{int(args.building_distance_m)}.csv"
    )

    if not args.skip_manifest:
        run([
            PYTHON, "scripts/build_manifests.py",
            "--dataset-root", args.dataset_root,
            "--scene-id", args.scene_id,
        ], args.dry_run)

    if not args.skip_train:
        run([
            PYTHON, "scripts/train.py",
            "--config", args.config,
            "--train-scene-id", args.scene_id,
        ], args.dry_run)

    if not args.skip_lookup:
        for split in args.splits:
            run([
                PYTHON, "scripts/build_offline_lookup_table.py",
                "--config", args.config,
                "--ckpt", args.ckpt,
                "--split", split,
                "--scene-id", args.scene_id,
                "--digital-subcarriers", *[str(x) for x in args.digital_subcarriers],
                "--snrs", *[str(x) for x in args.snrs],
                "--out", f"{out_prefix}_{split}.csv",
            ], args.dry_run)

    if not args.skip_trajectory:
        cmd = [
            PYTHON, "scripts/convert_magician_trajectories.py",
            "--dataset-root", args.dataset_root,
            "--out", trajectory_out,
            "--scene-id", args.scene_id,
            "--scene-unit-m", str(args.scene_unit_m),
            "--uav-min-altitude-m", str(args.uav_min_altitude_m),
            "--building-center-x-m", str(args.building_distance_m),
            "--vehicle-z-m", str(args.vehicle_z_m),
            "--trajectory-id-from-filename",
        ]
        if args.trajectory_inputs:
            cmd = [x for x in cmd if x not in ["--dataset-root", args.dataset_root]]
            cmd.extend(["--inputs", *args.trajectory_inputs])
        if args.trajectory_input_dir:
            cmd = [x for x in cmd if x not in ["--dataset-root", args.dataset_root]]
            cmd.extend(["--input-dir", args.trajectory_input_dir, "--glob", args.trajectory_glob])
        run(cmd, args.dry_run)

    print("\nPipeline finished." if not args.dry_run else "\nDry run finished.")


if __name__ == "__main__":
    main()
