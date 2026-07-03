"""Convert MAGICIAN UAV trajectory CSVs to global meter coordinates for A2G simulation.

Input CSV format can be either raw MAGICIAN coordinates:
    scene_id,trajectory_id,step,uav_x,uav_y,uav_z,...

or local meter coordinates:
    scene_id,trajectory_id,step,uav_x_m,uav_y_m,uav_z_m,...

The output is a single CSV with fields consumed by src.channel.a2g.A2GChannel:
    uav_x_global_m,uav_y_global_m,uav_z_global_m,vehicle_x_m,vehicle_y_m,vehicle_z_m
"""
import argparse
import csv
import glob
import math
import json
import os
import sys
from pathlib import Path


def read_rows(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty trajectory csv: {path}")
    return rows


def dataset_pose_inputs(dataset_root, scene_id):
    scene_dir = Path(dataset_root) / "scenes" / scene_id
    if not scene_dir.exists():
        raise FileNotFoundError(f"scene not found: {scene_dir}")
    containers = []
    obs = scene_dir / "observations"
    if obs.is_dir():
        containers.append(obs)
    containers.extend(sorted(p for p in scene_dir.glob("damage_config_*") if p.is_dir()))
    for container in containers:
        for traj_dir in sorted(p for p in container.iterdir() if p.is_dir()):
            pose_dir = traj_dir / "poses"
            if pose_dir.is_dir() and list(pose_dir.glob("*.json")):
                yield container.name, traj_dir.name, pose_dir


def rows_from_pose_dir(scene_id, container_name, traj_id, pose_dir):
    rows = []
    for i, path in enumerate(sorted(pose_dir.glob("*.json"))):
        obj = json.loads(path.read_text())
        center = obj.get("camera_center")
        if center is None:
            continue
        if center and isinstance(center[0], list):
            center = center[0]
        rows.append({
            "scene_id": scene_id,
            "observation_group": container_name,
            "trajectory_id": traj_id,
            "step": int(obj.get("frame_id", i)),
            "uav_x": float(center[0]),
            "uav_y": float(center[1]),
            "uav_z": float(center[2]),
        })
    if not rows:
        raise ValueError(f"no camera_center rows found in {pose_dir}")
    return rows


def as_float(row, key, default=None):
    if key in row and row[key] not in (None, ""):
        return float(row[key])
    if default is not None:
        return float(default)
    raise KeyError(key)


def convert_rows(rows, args, fallback_traj_id, source_name):
    has_meter = all(k in rows[0] for k in ("uav_x_m", "uav_y_m", "uav_z_m"))
    has_raw = all(k in rows[0] for k in ("uav_x", "uav_y", "uav_z"))
    if not has_meter and not has_raw:
        raise ValueError(
            f"{path} must contain either uav_x/uav_y/uav_z or uav_x_m/uav_y_m/uav_z_m columns"
        )

    if has_meter:
        local = [(as_float(r, "uav_x_m"), as_float(r, "uav_y_m"), as_float(r, "uav_z_m")) for r in rows]
    else:
        raw_z_min = min(as_float(r, "uav_z") for r in rows)
        local = []
        for r in rows:
            x_m = args.scene_unit_m * as_float(r, "uav_x")
            y_m = args.scene_unit_m * as_float(r, "uav_y")
            z_m = args.uav_min_altitude_m + args.scene_unit_m * (as_float(r, "uav_z") - raw_z_min)
            local.append((x_m, y_m, z_m))

    x_center = 0.5 * (min(p[0] for p in local) + max(p[0] for p in local))
    y_center = 0.5 * (min(p[1] for p in local) + max(p[1] for p in local))

    out = []
    prev = None
    scene_id = args.scene_id
    for i, (r, (x_m, y_m, z_m)) in enumerate(zip(rows, local)):
        traj_id = r.get("trajectory_id") or source_name or str(fallback_traj_id)
        if args.trajectory_id_from_filename and source_name:
            traj_id = source_name
        row_scene = scene_id or r.get("scene_id", "")
        step = int(float(r.get("step", i)))

        gx = args.building_center_x_m + (x_m - x_center)
        gy = args.building_center_y_m + (y_m - y_center)
        gz = args.building_center_z_m + z_m
        vx, vy, vz = args.vehicle_x_m, args.vehicle_y_m, args.vehicle_z_m

        if prev is None:
            step_distance = 0.0
        else:
            step_distance = math.sqrt((gx - prev[0]) ** 2 + (gy - prev[1]) ** 2 + (gz - prev[2]) ** 2)
        prev = (gx, gy, gz)

        horizontal = math.hypot(gx - vx, gy - vy)
        distance_3d = math.sqrt(horizontal * horizontal + (gz - vz) ** 2)
        elevation = math.degrees(math.atan2(max(gz - vz, 0.0), max(horizontal, 1e-6)))
        out.append({
            "scene_id": row_scene,
            "trajectory_id": traj_id,
            "step": step,
            "uav_x_global_m": gx,
            "uav_y_global_m": gy,
            "uav_z_global_m": gz,
            "vehicle_x_m": vx,
            "vehicle_y_m": vy,
            "vehicle_z_m": vz,
            "building_center_x_m": args.building_center_x_m,
            "building_center_y_m": args.building_center_y_m,
            "building_center_z_m": args.building_center_z_m,
            "step_distance_m": step_distance,
            "vehicle_horizontal_distance_m": horizontal,
            "vehicle_3d_distance_m": distance_3d,
            "elevation_angle_deg": elevation,
        })
    return out


def convert_one(path, args, fallback_traj_id):
    rows = read_rows(path)
    source_name = os.path.splitext(os.path.basename(path))[0]
    return convert_rows(rows, args, fallback_traj_id, source_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=None, help="Input trajectory CSV files.")
    ap.add_argument("--input-dir", default=None, help="Directory containing trajectory CSV files.")
    ap.add_argument("--glob", default="*.csv", help="Glob pattern used with --input-dir.")
    ap.add_argument("--dataset-root", default=None, help="Read trajectory poses from dataset scenes/<scene>/{observations,damage_config_*}/traj*/poses.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene-id", default=None)
    ap.add_argument("--scene-unit-m", type=float, default=2.0, help="Meters per raw MAGICIAN coordinate unit.")
    ap.add_argument("--uav-min-altitude-m", type=float, default=20.0)
    ap.add_argument("--building-center-x-m", type=float, default=2000.0)
    ap.add_argument("--building-center-y-m", type=float, default=0.0)
    ap.add_argument("--building-center-z-m", type=float, default=0.0)
    ap.add_argument("--vehicle-x-m", type=float, default=0.0)
    ap.add_argument("--vehicle-y-m", type=float, default=0.0)
    ap.add_argument("--vehicle-z-m", type=float, default=1.5)
    ap.add_argument("--trajectory-id-from-filename", action="store_true")
    args = ap.parse_args()

    rows = []
    input_count = 0
    if args.dataset_root:
        if not args.scene_id:
            raise SystemExit("--scene-id is required with --dataset-root")
        for i, (container_name, traj_id, pose_dir) in enumerate(dataset_pose_inputs(args.dataset_root, args.scene_id)):
            pose_rows = rows_from_pose_dir(args.scene_id, container_name, traj_id, pose_dir)
            rows.extend(convert_rows(pose_rows, args, i, traj_id))
            input_count += 1
    else:
        paths = []
        if args.inputs:
            paths.extend(args.inputs)
        if args.input_dir:
            paths.extend(sorted(glob.glob(os.path.join(args.input_dir, args.glob))))
        paths = sorted(dict.fromkeys(paths))
        if not paths:
            raise SystemExit("No input trajectory CSV files found. Use --inputs, --input-dir, or --dataset-root.")
        for i, path in enumerate(paths):
            rows.extend(convert_one(path, args, i))
        input_count = len(paths)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fieldnames = [
        "scene_id", "trajectory_id", "step",
        "uav_x_global_m", "uav_y_global_m", "uav_z_global_m",
        "vehicle_x_m", "vehicle_y_m", "vehicle_z_m",
        "building_center_x_m", "building_center_y_m", "building_center_z_m",
        "step_distance_m", "vehicle_horizontal_distance_m", "vehicle_3d_distance_m", "elevation_angle_deg",
    ]
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Converted {input_count} trajectories, wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
