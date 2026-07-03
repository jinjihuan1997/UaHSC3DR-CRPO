"""Export MAGICIAN test_memory RGB-D frames to the HDA SemCom dataset layout."""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def _numeric_key(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.name


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _parse_ids(value):
    if value is None:
        return None
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def _parse_trajectory_map(value):
    if value is None:
        return None
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"trajectory map item must be SOURCE:TARGET, got {item!r}")
        source, target = item.split(":", 1)
        out.append((int(source), int(target)))
    return out


def _frame_visibility(frame_path):
    frame = torch.load(frame_path, map_location="cpu")
    rgb = _to_numpy(frame["rgb"])[0]
    zbuf = _to_numpy(frame["zbuf"])[0, ..., 0]
    mask = _to_numpy(frame["mask"])[0, ..., 0].astype(bool)
    valid = mask & np.isfinite(zbuf) & (zbuf > 0.0)
    white_ratio = float(np.mean(np.all(rgb > 0.98, axis=2)))
    return {
        "rgb_std": float(rgb.std()),
        "white_ratio": white_ratio,
        "valid_depth_ratio": float(valid.mean()),
    }


def _first_visible_frame(frame_paths, min_rgb_std, max_white_ratio, min_valid_depth_ratio):
    for idx, frame_path in enumerate(frame_paths):
        stats = _frame_visibility(frame_path)
        if (
            stats["rgb_std"] >= min_rgb_std
            and stats["white_ratio"] <= max_white_ratio
            and stats["valid_depth_ratio"] >= min_valid_depth_ratio
        ):
            return idx, stats
    return None, None


def _camera_center_from_rt(r, t):
    r = np.asarray(r, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    return (-t @ np.linalg.inv(r)).astype(np.float32)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _export_frame(frame_path, out_dir, scene, traj, local_idx):
    frame = torch.load(frame_path, map_location="cpu")
    rgb = _to_numpy(frame["rgb"])[0]
    zbuf = _to_numpy(frame["zbuf"])[0, ..., 0]
    mask = _to_numpy(frame["mask"])[0, ..., 0].astype(bool)
    r = _to_numpy(frame["R"])[0].astype(np.float32)
    t = _to_numpy(frame["T"])[0].astype(np.float32)
    zfar = float(frame.get("zfar", 750.0))

    height, width = rgb.shape[:2]
    image_name = f"{local_idx:06d}.png"
    clean_path = out_dir / "clean_images" / image_name
    depth_path = out_dir / "depth" / image_name
    valid_path = out_dir / "valid_masks" / image_name
    pose_path = out_dir / "poses" / f"{local_idx:06d}.json"

    clean_path.parent.mkdir(parents=True, exist_ok=True)
    depth_path.parent.mkdir(parents=True, exist_ok=True)
    valid_path.parent.mkdir(parents=True, exist_ok=True)
    pose_path.parent.mkdir(parents=True, exist_ok=True)

    rgb_u8 = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    valid = mask & np.isfinite(zbuf) & (zbuf > 0.0)
    depth_scale = zfar / 65535.0
    depth_u16 = np.zeros_like(zbuf, dtype=np.uint16)
    depth_u16[valid] = np.clip(zbuf[valid] / depth_scale, 1, 65535).astype(np.uint16)
    valid_u8 = valid.astype(np.uint8) * 255

    Image.fromarray(rgb_u8, mode="RGB").save(clean_path)
    Image.fromarray(depth_u16, mode="I;16").save(depth_path)
    Image.fromarray(valid_u8, mode="L").save(valid_path)

    camera_center = _camera_center_from_rt(r, t)
    focal_scale = min(width, height) / 2.0
    focal = math.sqrt(3.0) * focal_scale
    pose = {
        "scene_id": scene,
        "damage_enabled": False,
        "damage_config_id": None,
        "trajectory_id": traj,
        "frame_id": int(local_idx),
        "source_frame_id": int(_numeric_key(frame_path)),
        "image_name": image_name,
        "rgb_image": f"clean_images/{image_name}",
        "depth_image": f"depth/{image_name}",
        "valid_mask": f"valid_masks/{image_name}",
        "width": int(width),
        "height": int(height),
        "zfar": zfar,
        "R": [r.tolist()],
        "T": [t.tolist()],
        "camera_center": [camera_center.tolist()],
        "fx": float(focal),
        "fy": float(focal),
        "cx": float((width - 1) / 2.0),
        "cy": float((height - 1) / 2.0),
        "camera_model": "pytorch3d_fov_perspective",
        "coordinate_system": "MAGICIAN/PyTorch3D",
        "depth_encoding": {
            "format": "uint16_png",
            "scale": depth_scale,
            "unit": "scene_units",
            "invalid_value": 0,
            "depth_value": "uint16_value * scale",
        },
        "valid_mask_encoding": {
            "format": "uint8_png",
            "valid_value": 255,
            "invalid_value": 0,
        },
    }
    _write_json(pose_path, pose)
    return {
        "scene_id": scene,
        "damage_config_id": None,
        "trajectory_id": traj,
        "frame_id": int(local_idx),
        "image_name": image_name,
        "clean_image": str(clean_path.relative_to(args.output_root)),
        "depth": str(depth_path.relative_to(args.output_root)),
        "valid_mask": str(valid_path.relative_to(args.output_root)),
        "camera_pose": str(pose_path.relative_to(args.output_root)),
        "clean_image_role": "ground_truth",
        "snr_generation": "online_random_during_training",
        "channel_generation": "online_random_during_training",
        "width": int(width),
        "height": int(height),
    }


def export_trajectory(scene_root, output_root, scene, traj_idx, max_frames, target_traj_idx=None):
    target_traj_idx = traj_idx if target_traj_idx is None else target_traj_idx
    traj = f"traj{target_traj_idx}"
    memory_dir = scene_root / scene / "test_memory_0" / "training" / str(traj_idx)
    frame_dir = memory_dir / "frames_highres"
    if not frame_dir.exists():
        frame_dir = memory_dir / "frames"
    if not frame_dir.exists():
        raise FileNotFoundError(f"missing MAGICIAN memory frames: {memory_dir}")
    frame_paths = sorted(frame_dir.glob("*.pt"), key=_numeric_key)
    start_idx = 0
    start_stats = None
    if args.skip_invisible_start:
        start_idx, start_stats = _first_visible_frame(
            frame_paths,
            args.min_start_rgb_std,
            args.max_start_white_ratio,
            args.min_start_valid_depth_ratio,
        )
        if start_idx is None:
            raise RuntimeError(
                f"no visible start frame found for {scene}/{traj}; "
                f"thresholds: rgb_std>={args.min_start_rgb_std}, "
                f"white_ratio<={args.max_start_white_ratio}, "
                f"valid_depth_ratio>={args.min_start_valid_depth_ratio}"
            )
        frame_paths = frame_paths[start_idx:]
    if max_frames > 0:
        frame_paths = frame_paths[:max_frames]
    if not frame_paths:
        raise FileNotFoundError(f"no .pt frames under {frame_dir}")

    out_dir = output_root / "scenes" / scene / "observations" / traj
    rows = [_export_frame(path, out_dir, scene, traj, idx) for idx, path in enumerate(frame_paths)]
    _write_jsonl(out_dir / "frame_manifest.jsonl", rows)
    _write_json(out_dir / "trajectory_summary.json", {
        "scene_id": scene,
        "damage_config_id": None,
        "trajectory_id": traj,
        "num_frames": len(rows),
        "source_start_frame": int(_numeric_key(frame_paths[0])),
        "skipped_source_frames": int(start_idx),
        "start_visibility": start_stats or _frame_visibility(frame_paths[0]),
        "damage_enabled": False,
        "source_memory_dir": str(memory_dir),
        "source_trajectory_id": int(traj_idx),
    })
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scenes", required=True, help="Comma-separated scene ids.")
    parser.add_argument("--num-trajectories", type=int, default=5)
    parser.add_argument("--trajectory-ids", default=None,
                        help="Comma-separated source trajectory ids to export. Overrides --num-trajectories.")
    parser.add_argument("--trajectory-map", default=None,
                        help="Comma-separated SOURCE:TARGET trajectory ids, e.g. 13:5.")
    parser.add_argument("--max-frames", type=int, default=102)
    parser.add_argument("--skip-invisible-start", action="store_true",
                        help="Start each exported trajectory at the first visible, non-white source frame.")
    parser.add_argument("--min-start-rgb-std", type=float, default=0.02)
    parser.add_argument("--max-start-white-ratio", type=float, default=0.98)
    parser.add_argument("--min-start-valid-depth-ratio", type=float, default=0.05)
    global args
    args = parser.parse_args()

    scene_root = Path(args.scene_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    scenes = [item.strip() for item in args.scenes.split(",") if item.strip()]
    output_root.mkdir(parents=True, exist_ok=True)
    total = 0
    for scene in scenes:
        trajectory_map = _parse_trajectory_map(args.trajectory_map)
        if trajectory_map is None:
            trajectory_map = [(idx, idx) for idx in (_parse_ids(args.trajectory_ids) or list(range(args.num_trajectories)))]
        for idx, target_idx in trajectory_map:
            count = export_trajectory(scene_root, output_root, scene, idx, args.max_frames, target_idx)
            total += count
            print(f"[exported] {scene}/traj{target_idx} source_traj={idx} frames={count}", flush=True)
    print(f"exported_frames={total}")


if __name__ == "__main__":
    main()
