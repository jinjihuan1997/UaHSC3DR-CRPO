"""Export one MAGICIAN PNG/JSON trajectory to GSFusion's Replica-like format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def _load_pose(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _c2w_from_pose(pose: dict) -> np.ndarray:
    if pose.get("coordinate_system") != "MAGICIAN/PyTorch3D":
        raise ValueError(f"Unsupported coordinate_system: {pose.get('coordinate_system')}")
    r = np.asarray(pose["R"][0], dtype=np.float64)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = r @ np.diag([-1.0, -1.0, 1.0])
    c2w[:3, 3] = np.asarray(pose["camera_center"][0], dtype=np.float64)
    return c2w


def _projection_from_pose(pose: dict) -> np.ndarray:
    full = np.asarray(pose["projection_matrix"][0], dtype=np.float64)
    w2v = np.asarray(pose["world_to_view_matrix"][0], dtype=np.float64)
    return np.linalg.inv(w2v) @ full


def _intrinsics_from_pose(pose: dict, width: int, height: int) -> tuple[float, float, float, float]:
    p = _projection_from_pose(pose)
    focal_scale = min(width, height) / 2.0
    fx = abs(float(p[0, 0])) * focal_scale
    fy = abs(float(p[1, 1])) * focal_scale
    return fx, fy, (width - 1.0) / 2.0, (height - 1.0) / 2.0


def _resize_image(image: Image.Image, width: int, height: int, is_depth: bool) -> Image.Image:
    resampling = getattr(Image, "Resampling", Image)
    mode = resampling.NEAREST if is_depth else resampling.LANCZOS
    return image.resize((width, height), mode)


def _write_config(
    path: Path,
    sequence_path: Path,
    output_path: Path,
    optim_params_path: Path,
    width: int,
    height: int,
    intrinsics: tuple[float, float, float, float],
    depth_scale: float,
    max_frames: int,
    map_dim: tuple[float, float, float],
    map_res: float,
) -> None:
    fx, fy, cx, cy = intrinsics
    dim = ", ".join(str(float(v)).rstrip("0").rstrip(".") for v in map_dim)
    text = f"""%YAML:1.2
map:
  dim:                        [{dim}]
  res:                        {map_res}

data:
  truncation_boundary_factor: 8
  max_weight:                 100

sensor:
  width:                      {int(width)}
  height:                     {int(height)}
  fx:                         {float(fx)}
  fy:                         {float(fy)}
  cx:                         {float(cx)}
  cy:                         {float(cy)}
  near_plane:                 0.1
  far_plane:                  200.0

reader:
  reader_type:                "replica"
  sequence_path:              "{sequence_path}"
  ground_truth_file:          "{sequence_path / "traj.txt"}"
  inverse_scale:              {float(depth_scale)}
  fps:                        0.0
  drop_frames:                false
  verbose:                    0

app:
  enable_ground_truth:        true
  optim_params_path:          "{optim_params_path}"
  ply_path:                   "{output_path / "point_cloud"}"
  mesh_path:                  "{output_path / "mesh"}"
  slice_path:                 ""
  structure_path:             ""
  integration_rate:           1
  rendering_rate:             1
  meshing_rate:               0
  max_frames:                 {int(max_frames)}
  log_file:                   "{output_path / "log.tsv"}"
"""
    path.write_text(text)


def _depth_stats(depth_u16: np.ndarray, scale: float) -> dict:
    valid = depth_u16 > 0
    values = depth_u16[valid].astype(np.float64) * scale
    if values.size == 0:
        return {"valid_ratio": 0.0, "min": None, "median": None, "p95": None, "max": None}
    return {
        "valid_ratio": float(valid.mean()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def export(args: argparse.Namespace) -> dict:
    traj_root = args.traj_root.resolve()
    output = args.output.resolve()
    sequence = output / "sequence"
    results = sequence / "results"
    gsfusion_output = output / "gsfusion_output"
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    results.mkdir(parents=True, exist_ok=True)
    gsfusion_output.mkdir(parents=True, exist_ok=True)

    pose_paths = sorted((traj_root / "poses").glob("*.json"))
    if args.max_frames is not None:
        pose_paths = pose_paths[: args.max_frames]
    if not pose_paths:
        raise FileNotFoundError(f"No pose JSON files found under {traj_root / 'poses'}")

    first_pose = _load_pose(pose_paths[0])
    src_width = int(first_pose["width"])
    src_height = int(first_pose["height"])
    width = int(round(src_width * args.resize))
    height = int(round(src_height * args.resize))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid output size {width}x{height} from resize={args.resize}")

    depth_scale = float(first_pose["depth_encoding"]["scale"])
    intrinsics = _intrinsics_from_pose(first_pose, width, height)
    c2w_world = []
    frames = []

    for out_i, pose_path in enumerate(pose_paths):
        pose = _load_pose(pose_path)
        image_name = pose["image_name"]
        rgb_path = traj_root / "clean_images" / image_name
        depth_path = traj_root / "depth" / image_name
        if not rgb_path.exists():
            raise FileNotFoundError(rgb_path)
        if not depth_path.exists():
            raise FileNotFoundError(depth_path)

        rgb = Image.open(rgb_path).convert("RGB")
        depth = Image.open(depth_path)
        if rgb.size != (src_width, src_height):
            raise ValueError(f"{rgb_path} size {rgb.size}, expected {(src_width, src_height)}")
        if depth.size != (src_width, src_height):
            raise ValueError(f"{depth_path} size {depth.size}, expected {(src_width, src_height)}")
        if args.resize != 1.0:
            rgb = _resize_image(rgb, width, height, is_depth=False)
            depth = _resize_image(depth, width, height, is_depth=True)

        rgb.save(results / f"frame{out_i:06d}.jpg", quality=args.jpeg_quality)
        depth.save(results / f"depth{out_i:06d}.png")

        c2w = _c2w_from_pose(pose)
        c2w_world.append(c2w)
        depth_u16 = np.asarray(depth)
        frames.append({
            "frame_index": out_i,
            "source_frame": image_name,
            "rgb": str(rgb_path),
            "depth_image": str(depth_path),
            "camera_center": pose["camera_center"][0],
            "depth_stats": _depth_stats(depth_u16, depth_scale),
        })

    world_from_local = c2w_world[0] if args.normalize_poses else np.eye(4, dtype=np.float64)
    local_from_world = np.linalg.inv(world_from_local)
    c2w_out = [local_from_world @ c2w if args.normalize_poses else c2w for c2w in c2w_world]
    traj_lines = [" ".join(f"{value:.9g}" for value in c2w.reshape(-1)) for c2w in c2w_out]
    (sequence / "traj.txt").write_text("\n".join(traj_lines) + "\n")
    np.savetxt(output / "world_from_local.txt", world_from_local, fmt="%.12g")

    rotations = np.stack([c2w[:3, :3] for c2w in c2w_out])
    dets = np.linalg.det(rotations)
    ortho = np.matmul(np.swapaxes(rotations, 1, 2), rotations)
    ortho_err = np.max(np.abs(ortho - np.eye(3)), axis=(1, 2))
    config_path = output / "gsfusion_config.yaml"
    _write_config(
        config_path,
        sequence,
        gsfusion_output,
        args.gsfusion_root.resolve() / "parameter/optimization_params_replica.json",
        width,
        height,
        intrinsics,
        depth_scale,
        len(c2w_out),
        tuple(args.map_dim),
        args.map_res,
    )

    report = {
        "source_trajectory": str(traj_root),
        "output": str(output),
        "sequence_path": str(sequence),
        "config_path": str(config_path),
        "num_frames": len(c2w_out),
        "source_size": {"width": src_width, "height": src_height},
        "gsfusion_size": {"width": width, "height": height, "resize": args.resize},
        "intrinsics": {
            "fx": intrinsics[0],
            "fy": intrinsics[1],
            "cx": intrinsics[2],
            "cy": intrinsics[3],
            "pure_projection_matrix": _projection_from_pose(first_pose).tolist(),
            "focal_scale": min(width, height) / 2.0,
        },
        "depth_encoding": {
            "stored_value": "uint16",
            "depth_value": "uint16_value * depth_scale",
            "depth_scale": depth_scale,
        },
        "pose": {
            "input_to_gsfusion": "local C2W" if args.normalize_poses else "world C2W",
            "world_from_local": str(output / "world_from_local.txt"),
            "formula": "C2W[:3,:3] = PyTorch3D_R @ diag(-1,-1,1); C2W[:3,3] = camera_center",
            "det_min": float(dets.min()),
            "det_max": float(dets.max()),
            "orthonormal_error_max": float(ortho_err.max()),
            "first_pose_is_identity": bool(np.allclose(c2w_out[0], np.eye(4), atol=1e-6)),
        },
        "file_layout": {
            "rgb_pattern": "sequence/results/frame%06d.jpg",
            "depth_pattern": "sequence/results/depth%06d.png",
            "traj": "sequence/traj.txt",
        },
        "frames_sample": frames[:3] + frames[-3:] if len(frames) > 6 else frames,
        "gsfusion_command": f"cd {args.gsfusion_root.resolve()} && ./build/app/gsfusion {config_path}",
    }
    (output / "input_audit_report.json").write_text(json.dumps(report, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gsfusion-root", type=Path, default=Path("/home/king/Downloads/Projects/TCOM/GSFusion"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--resize", type=float, default=0.5)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--normalize-poses", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--map-dim", type=float, nargs=3, default=[120.0, 120.0, 120.0])
    parser.add_argument("--map-res", type=float, default=0.10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    report = export(parse_args())
    print("[exported]", report["output"])
    print("[config]", report["config_path"])
    print("[run]", report["gsfusion_command"])


if __name__ == "__main__":
    main()
