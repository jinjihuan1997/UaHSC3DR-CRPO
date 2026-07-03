#!/usr/bin/env python3
"""Export MAGICIAN trajectory frames to a GSFusion Replica-style RGB-D dataset."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
TCOM_ROOT = ROOT.parent
DEFAULT_TRAJ_ROOT = (
    ROOT
    / "data/Macarons++/macarons++/fushimi"
    / "test_memory_0/training/0"
)
DEFAULT_OUTPUT = ROOT / "results" / "gsfusion_fushimi" / "test_memory_0"
DEFAULT_GSFUSION_ROOT = TCOM_ROOT / "GSFusion"


def sorted_frame_paths(frames_dir: Path) -> list[Path]:
    paths = list(frames_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt frames found in {frames_dir}")

    def key(path: Path):
        try:
            return (0, int(path.stem))
        except ValueError:
            return (1, path.name)

    return sorted(paths, key=key)


def as_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def frame_rgb_uint8(frame: dict) -> np.ndarray:
    rgb = frame["rgb"]
    if rgb.ndim == 4:
        rgb = rgb[0]
    arr = as_numpy(rgb).astype(np.float32)
    if arr.max(initial=0.0) <= 1.0:
        arr *= 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def frame_depth_mask(frame: dict) -> tuple[np.ndarray, np.ndarray]:
    depth = frame.get("zbuf")
    if depth is None:
        raise KeyError("Frame does not contain 'zbuf'.")
    if depth.ndim == 4:
        depth = depth[0, ..., 0]
    elif depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth_np = as_numpy(depth).astype(np.float32)

    mask = frame.get("mask")
    if mask is None:
        valid = np.ones_like(depth_np, dtype=bool)
    else:
        if mask.ndim == 4:
            mask = mask[0, ..., 0]
        elif mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        valid = as_numpy(mask).astype(bool)

    valid &= np.isfinite(depth_np) & (depth_np > 0)
    return depth_np, valid


def corrupt_valid_mask(
    valid: np.ndarray,
    keep_ratio: float,
    mode: str,
    block_size: int,
    seed: int,
    frame_index: int,
) -> np.ndarray:
    keep_ratio = float(keep_ratio)
    if keep_ratio < 0.0 or keep_ratio > 1.0:
        raise ValueError(f"depth_keep_ratio must be in [0, 1], got {keep_ratio}")
    valid = valid.astype(bool, copy=False)
    n_valid = int(valid.sum())
    if n_valid == 0 or keep_ratio >= 1.0:
        return valid.copy()

    rng = np.random.default_rng(int(seed) + int(frame_index) * 1000003)
    keep = np.zeros_like(valid, dtype=bool)
    target = int(round(keep_ratio * n_valid))
    if target <= 0:
        return keep

    if mode == "pixel_dropout":
        ys, xs = np.nonzero(valid)
        chosen = rng.choice(len(ys), size=min(target, len(ys)), replace=False)
        keep[ys[chosen], xs[chosen]] = True
        return keep

    if mode != "block_dropout":
        raise ValueError(f"Unsupported depth_corruption_mode: {mode}")
    if block_size <= 0:
        raise ValueError(f"depth_block_size must be positive, got {block_size}")

    blocks = []
    height, width = valid.shape
    for y0 in range(0, height, block_size):
        y1 = min(y0 + block_size, height)
        for x0 in range(0, width, block_size):
            x1 = min(x0 + block_size, width)
            count = int(valid[y0:y1, x0:x1].sum())
            if count > 0:
                blocks.append((y0, y1, x0, x1, count))
    order = rng.permutation(len(blocks))
    kept = 0
    for idx in order:
        y0, y1, x0, x1, count = blocks[int(idx)]
        gap = target - kept
        overshoot = kept + count - target
        if count <= gap or overshoot < gap:
            keep[y0:y1, x0:x1] |= valid[y0:y1, x0:x1]
            kept += count
        if kept >= target:
            break
    return keep


def save_depth_png(
    path: Path,
    depth_m: np.ndarray,
    valid: np.ndarray,
    depth_scale: float,
    max_depth_m: float | None,
) -> dict:
    depth = depth_m.copy()
    depth[~valid] = 0.0
    if max_depth_m is not None:
        depth[depth > max_depth_m] = 0.0
    scaled = np.rint(depth * depth_scale)
    clipped_pixels = int((scaled > np.iinfo(np.uint16).max).sum())
    scaled = np.clip(scaled, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    Image.fromarray(scaled, mode="I;16").save(path)

    valid_after = scaled > 0
    if np.any(valid_after):
        depth_kept = scaled[valid_after].astype(np.float32) / depth_scale
        return {
            "valid_pixels": int(valid_after.sum()),
            "depth_min": float(depth_kept.min()),
            "depth_max": float(depth_kept.max()),
            "depth_mean": float(depth_kept.mean()),
            "clipped_pixels": clipped_pixels,
        }
    return {
        "valid_pixels": 0,
        "depth_min": None,
        "depth_max": None,
        "depth_mean": None,
        "clipped_pixels": clipped_pixels,
    }


def pytorch3d_c2w(frame: dict, camera_axis: str) -> np.ndarray:
    """Convert saved PyTorch3D R/T to a column-vector camera-to-world matrix.

    PyTorch3D transforms row-vector points as X_cam = X_world R + T.
    Therefore C = -T R^T and column-vector C2W rotation is R.
    GSFusion's pinhole sensor follows OpenCV-style camera axes, so the default
    applies x/y flips from PyTorch3D view axes to x-right/y-down/z-forward.
    """
    r = frame["R"]
    t = frame["T"]
    if r.ndim == 3:
        r = r[0]
    if t.ndim == 2:
        t = t[0]
    r_np = as_numpy(r).astype(np.float64)
    t_np = as_numpy(t).astype(np.float64).reshape(3)

    c2w = np.eye(4, dtype=np.float64)
    rotation = r_np
    if camera_axis == "opencv":
        rotation = rotation @ np.diag([-1.0, -1.0, 1.0])
    elif camera_axis != "pytorch3d":
        raise ValueError(f"Unsupported camera_axis: {camera_axis}")

    c2w[:3, :3] = rotation
    c2w[:3, 3] = -r_np @ t_np
    return c2w


def intrinsics_from_fov(width: int, height: int, fov_deg: float, mode: str) -> tuple[float, float, float, float]:
    fov_rad = math.radians(float(fov_deg))
    if not 0.0 < fov_rad < math.pi:
        raise ValueError(f"fov_deg must be in (0, 180), got {fov_deg}")
    if mode == "vertical":
        fy = height / (2.0 * math.tan(fov_rad / 2.0))
        fx = fy
    elif mode == "horizontal":
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx
    elif mode == "min_side":
        focal = min(width, height) / (2.0 * math.tan(fov_rad / 2.0))
        fx = focal
        fy = focal
    else:
        raise ValueError(f"Unsupported fov_mode: {mode}")
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return fx, fy, cx, cy


def write_config(
    path: Path,
    args: argparse.Namespace,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> None:
    output_root = path.parent.resolve()
    optim_params = args.optim_params.resolve()
    log_file = output_root / "gsfusion_log.tsv"
    ply_path = output_root / "point_cloud"
    mesh_path = "" if args.disable_mesh else str(output_root / "mesh")
    map_dim = ", ".join(str(float(v)).rstrip("0").rstrip(".") for v in args.map_dim)

    text = f"""%YAML:1.2
map:
  dim:                        [{map_dim}]
  res:                        {args.map_res}

data:
  truncation_boundary_factor: {args.truncation_boundary_factor}
  max_weight:                 {args.max_weight}

sensor:
  width:                      {width}
  height:                     {height}
  fx:                         {fx}
  fy:                         {fy}
  cx:                         {cx}
  cy:                         {cy}
  near_plane:                 {args.near_plane}
  far_plane:                  {args.far_plane}

reader:
  reader_type:                "replica"
  sequence_path:              "{output_root}"
  ground_truth_file:          "{output_root / "traj.txt"}"
  inverse_scale:              {1.0 / args.depth_scale}
  fps:                        0.0
  drop_frames:                false
  verbose:                    {args.reader_verbose}

app:
  enable_ground_truth:        true
  optim_params_path:          "{optim_params}"
  ply_path:                   "{ply_path}"
  mesh_path:                  "{mesh_path}"
  slice_path:                 ""
  structure_path:             ""
  integration_rate:           {args.integration_rate}
  rendering_rate:             {args.rendering_rate}
  meshing_rate:               {args.meshing_rate}
  max_frames:                 {args.gsfusion_max_frames}
  log_file:                   "{log_file}"
"""
    path.write_text(text)


def export_dataset(args: argparse.Namespace) -> dict:
    traj_root = args.traj_root.resolve()
    frames_dir = traj_root / args.frames_subdir
    output = args.output.resolve()
    results_dir = output / "results"

    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Pass --overwrite to replace it.")
        shutil.rmtree(output)
    results_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted_frame_paths(frames_dir)
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise RuntimeError("No frames selected for export.")

    first = torch.load(frame_paths[0], map_location="cpu")
    rgb0 = frame_rgb_uint8(first)
    height, width = rgb0.shape[:2]

    if args.fx is None or args.fy is None:
        fx, fy, cx, cy = intrinsics_from_fov(width, height, args.fov_deg, args.fov_mode)
    else:
        fx = float(args.fx)
        fy = float(args.fy)
        cx = float(args.cx) if args.cx is not None else (width - 1.0) / 2.0
        cy = float(args.cy) if args.cy is not None else (height - 1.0) / 2.0

    rows = []
    traj_lines = []
    total_valid = 0
    total_clipped = 0
    depth_max_seen = 0.0
    for out_i, frame_path in enumerate(frame_paths):
        frame = first if out_i == 0 else torch.load(frame_path, map_location="cpu")
        rgb = frame_rgb_uint8(frame)
        if rgb.shape[:2] != (height, width):
            raise ValueError(f"Frame {frame_path} has resolution {rgb.shape[:2]}, expected {(height, width)}")

        rgb_name = f"frame{out_i:06d}.{args.rgb_ext}"
        depth_name = f"depth{out_i:06d}.png"
        Image.fromarray(rgb).save(results_dir / rgb_name, quality=args.rgb_quality)

        depth, valid = frame_depth_mask(frame)
        if depth.shape != (height, width):
            raise ValueError(f"Depth {frame_path} has shape {depth.shape}, expected {(height, width)}")
        original_valid_pixels = int(valid.sum())
        kept_valid = corrupt_valid_mask(
            valid,
            keep_ratio=args.depth_keep_ratio,
            mode=args.depth_corruption_mode,
            block_size=args.depth_block_size,
            seed=args.seed,
            frame_index=out_i,
        )
        stats = save_depth_png(results_dir / depth_name, depth, kept_valid, args.depth_scale, args.max_depth_m)

        c2w = pytorch3d_c2w(frame, args.camera_axis)
        traj_lines.append(" ".join(f"{value:.9g}" for value in c2w.reshape(-1)))

        valid_depth = depth[valid & np.isfinite(depth) & (depth > 0)]
        original_max = float(valid_depth.max()) if valid_depth.size else 0.0
        depth_max_seen = max(depth_max_seen, original_max)
        total_valid += int(stats["valid_pixels"])
        total_clipped += int(stats["clipped_pixels"])
        rows.append({
            "frame_index": out_i,
            "source_frame": str(frame_path),
            "rgb": str(results_dir / rgb_name),
            "depth": str(results_dir / depth_name),
            "valid_pixels": stats["valid_pixels"],
            "depth_min": stats["depth_min"],
            "depth_max": stats["depth_max"],
            "depth_mean": stats["depth_mean"],
            "source_depth_max": original_max,
            "clipped_pixels": stats["clipped_pixels"],
            "original_valid_pixels": original_valid_pixels,
            "kept_valid_pixels": stats["valid_pixels"],
            "depth_keep_ratio_actual": float(stats["valid_pixels"] / original_valid_pixels) if original_valid_pixels else 0.0,
        })

    (output / "traj.txt").write_text("\n".join(traj_lines) + "\n")
    write_config(output / "config.yaml", args, width, height, fx, fy, cx, cy)

    manifest = {
        "exporter": Path(__file__).name,
        "source_traj_root": str(traj_root),
        "source_frames_dir": str(frames_dir),
        "output": str(output),
        "format": "GSFusion Replica-style RGB-D",
        "num_frames": len(frame_paths),
        "width": width,
        "height": height,
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "fov_deg": args.fov_deg, "fov_mode": args.fov_mode},
        "depth": {
            "png_scale": args.depth_scale,
            "inverse_scale": 1.0 / args.depth_scale,
            "invalid_depth_value": 0,
            "max_depth_m": args.max_depth_m,
            "max_source_depth_m": depth_max_seen,
            "total_valid_pixels": total_valid,
            "total_clipped_pixels": total_clipped,
            "keep_ratio_requested": args.depth_keep_ratio,
            "corruption_mode": args.depth_corruption_mode,
            "block_size": args.depth_block_size,
            "seed": args.seed,
            "keep_ratio_actual": float(total_valid / sum(row["original_valid_pixels"] for row in rows)) if rows else 0.0,
        },
        "rgb": {
            "extension": args.rgb_ext,
            "quality": args.rgb_quality,
        },
        "pose": {
            "source": "PyTorch3D FoVPerspectiveCameras R/T",
            "camera_axis": args.camera_axis,
            "c2w_formula": "R_c2w=R[@diag(-1,-1,1) when opencv], C=-R@T",
        },
        "gsfusion_command": f"cd {args.gsfusion_root.resolve()} && ./build/app/gsfusion {output / 'config.yaml'}",
        "frames": rows,
    }
    (output / "export_manifest.json").write_text(json.dumps(manifest, indent=2))
    with (output / "integrity.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame_index",
                "source_frame",
                "rgb",
                "depth",
                "original_valid_pixels",
                "kept_valid_pixels",
                "depth_keep_ratio_actual",
                "depth_min",
                "depth_max",
                "depth_mean",
                "source_depth_max",
                "clipped_pixels",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    if total_clipped:
        print(f"[WARN] {total_clipped} depth pixels exceeded uint16 after scaling and were clipped.")
        print("[WARN] Use a smaller --depth-scale if this is unexpected.")

    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MAGICIAN .pt RGB-D trajectory frames for GSFusion.")
    parser.add_argument("--traj-root", type=Path, default=DEFAULT_TRAJ_ROOT)
    parser.add_argument("--frames-subdir", default="frames_highres", choices=["frames", "frames_highres"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit exported frames for smoke tests.")
    parser.add_argument("--rgb-ext", choices=["jpg", "png"], default="jpg")
    parser.add_argument("--rgb-quality", "--jpeg-quality", dest="rgb_quality", type=int, default=95)

    parser.add_argument("--depth-scale", type=float, default=100.0, help="Stored uint16 depth = depth_m * scale.")
    parser.add_argument("--max-depth-m", type=float, default=None, help="Set depths beyond this to 0 before export.")
    parser.add_argument("--depth-keep-ratio", type=float, default=1.0)
    parser.add_argument("--depth-corruption-mode", choices=["block_dropout", "pixel_dropout"], default="block_dropout")
    parser.add_argument("--depth-block-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera-axis", choices=["opencv", "pytorch3d"], default="opencv")

    parser.add_argument("--fov-deg", type=float, default=60.0)
    parser.add_argument("--fov-mode", choices=["vertical", "horizontal", "min_side"], default="vertical")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)

    parser.add_argument("--gsfusion-root", type=Path, default=DEFAULT_GSFUSION_ROOT)
    parser.add_argument("--optim-params", type=Path, default=DEFAULT_GSFUSION_ROOT / "parameter/optimization_params_replica.json")
    parser.add_argument("--map-dim", type=float, nargs=3, default=[160.0, 160.0, 160.0])
    parser.add_argument("--map-res", type=float, default=0.05)
    parser.add_argument("--truncation-boundary-factor", type=int, default=8)
    parser.add_argument("--max-weight", type=int, default=100)
    parser.add_argument("--near-plane", type=float, default=0.4)
    parser.add_argument("--far-plane", type=float, default=150.0)
    parser.add_argument("--integration-rate", type=int, default=1)
    parser.add_argument("--rendering-rate", type=int, default=1)
    parser.add_argument("--meshing-rate", type=int, default=0)
    parser.add_argument("--disable-mesh", action="store_true")
    parser.add_argument("--gsfusion-max-frames", type=int, default=-1)
    parser.add_argument("--reader-verbose", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = export_dataset(args)
    print("[INFO] Exported GSFusion dataset")
    print(f"  output: {manifest['output']}")
    print(f"  frames: {manifest['num_frames']}")
    print(f"  size:   {manifest['width']}x{manifest['height']}")
    print(f"  config: {Path(manifest['output']) / 'config.yaml'}")
    print(f"  run:    {manifest['gsfusion_command']}")


if __name__ == "__main__":
    main()
