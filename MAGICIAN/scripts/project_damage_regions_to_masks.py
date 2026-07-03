#!/usr/bin/env python3
"""Project MAGICIAN 3D damage regions into COLMAP image masks."""

from __future__ import annotations

import argparse
import json
import pickle
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import lmdb
import numpy as np
from PIL import Image, ImageFilter


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
}


def read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected end of COLMAP binary file.")
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path: Path) -> Dict[int, dict]:
    cameras = {}
    with path.open("rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id {model_id}.")
            model_name, num_params = CAMERA_MODELS[model_id]
            params = np.asarray(read_next_bytes(fid, 8 * num_params, "d" * num_params), dtype=np.float64)
            cameras[camera_id] = {
                "camera_id": camera_id,
                "model_id": model_id,
                "model_name": model_name,
                "width": int(width),
                "height": int(height),
                "params": params,
            }
    return cameras


def read_images_binary(path: Path) -> List[dict]:
    images = []
    with path.open("rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(fid, 4, "i")[0]
            qvec = np.asarray(read_next_bytes(fid, 32, "dddd"), dtype=np.float64)
            tvec = np.asarray(read_next_bytes(fid, 24, "ddd"), dtype=np.float64)
            camera_id = read_next_bytes(fid, 4, "i")[0]

            name_bytes = bytearray()
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                if char == b"":
                    raise EOFError("Unexpected EOF while reading image name.")
                name_bytes.extend(char)
            name = name_bytes.decode("utf-8")

            num_points2d = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2d * 24, 1)
            images.append({
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            })
    return sorted(images, key=lambda item: item["name"])


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float64)


def camera_intrinsics(camera: dict) -> Tuple[float, float, float, float]:
    params = camera["params"]
    if camera["model_name"] == "SIMPLE_PINHOLE":
        f, cx, cy = params
        return float(f), float(f), float(cx), float(cy)
    if camera["model_name"] == "PINHOLE":
        fx, fy, cx, cy = params
        return float(fx), float(fy), float(cx), float(cy)
    raise ValueError(f"Unsupported camera model: {camera['model_name']}")


def open_lmdb_readonly(lmdb_dir: Path):
    return lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, max_readers=1)


def load_points_from_lmdb(lmdb_dir: Path, key: str) -> np.ndarray:
    env = open_lmdb_readonly(lmdb_dir)
    try:
        with env.begin(write=False) as txn:
            payload = txn.get(key.encode("utf-8"))
        if payload is None:
            raise KeyError(f"LMDB key not found: {key}")
        record = pickle.loads(payload)
    finally:
        env.close()

    if "points" not in record:
        raise KeyError(f"LMDB record {key} does not contain 'points'.")
    points = np.asarray(record["points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {points.shape}")
    return points[np.isfinite(points).all(axis=1)]


def load_damage_regions(path: Path) -> dict:
    with path.open("r") as f:
        data = json.load(f)
    if not data.get("regions"):
        raise ValueError(f"No damage regions found in {path}")
    for region in data["regions"]:
        if "sigma" not in region:
            region["sigma"] = float(region["radius"]) / 2.0
    return data


def compute_damage_weights(points: np.ndarray, damage_regions: dict, mode: str) -> np.ndarray:
    weights = np.zeros(points.shape[0], dtype=np.float32)
    for region in damage_regions["regions"]:
        center = np.asarray(region["center"], dtype=np.float64)
        radius = float(region["radius"])
        sigma = max(float(region.get("sigma", radius / 2.0)), 1e-12)
        dist2 = np.sum((points - center.reshape(1, 3)) ** 2, axis=1)
        if mode == "soft":
            region_weight = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
        elif mode == "hard":
            region_weight = (dist2 <= radius * radius).astype(np.float32)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        weights = np.maximum(weights, region_weight)
    return np.clip(weights, 0.0, 1.0)


def sample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[idx]


def project_points(points: np.ndarray, image: dict, camera: dict) -> Tuple[np.ndarray, np.ndarray]:
    R = qvec2rotmat(image["qvec"])
    t = image["tvec"].reshape(1, 3)
    xyz_cam = points @ R.T + t
    valid = xyz_cam[:, 2] > 1e-8
    xyz_cam = xyz_cam[valid]
    if xyz_cam.shape[0] == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    fx, fy, cx, cy = camera_intrinsics(camera)
    u = fx * (xyz_cam[:, 0] / xyz_cam[:, 2]) + cx
    v = fy * (xyz_cam[:, 1] / xyz_cam[:, 2]) + cy
    inside = (
        (u >= 0) & (u < camera["width"]) &
        (v >= 0) & (v < camera["height"])
    )
    u_idx = np.rint(u[inside]).astype(np.int32)
    v_idx = np.rint(v[inside]).astype(np.int32)
    rounded_inside = (
        (u_idx >= 0) & (u_idx < camera["width"]) &
        (v_idx >= 0) & (v_idx < camera["height"])
    )
    return u_idx[rounded_inside], v_idx[rounded_inside]


def project_points_pytorch3d_metadata(points: np.ndarray, pose_metadata: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points with MAGICIAN-exported PyTorch3D camera metadata.

    Returns integer pixel coordinates and the indexes of the input points that
    survived the camera frustum check.
    """
    import torch
    from pytorch3d.renderer import FoVPerspectiveCameras

    if points.shape[0] == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
        )

    width = int(pose_metadata["width"])
    height = int(pose_metadata["height"])
    min_side = float(min(width, height))
    R = torch.as_tensor(pose_metadata["R"], dtype=torch.float32)
    T = torch.as_tensor(pose_metadata["T"], dtype=torch.float32)
    if R.ndim == 2:
        R = R.unsqueeze(0)
    if T.ndim == 1:
        T = T.unsqueeze(0)
    zfar = float(pose_metadata.get("zfar", 100.0))
    camera = FoVPerspectiveCameras(R=R, T=T, zfar=zfar, device="cpu")
    pts = torch.as_tensor(points, dtype=torch.float32)

    projected = camera.get_full_projection_transform().transform_points(pts)
    view = camera.get_world_to_view_transform().transform_points(pts)

    min_ndc_x = width / min_side - ((width - 1) / (min_side - 1)) * 2.0
    max_ndc_x = width / min_side
    min_ndc_y = height / min_side - ((height - 1) / (min_side - 1)) * 2.0
    max_ndc_y = height / min_side

    valid = (
        (projected[:, 0] >= min_ndc_x) &
        (projected[:, 0] <= max_ndc_x) &
        (projected[:, 1] >= min_ndc_y) &
        (projected[:, 1] <= max_ndc_y) &
        (view[:, 2] > 0.0)
    )
    if not bool(valid.any()):
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int64),
        )

    projected_np = projected[valid].detach().cpu().numpy()
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1).detach().cpu().numpy().astype(np.int64)
    u = np.rint((width / min_side - projected_np[:, 0]) * (min_side - 1.0) / 2.0).astype(np.int32)
    v = np.rint((height / min_side - projected_np[:, 1]) * (min_side - 1.0) / 2.0).astype(np.int32)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u[inside], v[inside], valid_idx[inside]


def mask_name_for_image(image_name: str) -> str:
    return Path(image_name).with_suffix(".png").name


def save_overlay(image_path: Path, mask: Image.Image, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    mask_arr = np.asarray(mask, dtype=np.uint8) > 0
    arr = np.asarray(image, dtype=np.float32)
    red = np.asarray([255.0, 0.0, 0.0], dtype=np.float32)
    arr[mask_arr] = arr[mask_arr] * 0.45 + red * 0.55
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 3D damage regions to COLMAP image masks.")
    parser.add_argument("--colmap-dataset", required=True)
    parser.add_argument("--lmdb-dir", required=True)
    parser.add_argument("--lmdb-key", required=True)
    parser.add_argument("--damage-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=("soft", "hard"), default="soft")
    parser.add_argument("--damage-threshold", type=float, default=0.05)
    parser.add_argument("--dilate-radius", type=int, default=8)
    parser.add_argument("--max-damage-points", type=int, default=200000)
    parser.add_argument("--save-overlay", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    colmap_dataset = Path(args.colmap_dataset).expanduser()
    sparse_dir = colmap_dataset / "sparse" / "0"
    output_dir = Path(args.output_dir).expanduser()
    masks_dir = output_dir / "masks"
    overlays_dir = output_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlay:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")
    points = load_points_from_lmdb(Path(args.lmdb_dir).expanduser(), args.lmdb_key)
    damage_regions = load_damage_regions(Path(args.damage_json).expanduser())
    weights = compute_damage_weights(points, damage_regions, args.mode)
    damage_points = points[weights > args.damage_threshold]
    damage_points = sample_points(damage_points, args.max_damage_points, args.seed)

    per_image = []
    for image in images:
        camera = cameras[image["camera_id"]]
        mask_arr = np.zeros((camera["height"], camera["width"]), dtype=np.uint8)
        u, v = project_points(damage_points, image, camera)
        if u.size:
            mask_arr[v, u] = 255
        mask = Image.fromarray(mask_arr, mode="L")
        if args.dilate_radius > 0:
            mask = mask.filter(ImageFilter.MaxFilter(size=2 * args.dilate_radius + 1))

        out_name = mask_name_for_image(image["name"])
        mask.save(masks_dir / out_name)

        if args.save_overlay:
            image_path = colmap_dataset / "images" / image["name"]
            save_overlay(image_path, mask, overlays_dir / out_name)

        mask_pixels = int(np.asarray(mask).astype(bool).sum())
        mask_ratio = float(mask_pixels / (camera["height"] * camera["width"]))
        per_image.append({
            "image_name": image["name"],
            "mask_name": out_name,
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
        })

    ratios = [item["mask_ratio"] for item in per_image]
    manifest = {
        "colmap_dataset": str(colmap_dataset),
        "lmdb_dir": args.lmdb_dir,
        "lmdb_key": args.lmdb_key,
        "damage_json": args.damage_json,
        "num_images": len(images),
        "total_points": int(points.shape[0]),
        "total_damage_points": int(damage_points.shape[0]),
        "damage_threshold": args.damage_threshold,
        "dilate_radius": args.dilate_radius,
        "per_image_mask_pixels": {item["image_name"]: item["mask_pixels"] for item in per_image},
        "per_image_mask_ratio": {item["image_name"]: item["mask_ratio"] for item in per_image},
        "images": per_image,
    }
    with (output_dir / "mask_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[INFO] total points           : {points.shape[0]}")
    print(f"[INFO] selected damage points : {damage_points.shape[0]}")
    print(f"[INFO] num images             : {len(images)}")
    print(f"[INFO] average mask ratio     : {float(np.mean(ratios)) if ratios else 0.0:.6f}")
    print(f"[INFO] max mask ratio         : {float(np.max(ratios)) if ratios else 0.0:.6f}")
    print(f"[INFO] output path            : {output_dir}")


if __name__ == "__main__":
    main()
