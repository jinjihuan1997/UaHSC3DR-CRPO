#!/usr/bin/env python3
"""
Create a debug PLY that highlights MAGICIAN damage regions on the saved point cloud.

This script is standalone: it reads an existing MAGICIAN LMDB record and an existing
damage_regions.json, then writes an ASCII PLY for CloudCompare, MeshLab, or Blender.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable, Tuple

import lmdb
import numpy as np


def open_lmdb_readonly(lmdb_dir: Path):
    if lmdb_dir.is_file():
        lmdb_dir = lmdb_dir.parent
    return lmdb.open(
        str(lmdb_dir),
        readonly=True,
        lock=False,
        readahead=False,
        max_readers=1,
    )


def parse_rgb_color(value: str) -> np.ndarray:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"Color must be formatted as R,G,B, got '{value}'."
        )
    try:
        color = np.asarray([int(part) for part in parts], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Color must contain integer R,G,B values, got '{value}'."
        ) from exc
    if np.any(color < 0) or np.any(color > 255):
        raise argparse.ArgumentTypeError(
            f"Color values must be in [0,255], got '{value}'."
        )
    return color


def normalize_colors_to_uint8(colors, n_points: int) -> Tuple[np.ndarray, bool]:
    if colors is None:
        return np.full((n_points, 3), 20, dtype=np.uint8), False

    colors = np.asarray(colors)
    if colors.ndim != 2 or colors.shape[0] != n_points or colors.shape[1] < 3:
        print("[WARN] LMDB points_color is missing or malformed. Using gray.")
        return np.full((n_points, 3), 20, dtype=np.uint8), False

    colors = colors[:, :3]
    colors = np.nan_to_num(colors, nan=0.5, posinf=1.0, neginf=0.0)

    if np.issubdtype(colors.dtype, np.floating):
        cmax = float(np.max(colors)) if colors.size else 1.0
        cmin = float(np.min(colors)) if colors.size else 0.0
        if cmax <= 1.5 and cmin >= -0.5:
            colors = colors * 255.0

    return np.clip(colors, 0, 255).astype(np.uint8), True


def load_lmdb_points(lmdb_dir: Path, lmdb_key: str) -> Tuple[np.ndarray, np.ndarray, bool]:
    env = open_lmdb_readonly(lmdb_dir)
    try:
        with env.begin(write=False) as txn:
            payload = txn.get(lmdb_key.encode("utf-8"))

            if payload is None:
                available = []
                cursor = txn.cursor()
                for idx, (key, _) in enumerate(cursor):
                    if idx >= 20:
                        break
                    try:
                        available.append(key.decode("utf-8"))
                    except UnicodeDecodeError:
                        available.append(repr(key))
                raise KeyError(
                    f"LMDB key '{lmdb_key}' not found. First keys: {available}"
                )

        record = pickle.loads(payload)
    finally:
        env.close()

    if "points" not in record:
        raise KeyError(f"LMDB record '{lmdb_key}' does not contain 'points'.")

    points = np.asarray(record["points"], dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {points.shape}")

    colors, has_original_colors = normalize_colors_to_uint8(
        record.get("points_color"), points.shape[0]
    )
    return points, colors, has_original_colors


def load_damage_regions(damage_json: Path) -> dict:
    with damage_json.open("r") as f:
        data = json.load(f)

    regions = data.get("regions", [])
    if not regions:
        raise ValueError(f"No regions found in {damage_json}")

    for idx, region in enumerate(regions):
        if "center" not in region:
            raise ValueError(f"Region {idx} has no center.")
        if "radius" not in region:
            raise ValueError(f"Region {idx} has no radius.")
        if "sigma" not in region:
            region["sigma"] = float(region["radius"]) / 2.0

    return data


def sanitize_and_sample(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    total_before = points.shape[0]
    finite = np.isfinite(points).all(axis=1)
    if not np.all(finite):
        print(f"[WARN] Removing {int(np.sum(~finite))} non-finite points.")
        points = points[finite]
        colors = colors[finite]

    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        keep.sort()
        points = points[keep]
        colors = colors[keep]

    if points.shape[0] == 0:
        raise ValueError("No valid points remain after filtering/sampling.")

    return points, colors, total_before


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


def apply_highlight_colors(
    base_colors: np.ndarray,
    has_original_colors: bool,
    damage_weights: np.ndarray,
    threshold: float,
    background_scale: float,
) -> Tuple[np.ndarray, int]:
    original = base_colors.astype(np.float32)
    if has_original_colors:
        output = np.clip(original * background_scale, 0, 255)
    else:
        output = np.full_like(original, 20.0)

    highlight_mask = damage_weights > threshold
    if np.any(highlight_mask):
        red = np.asarray([255.0, 20.0, 20.0], dtype=np.float32)
        yellow = np.asarray([255.0, 230.0, 20.0], dtype=np.float32)
        weights = damage_weights[highlight_mask].reshape(-1, 1)
        output[highlight_mask] = red * (1.0 - weights) + yellow * weights

    return np.clip(output, 0, 255).astype(np.uint8), int(np.sum(highlight_mask))


def write_ascii_ply(
    output: Path,
    points: np.ndarray,
    colors: np.ndarray,
    damage_weights: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float damage_weight\n")
        f.write("end_header\n")

        for point, color, weight in zip(points, colors, damage_weights):
            f.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} "
                f"{float(weight):.8f}\n"
            )


def process_one(
    lmdb_dir: Path,
    lmdb_key: str,
    damage_json: Path,
    output: Path,
    mode: str,
    threshold: float,
    max_points: int,
    seed: int,
    background_scale: float,
) -> None:
    print(f"[INFO] LMDB dir          : {lmdb_dir}")
    print(f"[INFO] LMDB key          : {lmdb_key}")
    print(f"[INFO] damage json path  : {damage_json}")
    print(f"[INFO] output path       : {output}")

    points, colors, has_original_colors = load_lmdb_points(lmdb_dir, lmdb_key)
    damage_regions = load_damage_regions(damage_json)
    points, colors, total_before = sanitize_and_sample(points, colors, max_points, seed)
    damage_weights = compute_damage_weights(points, damage_regions, mode)
    output_colors, highlighted = apply_highlight_colors(
        colors,
        has_original_colors,
        damage_weights,
        threshold,
        background_scale,
    )
    write_ascii_ply(output, points, output_colors, damage_weights)

    nonzero_ratio = float(np.mean(damage_weights > 0.0)) if damage_weights.size else 0.0
    print(f"[INFO] total points before sampling : {total_before}")
    print(f"[INFO] total points after sampling  : {points.shape[0]}")
    print(f"[INFO] highlighted points           : {highlighted}")
    print(f"[INFO] threshold                    : {threshold}")
    print(f"[INFO] damage_weight min            : {float(np.min(damage_weights)):.8f}")
    print(f"[INFO] damage_weight max            : {float(np.max(damage_weights)):.8f}")
    print(f"[INFO] damage_weight mean           : {float(np.mean(damage_weights)):.8f}")
    print(f"[INFO] damage nonzero ratio         : {nonzero_ratio:.8f}")

    if highlighted == 0:
        print(
            "[WARN] Highlighted points is 0. Try lowering --threshold to 0.01 "
            "or increasing damage_radius_ratio / damage_soft_sigma_ratio."
        )


def infer_scene_from_batch_root(batch_root: Path) -> str:
    # Expected: <dataset>/<scene>/test_memory_0/training
    try:
        return batch_root.parent.parent.name
    except IndexError as exc:
        raise ValueError(f"Cannot infer scene from batch root: {batch_root}") from exc


def iter_batch_damage_jsons(batch_root: Path) -> Iterable[Path]:
    return sorted(batch_root.glob("*/damage_debug/damage_regions.json"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Highlight MAGICIAN damage regions on an LMDB point cloud."
    )
    parser.add_argument("--lmdb-dir", required=True, help="MAGICIAN result LMDB directory.")
    parser.add_argument("--lmdb-key", help="LMDB key, e.g. neuschwanstein/0.")
    parser.add_argument("--damage-json", help="Path to damage_regions.json.")
    parser.add_argument(
        "--output",
        help=(
            "Output PLY path for single mode. In batch mode this is treated as an "
            "output directory; default: results/debug_view."
        ),
    )
    parser.add_argument("--mode", choices=("soft", "hard"), default="soft")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument(
        "--max-points",
        type=int,
        default=300000,
        help="Random downsample limit. Use 0 to keep all points.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for downsampling.")
    parser.add_argument(
        "--background-scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied to non-highlighted original colors. Default keeps original colors. "
            "If no original colors are present, non-highlighted points use [20,20,20]."
        ),
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.75,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--damage-color",
        type=parse_rgb_color,
        default=parse_rgb_color("255,0,0"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--highlight-color",
        type=parse_rgb_color,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--highlight-core-color",
        type=parse_rgb_color,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--core-threshold",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--batch-root",
        help="Batch root, e.g. data/.../<scene>/test_memory_0/training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lmdb_dir = Path(args.lmdb_dir).expanduser()

    if args.batch_root:
        batch_root = Path(args.batch_root).expanduser()
        scene = infer_scene_from_batch_root(batch_root)
        output_dir = Path(args.output).expanduser() if args.output else Path("results/debug_view")
        damage_jsons = list(iter_batch_damage_jsons(batch_root))
        if not damage_jsons:
            raise FileNotFoundError(f"No damage_regions.json files found under {batch_root}")

        for damage_json in damage_jsons:
            traj_id = damage_json.parent.parent.name
            lmdb_key = f"{scene}/{traj_id}"
            output = output_dir / f"{scene}_{traj_id}_damage_highlight.ply"
            process_one(
                lmdb_dir=lmdb_dir,
                lmdb_key=lmdb_key,
                damage_json=damage_json,
                output=output,
                mode=args.mode,
                threshold=args.threshold,
                max_points=args.max_points,
                seed=args.seed,
                background_scale=args.background_scale,
            )
        return

    if not args.lmdb_key:
        raise ValueError("--lmdb-key is required unless --batch-root is used.")
    if not args.damage_json:
        raise ValueError("--damage-json is required unless --batch-root is used.")
    if not args.output:
        raise ValueError("--output is required unless --batch-root is used.")

    process_one(
        lmdb_dir=lmdb_dir,
        lmdb_key=args.lmdb_key,
        damage_json=Path(args.damage_json).expanduser(),
        output=Path(args.output).expanduser(),
        mode=args.mode,
        threshold=args.threshold,
        max_points=args.max_points,
        seed=args.seed,
        background_scale=args.background_scale,
    )


if __name__ == "__main__":
    main()
