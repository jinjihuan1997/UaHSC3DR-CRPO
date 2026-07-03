#!/usr/bin/env python3
"""Run ROI semantic wireless digital transmission pipeline for a MAGICIAN trajectory."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macarons.communication.roi_semantic_codec import encode_decode_roi_semantic, select_semantic_mode
from macarons.communication.semantic_modes import get_default_modes
from macarons.communication.wireless_capacity import capacity_bits_per_slot

from project_damage_regions_to_masks import (
    compute_damage_weights,
    load_damage_regions,
    load_points_from_lmdb,
    mask_name_for_image,
    project_points,
    read_cameras_binary,
    read_images_binary,
    sample_points,
    save_overlay,
)

try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ROI semantic communication pipeline.")
    parser.add_argument("--input-dataset", required=True)
    parser.add_argument("--lmdb-dir", required=True)
    parser.add_argument("--lmdb-key", required=True)
    parser.add_argument("--damage-json", required=True)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--capacity-bits-per-frame", type=float, default=None)
    parser.add_argument("--bandwidth-hz", type=float, default=None)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--slot-time-s", type=float, default=None)
    parser.add_argument("--damage-threshold", type=float, default=0.05)
    parser.add_argument("--dilate-radius", type=int, default=8)
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--mode-selection", choices=("fixed", "capacity"), default="fixed")
    parser.add_argument("--fixed-mode", default="M2")
    parser.add_argument("--mode", choices=("soft", "hard"), default="soft", help="3D damage weight mode.")
    parser.add_argument("--max-damage-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_capacity(args: argparse.Namespace) -> float | None:
    if args.capacity_bits_per_frame is not None:
        return capacity_bits_per_slot(fixed_capacity_bits=args.capacity_bits_per_frame)
    if args.bandwidth_hz is not None or args.snr_db is not None or args.slot_time_s is not None:
        return capacity_bits_per_slot(
            bandwidth_hz=args.bandwidth_hz,
            snr_db=args.snr_db,
            slot_time_s=args.slot_time_s,
        )
    return None


def generate_masks(args: argparse.Namespace, input_dataset: Path, output_dataset: Path) -> list[dict]:
    sparse_dir = input_dataset / "sparse" / "0"
    masks_dir = output_dataset / "masks"
    overlays_dir = output_dataset / "overlays"
    if masks_dir.exists():
        shutil.rmtree(masks_dir)
    if args.save_overlays and overlays_dir.exists():
        shutil.rmtree(overlays_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")
    points = load_points_from_lmdb(Path(args.lmdb_dir).expanduser(), args.lmdb_key)
    damage_regions = load_damage_regions(Path(args.damage_json).expanduser())
    weights = compute_damage_weights(points, damage_regions, args.mode)
    damage_points = points[weights > args.damage_threshold]
    damage_points = sample_points(damage_points, args.max_damage_points, args.seed)

    mask_rows = []
    for image in images:
        camera = cameras[image["camera_id"]]
        mask_arr = np.zeros((camera["height"], camera["width"]), dtype=np.uint8)
        u, v = project_points(damage_points, image, camera)
        if u.size:
            mask_arr[v, u] = 255
        mask = Image.fromarray(mask_arr, mode="L")
        if args.dilate_radius > 0:
            mask = mask.filter(ImageFilter.MaxFilter(size=2 * args.dilate_radius + 1))

        mask_name = mask_name_for_image(image["name"])
        mask_path = masks_dir / mask_name
        mask.save(mask_path)
        if args.save_overlays:
            save_overlay(input_dataset / "images" / image["name"], mask, overlays_dir / mask_name)

        mask_pixels = int(np.asarray(mask, dtype=np.uint8).astype(bool).sum())
        mask_ratio = float(mask_pixels / (camera["height"] * camera["width"]))
        mask_rows.append({
            "image_name": image["name"],
            "mask_name": mask_name,
            "mask_path": str(mask_path),
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
        })

    with (output_dataset / "mask_manifest.json").open("w") as f:
        json.dump({
            "input_dataset": str(input_dataset),
            "lmdb_dir": args.lmdb_dir,
            "lmdb_key": args.lmdb_key,
            "damage_json": args.damage_json,
            "damage_threshold": args.damage_threshold,
            "dilate_radius": args.dilate_radius,
            "num_images": len(mask_rows),
            "total_damage_points": int(damage_points.shape[0]),
            "images": mask_rows,
        }, f, indent=2)

    print(f"[INFO] generated masks       : {masks_dir}")
    print(f"[INFO] selected damage pts  : {damage_points.shape[0]}")
    return mask_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dataset = Path(args.input_dataset).expanduser()
    output_dataset = Path(args.output_dataset).expanduser()
    output_images = output_dataset / "images"
    output_sparse = output_dataset / "sparse" / "0"
    if output_images.exists():
        shutil.rmtree(output_images)
    output_images.mkdir(parents=True, exist_ok=True)
    output_sparse.parent.mkdir(parents=True, exist_ok=True)

    if output_sparse.exists():
        shutil.rmtree(output_sparse)
    shutil.copytree(input_dataset / "sparse" / "0", output_sparse)

    modes = get_default_modes()
    modes_by_id = {mode.mode_id: mode for mode in modes}
    if args.fixed_mode not in modes_by_id:
        raise ValueError(f"Unknown fixed mode {args.fixed_mode}. Available: {sorted(modes_by_id)}")
    capacity_bits = resolve_capacity(args)
    if args.mode_selection == "capacity" and capacity_bits is None:
        raise ValueError("capacity mode requires --capacity-bits-per-frame or bandwidth/snr/slot-time.")

    mask_rows = generate_masks(args, input_dataset, output_dataset)
    mask_by_image = {row["image_name"]: row for row in mask_rows}
    bit_rows = []

    for image_name, mask_info in mask_by_image.items():
        image_path = input_dataset / "images" / image_name
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_info["mask_path"]).convert("L")
        if mask.size != image.size:
            mask = mask.resize(image.size, RESAMPLE_NEAREST)

        if args.mode_selection == "fixed":
            selected_mode = modes_by_id[args.fixed_mode]
            result = encode_decode_roi_semantic(image, mask, selected_mode)
            capacity_satisfied = True if capacity_bits is None else result["total_bits"] <= capacity_bits
        else:
            selection = select_semantic_mode(image, mask, modes, capacity_bits)
            selected_mode = selection["mode"]
            result = encode_decode_roi_semantic(image, mask, selected_mode)
            capacity_satisfied = bool(selection["capacity_satisfied"])

        out_image_path = output_images / image_name
        save_kwargs = {}
        if out_image_path.suffix.lower() in (".jpg", ".jpeg"):
            save_kwargs = {"quality": 100, "subsampling": 0}
        result["received_image"].save(out_image_path, **save_kwargs)
        cap_value = float(capacity_bits) if capacity_bits is not None else float("nan")
        bit_rows.append({
            "image_name": image_name,
            "selected_mode": selected_mode.mode_id,
            "capacity_bits": cap_value,
            "capacity_satisfied": capacity_satisfied,
            "mask_pixels": mask_info["mask_pixels"],
            "mask_ratio": mask_info["mask_ratio"],
            "bg_bits": result["bg_bits"],
            "roi_bits": result["roi_bits"],
            "metadata_bits": result["metadata_bits"],
            "total_bits": result["total_bits"],
            "total_mbits": result["total_bits"] / 1e6,
        })

    bit_fields = [
        "image_name",
        "selected_mode",
        "capacity_bits",
        "capacity_satisfied",
        "mask_pixels",
        "mask_ratio",
        "bg_bits",
        "roi_bits",
        "metadata_bits",
        "total_bits",
        "total_mbits",
    ]
    write_csv(output_dataset / "bit_report.csv", bit_rows, bit_fields)

    mode_rows = []
    for mode in modes:
        rows = [row for row in bit_rows if row["selected_mode"] == mode.mode_id]
        if not rows:
            mean_bits = 0.0
            mean_mask_ratio = 0.0
        else:
            mean_bits = float(np.mean([row["total_bits"] for row in rows]))
            mean_mask_ratio = float(np.mean([row["mask_ratio"] for row in rows]))
        mode_rows.append({
            "mode_id": mode.mode_id,
            "count": len(rows),
            "mean_bits": mean_bits,
            "mean_mask_ratio": mean_mask_ratio,
        })
    write_csv(output_dataset / "semantic_mode_report.csv", mode_rows, [
        "mode_id",
        "count",
        "mean_bits",
        "mean_mask_ratio",
    ])

    config = vars(args).copy()
    config["capacity_bits_resolved"] = capacity_bits
    config["semantic_modes"] = [mode.__dict__ for mode in modes]
    with (output_dataset / "comm_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    mean_bits = float(np.mean([row["total_bits"] for row in bit_rows])) if bit_rows else 0.0
    satisfied_values = [bool(row["capacity_satisfied"]) for row in bit_rows if not math.isnan(row["capacity_bits"])]
    satisfied_ratio = float(np.mean(satisfied_values)) if satisfied_values else 1.0
    print(f"[INFO] output dataset            : {output_dataset}")
    print(f"[INFO] received images           : {output_images}")
    print(f"[INFO] mean bits/frame           : {mean_bits:.2f}")
    print(f"[INFO] mean Mbits/frame          : {mean_bits / 1e6:.6f}")
    print(f"[INFO] capacity satisfied ratio  : {satisfied_ratio:.4f}")
    print("[INFO] mode usage:")
    for row in mode_rows:
        ratio = row["count"] / len(bit_rows) if bit_rows else 0.0
        print(f"  {row['mode_id']}: count={row['count']}, ratio={ratio:.3f}, mean_bits={row['mean_bits']:.2f}")


if __name__ == "__main__":
    main()
