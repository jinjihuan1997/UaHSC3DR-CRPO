#!/usr/bin/env python3
"""Generate HDA ROI semantic communication training data from fresh MAGICIAN runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT_DEFAULT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from project_damage_regions_to_masks import (  # noqa: E402
    compute_damage_weights,
    load_damage_regions,
    load_points_from_lmdb,
    project_points_pytorch3d_metadata,
)


DEFAULT_SCENES = [
    "neuschwanstein",
    "bridge",
    "eiffel",
    "alhambra",
    "dunnottar",
    "pisa",
    "bannerman",
    "redeemer",
    "pantheon",
    "fushimi",
    "liberty",
    "colosseum",
    "sofia_church",
    "barts",
    "sestino_museum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(PROJECT_ROOT_DEFAULT))
    parser.add_argument("--scene-root", default=None)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--scenes", default="all", help="'all' or comma-separated scene ids.")
    parser.add_argument("--num-trajectories", type=int, default=5)
    parser.add_argument("--damage-num-configs", type=int, default=1)
    parser.add_argument("--record-damage", action="store_true", help="Record damage/ROI annotations in addition to base observations.")
    parser.add_argument("--damage-generation-scope", choices=("scene_level", "trajectory_level"), default="scene_level")
    parser.add_argument("--damage-seed", type=int, default=2026)
    parser.add_argument("--damage-beta", type=float, default=1.0)
    parser.add_argument("--damage-num-regions", type=int, default=3)
    parser.add_argument("--damage-radius-ratio", type=float, default=0.03)
    parser.add_argument("--damage-soft-sigma-ratio", type=float, default=0.02)
    parser.add_argument("--damage-threshold", type=float, default=0.05)
    parser.add_argument("--damage-mode", choices=("soft", "hard"), default="soft")
    parser.add_argument("--dilate-radius", type=int, default=8)
    parser.add_argument("--image-width", type=int, default=1824)
    parser.add_argument("--image-height", type=int, default=1024)
    parser.add_argument(
        "--transmit-high-resolution",
        action="store_true",
        help="Deprecated compatibility flag; high-resolution observation export is always enabled.",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="Deprecated compatibility flag; depth is exported by default as base observation data.",
    )
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--save-priority-maps", action="store_true")
    parser.add_argument("--max-damage-points", type=int, default=200000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Deprecated compatibility flag. Existing complete trajectories are kept and skipped.",
    )
    parser.add_argument(
        "--force-clean",
        action="store_true",
        help="Delete output-root before generation. This restores the old destructive behavior.",
    )
    return parser.parse_args()


def resolve_scene_root(project_root: Path, scene_root_arg: Optional[str]) -> Path:
    candidates = []
    if scene_root_arg:
        candidates.append(Path(scene_root_arg).expanduser())
    candidates.extend([
        project_root / "data" / "Macarons++" / "macarons++",
        project_root / "data" / "macarons++",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find scene root. Pass --scene-root, for example "
        f"{project_root / 'data' / 'Macarons++' / 'macarons++'}"
    )


def resolve_scenes(scene_root: Path, scenes_arg: str) -> List[str]:
    if scenes_arg.strip().lower() != "all":
        return [item.strip() for item in scenes_arg.split(",") if item.strip()]
    template = [scene for scene in DEFAULT_SCENES if (scene_root / scene).exists()]
    if template:
        return template
    return sorted(path.name for path in scene_root.iterdir() if (path / "settings.json").exists())


def stable_resolved_seed(seed: int, scene_id: str, damage_config_id: str, scope: str, trajectory_id: str = "") -> int:
    key_items = [str(seed), scene_id, damage_config_id]
    if scope == "trajectory_level":
        key_items.append(trajectory_id)
    digest = hashlib.sha256(":".join(key_items).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> List[dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f if line.strip()]


def image_sort_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def prepare_output_root(output_root: Path, overwrite: bool, force_clean: bool) -> None:
    if output_root.exists():
        if force_clean:
            shutil.rmtree(output_root)
        elif overwrite:
            print("[WARN] --overwrite is deprecated and no longer deletes existing dataset files.")
            print("[WARN] Existing complete trajectories will be reused; use --force-clean to start from scratch.")
    (output_root / "scenes").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)
    (output_root / "reports").mkdir(parents=True, exist_ok=True)


def trajectory_dir(output_root: Path, scene_id: str, damage_config_id: str, traj_idx: int) -> Path:
    return output_root / "scenes" / scene_id / damage_config_id / f"traj{traj_idx}"


def config_ids_for_run(args: argparse.Namespace) -> List[str]:
    if not args.record_damage:
        return ["observations"]
    return [f"damage_config_{idx + 1:04d}" for idx in range(args.damage_num_configs)]


def run_id_for_config(args: argparse.Namespace, config_id: str) -> str:
    if not args.record_damage:
        return "magician_observations"
    return f"hda_roi_semcom_{config_id}"


def required_sample_paths(row: dict, args: argparse.Namespace) -> List[str]:
    paths = [
        row.get("clean_image"),
        row.get("depth"),
        row.get("valid_mask"),
        row.get("camera_pose"),
    ]
    if args.record_damage:
        paths.append(row.get("roi_mask"))
    if args.save_overlays:
        paths.append(row.get("overlay"))
    if args.save_priority_maps:
        paths.append(row.get("roi_priority_map"))
    return [path for path in paths if path]


def is_existing_trajectory_complete(
    output_root: Path,
    scene_id: str,
    damage_config_id: str,
    traj_idx: int,
    args: argparse.Namespace,
) -> bool:
    traj_id = f"traj{traj_idx}"
    traj_dir = trajectory_dir(output_root, scene_id, damage_config_id, traj_idx)
    manifest_path = traj_dir / "frame_manifest.jsonl"
    summary_path = traj_dir / "trajectory_summary.json"
    if not manifest_path.exists() or not summary_path.exists():
        return False
    try:
        rows = read_jsonl(manifest_path)
    except Exception:
        return False
    if not rows:
        return False
    for row in rows:
        if row.get("scene_id") != scene_id or row.get("trajectory_id") != traj_id:
            return False
        if args.record_damage and row.get("damage_config_id") != damage_config_id:
            return False
        for rel_path in required_sample_paths(row, args):
            if not (output_root / rel_path).exists():
                return False
        if not row.get("depth") or not row.get("valid_mask"):
            return False
        if args.save_overlays and not row.get("overlay"):
            return False
        if args.save_priority_maps and not row.get("roi_priority_map"):
            return False
    return True


def load_existing_trajectory(
    output_root: Path,
    scene_id: str,
    damage_config_id: str,
    traj_idx: int,
    all_sample_rows: List[dict],
    roi_rows: List[dict],
    record_damage: bool = True,
) -> dict:
    traj_dir = trajectory_dir(output_root, scene_id, damage_config_id, traj_idx)
    rows = read_jsonl(traj_dir / "frame_manifest.jsonl")
    with (traj_dir / "trajectory_summary.json").open("r") as f:
        summary = json.load(f)
    drop_trajectory_rows(all_sample_rows, roi_rows, scene_id, damage_config_id, f"traj{traj_idx}")
    all_sample_rows.extend(rows)
    if record_damage:
        roi_rows.extend(rows)
    print(f"[SKIP] Reusing existing {scene_id}/{damage_config_id}/traj{traj_idx}")
    return summary


def drop_trajectory_rows(
    all_sample_rows: List[dict],
    roi_rows: List[dict],
    scene_id: str,
    damage_config_id: str,
    trajectory_id: str,
) -> None:
    def keep(row: dict) -> bool:
        same_scene_traj = row.get("scene_id") == scene_id and row.get("trajectory_id") == trajectory_id
        same_damage_config = row.get("damage_config_id") in (damage_config_id, None)
        return not (same_scene_traj and same_damage_config)

    all_sample_rows[:] = [row for row in all_sample_rows if keep(row)]
    roi_rows[:] = [row for row in roi_rows if keep(row)]


def existing_raw_trajectory_available(
    output_root: Path,
    project_root: Path,
    scene_root: Path,
    scene_id: str,
    damage_config_id: str,
    traj_idx: int,
    record_damage: bool = True,
) -> bool:
    traj_dir = trajectory_dir(output_root, scene_id, damage_config_id, traj_idx)
    damage_json = scene_root / scene_id / "test_memory_0" / "training" / str(traj_idx) / "damage_debug" / "damage_regions.json"
    run_id = f"hda_roi_semcom_{damage_config_id}" if record_damage else "magician_observations"
    lmdb_dir = project_root / "results" / "scene_exploration" / f"{run_id}_lmdb"
    base_available = (
        (traj_dir / "clean_images").exists()
        and any((traj_dir / "clean_images").glob("*.png"))
        and (traj_dir / "poses").exists()
        and (traj_dir / "depth").exists()
        and (traj_dir / "valid_masks").exists()
    )
    if not record_damage:
        return base_available
    return base_available and damage_json.exists() and lmdb_dir.exists()


def make_magician_config(
    project_root: Path,
    scene_root: Path,
    output_root: Path,
    scenes: List[str],
    args: argparse.Namespace,
    damage_config_id: str,
    damage_config_index: int,
) -> Path:
    template_path = project_root / "configs" / "test" / "test_in_default_scenes_config.json"
    with template_path.open("r") as f:
        config = json.load(f)

    seed_scope = "scene" if args.damage_generation_scope == "scene_level" else "trajectory"
    run_id = run_id_for_config(args, damage_config_id)
    config.update({
        "dataset_path": str(scene_root),
        "test_scenes": scenes,
        "results_json_name": f"{run_id}_results.json",
        "lmdb_dir_name": f"{run_id}_lmdb",
        "compute_collision": False,
        "transmit_high_resolution": bool(args.transmit_high_resolution),
        "planning_image_height": 256,
        "planning_image_width": 456,
        "reconstruction_image_height": int(args.image_height),
        "reconstruction_image_width": int(args.image_width),
        "max_trajectories_per_scene": int(args.num_trajectories),
        "damage_aware_planning": bool(args.record_damage),
        "damage_alpha": 1.0,
        "damage_beta": float(args.damage_beta),
        "damage_lambda_comm": 0.0,
        "damage_gain_normalization": "match_general_mean",
        "damage_gain_scale": 10000.0,
        "damage_mode": "random_3d_spheres",
        "damage_seed": int(args.damage_seed + damage_config_index),
        "damage_seed_scope": seed_scope,
        "damage_num_regions": int(args.damage_num_regions),
        "damage_radius_ratio": float(args.damage_radius_ratio),
        "damage_soft_sigma_ratio": float(args.damage_soft_sigma_ratio),
        "damage_target_observations": 1,
        "save_damage_debug": bool(args.record_damage),
        "save_damage_metrics": bool(args.record_damage),
        "export_hda_roi_dataset": True,
        "hda_roi_export_root": str(output_root),
        "hda_export_dataset_id": damage_config_id,
        "hda_roi_save_depth": True,
        "hda_roi_save_valid_mask": True,
        "hda_roi_skip_existing_trajectories": True,
    })

    if args.record_damage:
        config["hda_roi_damage_config_id"] = damage_config_id
    else:
        config.pop("hda_roi_damage_config_id", None)

    config_path = project_root / "configs" / "test" / f"_generated_{run_id}.json"
    write_json(config_path, config)
    return config_path


def run_magician(project_root: Path, config_path: Path) -> None:
    print(f"[INFO] Running MAGICIAN with {config_path.name}")
    subprocess.run(
        [sys.executable, "test_magician_planning.py", "-c", config_path.name],
        cwd=str(project_root),
        check=True,
    )


def enrich_damage_config(
    damage_regions: dict,
    args: argparse.Namespace,
    scene_id: str,
    damage_config_id: str,
    config_seed: int,
) -> dict:
    regions = []
    for idx, region in enumerate(damage_regions.get("regions", [])):
        enriched = dict(region)
        enriched.setdefault("region_id", idx)
        enriched.setdefault("priority", 1.0)
        enriched.setdefault("source_point_index", region.get("center_candidate_index"))
        regions.append(enriched)

    resolved_seed = stable_resolved_seed(
        config_seed,
        scene_id,
        damage_config_id,
        args.damage_generation_scope,
    )
    return {
        "scene_id": scene_id,
        "damage_config_id": damage_config_id,
        "damage_seed": int(config_seed),
        "resolved_seed": int(damage_regions.get("seed", resolved_seed)),
        "damage_generation_scope": args.damage_generation_scope,
        "roi_annotation_type": "generated_pseudo_roi",
        "roi_generation_method": "3d_damage_region_projection",
        "center_source": damage_regions.get("center_source", "unknown"),
        "damage_num_regions": int(args.damage_num_regions),
        "damage_radius_ratio": float(args.damage_radius_ratio),
        "damage_soft_sigma_ratio": float(args.damage_soft_sigma_ratio),
        "damage_threshold": float(args.damage_threshold),
        "damage_mode": args.damage_mode,
        "dilate_radius": int(args.dilate_radius),
        "scene_bbox_min": damage_regions.get("scene_bbox_min"),
        "scene_bbox_max": damage_regions.get("scene_bbox_max"),
        "scene_scale": damage_regions.get("scene_scale"),
        "damage_regions": regions,
    }


def sample_damage_points(points: np.ndarray, weights: np.ndarray, threshold: float, max_points: int, seed: int):
    idx = np.nonzero(weights > threshold)[0]
    if max_points > 0 and idx.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(idx, size=max_points, replace=False)
    return points[idx], weights[idx], idx


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def count_components(mask: np.ndarray) -> int:
    try:
        from scipy import ndimage

        _, count = ndimage.label(mask.astype(np.uint8))
        return int(count)
    except Exception:
        ys, xs = np.nonzero(mask)
        remaining = set(zip(ys.tolist(), xs.tolist()))
        count = 0
        while remaining:
            count += 1
            start = remaining.pop()
            queue = deque([start])
            while queue:
                y, x = queue.popleft()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    item = (ny, nx)
                    if item in remaining:
                        remaining.remove(item)
                        queue.append(item)
        return count


def save_overlay(clean_path: Path, mask_arr: np.ndarray, out_path: Path) -> None:
    image = Image.open(clean_path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    red = np.asarray([255.0, 0.0, 0.0], dtype=np.float32)
    roi = mask_arr > 0
    arr[roi] = arr[roi] * 0.35 + red * 0.65
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB").save(out_path)


def project_masks_for_trajectory(
    output_root: Path,
    project_root: Path,
    scene_root: Path,
    lmdb_dir: Path,
    scene_id: str,
    damage_config_id: str,
    traj_idx: int,
    args: argparse.Namespace,
    all_sample_rows: List[dict],
    roi_rows: List[dict],
    missing: dict,
) -> Optional[dict]:
    traj_id = f"traj{traj_idx}"
    if is_existing_trajectory_complete(output_root, scene_id, damage_config_id, traj_idx, args):
        return load_existing_trajectory(
            output_root,
            scene_id,
            damage_config_id,
            traj_idx,
            all_sample_rows,
            roi_rows,
            record_damage=bool(args.record_damage),
        )

    drop_trajectory_rows(all_sample_rows, roi_rows, scene_id, damage_config_id, traj_id)
    traj_dir = trajectory_dir(output_root, scene_id, damage_config_id, traj_idx)
    clean_dir = traj_dir / "clean_images"
    pose_dir = traj_dir / "poses"
    masks_dir = traj_dir / "roi_masks"
    priority_dir = traj_dir / "roi_priority_maps"
    overlays_dir = traj_dir / "overlays"
    masks_dir.mkdir(parents=True, exist_ok=True)
    if args.save_priority_maps:
        priority_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlays_dir.mkdir(parents=True, exist_ok=True)

    clean_images = sorted(clean_dir.glob("*.png"), key=image_sort_key)
    if not clean_images:
        missing["failed_image_saves"].append(str(clean_dir))
        return None

    if not args.record_damage:
        manifest_path = traj_dir / "frame_manifest.jsonl"
        if not manifest_path.exists():
            missing["failed_image_saves"].append(str(manifest_path))
            return None
        rows = read_jsonl(manifest_path)
        drop_trajectory_rows(all_sample_rows, roi_rows, scene_id, damage_config_id, traj_id)
        for row in rows:
            row.setdefault("damage_enabled", False)
            row.setdefault("damage_config_id", None)
            row.setdefault("roi_ratio", 0.0)
        all_sample_rows.extend(rows)
        summary = {
            "scene_id": scene_id,
            "damage_config_id": None,
            "trajectory_id": traj_id,
            "num_frames": len(rows),
            "damage_enabled": False,
            "lmdb_key": f"{scene_id}/{traj_idx}",
        }
        write_json(traj_dir / "trajectory_summary.json", summary)
        return summary

    damage_json_src = scene_root / scene_id / "test_memory_0" / "training" / str(traj_idx) / "damage_debug" / "damage_regions.json"
    if not damage_json_src.exists():
        missing["failed_mask_projections"].append(str(damage_json_src))
        return None

    damage_regions = load_damage_regions(damage_json_src)
    config_seed = int(args.damage_seed)
    damage_config = enrich_damage_config(damage_regions, args, scene_id, damage_config_id, config_seed)
    write_json(output_root / "scenes" / scene_id / damage_config_id / "damage_config.json", damage_config)

    try:
        points = load_points_from_lmdb(lmdb_dir, f"{scene_id}/{traj_idx}")
    except Exception as exc:
        missing["failed_mask_projections"].append(f"{scene_id}/{traj_idx}: {exc}")
        return None

    weights = compute_damage_weights(points, damage_regions, args.damage_mode)
    damage_points, damage_point_weights, _ = sample_damage_points(
        points,
        weights,
        args.damage_threshold,
        args.max_damage_points,
        seed=args.damage_seed + traj_idx,
    )

    frame_rows = []
    roi_ratios = []
    for clean_path in clean_images:
        frame_id = int(clean_path.stem)
        pose_path = pose_dir / f"{frame_id:06d}.json"
        if not pose_path.exists():
            missing["invalid_camera_metadata"].append(str(pose_path))
            continue
        with pose_path.open("r") as f:
            pose_metadata = json.load(f)

        width = int(pose_metadata["width"])
        height = int(pose_metadata["height"])
        mask_arr = np.zeros((height, width), dtype=np.uint8)
        priority_arr = np.zeros((height, width), dtype=np.uint8)
        u, v, valid_idx = project_points_pytorch3d_metadata(damage_points, pose_metadata)
        if u.size:
            mask_arr[v, u] = 255
            if args.save_priority_maps:
                pixel_weights = np.clip(damage_point_weights[valid_idx] * 255.0, 0, 255).astype(np.uint8)
                np.maximum.at(priority_arr, (v, u), pixel_weights)

        mask_img = Image.fromarray(mask_arr, mode="L")
        if args.dilate_radius > 0:
            mask_img = mask_img.filter(ImageFilter.MaxFilter(size=2 * args.dilate_radius + 1))
        mask_arr = (np.asarray(mask_img, dtype=np.uint8) > 0).astype(np.uint8) * 255
        mask_path = masks_dir / clean_path.name
        Image.fromarray(mask_arr, mode="L").save(mask_path)

        priority_path = None
        if args.save_priority_maps:
            priority_img = Image.fromarray(priority_arr, mode="L")
            if args.dilate_radius > 0:
                priority_img = priority_img.filter(ImageFilter.MaxFilter(size=2 * args.dilate_radius + 1))
            priority_path = priority_dir / clean_path.name
            priority_img.save(priority_path)

        overlay_path = None
        if args.save_overlays:
            overlay_path = overlays_dir / clean_path.name
            save_overlay(clean_path, mask_arr, overlay_path)

        mask_bool = mask_arr > 0
        roi_pixels = int(mask_bool.sum())
        total_pixels = int(width * height)
        roi_ratio = float(roi_pixels / total_pixels)
        roi_ratios.append(roi_ratio)
        bbox = bbox_from_mask(mask_bool)
        if bbox is None:
            bbox_xmin = bbox_ymin = bbox_xmax = bbox_ymax = -1
            bbox_area = 0
            bbox_area_ratio = 0.0
            bbox_to_mask_ratio = 0.0
            missing["empty_roi_masks"].append(str(mask_path))
        else:
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox
            bbox_area = int((bbox_xmax - bbox_xmin) * (bbox_ymax - bbox_ymin))
            bbox_area_ratio = float(bbox_area / total_pixels)
            bbox_to_mask_ratio = float(bbox_area / max(roi_pixels, 1))
        num_components = count_components(mask_bool) if roi_pixels > 0 else 0

        depth_path = traj_dir / "depth" / clean_path.name
        valid_mask_path = traj_dir / "valid_masks" / clean_path.name
        if not depth_path.exists():
            missing["failed_image_saves"].append(str(depth_path))
            continue
        if not valid_mask_path.exists():
            missing["failed_image_saves"].append(str(valid_mask_path))
            continue
        depth_rel = depth_path.relative_to(output_root).as_posix()
        valid_mask_rel = valid_mask_path.relative_to(output_root).as_posix()

        row = {
            "scene_id": scene_id,
            "damage_config_id": damage_config_id,
            "trajectory_id": traj_id,
            "frame_id": frame_id,
            "image_name": clean_path.name,
            "clean_image": clean_path.relative_to(output_root).as_posix(),
            "roi_mask": mask_path.relative_to(output_root).as_posix(),
            "roi_priority_map": priority_path.relative_to(output_root).as_posix() if priority_path else None,
            "overlay": overlay_path.relative_to(output_root).as_posix() if overlay_path else None,
            "depth": depth_rel,
            "valid_mask": valid_mask_rel,
            "camera_pose": pose_path.relative_to(output_root).as_posix(),
            "clean_image_role": "ground_truth",
            "roi_annotation_type": "generated_pseudo_roi",
            "roi_generation_method": "3d_damage_region_projection",
            "roi_mask_role": "training_pseudo_label",
            "roi_policy": "roi_to_analog_background_to_digital",
            "snr_generation": "online_random_during_training",
            "channel_generation": "online_random_during_training",
            "width": width,
            "height": height,
            "roi_pixels": roi_pixels,
            "total_pixels": total_pixels,
            "roi_ratio": roi_ratio,
            "bbox_xmin": bbox_xmin,
            "bbox_ymin": bbox_ymin,
            "bbox_xmax": bbox_xmax,
            "bbox_ymax": bbox_ymax,
            "bbox_area": bbox_area,
            "bbox_area_ratio": bbox_area_ratio,
            "bbox_to_mask_ratio": bbox_to_mask_ratio,
            "num_connected_components": num_components,
        }
        frame_rows.append(row)
        all_sample_rows.append(row)
        roi_rows.append(row)

    write_jsonl(traj_dir / "frame_manifest.jsonl", frame_rows)
    summary = {
        "scene_id": scene_id,
        "damage_config_id": damage_config_id,
        "trajectory_id": traj_id,
        "num_frames": len(frame_rows),
        "mean_roi_ratio": float(np.mean(roi_ratios)) if roi_ratios else 0.0,
        "median_roi_ratio": float(np.median(roi_ratios)) if roi_ratios else 0.0,
        "min_roi_ratio": float(np.min(roi_ratios)) if roi_ratios else 0.0,
        "max_roi_ratio": float(np.max(roi_ratios)) if roi_ratios else 0.0,
        "lmdb_key": f"{scene_id}/{traj_idx}",
        "source_damage_json": str(damage_json_src),
    }
    write_json(traj_dir / "trajectory_summary.json", summary)
    return summary


def split_trajectory_samples(samples: List[dict]):
    grouped = defaultdict(list)
    for row in samples:
        grouped[(row["scene_id"], row.get("damage_config_id", "observations"), row["trajectory_id"])].append(row)

    by_scene_cfg = defaultdict(list)
    for scene_id, damage_config_id, traj_id in grouped:
        by_scene_cfg[(scene_id, damage_config_id)].append(traj_id)

    split_rows = {"train": [], "val": [], "test": []}
    for key, traj_ids in by_scene_cfg.items():
        traj_ids = sorted(set(traj_ids), key=lambda item: int(item.replace("traj", "")))
        n = len(traj_ids)
        if n >= 5:
            train_ids, val_ids, test_ids = traj_ids[:3], traj_ids[3:4], traj_ids[4:]
        else:
            train_end = max(1, int(np.ceil(n * 0.6)))
            val_end = max(train_end + 1, int(np.ceil(n * 0.8))) if n > 1 else train_end
            train_ids = traj_ids[:train_end]
            val_ids = traj_ids[train_end:val_end]
            test_ids = traj_ids[val_end:]
            if not val_ids and n > 1:
                val_ids = [traj_ids[-1]]
            if not test_ids and n > 2:
                test_ids = [traj_ids[-1]]
        mapping = {
            "train": set(train_ids),
            "val": set(val_ids),
            "test": set(test_ids),
        }
        for split, ids in mapping.items():
            for traj_id in ids:
                split_rows[split].extend(grouped[(key[0], key[1], traj_id)])
    return split_rows


def split_scene_samples(samples: List[dict]):
    scene_ids = sorted(set(row["scene_id"] for row in samples))
    n = len(scene_ids)
    train_end = max(1, int(np.ceil(n * 0.7)))
    val_end = max(train_end + 1, int(np.ceil(n * 0.85))) if n > 1 else train_end
    split_scene_ids = {
        "train": set(scene_ids[:train_end]),
        "val": set(scene_ids[train_end:val_end]),
        "test": set(scene_ids[val_end:]),
    }
    if not split_scene_ids["val"] and n > 1:
        split_scene_ids["val"] = {scene_ids[-1]}
    if not split_scene_ids["test"] and n > 2:
        split_scene_ids["test"] = {scene_ids[-1]}
    return {
        split: [row for row in samples if row["scene_id"] in ids]
        for split, ids in split_scene_ids.items()
    }


def damage_config_index(damage_config_id: str) -> int:
    try:
        return max(0, int(damage_config_id.rsplit("_", 1)[1]) - 1)
    except (IndexError, ValueError):
        return 0


def build_damage_summary_rows(output_root: Path, args: argparse.Namespace, samples: List[dict]) -> List[dict]:
    if not args.record_damage:
        return []
    rows = []
    keys = sorted(set((row["scene_id"], row["damage_config_id"]) for row in samples if row.get("damage_config_id") is not None))
    for scene_id, damage_config_id in keys:
        config_idx = damage_config_index(damage_config_id)
        damage_config_path = output_root / "scenes" / scene_id / damage_config_id / "damage_config.json"
        center_source = "unknown"
        if damage_config_path.exists():
            try:
                with damage_config_path.open("r") as f:
                    center_source = json.load(f).get("center_source", center_source)
            except (OSError, json.JSONDecodeError):
                pass
        flat_ratios = [
            row["roi_ratio"]
            for row in samples
            if row["scene_id"] == scene_id and row.get("damage_config_id") == damage_config_id
        ]
        rows.append({
            "scene_id": scene_id,
            "damage_config_id": damage_config_id,
            "damage_seed": int(args.damage_seed + config_idx),
            "resolved_seed": stable_resolved_seed(
                int(args.damage_seed + config_idx),
                scene_id,
                damage_config_id,
                args.damage_generation_scope,
            ),
            "damage_generation_scope": args.damage_generation_scope,
            "center_source": center_source,
            "damage_num_regions": int(args.damage_num_regions),
            "damage_radius_ratio": float(args.damage_radius_ratio),
            "damage_soft_sigma_ratio": float(args.damage_soft_sigma_ratio),
            "damage_threshold": float(args.damage_threshold),
            "damage_mode": args.damage_mode,
            "dilate_radius": int(args.dilate_radius),
            "num_trajectories": len(set(
                row["trajectory_id"]
                for row in samples
                if row["scene_id"] == scene_id and row.get("damage_config_id") == damage_config_id
            )),
            "num_frames": int(len(flat_ratios)),
            "mean_roi_ratio": float(np.mean(flat_ratios)) if flat_ratios else 0.0,
            "median_roi_ratio": float(np.median(flat_ratios)) if flat_ratios else 0.0,
            "min_roi_ratio": float(np.min(flat_ratios)) if flat_ratios else 0.0,
            "max_roi_ratio": float(np.max(flat_ratios)) if flat_ratios else 0.0,
        })
    return rows


def write_incremental_outputs(
    output_root: Path,
    args: argparse.Namespace,
    all_samples: List[dict],
    roi_rows: List[dict],
    missing: dict,
) -> dict:
    manifests_dir = output_root / "manifests"
    reports_dir = output_root / "reports"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(manifests_dir / "all_samples.jsonl", all_samples)
    trajectory_split = split_trajectory_samples(all_samples)
    for split, rows in trajectory_split.items():
        write_jsonl(manifests_dir / f"{split}_trajectory_split.jsonl", rows)
    scene_split = split_scene_samples(all_samples)
    for split, rows in scene_split.items():
        write_jsonl(manifests_dir / f"{split}_scene_split.jsonl", rows)

    pd.DataFrame(roi_rows).to_csv(reports_dir / "roi_area_statistics.csv", index=False)
    damage_summary_rows = build_damage_summary_rows(output_root, args, all_samples)
    pd.DataFrame(damage_summary_rows).to_csv(reports_dir / "damage_generation_summary.csv", index=False)
    write_json(reports_dir / "missing_files_report.json", missing)

    roi_ratios = [float(row.get("roi_ratio", 0.0)) for row in all_samples]
    dataset_summary = {
        "dataset_name": "hda_roi_semcom_dataset",
        "output_root": str(output_root),
        "total_scenes": len(set(row["scene_id"] for row in all_samples)),
        "total_damage_configs": len(set(row.get("damage_config_id") for row in all_samples if row.get("damage_config_id") is not None)),
        "total_trajectories": len(set((row["scene_id"], row.get("damage_config_id"), row["trajectory_id"]) for row in all_samples)),
        "total_valid_samples": len(all_samples),
        "scene_ids": sorted(set(row["scene_id"] for row in all_samples)),
        "trajectory_ids": sorted(set(row["trajectory_id"] for row in all_samples)),
        "mean_roi_ratio": float(np.mean(roi_ratios)) if roi_ratios else 0.0,
        "median_roi_ratio": float(np.median(roi_ratios)) if roi_ratios else 0.0,
        "min_roi_ratio": float(np.min(roi_ratios)) if roi_ratios else 0.0,
        "max_roi_ratio": float(np.max(roi_ratios)) if roi_ratios else 0.0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "incremental": True,
        "notes": [
            "Clean images are MAGICIAN-rendered UAV observations.",
            "ROI masks are generated pseudo labels projected from synthetic 3D damage regions when --record-damage is enabled.",
            "SNR and channel conditions are generated online during communication model training.",
            "This dataset is generated independently of downstream reconstruction evaluators.",
        ],
    }
    write_json(reports_dir / "dataset_summary.json", dataset_summary)
    write_dataset_loader_example(output_root)
    print(f"[SYNC] manifests/reports updated: {len(all_samples)} samples")
    return dataset_summary


def write_dataset_loader_example(output_root: Path) -> None:
    content = '''#!/usr/bin/env python3
"""Example PyTorch Dataset for HDA ROI semantic communication data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class HDAROISemComDataset(Dataset):
    def __init__(self, dataset_root, manifest):
        self.dataset_root = Path(dataset_root)
        manifest_path = self.dataset_root / manifest
        with manifest_path.open("r") as f:
            self.samples = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.samples)

    def _load_image(self, relative_path, mode):
        path = self.dataset_root / relative_path
        arr = np.asarray(Image.open(path).convert(mode), dtype=np.float32) / 255.0
        if mode == "RGB":
            arr = arr.transpose(2, 0, 1)
        else:
            arr = arr[None, ...]
        return torch.from_numpy(arr)

    def __getitem__(self, index):
        sample = self.samples[index]
        priority_path = sample.get("roi_priority_map")
        return {
            "image": self._load_image(sample["clean_image"], "RGB"),
            "mask": (self._load_image(sample["roi_mask"], "L") > 0).float() if sample.get("roi_mask") else None,
            "priority": self._load_image(priority_path, "L") if priority_path else None,
            "depth": self._load_image(sample["depth"], "I") if sample.get("depth") else None,
            "valid_mask": (self._load_image(sample["valid_mask"], "L") > 0).float() if sample.get("valid_mask") else None,
            "scene_id": sample["scene_id"],
            "damage_config_id": sample.get("damage_config_id"),
            "trajectory_id": sample["trajectory_id"],
            "frame_id": sample["frame_id"],
            "image_path": str(self.dataset_root / sample["clean_image"]),
            "mask_path": str(self.dataset_root / sample["roi_mask"]) if sample.get("roi_mask") else None,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", default="manifests/train_trajectory_split.jsonl")
    args = parser.parse_args()
    dataset = HDAROISemComDataset(args.dataset_root, args.manifest)
    print(f"Loaded {len(dataset)} samples")
    if len(dataset):
        sample = dataset[0]
        print(sample["image"].shape, sample["mask"].shape, sample["scene_id"], sample["trajectory_id"])


if __name__ == "__main__":
    main()
'''
    path = output_root / "dataset_loader_example.py"
    path.write_text(content)
    path.chmod(0o755)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    scene_root = resolve_scene_root(project_root, args.scene_root)
    output_root = Path(args.output_root).expanduser().resolve()
    args.transmit_high_resolution = True

    scenes = resolve_scenes(scene_root, args.scenes)
    if not scenes:
        raise ValueError(f"No scenes found in {scene_root}")
    prepare_output_root(output_root, args.overwrite, args.force_clean)

    all_samples: List[dict] = []
    roi_rows: List[dict] = []
    trajectory_summaries = []
    missing = {
        "failed_trajectory_runs": [],
        "failed_image_saves": [],
        "failed_mask_projections": [],
        "empty_roi_masks": [],
        "invalid_camera_metadata": [],
        "skipped_frames": [],
    }
    write_incremental_outputs(output_root, args, all_samples, roi_rows, missing)

    for config_idx, damage_config_id in enumerate(config_ids_for_run(args)):
        run_id = run_id_for_config(args, damage_config_id)
        lmdb_dir = project_root / "results" / "scene_exploration" / f"{run_id}_lmdb"

        for scene_id in scenes:
            for traj_idx in range(args.num_trajectories):
                if (
                    not is_existing_trajectory_complete(output_root, scene_id, damage_config_id, traj_idx, args)
                    and not existing_raw_trajectory_available(
                        output_root,
                        project_root,
                        scene_root,
                        scene_id,
                        damage_config_id,
                        traj_idx,
                        record_damage=bool(args.record_damage),
                    )
                ):
                    continue
                summary = project_masks_for_trajectory(
                    output_root,
                    project_root,
                    scene_root,
                    lmdb_dir,
                    scene_id,
                    damage_config_id,
                    traj_idx,
                    args,
                    all_samples,
                    roi_rows,
                    missing,
                )
                if summary is not None:
                    trajectory_summaries.append(summary)
                    write_incremental_outputs(output_root, args, all_samples, roi_rows, missing)

            scene_complete = all(
                is_existing_trajectory_complete(output_root, scene_id, damage_config_id, traj_idx, args)
                for traj_idx in range(args.num_trajectories)
            )
            if scene_complete:
                print(f"[SKIP] All requested trajectories already exist for {scene_id}/{damage_config_id}")
                continue

            config_path = make_magician_config(
                project_root, scene_root, output_root, [scene_id], args, damage_config_id, config_idx
            )
            try:
                run_magician(project_root, config_path)
            except subprocess.CalledProcessError as exc:
                missing["failed_trajectory_runs"].append(f"{scene_id}/{damage_config_id}: {exc}")

            for traj_idx in range(args.num_trajectories):
                summary = project_masks_for_trajectory(
                    output_root,
                    project_root,
                    scene_root,
                    lmdb_dir,
                    scene_id,
                    damage_config_id,
                    traj_idx,
                    args,
                    all_samples,
                    roi_rows,
                    missing,
                )
                if summary is None:
                    continue
                trajectory_summaries.append(summary)
                write_incremental_outputs(output_root, args, all_samples, roi_rows, missing)

    dataset_summary = write_incremental_outputs(output_root, args, all_samples, roi_rows, missing)
    manifests_dir = output_root / "manifests"
    reports_dir = output_root / "reports"

    print(f"[DONE] Dataset written to: {output_root}")
    print(f"[DONE] valid samples: {len(all_samples)}")
    print(f"[DONE] mean ROI ratio: {dataset_summary['mean_roi_ratio']:.6f}")
    print(f"[DONE] manifests: {manifests_dir}")
    print(f"[DONE] reports: {reports_dir}")


if __name__ == "__main__":
    main()
