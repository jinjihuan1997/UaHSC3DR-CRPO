"""Export degraded RGB-D sequences in a Replica-like format for GSFusion."""

import argparse
import csv
import zlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HDAROIDataset
from src.models.hda_roi import HDAROISystem
from src.utils.config import load_config


def _normalize_traj_id(value):
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith("traj"):
        return text
    return f"traj{int(float(text))}"


def _parse_traj_ids(value):
    if value is None:
        return None
    out = {_normalize_traj_id(x) for x in value.split(",") if x.strip()}
    return out or None


def _build_dataset(cfg, split, manifest_rel=None, scene_id_filter=None):
    d = cfg["data"]
    manifest = manifest_rel or {"train": d["train_manifest"], "val": d["val_manifest"], "test": d["test_manifest"]}[split]
    return HDAROIDataset(
        root=d["root"],
        manifest_rel=manifest,
        roi_source=d["roi_source"],
        image_size=d["image_size"],
        soft_mask_blur=d.get("soft_mask_blur", 9),
        mode="val",
        use_original_resolution=d.get("use_original_resolution", False),
        aux_payload_source=d.get("aux_payload_source", "zlib_crop"),
        aux_payload_size=d.get("aux_payload_size"),
        scene_id_filter=scene_id_filter,
        min_rgb_std=d.get("min_rgb_std", 0.0),
    )


def _read_lookup(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["sample_idx"] = int(float(row["sample_idx"]))
        row["frame_id_int"] = int(float(row.get("frame_id", row["sample_idx"])))
        row["snr_db"] = float(row["snr_db"])
        row["k_d"] = int(float(row["k_d"]))
        row["rgb_psnr"] = float(row["rgb_psnr"])
        row["depth_payload_bits"] = float(row["depth_payload_bits"])
        row["depth_bit_budget"] = float(row["depth_bit_budget"])
    return rows


def _dataset_index_by_frame_key(ds):
    index = {}
    for idx, rec in enumerate(ds.samples):
        scene_id = rec.get("scene_id", "")
        trajectory_id = _normalize_traj_id(rec.get("trajectory_id", ""))
        frame_id = rec.get("frame_id", rec.get("frame", ""))
        if scene_id and trajectory_id and frame_id != "":
            index[(scene_id, trajectory_id, int(float(frame_id)))] = idx
    return index


def _sample_index_for_lookup_row(row, ds, frame_key_to_idx):
    has_frame_key = (
        row.get("scene_id", "")
        and row.get("trajectory_id", "")
        and row.get("frame_id", "") != ""
    )
    key = (
        row.get("scene_id", ""),
        _normalize_traj_id(row.get("trajectory_id", "")),
        int(row["frame_id_int"]),
    )
    if key in frame_key_to_idx:
        return frame_key_to_idx[key]
    if has_frame_key:
        raise KeyError(f"lookup row frame key is not in dataset manifest: {key}")
    sample_idx = int(row["sample_idx"])
    if 0 <= sample_idx < len(ds):
        return sample_idx
    raise IndexError(f"lookup row cannot be matched to dataset sample: {row}")


def _normalize_lookup_q(rows, norm_rows=None):
    psnrs = [row["rgb_psnr"] for row in (norm_rows or rows)]
    lo, hi = min(psnrs), max(psnrs)
    for row in rows:
        row["q_rgb"] = max(0.0, min(1.0, (row["rgb_psnr"] - lo) / max(hi - lo, 1e-12)))
        payload = row["depth_payload_bits"]
        row["r_depth"] = 1.0 if payload <= 0 else max(0.0, min(1.0, row["depth_bit_budget"] / payload))


def _resolve(root, rec, key):
    rel = rec.get(key)
    if not rel:
        return None
    root = Path(root)
    path = root / rel
    if path.exists():
        return path
    scene_id = rec.get("scene_id")
    trajectory_id = rec.get("trajectory_id")
    if scene_id and trajectory_id:
        matches = sorted((root / "scenes" / scene_id).glob(f"*/{trajectory_id}/{rel}"))
        if matches:
            return matches[0]
    return None


def _pose_json(root, rec):
    path = _resolve(root, rec, "camera_pose")
    if path is None:
        raise FileNotFoundError(
            "missing camera_pose for "
            f"scene={rec.get('scene_id')} trajectory={rec.get('trajectory_id')} "
            f"frame={rec.get('frame_id')}: {rec.get('camera_pose')}"
        )
    with path.open() as f:
        return json.load(f)


def _c2w_from_pose(pose):
    if pose.get("coordinate_system") == "MAGICIAN/PyTorch3D" and "R" in pose and "camera_center" in pose:
        # MAGICIAN/PyTorch3D stores row-vector world-to-view transforms:
        # X_cam = X_world @ R + T. The column-vector C2W rotation is R, and
        # GSFusion expects OpenCV-like camera axes (+X right, +Y down, +Z
        # forward), so flip PyTorch3D's +X-left/+Y-up axes while preserving Z.
        r = np.asarray(pose["R"][0], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = r @ np.diag([-1.0, -1.0, 1.0])
        c2w[:3, 3] = np.asarray(pose["camera_center"][0], dtype=np.float64)
        return c2w
    if "world_to_view_matrix" in pose:
        m = np.asarray(pose["world_to_view_matrix"][0], dtype=np.float64)
        try:
            return np.linalg.inv(m.T)
        except np.linalg.LinAlgError:
            pass
    c2w = np.eye(4, dtype=np.float64)
    if "R" in pose:
        c2w[:3, :3] = np.asarray(pose["R"][0], dtype=np.float64)
    if "camera_center" in pose:
        c2w[:3, 3] = np.asarray(pose["camera_center"][0], dtype=np.float64)
    return c2w


def _intrinsics_from_pose(pose, width, height):
    if all(key in pose for key in ("fx", "fy", "cx", "cy")):
        fx = float(pose["fx"])
        fy = float(pose["fy"])
        cx = float(pose["cx"])
        cy = float(pose["cy"])
    elif "projection_matrix" in pose and "world_to_view_matrix" in pose:
        # projection_matrix is the full PyTorch3D world-to-clip transform, not
        # the pure camera projection. Remove world_to_view before deriving the
        # pixel focal length. PyTorch3D NDC uses the shorter image side as the
        # [-1, 1] scale for non-square images.
        full = np.asarray(pose["projection_matrix"][0], dtype=np.float64)
        w2v = np.asarray(pose["world_to_view_matrix"][0], dtype=np.float64)
        p = np.linalg.inv(w2v) @ full
        focal_scale = min(width, height) / 2.0
        fx = abs(float(p[0, 0])) * focal_scale
        fy = abs(float(p[1, 1])) * focal_scale
    elif "projection_matrix" in pose:
        p = np.asarray(pose["projection_matrix"][0], dtype=np.float64)
        focal_scale = min(width, height) / 2.0
        fx = abs(float(p[0, 0])) * focal_scale
        fy = abs(float(p[1, 1])) * focal_scale
    else:
        fx = 0.7 * width
        fy = 0.7 * height
    if "cx" not in locals():
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
    return fx, fy, cx, cy


def _load_depth_u16(path):
    arr = np.asarray(Image.open(path))
    if arr.dtype != np.uint16:
        arr = np.clip(arr.astype(np.float32), 0, 65535).astype(np.uint16)
    return arr


def _degrade_depth_blocks(depth_u16, r_depth, n_blocks, seed):
    n_blocks = max(int(n_blocks), 1)
    r_depth = max(0.0, min(1.0, float(r_depth)))
    keep = int(math.floor(r_depth * n_blocks + 1e-9))
    if r_depth > 0.0:
        keep = max(keep, 1)
    if keep >= n_blocks:
        return depth_u16
    out = depth_u16.copy()
    grid_h = int(math.floor(math.sqrt(n_blocks)))
    grid_w = int(math.ceil(n_blocks / max(grid_h, 1)))
    blocks = [(i, j) for i in range(grid_h) for j in range(grid_w)][:n_blocks]
    rng = random.Random(seed)
    rng.shuffle(blocks)
    keep_set = set(blocks[:keep])
    h, w = out.shape[:2]
    for i, j in blocks:
        if (i, j) in keep_set:
            continue
        y0, y1 = int(i * h / grid_h), int((i + 1) * h / grid_h)
        x0, x1 = int(j * w / grid_w), int((j + 1) * w / grid_w)
        out[y0:y1, x0:x1] = 0
    return out


def _stable_depth_seed(cond_name, local_idx):
    return zlib.crc32(f"{cond_name}:{local_idx}".encode("utf-8")) & 0xFFFFFFFF


def _write_gsfusion_config(path, sequence_path, output_path, optim_params_path, width, height, intrinsics, depth_scale, max_frames):
    fx, fy, cx, cy = intrinsics
    text = f"""%YAML:1.2
map:
  dim:                        [120, 120, 120]
  res:                        0.10

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
  ground_truth_file:          "{sequence_path}/traj.txt"
  inverse_scale:              {float(depth_scale)}
  fps:                        0.0
  drop_frames:                false
  verbose:                    0

app:
  enable_ground_truth:        true
  optim_params_path:          "{optim_params_path}"
  ply_path:                   "{output_path}/point_cloud"
  mesh_path:                  "{output_path}/mesh"
  slice_path:                 ""
  structure_path:             ""
  integration_rate:           1
  rendering_rate:             1
  meshing_rate:               0
  max_frames:                 {int(max_frames)}
  log_file:                   "{output_path}/log.tsv"
"""
    path.write_text(text)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--q-normalization-lookup", default=None,
                    help="Optional lookup CSV used only to compute the global RGB PSNR normalization range.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--semcom-ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--manifest", default=None,
                    help="Optional manifest path relative to data.root, overriding --split manifest.")
    ap.add_argument("--scene-id", default=None,
                    help="Only use samples from this scene_id. Omit to use all scenes in the manifest.")
    ap.add_argument("--trajectory-ids", default=None,
                    help="Comma-separated trajectories to include, e.g. 0,1,2 or traj0,traj1.")
    ap.add_argument("--out", default="outputs/gsfusion_conditions")
    ap.add_argument("--gsfusion-root", default="../GSFusion")
    ap.add_argument("--num-conditions", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--n-depth-blocks", type=int, default=16)
    ap.add_argument("--no-normalize-poses", action="store_true",
                    help="Write raw C2W poses instead of local poses relative to the first frame.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = args.device or cfg.get("train", {}).get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    rows = _read_lookup(args.lookup)
    norm_rows = _read_lookup(args.q_normalization_lookup) if args.q_normalization_lookup else None
    _normalize_lookup_q(rows, norm_rows=norm_rows)
    ds = _build_dataset(cfg, args.split, manifest_rel=args.manifest, scene_id_filter=args.scene_id)
    frame_key_to_idx = _dataset_index_by_frame_key(ds)
    traj_ids = _parse_traj_ids(args.trajectory_ids)
    if traj_ids is not None:
        rows = [
            row for row in rows
            if _normalize_traj_id(row.get("trajectory_id", "")) in traj_ids
        ]
    if not rows:
        raise SystemExit("lookup has no rows after scene/trajectory filtering")
    model = HDAROISystem(cfg).to(device)
    model.load_state_dict(torch.load(args.semcom_ckpt, map_location=device))
    model.eval()

    grouped = defaultdict(list)
    for row in rows:
        key = (row.get("trajectory_id", ""), row["snr_db"], row["k_d"])
        grouped[key].append(row)
    groups = []
    for key, items in grouped.items():
        items = sorted(items, key=lambda r: r["frame_id_int"])
        if len(items) >= 2:
            groups.append((key, items))
    short_groups = [(key, len(items)) for key, items in groups if len(items) < args.max_frames]
    if short_groups:
        print(
            f"ERROR: lookup has condition groups shorter than --max-frames={args.max_frames}; "
            "rebuild lookup or lower --max-frames only if intentional."
        )
        for key, count in short_groups[:20]:
            print(f"  {key}: rows={count}")
        raise SystemExit(1)
    rng = random.Random(args.seed)
    rng.shuffle(groups)
    if args.num_conditions > 0:
        groups = groups[:args.num_conditions]

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    gsfusion_root = Path(args.gsfusion_root).resolve()
    optim_params = gsfusion_root / "parameter/optimization_params_degraded.json"
    index = []

    for cond_idx, ((traj_id, snr_db, k_d), items) in enumerate(groups):
        items = items[:args.max_frames]
        cond_name = f"cond_{cond_idx:03d}_{traj_id or 'traj'}_snr{snr_db:g}_kd{k_d}"
        cond_dir = out_root / cond_name
        seq_dir = cond_dir / "sequence"
        results_dir = seq_dir / "results"
        output_dir = cond_dir / "gsfusion_output"
        results_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_meta = []
        c2w_list = []
        first_pose = None
        width = height = None
        depth_scale = 1.0 / 6553.5

        for local_idx, row in enumerate(items):
            sample_idx = _sample_index_for_lookup_row(row, ds, frame_key_to_idx)
            sample = ds[sample_idx]
            rec = ds.samples[sample_idx]
            pose = _pose_json(cfg["data"]["root"], rec)
            if first_pose is None:
                first_pose = pose
            depth_scale = float(pose.get("depth_encoding", {}).get("scale", depth_scale))

            rgb = sample["rgb"].unsqueeze(0).to(device)
            depth = sample["depth"].unsqueeze(0).to(device)
            valid_mask = sample["valid_mask"].unsqueeze(0).to(device)
            depth_bits = sample["depth_payload_bits"].view(1).to(device)
            cfg["channel"]["resource_allocation"] = "fixed_subcarriers"
            cfg["channel"]["digital_resource_ratio"] = int(k_d) / float(cfg["channel"].get("data_subcarriers", 24))
            model.cfg = cfg
            out = model(rgb, depth, valid_mask, float(snr_db), training=False, aux_payload_bits=depth_bits)
            rgb_hat = out["rgb_hat"].detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
            rgb_u8 = (rgb_hat * 255.0 + 0.5).astype(np.uint8)
            Image.fromarray(rgb_u8).save(results_dir / f"frame{local_idx:06d}.jpg", quality=95)

            depth_path = Path(ds._resolve(rec, "depth"))
            depth_u16 = _load_depth_u16(depth_path)
            if rgb_u8.shape[:2] != depth_u16.shape[:2]:
                depth_u16 = np.asarray(Image.fromarray(depth_u16).resize((rgb_u8.shape[1], rgb_u8.shape[0]), Image.NEAREST))
            degraded_depth = _degrade_depth_blocks(
                depth_u16,
                row["r_depth"],
                args.n_depth_blocks,
                seed=_stable_depth_seed(cond_name, local_idx),
            )
            Image.fromarray(degraded_depth).save(results_dir / f"depth{local_idx:06d}.png")

            c2w = _c2w_from_pose(pose)
            c2w_list.append(c2w)
            height, width = rgb_u8.shape[:2]
            frame_meta.append({
                "local_frame": local_idx,
                "sample_idx": sample_idx,
                "scene_id": rec.get("scene_id", ""),
                "trajectory_id": rec.get("trajectory_id", ""),
                "frame_id": rec.get("frame_id", row["frame_id_int"]),
                "clean_image": str(_resolve(cfg["data"]["root"], rec, "clean_image")),
                "depth_image": str(depth_path),
                "q_rgb": row["q_rgb"],
                "r_depth": row["r_depth"],
                "rgb_psnr_lookup": row["rgb_psnr"],
                "depth_payload_bits": row["depth_payload_bits"],
                "depth_bit_budget": row["depth_bit_budget"],
            })

        if args.no_normalize_poses or not c2w_list:
            local_c2w_list = c2w_list
        else:
            base_inv = np.linalg.inv(c2w_list[0])
            local_c2w_list = [base_inv @ c2w for c2w in c2w_list]
        traj_lines = [" ".join(f"{v:.9g}" for v in c2w.reshape(-1)) for c2w in local_c2w_list]
        (seq_dir / "traj.txt").write_text("\n".join(traj_lines) + "\n")
        intrinsics = _intrinsics_from_pose(first_pose or {}, width, height)
        config_path = cond_dir / "gsfusion_config.yaml"
        _write_gsfusion_config(
            config_path,
            seq_dir.resolve(),
            output_dir.resolve(),
            optim_params.resolve(),
            width,
            height,
            intrinsics,
            depth_scale,
            len(frame_meta),
        )
        meta = {
            "condition": cond_name,
            "snr_db": snr_db,
            "k_d": k_d,
            "q_rgb": float(np.mean([m["q_rgb"] for m in frame_meta])),
            "r_depth": float(np.mean([m["r_depth"] for m in frame_meta])),
            "config_path": str(config_path.resolve()),
            "sequence_path": str(seq_dir.resolve()),
            "output_path": str(output_dir.resolve()),
            "frames": frame_meta,
        }
        (cond_dir / "condition_meta.json").write_text(json.dumps(meta, indent=2))
        index.append(meta)
        print(f"[exported] {cond_name} frames={len(frame_meta)}")

    (out_root / "conditions_index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {out_root / 'conditions_index.json'}")


if __name__ == "__main__":
    main()
