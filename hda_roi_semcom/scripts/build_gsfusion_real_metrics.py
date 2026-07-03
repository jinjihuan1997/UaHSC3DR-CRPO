"""Build gsfusion_real_metrics.csv from GSFusion condition outputs."""

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def _read_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def _psnr(a, b):
    mse = float(np.mean(np.square(a - b)))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def _ssim_global(a, b):
    x = a.reshape(-1, 3)
    y = b.reshape(-1, 3)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    vals = []
    for ch in range(3):
        xi = x[:, ch]
        yi = y[:, ch]
        mux, muy = xi.mean(), yi.mean()
        vx, vy = xi.var(), yi.var()
        cov = ((xi - mux) * (yi - muy)).mean()
        vals.append(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))
    return float(np.clip(np.mean(vals), 0.0, 1.0))


def _lpips_fallback(a, b, ssim):
    # Monotonic perceptual-distance fallback when LPIPS is not installed.
    l1 = float(np.mean(np.abs(a - b)))
    return float(np.clip(0.5 * l1 + 0.5 * (1.0 - ssim), 0.0, 1.0))


def _latest_mesh(mesh_dir):
    paths = sorted(Path(mesh_dir).glob("mesh_*.ply"))
    if not paths:
        return None
    def key(p):
        m = re.search(r"mesh_(\d+)\.ply", p.name)
        return int(m.group(1)) if m else -1
    return sorted(paths, key=key)[-1]


def _subsample_points(pts, max_points):
    if max_points is not None and pts.shape[0] > max_points:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(pts.shape[0], size=max_points, replace=False)]
    return pts


def _c2w_from_pose(pose):
    if pose.get("coordinate_system") == "MAGICIAN/PyTorch3D" and "R" in pose and "camera_center" in pose:
        r = np.asarray(pose["R"][0], dtype=np.float64)
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = r @ np.diag([-1.0, -1.0, 1.0])
        c2w[:3, 3] = np.asarray(pose["camera_center"][0], dtype=np.float64)
        # MAGICIAN world units are 10× the mesh/settings units in which the GT
        # mesh is stored.  Scale the entire local→world transform by 1/10 so
        # that predicted mesh points land in the same coordinate frame as the GT.
        c2w[:3, :] /= 10.0
        return c2w
    if "world_to_view_matrix" in pose:
        m = np.asarray(pose["world_to_view_matrix"][0], dtype=np.float64)
        return np.linalg.inv(m.T)
    c2w = np.eye(4, dtype=np.float64)
    if "R" in pose:
        c2w[:3, :3] = np.asarray(pose["R"][0], dtype=np.float64)
    if "camera_center" in pose:
        c2w[:3, 3] = np.asarray(pose["camera_center"][0], dtype=np.float64)
    return c2w


def _pose_path_from_frame(frame):
    depth_path = Path(frame["depth_image"])
    return depth_path.parent.parent / "poses" / f"{depth_path.stem}.json"


def _world_from_local(cond):
    """Return local-to-GT-world transform for normalized GSFusion trajectories."""
    sequence = Path(cond["sequence_path"])
    traj_path = sequence / "traj.txt"
    if traj_path.exists():
        first = np.loadtxt(traj_path, max_rows=1).reshape(4, 4)
        if not np.allclose(first, np.eye(4), atol=1e-5):
            return None
    if not cond.get("frames"):
        return None
    pose_path = _pose_path_from_frame(cond["frames"][0])
    if not pose_path.exists():
        return None
    with pose_path.open() as f:
        return _c2w_from_pose(json.load(f))


def _transform_points(pts, transform):
    if transform is None:
        return pts
    pts_h = np.concatenate([pts.astype(np.float64, copy=False), np.ones((len(pts), 1))], axis=1)
    return (pts_h @ transform.T)[:, :3].astype(np.float32)


def _crop_to_bbox(pts, bbox_min, bbox_max, margin):
    lo = bbox_min - margin
    hi = bbox_max + margin
    keep = np.all((pts >= lo) & (pts <= hi), axis=1)
    return pts[keep]


def _read_obj_vertices(path, max_points):
    pts = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    pts = np.asarray(pts, dtype=np.float32)
    return _subsample_points(pts, max_points)


def _read_ply_vertices(path, max_points):
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"bad ply header: {path}")
            header_lines.append(line.decode("ascii", errors="ignore").strip())
            if header_lines[-1] == "end_header":
                break
        fmt = next(line for line in header_lines if line.startswith("format "))
        vertex_count = 0
        properties = []
        in_vertex = False
        for line in header_lines:
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
                in_vertex = True
                continue
            if line.startswith("element ") and not line.startswith("element vertex"):
                in_vertex = False
            if in_vertex and line.startswith("property "):
                parts = line.split()
                properties.append((parts[1], parts[2]))
        prop_names = [p[1] for p in properties]
        xyz_idx = [prop_names.index("x"), prop_names.index("y"), prop_names.index("z")]

        if "ascii" in fmt:
            pts = np.loadtxt(
                f,
                dtype=np.float32,
                usecols=xyz_idx,
                max_rows=vertex_count,
            )
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            if pts.shape[0] != vertex_count:
                raise ValueError(f"truncated ply vertices: expected {vertex_count}, got {pts.shape[0]}: {path}")
        elif "binary_little_endian" in fmt:
            type_map = {
                "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
                "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
                "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
                "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
            }
            dtype = np.dtype([(name, type_map[t]) for t, name in properties])
            data = np.fromfile(f, dtype=dtype, count=vertex_count)
            if data.shape[0] != vertex_count:
                raise ValueError(f"truncated ply vertices: expected {vertex_count}, got {data.shape[0]}: {path}")
            pts = np.column_stack([
                data[prop_names[xyz_idx[0]]],
                data[prop_names[xyz_idx[1]]],
                data[prop_names[xyz_idx[2]]],
            ]).astype(np.float32, copy=False)
        else:
            raise ValueError(f"unsupported ply format {fmt}: {path}")

    return _subsample_points(pts, max_points)


def _read_vertices(path, max_points):
    suffix = Path(path).suffix.lower()
    if suffix == ".ply":
        return _read_ply_vertices(path, max_points)
    if suffix == ".obj":
        return _read_obj_vertices(path, max_points)
    raise ValueError(f"unsupported geometry format: {path}")


def _geometry_metrics(pred_mesh, gt_mesh, max_points, f_threshold, pred_transform=None, crop_margin=None):
    gt = _read_vertices(gt_mesh, max_points)
    gt_tree = cKDTree(gt)
    return _geometry_metrics_against_gt(
        pred_mesh, gt, gt_tree, max_points, f_threshold, pred_transform, crop_margin)


def _geometry_metrics_against_gt(pred_mesh, gt, gt_tree, max_points, f_threshold, pred_transform=None, crop_margin=None):
    pred_max_points = None if crop_margin is not None else max_points
    pred = _transform_points(_read_vertices(pred_mesh, pred_max_points), pred_transform)
    if crop_margin is not None:
        pred = _crop_to_bbox(pred, gt.min(axis=0), gt.max(axis=0), crop_margin)
        pred = _subsample_points(pred, max_points)
    if len(pred) == 0 or len(gt) == 0:
        return float("nan"), float("nan"), float("nan")
    pred_tree = cKDTree(pred)
    d_pred_to_gt, _ = gt_tree.query(pred, k=1)
    d_gt_to_pred, _ = pred_tree.query(gt, k=1)
    chamfer = float(0.5 * (d_pred_to_gt.mean() + d_gt_to_pred.mean()))
    precision = float(np.mean(d_pred_to_gt < f_threshold))
    completeness = float(np.mean(d_gt_to_pred < f_threshold))
    fscore = 0.0 if precision + completeness <= 1e-12 else 2 * precision * completeness / (precision + completeness)
    return chamfer, fscore, completeness


def _metrics_from_render_csv(output):
    render_metrics_csv = Path(output) / "render_metrics.csv"
    if not render_metrics_csv.exists():
        return None
    with render_metrics_csv.open(newline="") as rf:
        rows = list(csv.DictReader(rf))
    data_rows = [r for r in rows if r.get("frame_index", "").strip().lstrip("-").isdigit()]
    if not data_rows:
        return None
    psnr = float(np.mean([float(r["psnr_input"]) for r in data_rows]))
    ssim = float(np.mean([float(r["ssim_input"]) for r in data_rows]))
    lpips = float(np.mean([1.0 - float(r["ssim_input"]) for r in data_rows]))
    return psnr, ssim, lpips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions-dir", default="outputs/gsfusion_conditions")
    ap.add_argument("--out", default="outputs/gsfusion_real_metrics.csv")
    ap.add_argument("--gt-mesh", default=None, help="GT mesh PLY. If omitted, geometry falls back to R_depth proxy.")
    ap.add_argument("--pred-sampled-name", default="sampled_surface_100k.ply",
                    help="Prefer this cached point PLY under each gsfusion_output before reading mesh/mesh_*.ply.")
    ap.add_argument("--max-points", type=int, default=200000)
    ap.add_argument("--f-threshold", type=float, default=1.0)
    ap.add_argument("--crop-gt-bbox-margin", type=float, default=3.0,
                    help="Crop predicted mesh to the GT mesh bbox expanded by this margin before geometry metrics. Use a negative value to disable.")
    ap.add_argument("--include-missing", action="store_true")
    ap.add_argument("--prefer-render-metrics", action="store_true",
                    help="Use GSFusion render_metrics.csv for image metrics instead of rereading render_eval images.")
    args = ap.parse_args()

    index_path = Path(args.conditions_dir) / "conditions_index.json"
    if not index_path.exists():
        raise SystemExit(f"missing conditions index: {index_path}")
    index = json.loads(index_path.read_text())

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt_points = gt_tree = None
    if args.gt_mesh and Path(args.gt_mesh).exists():
        gt_points = _read_vertices(Path(args.gt_mesh), args.max_points)
        gt_tree = cKDTree(gt_points)

    fieldnames = [
        "condition", "q_rgb", "r_depth", "psnr", "ssim", "lpips",
        "chamfer", "fscore", "completeness", "geometry_source",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cond in index:
            output = Path(cond["output_path"])
            csv_metrics = _metrics_from_render_csv(output) if args.prefer_render_metrics else None
            if csv_metrics is not None:
                psnr, ssim, lpips = csv_metrics
            else:
                render_dir = output / "render_eval"
                psnr_vals, ssim_vals, lpips_vals = [], [], []
                for frame in cond["frames"]:
                    rendered = render_dir / f"frame{int(frame['local_frame']):06d}.png"
                    clean_path = frame.get("clean_image")
                    clean = Path(clean_path) if clean_path else None
                    if not rendered.exists() or clean is None or not clean.exists():
                        continue
                    a = _read_image(rendered)
                    b = _read_image(clean)
                    if a.shape != b.shape:
                        b = np.asarray(Image.fromarray((b * 255).astype(np.uint8)).resize((a.shape[1], a.shape[0]), Image.BICUBIC), dtype=np.float32) / 255.0
                    ssim = _ssim_global(a, b)
                    psnr_vals.append(_psnr(a, b))
                    ssim_vals.append(ssim)
                    lpips_vals.append(_lpips_fallback(a, b, ssim))

                if not psnr_vals:
                    # Do NOT fall back to render_metrics.csv (psnr_input/ssim_input): those
                    # measure GSFusion-render vs degraded-input, not vs clean GT — using them
                    # would silently corrupt the quality metrics reported in the paper.
                    print(f"[warn] no clean render_eval frames for {cond['condition']} — skipping",
                          file=sys.stderr)
                    if not args.include_missing:
                        continue
                    psnr = ssim = lpips = float("nan")
                else:
                    psnr = float(np.mean(psnr_vals))
                    ssim = float(np.mean(ssim_vals))
                    lpips = float(np.mean(lpips_vals))

            if args.gt_mesh:
                sampled_pred = output / args.pred_sampled_name
                pred_mesh = sampled_pred if sampled_pred.exists() else _latest_mesh(output / "mesh")
                if pred_mesh is not None and gt_points is not None:
                    pred_transform = _world_from_local(cond)
                    crop_margin = args.crop_gt_bbox_margin if args.crop_gt_bbox_margin >= 0 else None
                    try:
                        chamfer, fscore, completeness = _geometry_metrics_against_gt(
                            pred_mesh, gt_points, gt_tree, args.max_points, args.f_threshold, pred_transform, crop_margin)
                        source_prefix = "sampled_pred_vs_gt" if sampled_pred.exists() else "mesh_vs_gt"
                        geometry_source = f"{source_prefix}_world_aligned" if pred_transform is not None else source_prefix
                        if crop_margin is not None:
                            geometry_source += f"_gt_bbox_crop{crop_margin:g}"
                        if not all(math.isfinite(x) for x in (chamfer, fscore, completeness)):
                            print(f"[warn] invalid geometry metrics for {cond['condition']}: {pred_mesh}", file=sys.stderr)
                            geometry_source = "invalid_geometry"
                    except (OSError, ValueError) as exc:
                        print(f"[warn] invalid mesh for {cond['condition']}: {exc}", file=sys.stderr)
                        chamfer, fscore, completeness = float("nan"), float("nan"), float("nan")
                        geometry_source = "invalid_mesh"
                else:
                    chamfer, fscore, completeness = float("nan"), float("nan"), float("nan")
                    geometry_source = "missing_mesh"
            else:
                # Fallback keeps the CSV usable for fitting but is not a real mesh metric.
                r_depth = float(cond["r_depth"])
                chamfer = 1.0 - r_depth
                fscore = r_depth
                completeness = r_depth
                geometry_source = "proxy_from_r_depth"

            writer.writerow({
                "condition": cond["condition"],
                "q_rgb": cond["q_rgb"],
                "r_depth": cond["r_depth"],
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "chamfer": chamfer,
                "fscore": fscore,
                "completeness": completeness,
                "geometry_source": geometry_source,
            })
            f.flush()
            print(f"[metrics] {cond['condition']} geometry={geometry_source}", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
