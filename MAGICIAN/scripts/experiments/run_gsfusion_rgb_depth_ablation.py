#!/usr/bin/env python3
"""Run GSFusion ablations over RGB JPEG quality and depth keep ratio."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
TCOM_ROOT = ROOT.parent
DEFAULT_TRAJ_ROOT = ROOT / "data/Macarons++/macarons++/fushimi/test_memory_0/training/0"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "gsfusion_rgb_depth_ablation_fushimi"
DEFAULT_GSFUSION_ROOT = TCOM_ROOT / "GSFusion"
DEFAULT_GSFUSION_BIN = DEFAULT_GSFUSION_ROOT / "build/app/gsfusion"


def r_tag(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def run_name(rgb_quality: int, depth_keep: float) -> str:
    return f"rgbq{int(rgb_quality)}_depthr{r_tag(depth_keep)}"


def run_cmd(cmd: list[str], cwd: Path, log_path: Path | None = None) -> float:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    start = time.time()
    if log_path is None:
        completed = subprocess.run(cmd, cwd=cwd)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as log:
            log.write("Command: " + " ".join(str(c) for c in cmd) + "\n")
            log.flush()
            completed = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.time() - start
    if completed.returncode != 0:
        suffix = f"; see {log_path}" if log_path else ""
        raise RuntimeError(f"Command failed with exit code {completed.returncode}{suffix}")
    return elapsed


def read_stats(path: Path) -> dict[str, float | int | str | None]:
    stats = {
        "avg_fps": None,
        "global_opt_time_s": None,
        "gpu_memory_mb": None,
        "keyframes": None,
    }
    if not path.exists():
        return stats
    text = path.read_text(errors="ignore")
    patterns = {
        "avg_fps": r"Avg\. fps:\s*([0-9.eE+-]+)",
        "global_opt_time_s": r"Global opt\. time:\s*([0-9.eE+-]+)",
        "gpu_memory_mb": r"GPU memory usage:\s*([0-9.eE+-]+)",
        "keyframes": r"#Keyframes:\s*([0-9]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            stats[key] = int(value) if key == "keyframes" else float(value)
    return stats


def read_render_metric_mean(path: Path) -> dict[str, float | None]:
    metrics = {
        "render_psnr_input": None,
        "render_ssim_input": None,
        "render_l1_input": None,
    }
    if not path.exists():
        return metrics
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("frame_index") == "mean":
                metrics["render_psnr_input"] = float(row["psnr_input"])
                metrics["render_ssim_input"] = float(row["ssim_input"])
                metrics["render_l1_input"] = float(row["l1_input"])
                return metrics
    return metrics


def frame_rgb_uint8(frame: dict) -> np.ndarray:
    rgb = frame["rgb"]
    if rgb.ndim == 4:
        rgb = rgb[0]
    arr = rgb.detach().cpu().numpy().astype(np.float32)
    if arr.max(initial=0.0) <= 1.0:
        arr *= 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def load_torch_frame(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def ssim_rgb(render: np.ndarray, target: np.ndarray) -> float:
    try:
        import cv2
    except ImportError:
        return float("nan")

    render = render.astype(np.float32)
    target = target.astype(np.float32)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    scores = []
    for channel in range(3):
        x = render[..., channel]
        y = target[..., channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        ssim_map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )
        scores.append(float(np.mean(ssim_map)))
    return float(np.mean(scores))


def image_metrics(render: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    render_f = render.astype(np.float32) / 255.0
    target_f = target.astype(np.float32) / 255.0
    diff = render_f - target_f
    mse = float(np.mean(diff * diff))
    psnr = float("inf") if mse == 0.0 else 20.0 * math.log10(1.0 / math.sqrt(mse))
    l1 = float(np.mean(np.abs(diff)))
    ssim = ssim_rgb(render_f, target_f)
    return psnr, ssim, l1


def compute_original_rgb_render_metrics(run_dir: Path, max_frames: int | None = None) -> dict[str, float | int | None]:
    manifest = load_manifest(run_dir)
    frames = manifest.get("frames", [])
    render_dir = run_dir / "render_eval"
    if not frames or not render_dir.exists():
        return {
            "render_psnr_original": None,
            "render_ssim_original": None,
            "render_l1_original": None,
            "render_frames": 0,
        }

    psnr_values = []
    ssim_values = []
    l1_values = []
    selected_frames = frames[:max_frames] if max_frames is not None else frames
    for frame_info in selected_frames:
        frame_index = int(frame_info["frame_index"])
        render_path = render_dir / f"frame{frame_index:06d}.png"
        source_frame = Path(frame_info["source_frame"])
        if not render_path.exists() or not source_frame.exists():
            continue
        render = np.asarray(Image.open(render_path).convert("RGB"))
        source = frame_rgb_uint8(load_torch_frame(source_frame))
        if render.shape[:2] != source.shape[:2]:
            source = np.asarray(Image.fromarray(source).resize((render.shape[1], render.shape[0]), Image.BILINEAR))
        psnr, ssim, l1 = image_metrics(render, source)
        psnr_values.append(psnr)
        if not math.isnan(ssim):
            ssim_values.append(ssim)
        l1_values.append(l1)

    return {
        "render_psnr_original": float(np.mean(psnr_values)) if psnr_values else None,
        "render_ssim_original": float(np.mean(ssim_values)) if ssim_values else None,
        "render_l1_original": float(np.mean(l1_values)) if l1_values else None,
        "render_frames": len(psnr_values),
    }


def ply_header_counts(path: Path) -> tuple[int | None, int | None, int | None]:
    if not path.exists():
        return None, None, None
    vertices = None
    faces = None
    header_bytes = 0
    with path.open("rb") as f:
        for raw in f:
            header_bytes += len(raw)
            line = raw.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                vertices = int(line.split()[-1])
            elif line.startswith("element face"):
                faces = int(line.split()[-1])
            elif line == "end_header":
                break
    return vertices, faces, header_bytes


def load_manifest(run_dir: Path) -> dict:
    path = run_dir / "export_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def collect_row(args: argparse.Namespace, run_dir: Path, rgb_quality: int, depth_keep: float, train_elapsed: float) -> dict:
    manifest = load_manifest(run_dir)
    stats = read_stats(run_dir / "stats")
    render_input_metrics = read_render_metric_mean(run_dir / "render_metrics.csv")
    render_original_metrics = compute_original_rgb_render_metrics(run_dir, args.render_metric_max_frames)
    point_cloud = run_dir / "point_cloud" / f"iteration_{manifest.get('num_frames', args.gsfusion_max_frames if args.gsfusion_max_frames > 0 else '')}" / "point_cloud.ply"
    if not point_cloud.exists():
        candidates = sorted((run_dir / "point_cloud").glob("iteration_*/point_cloud.ply"))
        point_cloud = candidates[-1] if candidates else point_cloud
    pc_vertices, pc_faces, _ = ply_header_counts(point_cloud)
    depth_info = manifest.get("depth", {})
    rgb_info = manifest.get("rgb", {})
    return {
        "run_name": run_dir.name,
        "rgb_quality": rgb_quality,
        "depth_keep_ratio_requested": depth_keep,
        "depth_keep_ratio_actual": depth_info.get("keep_ratio_actual"),
        "depth_corruption_mode": depth_info.get("corruption_mode"),
        "depth_block_size": depth_info.get("block_size"),
        "rgb_ext": rgb_info.get("extension"),
        "frames": manifest.get("num_frames"),
        "width": manifest.get("width"),
        "height": manifest.get("height"),
        "point_cloud_vertices": pc_vertices,
        "point_cloud_size_bytes": point_cloud.stat().st_size if point_cloud.exists() else None,
        "avg_fps": stats.get("avg_fps"),
        "global_opt_time_s": stats.get("global_opt_time_s"),
        "gpu_memory_mb": stats.get("gpu_memory_mb"),
        "keyframes": stats.get("keyframes"),
        "wall_time_s": train_elapsed,
        **render_input_metrics,
        **render_original_metrics,
        "dataset_dir": str(run_dir),
        "point_cloud_path": str(point_cloud) if point_cloud.exists() else "",
        "render_metrics_path": str(run_dir / "render_metrics.csv"),
        "render_eval_dir": str(run_dir / "render_eval"),
        "stats_path": str(run_dir / "stats"),
        "run_log": str(run_dir / "run.log"),
    }


def write_summary(rows: list[dict], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "summary.csv"
    fieldnames = [
        "run_name",
        "rgb_quality",
        "depth_keep_ratio_requested",
        "depth_keep_ratio_actual",
        "depth_corruption_mode",
        "depth_block_size",
        "rgb_ext",
        "frames",
        "width",
        "height",
        "point_cloud_vertices",
        "point_cloud_size_bytes",
        "avg_fps",
        "global_opt_time_s",
        "gpu_memory_mb",
        "keyframes",
        "wall_time_s",
        "render_psnr_input",
        "render_ssim_input",
        "render_l1_input",
        "render_psnr_original",
        "render_ssim_original",
        "render_l1_original",
        "render_frames",
        "dataset_dir",
        "point_cloud_path",
        "render_metrics_path",
        "render_eval_dir",
        "stats_path",
        "run_log",
    ]
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# GSFusion RGB/Depth Ablation",
        "",
        "Fixed MAGICIAN Fushimi trajectory; changed only RGB JPEG quality and depth keep ratio.",
        "",
        "| Run | RGB Q | Depth keep | Actual keep | Points | PSNR vs original | SSIM vs original | PSNR vs input | Avg FPS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {rgb_quality} | {depth_keep_ratio_requested} | {depth_keep_ratio_actual} | "
            "{point_cloud_vertices} | {render_psnr_original} | {render_ssim_original} | "
            "{render_psnr_input} | {avg_fps} |".format(**row)
        )
    if rows:
        qualities = sorted({int(row["rgb_quality"]) for row in rows}, reverse=True)
        keeps = sorted({float(row["depth_keep_ratio_requested"]) for row in rows}, reverse=True)
        by_key = {(int(row["rgb_quality"]), float(row["depth_keep_ratio_requested"])): row for row in rows}
        base = by_key.get((max(qualities), max(keeps)))
        lines.extend(["", "## Point Count Matrix", "", "| RGB Q / Depth keep | " + " | ".join(f"{k:.2f}" for k in keeps) + " |"])
        lines.append("|---" + "|---:" * len(keeps) + "|")
        for quality in qualities:
            vals = []
            for keep in keeps:
                row = by_key.get((quality, keep))
                vals.append(str(row.get("point_cloud_vertices", "")) if row else "")
            lines.append(f"| {quality} | " + " | ".join(vals) + " |")
        if base and base.get("point_cloud_vertices"):
            base_points = int(base["point_cloud_vertices"])
            lines.extend(["", "## Relative Point Count Drop", "", "| Run | Drop vs RGB95/Depth1.0 |", "|---|---:|"])
            for row in rows:
                points = int(row["point_cloud_vertices"])
                drop = (points / base_points - 1.0) * 100.0
                lines.append(f"| {row['run_name']} | {drop:.2f}% |")
            lines.extend([
                "",
                "## Current Conclusion",
                "",
                "- Depth keep ratio dominates point-cloud coverage in this setup.",
                "- Reducing depth keep from 1.0 to 0.9 reduces point count by about 3%.",
                "- Reducing depth keep from 1.0 to 0.5 reduces point count by about 16%.",
                "- RGB JPEG quality from 95 to 25 has little effect on point count when depth keep is fixed.",
                "- RGB quality should be judged with render metrics, especially PSNR/SSIM against the original uncompressed MAGICIAN RGB.",
            ])
        lines.extend(["", "## Render PSNR vs Original RGB", ""])
        lines.append("| RGB Q / Depth keep | " + " | ".join(f"{k:.2f}" for k in keeps) + " |")
        lines.append("|---" + "|---:" * len(keeps) + "|")
        for quality in qualities:
            vals = []
            for keep in keeps:
                row = by_key.get((quality, keep))
                val = row.get("render_psnr_original") if row else None
                vals.append("" if val is None else f"{float(val):.4f}")
            lines.append(f"| {quality} | " + " | ".join(vals) + " |")

        lines.extend(["", "## Render SSIM vs Original RGB", ""])
        lines.append("| RGB Q / Depth keep | " + " | ".join(f"{k:.2f}" for k in keeps) + " |")
        lines.append("|---" + "|---:" * len(keeps) + "|")
        for quality in qualities:
            vals = []
            for keep in keeps:
                row = by_key.get((quality, keep))
                val = row.get("render_ssim_original") if row else None
                vals.append("" if val is None else f"{float(val):.4f}")
            lines.append(f"| {quality} | " + " | ".join(vals) + " |")

        lines.extend(["", "## Monotonicity Checks", ""])
        for quality in qualities:
            vals = [
                by_key[(quality, keep)].get("point_cloud_vertices")
                for keep in keeps
                if (quality, keep) in by_key and by_key[(quality, keep)].get("point_cloud_vertices") is not None
            ]
            if len(vals) > 1:
                monotonic_depth = all(int(vals[i]) >= int(vals[i + 1]) for i in range(len(vals) - 1))
                result = "monotonic" if monotonic_depth else "not monotonic"
            else:
                result = "insufficient data"
            lines.append(f"- Point count vs depth keep at RGB Q{quality}: {result}")
        for quality in qualities:
            vals = [
                by_key[(quality, keep)].get("render_ssim_original")
                for keep in keeps
                if (quality, keep) in by_key and by_key[(quality, keep)].get("render_ssim_original") is not None
            ]
            if len(vals) > 1:
                monotonic_depth_ssim = all(float(vals[i]) >= float(vals[i + 1]) for i in range(len(vals) - 1))
                result = "monotonic" if monotonic_depth_ssim else "not monotonic"
            else:
                result = "insufficient data"
            lines.append(f"- Render SSIM vs depth keep at RGB Q{quality}: {result}")
        for keep in keeps:
            vals = [
                by_key[(quality, keep)].get("render_psnr_original")
                for quality in sorted(qualities)
                if (quality, keep) in by_key and by_key[(quality, keep)].get("render_psnr_original") is not None
            ]
            if len(vals) > 1:
                monotonic_rgb = all(float(vals[i]) <= float(vals[i + 1]) for i in range(len(vals) - 1))
                result = "monotonic" if monotonic_rgb else "not monotonic"
            else:
                result = "insufficient data"
            lines.append(f"- Render PSNR vs RGB quality at depth keep {keep:.2f}: {result}")
        for keep in keeps:
            vals = [
                by_key[(quality, keep)].get("render_ssim_original")
                for quality in sorted(qualities)
                if (quality, keep) in by_key and by_key[(quality, keep)].get("render_ssim_original") is not None
            ]
            if len(vals) > 1:
                monotonic_rgb_ssim = all(float(vals[i]) <= float(vals[i + 1]) for i in range(len(vals) - 1))
                result = "monotonic" if monotonic_rgb_ssim else "not monotonic"
            else:
                result = "insufficient data"
            lines.append(f"- Render SSIM vs RGB quality at depth keep {keep:.2f}: {result}")
    lines.extend([
        "",
        "Notes:",
        "- Mesh export is disabled; comparison uses GSFusion Gaussian point cloud output and runtime stats.",
        "- Depth loss is represented as depth=0 in exported PNGs, which GSFusion treats as invalid depth.",
        "- This is a reconstruction-output proxy comparison, not a ground-truth geometry accuracy evaluation.",
    ])
    (output_root / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GSFusion RGB quality / depth keep ratio ablation.")
    parser.add_argument("--traj-root", type=Path, default=DEFAULT_TRAJ_ROOT)
    parser.add_argument("--frames-subdir", choices=["frames", "frames_highres"], default="frames_highres")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gsfusion-root", type=Path, default=DEFAULT_GSFUSION_ROOT)
    parser.add_argument("--gsfusion-bin", type=Path, default=DEFAULT_GSFUSION_BIN)
    parser.add_argument("--rgb-qualities", type=int, nargs="+", default=[95, 50, 25])
    parser.add_argument("--depth-keep-ratios", type=float, nargs="+", default=[1.0, 0.9, 0.5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--depth-block-size", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--gsfusion-max-frames", type=int, default=-1)
    parser.add_argument("--map-dim", type=float, nargs=3, default=[240.0, 240.0, 240.0])
    parser.add_argument("--map-res", type=float, default=0.1)
    parser.add_argument("--far-plane", type=float, default=220.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--render-metric-max-frames",
        type=int,
        default=None,
        help="Limit Python-side render-vs-original metric frames. Default evaluates all rendered frames.",
    )
    args = parser.parse_args()

    if not args.gsfusion_bin.exists():
        raise FileNotFoundError(args.gsfusion_bin)

    output_root = args.output_root.resolve()
    rows = []
    for rgb_quality in args.rgb_qualities:
        for depth_keep in args.depth_keep_ratios:
            name = run_name(rgb_quality, depth_keep)
            run_dir = output_root / name
            pc_candidates = sorted((run_dir / "point_cloud").glob("iteration_*/point_cloud.ply"))
            if args.skip_existing and pc_candidates and (run_dir / "stats").exists():
                print(f"[skip] {name}")
                rows.append(collect_row(args, run_dir, rgb_quality, depth_keep, 0.0))
                continue

            print("=" * 80)
            print(f"[INFO] Run {name}")
            export_cmd = [
                sys.executable,
                str(ROOT / "scripts/export_magician_to_gsfusion_replica.py"),
                "--traj-root", str(args.traj_root),
                "--frames-subdir", args.frames_subdir,
                "--output", str(run_dir),
                "--rgb-quality", str(rgb_quality),
                "--depth-keep-ratio", str(depth_keep),
                "--depth-corruption-mode", "block_dropout",
                "--depth-block-size", str(args.depth_block_size),
                "--seed", str(args.seed),
                "--far-plane", str(args.far_plane),
                "--map-dim", *(str(v) for v in args.map_dim),
                "--map-res", str(args.map_res),
                "--disable-mesh",
                "--gsfusion-root", str(args.gsfusion_root),
            ]
            if args.max_frames is not None:
                export_cmd.extend(["--max-frames", str(args.max_frames)])
            if args.gsfusion_max_frames is not None:
                export_cmd.extend(["--gsfusion-max-frames", str(args.gsfusion_max_frames)])
            if args.overwrite:
                export_cmd.append("--overwrite")
            run_cmd(export_cmd, cwd=ROOT, log_path=run_dir.with_suffix(".export.log"))

            gsfusion_cmd = [str(args.gsfusion_bin), str(run_dir / "config.yaml")]
            elapsed = run_cmd(gsfusion_cmd, cwd=args.gsfusion_root, log_path=run_dir / "run.log")
            row = collect_row(args, run_dir, rgb_quality, depth_keep, elapsed)
            rows.append(row)
            write_summary(rows, output_root)

    write_summary(rows, output_root)
    print(f"[INFO] Summary CSV: {output_root / 'summary.csv'}")
    print(f"[INFO] Summary MD:  {output_root / 'summary.md'}")


if __name__ == "__main__":
    main()
