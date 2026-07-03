"""Run GSFusion for exported degraded RGB-D conditions."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from cache_gsfusion_geometry_samples import _cache_one_mesh, _condition_seed


def _is_complete_output(output_path):
    render_metrics = output_path / "render_metrics.csv"
    if not render_metrics.exists():
        return False
    if "mean," not in render_metrics.read_text(errors="ignore"):
        return False
    if not any(output_path.glob("point_cloud/iteration_*/point_cloud.ply")):
        return False
    mesh = _latest_mesh(output_path / "mesh")
    sampled_mesh = sorted(output_path.glob("sampled_surface*.ply"))
    if mesh is None and not sampled_mesh:
        return False
    if mesh is not None:
        if mesh.stat().st_size == 0:
            return False
        return _ply_has_complete_last_line(mesh)
    return sampled_mesh[-1].stat().st_size > 0


def _latest_mesh(mesh_dir):
    paths = sorted(mesh_dir.glob("mesh_*.ply"))
    if not paths:
        return None

    def key(path):
        m = re.search(r"mesh_(\d+)\.ply", path.name)
        return int(m.group(1)) if m else -1

    return sorted(paths, key=key)[-1]


def _ply_has_complete_last_line(path):
    try:
        with path.open("rb") as f:
            f.seek(-1, 2)
            return f.read(1) == b"\n"
    except OSError:
        return False


def _free_gib(path):
    return shutil.disk_usage(path).free / (1024 ** 3)


def _parse_config_field(text, key):
    """Extract a numeric value from the OpenCV YAML config by key name."""
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*([^\s#]+)", text, re.MULTILINE)
    if m is None:
        raise ValueError(f"key '{key}' not found in config")
    return float(m.group(1))


def _build_resized_config(orig_config_path, resized_seq_dir, target_w, target_h):
    """
    Create a GSFusion config pointing to resized_seq_dir with sensor dims
    and intrinsics scaled to (target_w, target_h).
    Returns the path to the new config file.
    """
    text = orig_config_path.read_text()

    orig_w = _parse_config_field(text, "width")
    orig_h = _parse_config_field(text, "height")
    sx = target_w / orig_w
    sy = target_h / orig_h

    def scale(key, factor):
        return re.sub(
            rf"(^\s*{re.escape(key)}\s*:\s*)([^\s#]+)",
            lambda m: m.group(1) + f"{float(m.group(2)) * factor:.6g}",
            text,
            flags=re.MULTILINE,
        )

    text = scale("width",  sx);  text = re.sub(r"(width\s*:\s*)\d+\.\d+", lambda m: m.group(1) + str(int(target_w)), text)
    text = scale("height", sy);  text = re.sub(r"(height\s*:\s*)\d+\.\d+", lambda m: m.group(1) + str(int(target_h)), text)
    text = scale("fx", sx)
    text = scale("fy", sy)
    text = scale("cx", sx)
    text = scale("cy", sy)

    seq_str = str(resized_seq_dir.resolve())
    text = re.sub(r'(sequence_path\s*:\s*")[^"]*(")', rf'\g<1>{seq_str}\g<2>', text)
    text = re.sub(r'(ground_truth_file\s*:\s*")[^"]*(")', rf'\g<1>{seq_str}/traj.txt\g<2>', text)

    resized_config_path = orig_config_path.parent / "gsfusion_config_resized.yaml"
    resized_config_path.write_text(text)
    return resized_config_path


def _prepare_resized_sequence(seq_dir, target_w, target_h):
    """
    Create sequence_resized/ alongside sequence/ with images downsampled to
    (target_w, target_h). traj.txt is symlinked. Original files are untouched.
    Returns the resized sequence dir path.
    """
    resized_seq_dir = seq_dir.parent / "sequence_resized"
    resized_results = resized_seq_dir / "results"
    resized_results.mkdir(parents=True, exist_ok=True)

    traj_src = seq_dir / "traj.txt"
    traj_dst = resized_seq_dir / "traj.txt"
    if not traj_dst.exists():
        shutil.copy2(traj_src, traj_dst)

    src_results = seq_dir / "results"
    for src in sorted(src_results.iterdir()):
        dst = resized_results / src.name
        if dst.exists():
            continue
        img = Image.open(src)
        img_resized = img.resize((target_w, target_h), Image.LANCZOS)
        img_resized.save(dst)

    return resized_seq_dir


def _cache_and_maybe_prune_geometry(output_path, condition, sample_points, sample_name, prune_mesh):
    if sample_points <= 0:
        return
    out_path = output_path / sample_name
    mesh_dir = output_path / "mesh"
    if not out_path.exists():
        mesh = _latest_mesh(mesh_dir)
        if mesh is None:
            raise FileNotFoundError(f"cannot cache geometry for {condition}: no mesh under {mesh_dir}")
        seed = _condition_seed(0, condition)
        meta = _cache_one_mesh(mesh, out_path, sample_points, seed)
        print(
            f"[geometry-cache] {condition} points={meta['sampled_point_count']} "
            f"method={meta['method']} -> {out_path}",
            flush=True,
        )
    if prune_mesh and mesh_dir.exists():
        shutil.rmtree(mesh_dir)
        print(f"[geometry-prune] {condition} removed {mesh_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsfusion-root", default="../GSFusion")
    ap.add_argument("--conditions-dir", default="outputs/gsfusion_conditions")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--continue-on-error", action="store_true")
    ap.add_argument("--gsfusion-width", type=int, default=None,
                    help="Resize input images to this width at runtime (e.g. 912). "
                         "Original exported images are not modified.")
    ap.add_argument("--gsfusion-height", type=int, default=None,
                    help="Resize input images to this height at runtime (e.g. 512). "
                         "Original exported images are not modified.")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                    help="Minimum free GiB required before starting each GSFusion run.")
    ap.add_argument("--cache-geometry-sample-points", type=int, default=0,
                    help="After each complete output, cache this many mesh vertices to a small PLY. 0 disables.")
    ap.add_argument("--geometry-sample-name", default="sampled_surface_100k.ply")
    ap.add_argument("--prune-mesh-after-cache", action="store_true",
                    help="Delete gsfusion_output/mesh after the geometry cache file exists.")
    args = ap.parse_args()

    resize = args.gsfusion_width is not None and args.gsfusion_height is not None

    gsfusion_root = Path(args.gsfusion_root).resolve()
    exe = gsfusion_root / "build/app/gsfusion"
    if not exe.exists():
        raise SystemExit(f"GSFusion executable not found: {exe}")

    index_path = Path(args.conditions_dir) / "conditions_index.json"
    if not index_path.exists():
        raise SystemExit(f"missing conditions index: {index_path}")
    index = json.loads(index_path.read_text())
    if args.limit is not None:
        index = index[:args.limit]

    for i, cond in enumerate(index, start=1):
        config_path = Path(cond["config_path"])
        output_path = Path(cond["output_path"])
        if args.skip_existing and _is_complete_output(output_path):
            _cache_and_maybe_prune_geometry(
                output_path,
                cond["condition"],
                args.cache_geometry_sample_points,
                args.geometry_sample_name,
                args.prune_mesh_after_cache,
            )
            print(f"[skip {i}/{len(index)}] {cond['condition']}")
            continue
        print(f"[run {i}/{len(index)}] {cond['condition']}")
        free_gib = _free_gib(config_path.parent)
        if free_gib < args.min_free_gb:
            raise SystemExit(
                f"Only {free_gib:.2f} GiB free under {config_path.parent}; "
                f"refusing to start {cond['condition']} because it may leave partial outputs. "
                f"Free space or lower --min-free-gb to continue."
            )

        run_config = config_path
        if resize:
            seq_dir = Path(cond["sequence_path"])
            resized_seq_dir = _prepare_resized_sequence(
                seq_dir, args.gsfusion_width, args.gsfusion_height
            )
            run_config = _build_resized_config(
                config_path, resized_seq_dir, args.gsfusion_width, args.gsfusion_height
            )

        log_path = config_path.parent / "gsfusion_run.log"
        with log_path.open("w") as log_file:
            proc = subprocess.run(
                [str(exe), str(run_config)],
                cwd=str(gsfusion_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        log_text = log_path.read_text(errors="ignore")
        if proc.returncode != 0:
            print(log_text[-4000:])
            msg = f"GSFusion failed for {cond['condition']} with code {proc.returncode}"
            if args.continue_on_error:
                (config_path.parent / "gsfusion_failed.txt").write_text(msg + "\n")
                print(f"[failed] {msg}")
                continue
            raise SystemExit(msg)
        if "#Keyframes: 0" in log_text:
            msg = f"GSFusion produced no keyframes for {cond['condition']}"
            (config_path.parent / "gsfusion_empty.txt").write_text(msg + "\n")
            print(f"[empty] {msg}")
            if not args.continue_on_error:
                raise SystemExit(msg)
            continue
        _cache_and_maybe_prune_geometry(
            output_path,
            cond["condition"],
            args.cache_geometry_sample_points,
            args.geometry_sample_name,
            args.prune_mesh_after_cache,
        )
        print(f"[done] {cond['condition']} log={log_path}")


if __name__ == "__main__":
    main()
