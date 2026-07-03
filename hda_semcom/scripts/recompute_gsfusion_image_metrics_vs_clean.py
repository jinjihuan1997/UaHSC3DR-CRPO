"""Recompute GSFusion image metrics against clean GT frames.

The regular real-metrics builder can optionally read GSFusion's
render_metrics.csv, but that compares rendered frames against the input
sequence. For degraded sequences, the input is already channel-damaged. This
script updates only image-quality columns against the clean frame paths stored
in conditions_index.json while preserving existing geometry metrics.
"""

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

from build_gsfusion_real_metrics import _lpips_fallback, _psnr, _read_image, _ssim_global


def _condition_map(conditions_dir):
    index_path = Path(conditions_dir) / "conditions_index.json"
    if not index_path.exists():
        raise SystemExit(f"missing conditions index: {index_path}")
    return {item["condition"]: item for item in json.loads(index_path.read_text())}


def _read_rows(path):
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _resized_clean(clean_cache, path, shape):
    key = (str(path), shape[0], shape[1])
    if key in clean_cache:
        return clean_cache[key]
    clean = _read_image(path)
    if clean.shape != shape:
        clean = np.asarray(
            Image.fromarray((clean * 255).astype(np.uint8)).resize(
                (shape[1], shape[0]), Image.BICUBIC
            ),
            dtype=np.float32,
        ) / 255.0
    clean_cache[key] = clean
    return clean


def _image_metrics(cond, clean_cache):
    output = Path(cond["output_path"])
    render_dir = output / "render_eval"
    psnr_vals = []
    ssim_vals = []
    lpips_vals = []
    for frame in cond["frames"]:
        rendered = render_dir / f"frame{int(frame['local_frame']):06d}.png"
        clean = Path(frame["clean_image"])
        if not rendered.exists() or not clean.exists():
            continue
        a = _read_image(rendered)
        b = _resized_clean(clean_cache, clean, a.shape)
        ssim = _ssim_global(a, b)
        psnr_vals.append(_psnr(a, b))
        ssim_vals.append(ssim)
        lpips_vals.append(_lpips_fallback(a, b, ssim))

    if not psnr_vals:
        return None
    return {
        "psnr": float(np.mean(psnr_vals)),
        "ssim": float(np.mean(ssim_vals)),
        "lpips": float(np.mean(lpips_vals)),
    }


def _worker(item):
    idx, cond = item
    return idx, _image_metrics(cond, {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-in", required=True)
    ap.add_argument("--conditions-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress-every", type=int, default=25)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    rows, fieldnames = _read_rows(args.metrics_in)
    by_condition = _condition_map(args.conditions_dir)
    out_rows = []
    jobs = []
    for idx, row in enumerate(rows):
        cond = by_condition.get(row["condition"])
        if cond is None:
            raise SystemExit(f"condition missing from index: {row['condition']}")
        jobs.append((idx, cond))

    results = {}
    if args.workers > 1:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_worker, job) for job in jobs]
            for future in as_completed(futures):
                idx, metrics = future.result()
                results[idx] = metrics
                done += 1
                if args.progress_every > 0 and done % args.progress_every == 0:
                    print(f"[image-metrics] {done}/{len(rows)}", file=sys.stderr)
    else:
        clean_cache = {}
        for idx, cond in jobs:
            results[idx] = _image_metrics(cond, clean_cache)
            if args.progress_every > 0 and (idx + 1) % args.progress_every == 0:
                print(f"[image-metrics] {idx + 1}/{len(rows)} {cond['condition']}", file=sys.stderr)

    for idx, row in enumerate(rows):
        metrics = results[idx]
        if metrics is None:
            print(f"[warn] no clean image metrics for {row['condition']}", file=sys.stderr)
        else:
            row.update({name: f"{value:.12g}" for name, value in metrics.items()})
        out_rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {out_path} rows={len(out_rows)}")


if __name__ == "__main__":
    main()
