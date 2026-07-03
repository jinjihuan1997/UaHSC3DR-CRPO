#!/usr/bin/env python3
"""Evaluate ROI semantic images globally and inside damage masks."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None


def image_sort_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def load_mask(path: Path, size) -> np.ndarray:
    if not path.exists():
        return np.zeros((size[1], size[0]), dtype=bool)
    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, RESAMPLE_NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    err = mse(a, b)
    if err <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(err)))


def masked_psnr(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return psnr(ref[mask], pred[mask])


def masked_ssim(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if skimage_ssim is None or not np.any(mask):
        return float("nan")
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    ref_crop = ref[y0:y1, x0:x1]
    pred_crop = pred[y0:y1, x0:x1]
    mask_crop = mask[y0:y1, x0:x1]
    if ref_crop.shape[0] < 7 or ref_crop.shape[1] < 7:
        return float("nan")
    score_map = skimage_ssim(
        ref_crop.astype(np.uint8),
        pred_crop.astype(np.uint8),
        channel_axis=2,
        data_range=255,
        full=True,
    )[1]
    if score_map.ndim == 3:
        score_map = np.mean(score_map, axis=2)
    return float(np.mean(score_map[mask_crop]))


def read_bit_report(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", newline="") as f:
        return {row["image_name"]: row for row in csv.DictReader(f)}


def nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate damage ROI image quality.")
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--bit-report", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir).expanduser()
    pred_dir = Path(args.pred_dir).expanduser()
    mask_dir = Path(args.mask_dir).expanduser()
    output_csv = Path(args.output_csv).expanduser()
    bit_rows = read_bit_report(Path(args.bit_report).expanduser())

    if skimage_ssim is None:
        print("[WARN] skimage is not available. damage_ssim will be NaN.")

    ref_images = sorted([p for p in ref_dir.iterdir() if p.is_file()], key=image_sort_key)
    rows = []
    for ref_path in ref_images:
        pred_path = pred_dir / ref_path.name
        if not pred_path.exists():
            print(f"[WARN] Missing prediction image: {pred_path}")
            continue

        ref = load_rgb(ref_path)
        pred = load_rgb(pred_path)
        if ref.shape != pred.shape:
            raise ValueError(f"Image shape mismatch for {ref_path.name}: {ref.shape} vs {pred.shape}")

        mask_path = mask_dir / ref_path.with_suffix(".png").name
        mask = load_mask(mask_path, Image.open(ref_path).size)
        mask_pixels = int(mask.sum())
        mask_ratio = float(mask_pixels / mask.size)

        bit_row = bit_rows.get(ref_path.name, {})
        total_bits = float(bit_row.get("total_bits", "nan"))
        total_mbits = total_bits / 1e6 if not math.isnan(total_bits) else float("nan")

        full_score = psnr(ref, pred)
        damage_score = masked_psnr(ref, pred, mask)
        damage_ssim = masked_ssim(ref, pred, mask)
        rows.append({
            "image_name": ref_path.name,
            "full_psnr": full_score,
            "damage_psnr": damage_score,
            "damage_ssim": damage_ssim,
            "mask_pixels": mask_pixels,
            "mask_ratio": mask_ratio,
            "total_bits": total_bits,
            "total_mbits": total_mbits,
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        fieldnames = [
            "image_name",
            "full_psnr",
            "damage_psnr",
            "damage_ssim",
            "mask_pixels",
            "mask_ratio",
            "total_bits",
            "total_mbits",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mean_full = nanmean([r["full_psnr"] for r in rows])
    mean_damage = nanmean([r["damage_psnr"] for r in rows])
    mean_ssim = nanmean([r["damage_ssim"] for r in rows])
    mean_bits = nanmean([r["total_bits"] for r in rows])
    mean_mbits = nanmean([r["total_mbits"] for r in rows])
    psnr_per_mbit = mean_damage / mean_mbits if mean_mbits and not math.isnan(mean_mbits) else float("nan")

    print(f"[INFO] output csv                 : {output_csv}")
    print(f"[INFO] frames evaluated           : {len(rows)}")
    print(f"[INFO] mean full_psnr             : {mean_full:.4f}")
    print(f"[INFO] mean damage_psnr           : {mean_damage:.4f}")
    print(f"[INFO] mean damage_ssim           : {mean_ssim:.4f}")
    print(f"[INFO] mean bits/frame            : {mean_bits:.2f}")
    print(f"[INFO] mean Mbits/frame           : {mean_mbits:.6f}")
    print(f"[INFO] damage_psnr_per_mbit       : {psnr_per_mbit:.6f}")


if __name__ == "__main__":
    main()
