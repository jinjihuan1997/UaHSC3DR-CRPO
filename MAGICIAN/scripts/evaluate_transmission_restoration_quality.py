#!/usr/bin/env python3
"""Evaluate restoration quality for ROI semantic and traditional transmission methods."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def image_sort_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def list_images(image_dir: Path) -> List[Path]:
    images = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix in IMAGE_EXTENSIONS],
        key=image_sort_key,
    )
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return images


def parse_method(value: str) -> dict:
    parts = value.split(",", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--method must be name,pred_dir,bit_report"
        )
    return {
        "name": parts[0],
        "pred_dir": Path(parts[1]).expanduser(),
        "bit_report": Path(parts[2]).expanduser(),
    }


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)


def find_image(directory: Path, ref_name: str) -> Path | None:
    direct = directory / ref_name
    if direct.exists():
        return direct
    stem = Path(ref_name).stem
    for suffix in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_mask(mask_dir: Path, ref_name: str, size: tuple[int, int]) -> np.ndarray:
    mask_path = mask_dir / Path(ref_name).with_suffix(".png").name
    if not mask_path.exists():
        return np.zeros((size[1], size[0]), dtype=bool)
    mask = Image.open(mask_path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, RESAMPLE_NEAREST)
    return np.asarray(mask, dtype=np.uint8) > 0


def psnr_from_arrays(ref: np.ndarray, pred: np.ndarray) -> float:
    if ref.size == 0:
        return float("nan")
    err = float(np.mean((ref - pred) ** 2))
    if err <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(err)))


def masked_psnr(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return psnr_from_arrays(ref[mask], pred[mask])


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def full_ssim(ref: np.ndarray, pred: np.ndarray) -> float:
    if skimage_ssim is None:
        return float("nan")
    if ref.shape[0] < 7 or ref.shape[1] < 7:
        return float("nan")
    return float(skimage_ssim(
        ref.astype(np.uint8),
        pred.astype(np.uint8),
        channel_axis=2,
        data_range=255,
    ))


def damage_ssim_bbox(ref: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    if skimage_ssim is None:
        return float("nan")
    bbox = bbox_from_mask(mask)
    if bbox is None:
        return float("nan")
    x0, y0, x1, y1 = bbox
    ref_crop = ref[y0:y1, x0:x1]
    pred_crop = pred[y0:y1, x0:x1]
    if ref_crop.shape[0] < 7 or ref_crop.shape[1] < 7:
        return float("nan")
    return float(skimage_ssim(
        ref_crop.astype(np.uint8),
        pred_crop.astype(np.uint8),
        channel_axis=2,
        data_range=255,
    ))


def read_bit_report(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"bit_report.csv not found: {path}")
    mapping = {}
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            name = row.get("image_name")
            if name:
                mapping[name] = float(row["total_bits"])
                mapping[Path(name).with_suffix(".png").name] = float(row["total_bits"])
    return mapping


def nanmean(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary_table(rows: List[dict]) -> None:
    print("")
    print(
        f"{'Method':<30} {'Mbits/frame':>12} {'Full PSNR':>11} "
        f"{'Damage PSNR':>13} {'Damage SSIM':>13} {'Damage PSNR/Mbit':>18}"
    )
    for row in rows:
        print(
            f"{row['method']:<30} "
            f"{row['mean_mbits_per_frame']:>12.4f} "
            f"{row['mean_full_psnr']:>11.3f} "
            f"{row['mean_damage_psnr']:>13.3f} "
            f"{row['mean_damage_ssim']:>13.4f} "
            f"{row['damage_psnr_per_mbit']:>18.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate restoration quality across transmission methods.")
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--mask-dir", required=True)
    parser.add_argument("--method", action="append", required=True, type=parse_method)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-csv", required=True)
    args = parser.parse_args()

    if skimage_ssim is None:
        print("[WARN] skimage is not available. SSIM columns will be NaN.")

    ref_dir = Path(args.ref_dir).expanduser()
    mask_dir = Path(args.mask_dir).expanduser()
    ref_images = list_images(ref_dir)

    detail_rows = []
    summary_rows = []
    for method in args.method:
        bit_report = read_bit_report(method["bit_report"])
        method_rows = []
        for ref_path in ref_images:
            pred_path = find_image(method["pred_dir"], ref_path.name)
            if pred_path is None:
                print(f"[WARN] Missing prediction for {method['name']} / {ref_path.name}")
                continue

            ref = load_rgb(ref_path)
            pred = load_rgb(pred_path)
            if ref.shape != pred.shape:
                raise ValueError(
                    f"Shape mismatch for {method['name']} {ref_path.name}: {ref.shape} vs {pred.shape}"
                )
            width, height = Image.open(ref_path).size
            mask = load_mask(mask_dir, ref_path.name, (width, height))
            background_mask = ~mask
            mask_pixels = int(mask.sum())
            mask_ratio = float(mask_pixels / mask.size)
            total_bits = bit_report.get(ref_path.name, bit_report.get(ref_path.with_suffix(".png").name, float("nan")))
            total_mbits = total_bits / 1e6 if not math.isnan(total_bits) else float("nan")

            row = {
                "method": method["name"],
                "image_name": ref_path.name,
                "full_psnr": psnr_from_arrays(ref, pred),
                "damage_psnr": masked_psnr(ref, pred, mask),
                "background_psnr": masked_psnr(ref, pred, background_mask),
                "full_ssim": full_ssim(ref, pred),
                "damage_ssim": damage_ssim_bbox(ref, pred, mask),
                "mask_pixels": mask_pixels,
                "mask_ratio": mask_ratio,
                "total_bits": total_bits,
                "total_mbits": total_mbits,
            }
            method_rows.append(row)
            detail_rows.append(row)

        mean_mbits = nanmean([row["total_mbits"] for row in method_rows])
        mean_damage_psnr = nanmean([row["damage_psnr"] for row in method_rows])
        mean_damage_ssim = nanmean([row["damage_ssim"] for row in method_rows])
        summary_rows.append({
            "method": method["name"],
            "mean_full_psnr": nanmean([row["full_psnr"] for row in method_rows]),
            "mean_damage_psnr": mean_damage_psnr,
            "mean_background_psnr": nanmean([row["background_psnr"] for row in method_rows]),
            "mean_full_ssim": nanmean([row["full_ssim"] for row in method_rows]),
            "mean_damage_ssim": mean_damage_ssim,
            "mean_mbits_per_frame": mean_mbits,
            "damage_psnr_per_mbit": mean_damage_psnr / mean_mbits if mean_mbits and not math.isnan(mean_mbits) else float("nan"),
            "damage_ssim_per_mbit": mean_damage_ssim / mean_mbits if mean_mbits and not math.isnan(mean_mbits) else float("nan"),
            "num_frames": len(method_rows),
        })

    detail_fields = [
        "method",
        "image_name",
        "full_psnr",
        "damage_psnr",
        "background_psnr",
        "full_ssim",
        "damage_ssim",
        "mask_pixels",
        "mask_ratio",
        "total_bits",
        "total_mbits",
    ]
    summary_fields = [
        "method",
        "mean_full_psnr",
        "mean_damage_psnr",
        "mean_background_psnr",
        "mean_full_ssim",
        "mean_damage_ssim",
        "mean_mbits_per_frame",
        "damage_psnr_per_mbit",
        "damage_ssim_per_mbit",
        "num_frames",
    ]
    write_csv(Path(args.output_csv).expanduser(), detail_rows, detail_fields)
    write_csv(Path(args.summary_csv).expanduser(), summary_rows, summary_fields)
    print_summary_table(summary_rows)

    by_name = {row["method"]: row for row in summary_rows}
    roi = by_name.get("roi_semantic")
    if roi is not None:
        lowres = [row for row in summary_rows if row["method"].startswith("full_lowres")]
        if lowres:
            best_low_damage = max(lowres, key=lambda row: row["mean_damage_psnr"])
            relation = "higher" if roi["mean_damage_psnr"] > best_low_damage["mean_damage_psnr"] else "not higher"
            print(f"[INFO] ROI semantic damage PSNR is {relation} than best full_lowres ({best_low_damage['method']}).")
        jpeg_q30 = by_name.get("jpeg_q30")
        if jpeg_q30 is not None:
            close_bits = abs(roi["mean_mbits_per_frame"] - jpeg_q30["mean_mbits_per_frame"])
            print(
                f"[INFO] ROI vs jpeg_q30: damage_psnr {roi['mean_damage_psnr']:.3f} vs "
                f"{jpeg_q30['mean_damage_psnr']:.3f}, Mbits/frame delta {close_bits:.4f}."
            )
        best_eff = max(summary_rows, key=lambda row: row["damage_psnr_per_mbit"] if not math.isnan(row["damage_psnr_per_mbit"]) else -1.0)
        print(f"[INFO] Best damage PSNR/Mbit method: {best_eff['method']}")
    print(f"[INFO] per-frame CSV : {Path(args.output_csv).expanduser()}")
    print(f"[INFO] summary CSV   : {Path(args.summary_csv).expanduser()}")


if __name__ == "__main__":
    main()
