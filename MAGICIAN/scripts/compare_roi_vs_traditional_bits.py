#!/usr/bin/env python3
"""Compare ROI semantic transmission bits against traditional image transmission."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def image_sort_key(path: Path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def list_images(image_dir: Path) -> List[Path]:
    images = sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix in IMAGE_EXTENSIONS],
        key=image_sort_key,
    )
    if not images:
        raise FileNotFoundError(f"No png/jpg/jpeg images found in {image_dir}")
    return images


def jpeg_bits(image: Image.Image, quality: int) -> int:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality), optimize=True)
    return len(buffer.getvalue()) * 8


def read_roi_bit_report(path: Path) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"ROI bit report not found: {path}")
    rows = []
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        raise ValueError(f"ROI bit report is empty: {path}")

    def mean_field(field: str) -> float:
        values = []
        for row in rows:
            if field in row and row[field] not in ("", None):
                values.append(float(row[field]))
        return float(np.mean(values)) if values else float("nan")

    return {
        "num_frames": len(rows),
        "mean_total_bits": mean_field("total_bits"),
        "mean_bg_bits": mean_field("bg_bits"),
        "mean_roi_bits": mean_field("roi_bits"),
        "mean_metadata_bits": mean_field("metadata_bits"),
        "mean_mask_ratio": mean_field("mask_ratio"),
    }


def add_method(
    rows: List[dict],
    method: str,
    mean_bits: float,
    raw_bits: float,
    jpeg_q95_bits: float,
    num_frames: int,
    mean_bg_bits: float = float("nan"),
    mean_roi_bits: float = float("nan"),
    mean_metadata_bits: float = float("nan"),
) -> None:
    rows.append({
        "method": method,
        "mean_bits_per_frame": mean_bits,
        "mean_mbits_per_frame": mean_bits / 1e6,
        "compression_ratio_vs_raw": raw_bits / mean_bits if mean_bits > 0 else float("inf"),
        "bit_saving_vs_raw_percent": (1.0 - mean_bits / raw_bits) * 100.0 if raw_bits > 0 else float("nan"),
        "bit_saving_vs_jpeg_q95_percent": (
            (1.0 - mean_bits / jpeg_q95_bits) * 100.0
            if jpeg_q95_bits and jpeg_q95_bits > 0 else float("nan")
        ),
        "num_frames": num_frames,
        "mean_bg_bits": mean_bg_bits,
        "mean_roi_bits": mean_roi_bits,
        "mean_metadata_bits": mean_metadata_bits,
    })


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "mean_bits_per_frame",
        "mean_mbits_per_frame",
        "compression_ratio_vs_raw",
        "bit_saving_vs_raw_percent",
        "bit_saving_vs_jpeg_q95_percent",
        "num_frames",
        "mean_bg_bits",
        "mean_roi_bits",
        "mean_metadata_bits",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[dict]) -> None:
    print("")
    print(f"{'Method':<28} {'Mbits/frame':>14} {'Saving vs raw':>16} {'Ratio vs raw':>14}")
    for row in rows:
        print(
            f"{row['method']:<28} "
            f"{row['mean_mbits_per_frame']:>14.4f} "
            f"{row['bit_saving_vs_raw_percent']:>15.2f}% "
            f"{row['compression_ratio_vs_raw']:>13.2f}x"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare ROI semantic transmission bits with traditional strategies."
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--mask-dir", required=True, help="Accepted for bookkeeping; not used for bit counts.")
    parser.add_argument("--roi-bit-report", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--jpeg-qualities", default="95,75,50,30")
    parser.add_argument("--low-scales", default="0.25,0.125")
    parser.add_argument("--low-quality", type=int, default=30)
    args = parser.parse_args()

    image_dir = Path(args.image_dir).expanduser()
    output_csv = Path(args.output_csv).expanduser()
    images = list_images(image_dir)
    jpeg_qualities = parse_int_list(args.jpeg_qualities)
    low_scales = parse_float_list(args.low_scales)

    first = Image.open(images[0]).convert("RGB")
    width, height = first.size
    raw_bits = float(width * height * 3 * 8)
    num_frames = len(images)

    jpeg_means = {}
    lowres_means = {}
    for quality in jpeg_qualities:
        bits = []
        for path in images:
            image = Image.open(path).convert("RGB")
            bits.append(jpeg_bits(image, quality))
        jpeg_means[quality] = float(np.mean(bits))

    for scale in low_scales:
        bits = []
        for path in images:
            image = Image.open(path).convert("RGB")
            low_size = (
                max(1, int(round(image.size[0] * scale))),
                max(1, int(round(image.size[1] * scale))),
            )
            low = image.resize(low_size, RESAMPLE_BICUBIC)
            bits.append(jpeg_bits(low, args.low_quality))
        lowres_means[scale] = float(np.mean(bits))

    if 95 in jpeg_means:
        jpeg_q95_bits = jpeg_means[95]
    else:
        q_ref = jpeg_qualities[0]
        jpeg_q95_bits = jpeg_means[q_ref]
        print(f"[WARN] q95 not requested; using jpeg_q{q_ref} as JPEG reference.")

    rows = []
    add_method(rows, "raw_rgb_fullres", raw_bits, raw_bits, jpeg_q95_bits, num_frames)
    for quality in jpeg_qualities:
        add_method(rows, f"jpeg_q{quality}", jpeg_means[quality], raw_bits, jpeg_q95_bits, num_frames)
    for scale in low_scales:
        add_method(
            rows,
            f"full_lowres_s{scale:g}_q{args.low_quality}",
            lowres_means[scale],
            raw_bits,
            jpeg_q95_bits,
            num_frames,
        )

    roi = read_roi_bit_report(Path(args.roi_bit_report).expanduser())
    add_method(
        rows,
        "roi_semantic",
        roi["mean_total_bits"],
        raw_bits,
        jpeg_q95_bits,
        int(roi["num_frames"]),
        mean_bg_bits=roi["mean_bg_bits"],
        mean_roi_bits=roi["mean_roi_bits"],
        mean_metadata_bits=roi["mean_metadata_bits"],
    )

    write_csv(output_csv, rows)
    print_table(rows)

    roi_row = next(row for row in rows if row["method"] == "roi_semantic")
    low_rows = [row for row in rows if row["method"].startswith("full_lowres_")]
    best_low = min(low_rows, key=lambda row: row["mean_bits_per_frame"]) if low_rows else None

    print("")
    print(f"[INFO] image dir                 : {image_dir}")
    print(f"[INFO] mask dir                  : {Path(args.mask_dir).expanduser()}")
    print(f"[INFO] roi bit report            : {Path(args.roi_bit_report).expanduser()}")
    print(f"[INFO] output csv                : {output_csv}")
    print(f"[INFO] raw RGB bits/frame        : {raw_bits:.2f}")
    print(f"[INFO] ROI saving vs raw         : {roi_row['bit_saving_vs_raw_percent']:.2f}%")
    print(f"[INFO] ROI saving vs JPEG q95    : {roi_row['bit_saving_vs_jpeg_q95_percent']:.2f}%")
    if best_low is not None:
        relation = "lower" if roi_row["mean_bits_per_frame"] < best_low["mean_bits_per_frame"] else "higher"
        print(
            f"[INFO] ROI semantic is {relation} than best listed full low-res "
            f"({best_low['method']}) in bits/frame."
        )
    if roi_row["bit_saving_vs_jpeg_q95_percent"] < 10.0:
        print("[WARN] ROI semantic does not save much compared with JPEG q95.")
        print("[WARN] Possible causes: large mask ratio, large ROI bbox, high roi_quality,")
        print("[WARN] low_scale/bg_quality not low enough, or bbox-based ROI instead of tiles/components.")


if __name__ == "__main__":
    main()
