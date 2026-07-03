#!/usr/bin/env python3
"""Make a contact sheet comparing restored transmission images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from PIL import Image, ImageDraw, ImageFont

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC

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
    parts = value.split(",", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--method must be name,path")
    return {"name": parts[0], "path": Path(parts[1]).expanduser()}


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


def load_tile(path: Path | None, size: tuple[int, int]) -> Image.Image:
    if path is None or not path.exists():
        return Image.new("RGB", size, (40, 40, 40))
    return Image.open(path).convert("RGB").resize(size, RESAMPLE_BICUBIC)


def draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font) -> None:
    x, y = xy
    draw.rectangle([x, y, x + 450, y + 18], fill=(0, 0, 0))
    draw.text((x + 4, y + 2), text, fill=(255, 255, 255), font=font)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a restoration comparison contact sheet.")
    parser.add_argument("--ref-dir", required=True)
    parser.add_argument("--overlay-dir", required=True)
    parser.add_argument("--method", action="append", required=True, type=parse_method)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-images", type=int, default=8)
    args = parser.parse_args()

    ref_dir = Path(args.ref_dir).expanduser()
    overlay_dir = Path(args.overlay_dir).expanduser()
    output = Path(args.output).expanduser()
    ref_images = list_images(ref_dir)[:args.num_images]

    tile_size = (456, 256)
    label_h = 24
    columns = [
        {"name": "Original", "path": ref_dir},
        {"name": "Damage Overlay", "path": overlay_dir},
    ] + args.method
    sheet_w = tile_size[0] * len(columns)
    sheet_h = (tile_size[1] + label_h) * len(ref_images)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row_idx, ref_path in enumerate(ref_images):
        y0 = row_idx * (tile_size[1] + label_h)
        for col_idx, column in enumerate(columns):
            x0 = col_idx * tile_size[0]
            if column["name"] == "Original":
                image_path = ref_path
            else:
                image_path = find_image(column["path"], ref_path.name)
            tile = load_tile(image_path, tile_size)
            sheet.paste(tile, (x0, y0 + label_h))
            label = f"{column['name']} | {ref_path.name}" if col_idx == 0 else column["name"]
            draw_label(draw, (x0, y0), label, font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(f"[INFO] wrote comparison sheet: {output}")
    print(f"[INFO] rows: {len(ref_images)}, columns: {len(columns)}")


if __name__ == "__main__":
    main()
