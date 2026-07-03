#!/usr/bin/env python3
"""Visualize depth/mask degradation for different r_aux values."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FRAME = (
    ROOT
    / "data/Macarons++/macarons++/fushimi"
    / "test_memory_raux_block_fushimi_r1p00_seed0/training/0/frames/0.pt"
)
DEFAULT_OUTPUT = ROOT / "results" / "raux_depth_mask_visuals"


def parse_r_values(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def r_tag(r: float) -> str:
    return f"{r:.2f}".replace(".", "p")


def infer_frame_idx(frame_path: Path) -> int:
    try:
        return int(frame_path.stem)
    except ValueError:
        return 0


def valid_depth_mask(depth: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return mask.bool() & torch.isfinite(depth) & (depth > 0)


def degrade_depth_block(valid: torch.Tensor, r: float, seed: int, block_size: int) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    keep_mask = torch.zeros_like(valid, dtype=torch.bool)
    n_valid_total = int(valid.sum().item())
    target_keep_total = int(round(float(r) * n_valid_total))
    if target_keep_total <= 0:
        return keep_mask

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    kept_total = 0
    batch_size, height, width = valid.shape[:3]

    for b in range(batch_size):
        valid_hw = valid[b, ..., 0] if valid.ndim == 4 else valid[b]
        blocks = []
        for y0 in range(0, height, block_size):
            y1 = min(y0 + block_size, height)
            for x0 in range(0, width, block_size):
                x1 = min(x0 + block_size, width)
                count = int(valid_hw[y0:y1, x0:x1].sum().item())
                if count > 0:
                    blocks.append((y0, y1, x0, x1, count))

        order = torch.randperm(len(blocks), generator=generator, device="cpu").tolist()
        for block_i in order:
            if kept_total >= target_keep_total:
                break
            y0, y1, x0, x1, count = blocks[block_i]
            current_gap = target_keep_total - kept_total
            overshoot = kept_total + count - target_keep_total
            if count <= current_gap or overshoot < current_gap:
                if valid.ndim == 4:
                    keep_mask[b, y0:y1, x0:x1, :] = valid[b, y0:y1, x0:x1, :]
                else:
                    keep_mask[b, y0:y1, x0:x1] = valid[b, y0:y1, x0:x1]
                kept_total += count

    return keep_mask


def degrade_depth(
    depth: torch.Tensor,
    mask: torch.Tensor,
    r: float,
    seed: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match macarons.testers.magician_planning.degrade_depth block mode."""
    r = float(r)
    if r < 0.0 or r > 1.0:
        raise ValueError(f"r must be in [0, 1], got {r}.")

    valid = valid_depth_mask(depth, mask)
    n_valid = int(valid.sum().item())
    if n_valid == 0 or r >= 1.0:
        return depth.clone(), mask.bool().clone()

    keep_mask = degrade_depth_block(valid, r=r, seed=seed, block_size=block_size)

    degraded_depth = depth.clone()
    degraded_mask = mask.bool() & keep_mask
    degraded_depth[mask.bool() & ~keep_mask] = 0.0
    return degraded_depth, degraded_mask


def tensor_image_to_uint8(rgb: torch.Tensor) -> np.ndarray:
    arr = rgb.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def depth_to_color(depth_hw: np.ndarray, valid_hw: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    denom = max(vmax - vmin, 1e-8)
    t = np.clip((depth_hw - vmin) / denom, 0.0, 1.0)

    c0 = np.array([36, 54, 120], dtype=np.float32)
    c1 = np.array([49, 204, 197], dtype=np.float32)
    c2 = np.array([250, 230, 85], dtype=np.float32)

    rgb = np.zeros((*depth_hw.shape, 3), dtype=np.float32)
    lower = t <= 0.5
    upper = ~lower
    rgb[lower] = c0 + (c1 - c0) * (t[lower][..., None] * 2.0)
    rgb[upper] = c1 + (c2 - c1) * ((t[upper][..., None] - 0.5) * 2.0)
    rgb[~valid_hw] = 0.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def mask_to_image(mask_hw: np.ndarray) -> np.ndarray:
    return (mask_hw.astype(np.uint8) * 255)


def dropped_overlay(rgb: np.ndarray, original_valid: np.ndarray, degraded_valid: np.ndarray) -> np.ndarray:
    dropped = original_valid & ~degraded_valid
    overlay = rgb.copy().astype(np.float32)
    overlay[~original_valid] *= 0.25
    red = np.array([255, 32, 32], dtype=np.float32)
    overlay[dropped] = overlay[dropped] * 0.25 + red * 0.75
    return np.clip(overlay, 0, 255).astype(np.uint8)


def add_label(image: Image.Image, text: str, height: int = 30) -> Image.Image:
    font = ImageFont.load_default()
    out = Image.new("RGB", (image.width, image.height + height), "white")
    out.paste(image.convert("RGB"), (0, height))
    draw = ImageDraw.Draw(out)
    draw.text((8, 9), text, fill=(0, 0, 0), font=font)
    return out


def build_sheet(rows: list[dict], output_path: Path) -> None:
    labeled_tiles = []
    for row in rows:
        depth = add_label(Image.open(row["depth_path"]), f"r={row['r']:.2f} depth")
        mask = add_label(Image.open(row["mask_path"]).convert("RGB"), f"mask kept={row['retention']:.3f}")
        overlay = add_label(Image.open(row["overlay_path"]), "red = dropped")
        labeled_tiles.append((depth, mask, overlay))

    margin = 8
    col_widths = [max(tile[col].width for tile in labeled_tiles) for col in range(3)]
    row_heights = [max(tile.height for tile in row) for row in labeled_tiles]
    width = sum(col_widths) + margin * (len(col_widths) + 1)
    height = sum(row_heights) + margin * (len(row_heights) + 1)
    sheet = Image.new("RGB", (width, height), "white")

    y = margin
    for row, row_height in zip(labeled_tiles, row_heights):
        x = margin
        for tile, col_width in zip(row, col_widths):
            sheet.paste(tile, (x, y))
            x += col_width + margin
        y += row_height + margin

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create visual depth/mask degradation images for r_aux values.")
    parser.add_argument("--frame", default=str(DEFAULT_FRAME), help="Path to a saved MAGICIAN frame .pt file.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT), help="Directory for PNG outputs.")
    parser.add_argument("--r-values", default="1.0,0.95,0.90,0.85,0.80,0.70")
    parser.add_argument("--degrade-block-size", type=int, default=16)
    parser.add_argument("--degrade-seed", type=int, default=0, help="Base seed used by the r_aux experiment.")
    parser.add_argument("--frame-idx", type=int, default=None, help="Override frame index for seed offset.")
    args = parser.parse_args()

    frame_path = Path(args.frame).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    r_values = parse_r_values(args.r_values)
    frame_idx = infer_frame_idx(frame_path) if args.frame_idx is None else int(args.frame_idx)
    effective_seed = int(args.degrade_seed) + frame_idx

    frame = torch.load(frame_path, map_location="cpu")
    rgb = tensor_image_to_uint8(frame["rgb"])
    depth = frame["zbuf"].detach().cpu()
    mask = frame["mask"].detach().cpu().bool()
    original_valid = valid_depth_mask(depth, mask)

    depth_hw = depth[0, ..., 0].numpy()
    original_valid_hw = original_valid[0, ..., 0].numpy()
    valid_depth_values = depth_hw[original_valid_hw]
    if valid_depth_values.size == 0:
        raise ValueError(f"No valid depth pixels in {frame_path}")
    vmin = float(np.percentile(valid_depth_values, 1.0))
    vmax = float(np.percentile(valid_depth_values, 99.0))

    frame_output_dir = output_dir / f"block_b{args.degrade_block_size}" / f"frame_{frame_idx:06d}"
    frame_output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(frame_output_dir / "rgb_reference.png")

    rows = []
    n_original = int(original_valid.sum().item())
    for r in r_values:
        degraded_depth, degraded_mask = degrade_depth(
            depth,
            mask,
            r=r,
            seed=effective_seed,
            block_size=args.degrade_block_size,
        )
        degraded_valid = valid_depth_mask(degraded_depth, degraded_mask)
        degraded_depth_hw = degraded_depth[0, ..., 0].numpy()
        degraded_valid_hw = degraded_valid[0, ..., 0].numpy()

        tag = f"r{r_tag(r)}"
        r_dir = frame_output_dir / tag
        r_dir.mkdir(parents=True, exist_ok=True)

        depth_img = depth_to_color(degraded_depth_hw, degraded_valid_hw, vmin=vmin, vmax=vmax)
        mask_img = mask_to_image(degraded_valid_hw)
        overlay_img = dropped_overlay(rgb, original_valid_hw, degraded_valid_hw)

        depth_path = r_dir / "depth.png"
        mask_path = r_dir / "mask.png"
        overlay_path = r_dir / "dropped_overlay.png"
        Image.fromarray(depth_img).save(depth_path)
        Image.fromarray(mask_img).save(mask_path)
        Image.fromarray(overlay_img).save(overlay_path)

        n_kept = int(degraded_valid.sum().item())
        retention = n_kept / max(1, n_original)
        rows.append(
            {
                "r": float(r),
                "requested_r": float(r),
                "frame_idx": frame_idx,
                "seed": effective_seed,
                "degrade_mode": "block",
                "degrade_block_size": int(args.degrade_block_size),
                "valid_before": n_original,
                "valid_after": n_kept,
                "retention": retention,
                "depth_path": str(depth_path),
                "mask_path": str(mask_path),
                "overlay_path": str(overlay_path),
            }
        )

    build_sheet(rows, frame_output_dir / "comparison_depth_mask_overlay.png")

    csv_path = frame_output_dir / "retention_report.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "requested_r",
                "frame_idx",
                "seed",
                "degrade_mode",
                "degrade_block_size",
                "valid_before",
                "valid_after",
                "retention",
                "depth_path",
                "mask_path",
                "overlay_path",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote visualizations to: {frame_output_dir}")
    print(f"Comparison sheet: {frame_output_dir / 'comparison_depth_mask_overlay.png'}")
    print(f"Retention report: {csv_path}")


if __name__ == "__main__":
    main()
