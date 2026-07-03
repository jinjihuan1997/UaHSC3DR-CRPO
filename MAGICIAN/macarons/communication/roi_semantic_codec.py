"""ROI-based task-aware digital semantic transmission codec."""

from __future__ import annotations

import io
from dataclasses import asdict
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageFilter

from .semantic_modes import SemanticMode, sort_modes_by_quality

try:
    RESAMPLE_BICUBIC = Image.Resampling.BICUBIC
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_BICUBIC = Image.BICUBIC
    RESAMPLE_NEAREST = Image.NEAREST


def _jpeg_roundtrip(image: Image.Image, quality: int) -> tuple[Image.Image, int]:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=int(quality), optimize=True)
    payload = buffer.getvalue()
    decoded = Image.open(io.BytesIO(payload)).convert("RGB")
    return decoded, len(payload) * 8


def _mask_to_bool(mask) -> np.ndarray:
    if isinstance(mask, Image.Image):
        mask_arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    else:
        mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    if mask_arr.dtype == np.bool_:
        return mask_arr
    return mask_arr > 0


def _mask_bbox(mask_bool: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _background_roundtrip(image: Image.Image, mode: SemanticMode) -> tuple[Image.Image, int]:
    width, height = image.size
    low_size = (
        max(1, int(round(width * mode.low_scale))),
        max(1, int(round(height * mode.low_scale))),
    )
    low = image.resize(low_size, RESAMPLE_BICUBIC)
    decoded_low, bg_bits = _jpeg_roundtrip(low, mode.bg_quality)
    return decoded_low.resize((width, height), RESAMPLE_BICUBIC), bg_bits


def encode_decode_roi_semantic(image: Image.Image, mask, mode: SemanticMode) -> dict:
    image = image.convert("RGB")
    mask_bool = _mask_to_bool(mask)
    if mask_bool.shape != (image.size[1], image.size[0]):
        mask_img = Image.fromarray(mask_bool.astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize(image.size, RESAMPLE_NEAREST)
        mask_bool = np.asarray(mask_img, dtype=np.uint8) > 0

    received, bg_bits = _background_roundtrip(image, mode)
    bbox = _mask_bbox(mask_bool)
    roi_bits = 0
    metadata_bits = 0
    if bbox is not None:
        roi_crop = image.crop(bbox)
        decoded_roi, roi_bits = _jpeg_roundtrip(roi_crop, mode.roi_quality)
        received.paste(decoded_roi, bbox)
        metadata_bits = int(mode.metadata_bits)

    total_bits = int(bg_bits + roi_bits + metadata_bits)
    return {
        "received_image": received,
        "bg_bits": int(bg_bits),
        "roi_bits": int(roi_bits),
        "metadata_bits": int(metadata_bits),
        "total_bits": total_bits,
        "roi_bbox": bbox,
        "mode": asdict(mode),
    }


def estimate_roi_semantic_bits(image: Image.Image, mask, mode: SemanticMode) -> dict:
    result = encode_decode_roi_semantic(image, mask, mode)
    result.pop("received_image")
    return result


def select_semantic_mode(image: Image.Image, mask, modes: Iterable[SemanticMode], capacity_bits: float) -> dict:
    quality_sorted = sort_modes_by_quality(modes)
    estimates = []
    for mode in reversed(quality_sorted):
        estimate = estimate_roi_semantic_bits(image, mask, mode)
        estimates.append(estimate)
        if estimate["total_bits"] <= capacity_bits:
            return {
                "mode": mode,
                "estimate": estimate,
                "capacity_satisfied": True,
            }

    lowest = quality_sorted[0]
    lowest_estimate = next(
        (estimate for estimate in estimates if estimate["mode"]["mode_id"] == lowest.mode_id),
        None,
    )
    if lowest_estimate is None:
        lowest_estimate = estimate_roi_semantic_bits(image, mask, lowest)
    return {
        "mode": lowest,
        "estimate": lowest_estimate,
        "capacity_satisfied": False,
    }
