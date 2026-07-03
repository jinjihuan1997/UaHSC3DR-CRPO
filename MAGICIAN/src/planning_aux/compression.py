"""Planning-level depth/mask compression utilities.

Three compression modes simulate auxiliary-channel bandwidth limits at planning time:

  block_dropout       -- drop aligned block_size x block_size tiles of valid pixels
                         (bit-identical to MAGICIAN's existing 'block' degrade_depth)
  pixel_dropout       -- drop individual valid pixels uniformly at random
  downsample_upsample -- area-mode bilinear downsample to scale*H x scale*W, then
                         nearest-upsample back; mask uses nearest throughout

All functions accept PyTorch tensors of shape [B, H, W] or [B, H, W, 1].
Return value: (degraded_depth, degraded_mask) with the same shape as input.

Nominal data ratio / compression ratio conventions
---------------------------------------------------
block_dropout / pixel_dropout:
  r                    = retention ratio (fraction of valid pixels kept)
  nominal_data_ratio   = r
  nominal_compression_ratio = 1 / r

downsample_upsample:
  scale                = linear spatial scale factor (0 < scale <= 1)
  nominal_data_ratio   = scale ** 2  (e.g. scale=0.5 -> 25% pixels)
  nominal_compression_ratio = 1 / scale ** 2

MAGICIAN internal convention: r_aux is passed to degrade_depth() as the 'r' parameter.
For downsample_upsample, r_aux IS the scale factor (not scale**2).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def pixel_dropout(
    depth: torch.Tensor,
    mask: torch.Tensor,
    r: float,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly keep r fraction of currently-valid depth pixels."""
    r = float(r)
    if r >= 1.0:
        return depth, mask.bool()

    valid = mask.bool() & torch.isfinite(depth) & (depth > 0)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return depth, mask.bool()

    target_keep = int(round(r * n_valid))
    if target_keep <= 0:
        return depth.clone(), torch.zeros_like(mask, dtype=torch.bool)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    valid_flat = valid.reshape(-1)
    valid_indices = valid_flat.nonzero(as_tuple=False).squeeze(1)
    perm = torch.randperm(len(valid_indices), generator=generator)
    keep_indices = valid_indices[perm[:target_keep]]

    keep_mask_flat = torch.zeros_like(valid_flat, dtype=torch.bool)
    keep_mask_flat[keep_indices] = True
    keep_mask = keep_mask_flat.view(valid.shape)

    degraded_depth = depth.clone()
    degraded_depth[valid & ~keep_mask] = 0.0
    return degraded_depth, mask.bool() & keep_mask


def block_dropout(
    depth: torch.Tensor,
    mask: torch.Tensor,
    r: float,
    seed: int = 0,
    block_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep r fraction of valid pixels by accepting or rejecting block_size x block_size tiles.

    Reproduces MAGICIAN's _degrade_depth_block so that planning-level compression
    and on-device degradation stay bit-identical when the same (r, seed, block_size) triple
    is used.
    """
    r = float(r)
    if r >= 1.0:
        return depth, mask.bool()
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    valid = mask.bool() & torch.isfinite(depth) & (depth > 0)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        return depth, mask.bool()

    target_keep = int(round(r * n_valid))
    if target_keep <= 0:
        return depth.clone(), torch.zeros_like(mask, dtype=torch.bool)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    keep_mask = torch.zeros_like(valid, dtype=torch.bool)
    kept_total = 0
    batch_size = valid.shape[0]
    height = valid.shape[1]
    width = valid.shape[2]

    for b in range(batch_size):
        valid_hw = valid[b, ..., 0] if valid.ndim == 4 else valid[b]
        blocks: list[tuple[int, int, int, int, int]] = []
        for y0 in range(0, height, block_size):
            y1 = min(y0 + block_size, height)
            for x0 in range(0, width, block_size):
                x1 = min(x0 + block_size, width)
                count = int(valid_hw[y0:y1, x0:x1].sum().item())
                if count > 0:
                    blocks.append((y0, y1, x0, x1, count))
        if not blocks:
            continue
        order = torch.randperm(len(blocks), generator=generator, device="cpu").tolist()
        for bi in order:
            if kept_total >= target_keep:
                break
            y0, y1, x0, x1, count = blocks[bi]
            gap = target_keep - kept_total
            overshoot = kept_total + count - target_keep
            if count <= gap or overshoot < gap:
                if valid.ndim == 4:
                    keep_mask[b, y0:y1, x0:x1, :] = valid[b, y0:y1, x0:x1, :]
                else:
                    keep_mask[b, y0:y1, x0:x1] = valid[b, y0:y1, x0:x1]
                kept_total += count

    degraded_depth = depth.clone()
    degraded_depth[valid & ~keep_mask] = 0.0
    return degraded_depth, mask.bool() & keep_mask


def downsample_upsample(
    depth: torch.Tensor,
    mask: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Area-mode downsample to scale*H x scale*W, then nearest upsample back to original size.

    scale is the linear spatial scale factor (0 < scale <= 1.0):
      scale=1.0   -> identity (no change)
      scale=0.5   -> 50% linear, 25% nominal pixels, 4x compression ratio
      scale=0.25  -> 25% linear, 6.25% nominal pixels, 16x compression ratio
      scale=0.125 -> 12.5% linear, 1.5625% nominal pixels, 64x compression ratio

    Depth uses area-mode downsampling (preserves local depth statistics).
    Mask uses nearest throughout (avoids creating spurious valid regions).
    Output shape is identical to input -- only information content is reduced.
    """
    scale = float(scale)
    if scale >= 1.0:
        return depth, mask.bool()
    if scale <= 0.0:
        return depth.clone(), torch.zeros_like(mask, dtype=torch.bool)

    ndim = depth.ndim
    if ndim == 4:
        depth_4d = depth.permute(0, 3, 1, 2).float()
        mask_4d = mask.float().permute(0, 3, 1, 2) if mask.ndim == 4 else mask.float().unsqueeze(1)
    else:
        depth_4d = depth.float().unsqueeze(1)
        mask_4d = mask.float().unsqueeze(1)

    _B, _C, H, W = depth_4d.shape
    H_low = max(1, int(H * scale))
    W_low = max(1, int(W * scale))

    depth_low = F.interpolate(depth_4d, size=(H_low, W_low), mode="area")
    mask_low = F.interpolate(mask_4d, size=(H_low, W_low), mode="nearest")
    depth_up = F.interpolate(depth_low, size=(H, W), mode="nearest")
    mask_up = F.interpolate(mask_low, size=(H, W), mode="nearest")

    # Prevent invalid-to-valid promotion: the original mask is the authority.
    # A pixel can only be valid in the output if it was already valid in the input.
    mask_up = mask_up * mask_4d

    if ndim == 4:
        return depth_up.permute(0, 2, 3, 1), mask_up.permute(0, 2, 3, 1).bool()
    return depth_up.squeeze(1), mask_up.squeeze(1).bool()


def compress_depth_mask(
    depth: torch.Tensor,
    mask: torch.Tensor,
    mode: str,
    r: float = 1.0,
    seed: int = 0,
    block_size: int = 16,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified entry point -- dispatches to the chosen compression mode.

    For downsample_upsample, pass scale= explicitly (linear spatial scale factor).
    If scale is None, r is used as the scale for backwards compatibility.
    """
    if mode == "block_dropout":
        return block_dropout(depth, mask, r=r, seed=seed, block_size=block_size)
    elif mode == "pixel_dropout":
        return pixel_dropout(depth, mask, r=r, seed=seed)
    elif mode == "downsample_upsample":
        s = scale if scale is not None else r
        return downsample_upsample(depth, mask, scale=s)
    else:
        raise ValueError(
            f"Unknown compression mode: {mode!r}. "
            "Choose from: block_dropout, pixel_dropout, downsample_upsample"
        )
