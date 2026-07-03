"""
随机抽取测试样本，生成原图 / ROI / 多 SNR 重建对比图。

用法：
    python scripts/visualize_comparisons.py \
        --config configs/default.yaml \
        --ckpt checkpoints/stage3_final.pt \
        --num-samples 8 \
        --snrs 0 10 20
"""
import os
import sys
import argparse
import random

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import HDAROIDataset
from src.models.hda_roi import HDAROISystem
from src.utils.metrics import region_psnr
from src.channel.channels import analog_noise_std, digital_link_budget_from_snr


def tensor_to_pil(x):
    x = x.detach().cpu().clamp(0, 1)
    if x.dim() == 3 and x.shape[0] > 3:
        x = x[:3]
    return TF.to_pil_image(x)


def make_roi_overlay(image_t, mask_t):
    img = image_t.detach().cpu().clamp(0, 1)
    mask = (mask_t.detach().cpu() > 0.5).float()
    red = torch.zeros_like(img)
    red[0] = 1.0
    alpha = 0.35 * mask
    overlay = img * (1.0 - alpha) + red * alpha
    return tensor_to_pil(overlay)


def draw_label(img, label):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    pad = 6
    bbox = draw.textbbox((0, 0), label, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, w + 2 * pad, h + 2 * pad], fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return out


def concat_row(images):
    widths = [im.width for im in images]
    heights = [im.height for im in images]
    canvas = Image.new("RGB", (sum(widths), max(heights)), (255, 255, 255))
    x = 0
    for im in images:
        canvas.paste(im.convert("RGB"), (x, 0))
        x += im.width
    return canvas


@torch.no_grad()
def render_sample(model, image, mask, snrs, cfg, device):
    image_b = image.unsqueeze(0).to(device)
    mask_b = mask.unsqueeze(0).to(device)
    tiles = [
        draw_label(tensor_to_pil(image), "clean"),
        draw_label(make_roi_overlay(image, mask), "roi mask"),
    ]
    for snr in snrs:
        out = model(image_b, mask_b, float(snr), training=False)
        recon = out["I_hat"][0]
        full, roi, bg = region_psnr(image_b, out["I_hat"], mask_b)
        budget = digital_link_budget_from_snr(float(snr), cfg, device=device)
        mcs = budget["mcs_index"]
        phy_rate = budget["phy_rate_mbps"]
        bits = budget["digital_budget_bits"].item()
        analog_symbols = int(budget["analog_symbols"].item())
        token_bits = cfg["model"]["feat_dim"] * cfg["channel"]["digital_quant_bits"]
        tokens = int(bits // token_bits)
        noise = analog_noise_std(float(snr)).item()
        mcs_label = "OUT" if int(mcs.item()) < 0 else str(int(mcs.item()))
        label = (f"{snr:g}dB MCS{mcs_label} {phy_rate.item():.2f}Mb/s Tok{tokens} A{analog_symbols}\n"
                 f"N {noise:.3f} "
                 f"PSNR full {full:.1f} roi {roi:.1f} bg {bg:.1f}")
        tiles.append(draw_label(tensor_to_pil(recon), label))
    return concat_row(tiles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--snrs", type=float, nargs="+", default=[0, 10, 20])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs/comparisons")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    d = cfg["data"]
    ds = HDAROIDataset(
        root=d["root"],
        manifest_rel=d["test_manifest"],
        roi_source=d["roi_source"],
        image_size=d["image_size"],
        soft_mask_blur=d.get("soft_mask_blur", 9),
        mode="val",
        use_original_resolution=d.get("use_original_resolution", False),
        aux_payload_source=d.get("aux_payload_source", "zlib_crop"),
        aux_payload_size=d.get("aux_payload_size"),
        min_rgb_std=d.get("min_rgb_std", 1e-4),
    )

    model = HDAROISystem(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(ds)), k=min(args.num_samples, len(ds)))
    for out_idx, ds_idx in enumerate(indices):
        image, mask = ds[ds_idx]
        row = render_sample(model, image, mask, args.snrs, cfg, device)
        path = os.path.join(args.out_dir, f"comparison_{out_idx:03d}_sample_{ds_idx:05d}.png")
        row.save(path)
        print(f"[saved] {path}")


if __name__ == "__main__":
    main()
