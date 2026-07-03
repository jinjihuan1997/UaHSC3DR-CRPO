"""
Render random single-sample reconstructions for chosen/random SNR and digital subcarrier allocations.
"""
import argparse
import os
import random
import sys
import time

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import HDAROIDataset
from src.models.hda_roi import HDAROISystem
from src.channel.channels import digital_link_budget_from_snr, analog_noise_std
from src.utils.metrics import region_psnr, ms_ssim_db


def tensor_to_pil(x):
    return TF.to_pil_image(x.detach().cpu().clamp(0, 1))


def draw_label(img, label):
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    pad = 8
    bbox = draw.multiline_textbbox((0, 0), label, font=font, spacing=4)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, w + 2 * pad, h + 2 * pad], fill=(0, 0, 0))
    draw.multiline_text((pad, pad), label, fill=(255, 255, 255), font=font, spacing=4)
    return out


def make_row(clean, recon, max_height=480):
    imgs = [clean.convert("RGB"), recon.convert("RGB")]
    scaled = []
    for im in imgs:
        if im.height > max_height:
            w = int(round(im.width * max_height / im.height))
            im = im.resize((w, max_height), Image.BICUBIC)
        scaled.append(im)
    canvas = Image.new("RGB", (sum(im.width for im in scaled), max(im.height for im in scaled)), (255, 255, 255))
    x = 0
    for im in scaled:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas


def build_dataset(cfg, split, scene_id_filter=None):
    d = cfg["data"]
    manifest = {"train": d["train_manifest"], "val": d["val_manifest"], "test": d["test_manifest"]}[split]
    return HDAROIDataset(
        root=d["root"],
        manifest_rel=manifest,
        roi_source=d["roi_source"],
        image_size=d["image_size"],
        soft_mask_blur=d.get("soft_mask_blur", 9),
        mode="train" if split == "train" else "val",
        use_original_resolution=d.get("use_original_resolution", False),
        aux_payload_source=d.get("aux_payload_source", "zlib_crop"),
        aux_payload_size=d.get("aux_payload_size"),
        min_rgb_std=d.get("min_rgb_std", 1e-4),
        scene_id_filter=(scene_id_filter if scene_id_filter is not None
                         else d.get("train_scene_id") if split == "train" else None),
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--scene-id", default=None,
                    help="Only use samples from this scene_id. Omit to use all scenes in the split.")
    ap.add_argument("--num-samples", type=int, default=2)
    ap.add_argument("--sample-indices", type=int, nargs="+", default=None)
    ap.add_argument("--snrs", type=float, nargs="+", default=None)
    ap.add_argument("--digital-subcarriers", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs/random_single")
    ap.add_argument("--max-height", type=int, default=480)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = cfg["train"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    rng = random.Random(args.seed)
    ds = build_dataset(cfg, args.split, scene_id_filter=args.scene_id)
    if args.sample_indices:
        indices = args.sample_indices[:args.num_samples]
    else:
        indices = rng.sample(range(len(ds)), k=min(args.num_samples, len(ds)))

    snr_choices = args.snrs if args.snrs is not None else cfg.get("eval", {}).get("snr_points", [10.0])
    k_choices = args.digital_subcarriers or cfg["channel"].get("digital_subcarrier_grid", [12])
    total_subcarriers = int(cfg["channel"].get("data_subcarriers", 24))

    model = HDAROISystem(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)

    # One warm-up pass keeps CUDA kernel setup out of the reported per-sample latency.
    first = ds[indices[0]]
    warm_rgb = first["rgb"].unsqueeze(0).to(device)
    warm_depth = first["depth"].unsqueeze(0).to(device)
    warm_mask = first["valid_mask"].unsqueeze(0).to(device)
    warm_aux = first["aux_payload_bits"].view(1).to(device)
    warm_cfg = model.cfg["channel"]
    warm_cfg["resource_allocation"] = "fixed_subcarriers"
    warm_cfg["digital_resource_ratio"] = float(k_choices[0]) / float(total_subcarriers)
    _ = model(warm_rgb, warm_depth, warm_mask, float(snr_choices[0]), training=False, aux_payload_bits=warm_aux)
    if device == "cuda":
        torch.cuda.synchronize()

    print("idx,snr_db,k_d,k_rgb,rgb_symbols,digital_re,aux_payload_bits,aux_budget_bits,aux_ok,aux_ratio,rgb_psnr,valid_psnr,invalid_psnr,ms_ssim_db,noise_std,load_latency_ms,forward_latency_ms,total_latency_ms,out_path")
    forward_latencies = []
    load_latencies = []
    total_latencies = []
    for n, idx in enumerate(indices):
        load_t0 = time.perf_counter()
        sample = ds[idx]
        load_latency_ms = (time.perf_counter() - load_t0) * 1000.0
        load_latencies.append(load_latency_ms)
        snr = float(rng.choice(snr_choices))
        k_d = int(rng.choice(k_choices))
        cfg["channel"]["resource_allocation"] = "fixed_subcarriers"
        cfg["channel"]["digital_resource_ratio"] = k_d / float(total_subcarriers)

        rgb = sample["rgb"].unsqueeze(0).to(device)
        depth = sample["depth"].unsqueeze(0).to(device)
        valid_mask = sample["valid_mask"].unsqueeze(0).to(device)
        aux_bits = sample["aux_payload_bits"].view(1).to(device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model(rgb, depth, valid_mask, snr, training=False, aux_payload_bits=aux_bits)
        if device == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        forward_latencies.append(latency_ms)
        total_latency_ms = load_latency_ms + latency_ms
        total_latencies.append(total_latency_ms)

        rgb_hat = out["rgb_hat"]
        full, valid, invalid = region_psnr(rgb, rgb_hat, valid_mask)
        msssim = ms_ssim_db(rgb, rgb_hat)
        aux = out["aux_info"]
        payload = float(aux["aux_payload_bits"].float().view(-1)[0].item())
        budget_bits = float(aux["aux_budget_bits"].float().view(-1)[0].item())
        aux_ok = float(aux["aux_success"].float().view(-1)[0].item())
        aux_ratio = 1.0 if payload <= 0 else min(1.0, budget_bits / payload)
        budget = digital_link_budget_from_snr(snr, cfg, device=device)
        rgb_symbols = int(float(budget["analog_symbols"].float().view(-1)[0].item()))
        digital_re = int(float(budget["digital_resource_elements"].float().view(-1)[0].item()))
        k_rgb = int(budget["analog_data_subcarriers"])
        noise = float(analog_noise_std(snr).item())

        clean_img = draw_label(tensor_to_pil(sample["rgb"]), f"clean\nsample {idx}")
        recon_label = (
            f"recon\nSNR {snr:g} dB  k_d {k_d}  k_rgb {k_rgb}\n"
            f"PSNR {full:.2f} dB  AuxOK {aux_ok:.0f}  AuxRatio {aux_ratio:.3f}\n"
            f"latency {latency_ms:.2f} ms"
        )
        recon_img = draw_label(tensor_to_pil(rgb_hat[0]), recon_label)
        row = make_row(clean_img, recon_img, args.max_height)
        out_path = os.path.join(args.out_dir, f"sample_{idx:05d}_snr{snr:g}_kd{k_d}.png")
        row.save(out_path)

        print(
            f"{idx},{snr:g},{k_d},{k_rgb},{rgb_symbols},{digital_re},"
            f"{payload:.0f},{budget_bits:.0f},{aux_ok:.0f},{aux_ratio:.4f},"
            f"{full:.4f},{valid:.4f},{invalid:.4f},{msssim:.4f},{noise:.6f},"
            f"{load_latency_ms:.3f},{latency_ms:.3f},{total_latency_ms:.3f},{out_path}"
        )

    if forward_latencies:
        print(f"average_load_latency_ms,{sum(load_latencies) / len(load_latencies):.3f}")
        print(f"average_forward_latency_ms,{sum(forward_latencies) / len(forward_latencies):.3f}")
        print(f"average_total_latency_ms,{sum(total_latencies) / len(total_latencies):.3f}")


if __name__ == "__main__":
    main()
