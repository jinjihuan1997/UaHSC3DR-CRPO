"""
Evaluate RGB JSCC quality and auxiliary depth/valid-mask digital delivery.

Usage:
    python scripts/evaluate.py --config configs/default.yaml --ckpt checkpoints/stage3_final.pt
"""
import os
import sys
import argparse

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import build_dataloader
from src.models.hda_roi import HDAROISystem
from src.utils.metrics import RegionPSNRMeter, ms_ssim_db
from src.channel.channels import analog_noise_std, digital_link_budget_from_snr


def build_lpips_metric(device):
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError(
            "LPIPS is not installed. Run: pip install lpips"
        ) from exc
    metric = lpips.LPIPS(net="alex").to(device)
    metric.eval()
    for p in metric.parameters():
        p.requires_grad_(False)
    return metric


@torch.no_grad()
def evaluate(model, loader, snr, device, lpips_metric):
    model.eval()
    meter = RegionPSNRMeter()
    acc = {"msssim": 0.0, "lpips": 0.0, "n": 0, "aux_ok": 0.0,
           "payload_bits": 0.0, "budget_bits": 0.0}
    for batch in loader:
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        aux_bits = batch["aux_payload_bits"].to(device)
        out = model(rgb, depth, valid_mask, snr, training=False, aux_payload_bits=aux_bits)
        rgb_hat = out["rgb_hat"]
        meter.update(rgb, rgb_hat, valid_mask)
        acc["msssim"] += ms_ssim_db(rgb, rgb_hat)
        acc["lpips"] += lpips_metric(rgb * 2.0 - 1.0, rgb_hat * 2.0 - 1.0).mean().item()
        acc["aux_ok"] += out["aux_info"]["aux_success"].float().mean().item()
        acc["payload_bits"] += out["aux_info"]["aux_payload_bits"].float().mean().item()
        acc["budget_bits"] += out["aux_info"]["aux_budget_bits"].float().mean().item()
        acc["n"] += 1
    n = max(acc["n"], 1)
    full, valid, invalid = meter.compute()
    return (full, valid, invalid, acc["msssim"] / n, acc["lpips"] / n,
            acc["aux_ok"] / n, acc["payload_bits"] / n, acc["budget_bits"] / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    loader = build_dataloader(cfg, "test")
    model = HDAROISystem(cfg).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    lpips_metric = build_lpips_metric(device)

    print(f"{'SNR(dB)':>8} {'MCS':>4} {'PHY(Mb/s)':>9} {'D_aux(bit)':>11} "
          f"{'Payload':>10} {'AuxOK':>6} {'RGBSym':>6} {'NoiseStd':>8} "
          f"{'RGB-PSNR':>9} {'Valid-PSNR':>10} {'Invalid':>8} {'MS-SSIM':>9} {'LPIPS':>8}")
    print("-" * 128)
    for snr in cfg["eval"]["snr_points"]:
        budget = digital_link_budget_from_snr(float(snr), cfg, device=device)
        mcs = budget["mcs_index"]
        phy_rate = budget["phy_rate_mbps"]
        analog_symbols = int(budget["analog_symbols"].item())
        noise = analog_noise_std(float(snr)).item()
        f, valid, invalid, m, lp, ok, payload, aux_budget = evaluate(
            model, loader, float(snr), device, lpips_metric)
        mcs_label = "OUT" if int(mcs.item()) < 0 else str(int(mcs.item()))
        print(f"{snr:>8} {mcs_label:>4} {phy_rate.item():>9.3f} {aux_budget:>11.0f} "
              f"{payload:>10.0f} {ok:>6.2f} {analog_symbols:>6} {noise:>8.4f} "
              f"{f:>9.2f} {valid:>10.2f} {invalid:>8.2f} {m:>9.2f} {lp:>8.4f}")


if __name__ == "__main__":
    main()
