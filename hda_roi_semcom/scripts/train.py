"""
三阶段训练主程序。

阶段1：训练语义编解码器（自编码，ROI 加权 + 傅里叶损失）
阶段2：冻结语义编解码器，训练模数分配 + 模拟信道收发机
阶段3：全网络端到端联合微调（codec 用更小学习率）

用法：
    python scripts/train.py --config configs/default.yaml
"""
import os
import sys
import argparse
import random

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.data.dataset import build_dataloader
from src.models.hda_roi import HDAROISystem
from src.losses.roi_loss import ROIAwareLoss
from src.utils.metrics import RegionPSNRMeter


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def sample_snr(cfg, B, device):
    points = cfg["channel"].get("train_snr_points")
    if points:
        values = torch.tensor(points, device=device, dtype=torch.float32)
        idx = torch.randint(0, len(points), (B,), device=device)
        return values[idx]
    lo, hi = cfg["channel"]["train_snr_min"], cfg["channel"]["train_snr_max"]
    return torch.empty(B, device=device).uniform_(lo, hi)


def sample_resource_point(cfg):
    ch = cfg["channel"]
    if ch.get("resource_allocation") != "discrete_subcarriers":
        return None
    data_subcarriers = int(ch.get("data_subcarriers", 24))
    grid = ch.get("digital_subcarrier_grid", list(range(1, data_subcarriers)))
    k = int(random.choice(grid))
    ch["digital_resource_ratio"] = k / float(data_subcarriers)
    return k


def save_ckpt(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  [saved] {path}")


def zero_mask_like(x):
    return torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)


def train_stage1(model, loader, loss_fn, cfg, device):
    print("\n=== Stage 1: RGB semantic codec ===")
    opt = torch.optim.Adam(model.codec_params(), lr=cfg["train"]["lr_stage1"])
    for ep in range(cfg["train"]["epochs_stage1"]):
        model.train()
        for it, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            rgb_hat, _ = model.forward_codec(rgb)
            loss = loss_fn.semantic_loss(rgb, rgb_hat, zero_mask_like(rgb))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.codec_params(), cfg["train"]["grad_clip"])
            opt.step()
            if it % cfg["train"]["log_every"] == 0:
                print(f"  [S1] ep{ep} it{it} loss={loss.item():.4f}")
    save_ckpt(model, os.path.join(cfg["train"]["save_dir"], "stage1.pt"))


def train_stage2(model, loader, loss_fn, cfg, device):
    print("\n=== Stage 2: RGB JSCC transceiver ===")
    for p in model.codec_params():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.transceiver_params(), lr=cfg["train"]["lr_stage2"])
    for ep in range(cfg["train"]["epochs_stage2"]):
        model.train()
        for it, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            aux_bits = batch["aux_payload_bits"].to(device)
            resource_k = sample_resource_point(cfg)
            snr = sample_snr(cfg, rgb.shape[0], device)
            out = model(rgb, depth, valid_mask, snr, training=True, aux_payload_bits=aux_bits)
            l_cd = (out["z"].detach() - out["z_hat"]).abs().mean()
            l_sd = loss_fn.semantic_loss(rgb, out["rgb_hat"], zero_mask_like(rgb))
            loss = l_sd + l_cd
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.transceiver_params(), cfg["train"]["grad_clip"])
            opt.step()
            if it % cfg["train"]["log_every"] == 0:
                aux_ok = out["aux_info"]["aux_success"].float().mean().item()
                resource_msg = "" if resource_k is None else f" k_d={resource_k}"
                print(f"  [S2] ep{ep} it{it} loss={loss.item():.4f} "
                      f"(sd={l_sd.item():.3f} cd={l_cd.item():.3f} aux_ok={aux_ok:.2f}{resource_msg})")
    save_ckpt(model, os.path.join(cfg["train"]["save_dir"], "stage2.pt"))


def train_stage3(model, loader, loss_fn, cfg, device):
    print("\n=== Stage 3: RGB JSCC end-to-end fine-tuning ===")
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam([
        {"params": model.codec_params(), "lr": cfg["train"]["lr_stage3_codec"]},
        {"params": model.transceiver_params(), "lr": cfg["train"]["lr_stage3_other"]},
    ])
    for ep in range(cfg["train"]["epochs_stage3"]):
        model.train()
        for it, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            valid_mask = batch["valid_mask"].to(device)
            aux_bits = batch["aux_payload_bits"].to(device)
            resource_k = sample_resource_point(cfg)
            snr = sample_snr(cfg, rgb.shape[0], device)
            out = model(rgb, depth, valid_mask, snr, training=True, aux_payload_bits=aux_bits)
            l_sd = loss_fn.semantic_loss(rgb, out["rgb_hat"], zero_mask_like(rgb), use_lpips=True)
            loss = l_sd
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            opt.step()
            if it % cfg["train"]["log_every"] == 0:
                aux_ok = out["aux_info"]["aux_success"].float().mean().item()
                resource_msg = "" if resource_k is None else f" k_d={resource_k}"
                print(f"  [S3] ep{ep} it{it} loss={loss.item():.4f} aux_ok={aux_ok:.2f}{resource_msg}")
    save_ckpt(model, os.path.join(cfg["train"]["save_dir"], "stage3_final.pt"))


@torch.no_grad()
def quick_val(model, loader, cfg, device, snr=6.0):
    model.eval()
    if cfg["channel"].get("resource_allocation") == "discrete_subcarriers":
        grid = cfg["channel"].get("digital_subcarrier_grid", [12])
        data_subcarriers = int(cfg["channel"].get("data_subcarriers", 24))
        cfg["channel"]["digital_resource_ratio"] = int(grid[len(grid) // 2]) / float(data_subcarriers)
    meter = RegionPSNRMeter()
    n = 0
    for batch in loader:
        rgb = batch["rgb"].to(device)
        depth = batch["depth"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        aux_bits = batch["aux_payload_bits"].to(device)
        out = model(rgb, depth, valid_mask, snr, training=False, aux_payload_bits=aux_bits)
        meter.update(rgb, out["rgb_hat"], valid_mask)
        n += 1
        if n >= 20:
            break
    full, valid, invalid = meter.compute()
    print(f"  [val@{snr}dB] full={full:.2f} valid={valid:.2f} invalid={invalid:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train-scene-id", default=None,
                    help="Train on one scene_id. Omit this option to use all training scenes.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.train_scene_id is not None:
        cfg["data"]["train_scene_id"] = args.train_scene_id
    set_seed(cfg["train"]["seed"])

    device = cfg["train"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU")
        device = "cpu"

    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val")
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    model = HDAROISystem(cfg).to(device)
    loss_fn = ROIAwareLoss(cfg["loss"]["lambda_roi"],
                           cfg["loss"]["lambda_fourier"],
                           cfg["loss"]["lambda_rate"],
                           cfg["loss"].get("lambda_lpips", 0.0))

    train_stage1(model, train_loader, loss_fn, cfg, device)
    quick_val(model, val_loader, cfg, device)
    train_stage2(model, train_loader, loss_fn, cfg, device)
    quick_val(model, val_loader, cfg, device)
    train_stage3(model, train_loader, loss_fn, cfg, device)
    quick_val(model, val_loader, cfg, device)
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
