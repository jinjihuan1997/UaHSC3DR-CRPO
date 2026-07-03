"""
DEPRECATED: 本脚本依赖 model.alloc（HDAROISystem 的旧分配模块），该模块已从
src/models/hda_roi.py 中移除。当前项目主线为 RL 资源分配（src/rl/），请改用：
    python scripts/sanity_check_crpo.py
"""
import sys
print(
    "ERROR: scripts/sanity_check.py is deprecated and cannot run.\n"
    "Use: python scripts/sanity_check_crpo.py",
    file=sys.stderr,
)
sys.exit(1)

# ── Everything below is dead code, kept for historical reference only ──────────
import os  # noqa: E402
import argparse  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config  # noqa: E402
from src.models.hda_roi import HDAROISystem  # noqa: E402
from src.losses.roi_loss import ROIAwareLoss  # noqa: E402
from src.utils.metrics import region_psnr  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["train"]["device"] = "cpu"          # 自检强制用 CPU
    cfg["data"]["image_size"] = 128         # 缩小尺寸加速自检
    cfg["model"]["analog_symbols"] = 64

    device = "cpu"
    B, H = 2, cfg["data"]["image_size"]
    I = torch.rand(B, 3, H, H)
    # 合成 ROI 掩码：中间方块为 ROI
    mask = torch.zeros(B, 1, H, H)
    mask[:, :, H//4:3*H//4, H//4:3*H//4] = 1.0

    model = HDAROISystem(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ok] model built, params = {n_params/1e6:.2f}M")

    loss_fn = ROIAwareLoss(cfg["loss"]["lambda_roi"],
                           cfg["loss"]["lambda_fourier"],
                           cfg["loss"]["lambda_rate"])

    # ---- 阶段1前向 ----
    I_hat1, z = model.forward_codec(I)
    assert I_hat1.shape == I.shape, f"codec output shape {I_hat1.shape}"
    l1 = loss_fn.semantic_loss(I, I_hat1, mask)
    l1.backward()
    print(f"[ok] stage1 forward/backward, loss={l1.item():.4f}, feat={tuple(z.shape)}")

    model.zero_grad()
    # ---- 完整前向 ----
    snr = torch.empty(B).uniform_(0, 12)
    out = model(I, mask, snr, training=True)
    assert out["I_hat"].shape == I.shape
    # 验证恒等：z_D + z_A == z（融合还原）
    z_D, z_A, mf = model.alloc.split(out["z"], mask)
    recon_err = (z_D + z_A - out["z"]).abs().max().item()
    print(f"[ok] full forward, split identity max-err={recon_err:.2e} (应≈0)")

    l_sd = loss_fn.semantic_loss(I, out["I_hat"], mask)
    l_cd = loss_fn.channel_loss(out["z"].detach(), out["z_hat"], out["mask_feat"])
    l_rate = loss_fn.rate_loss(out["z"].detach(), out["mask_feat"])
    loss = l_sd + l_cd + loss_fn.lambda_rate * l_rate
    loss.backward()
    print(f"[ok] full backward, loss={loss.item():.4f} "
          f"(sd={l_sd.item():.3f} cd={l_cd.item():.3f})")

    # ---- 指标 ----
    f, r, b = region_psnr(I, out["I_hat"], mask)
    print(f"[ok] metrics: full={f:.2f} roi={r:.2f} bg={b:.2f} dB (随机权重，仅验证可算)")

    # ---- 验证 ROI 加权确实生效 ----
    w_in = loss_fn.weighted_mse(I, I_hat1, mask).item()
    w_uniform = ((I - I_hat1) ** 2).mean().item()
    print(f"[ok] ROI 加权 MSE={w_in:.4f} vs 均匀 MSE={w_uniform:.4f} "
          f"(加权应更大，因 ROI 权重>1)")

    print("\n所有自检通过。可以接入真实数据集运行 train.py。")


if __name__ == "__main__":
    main()
