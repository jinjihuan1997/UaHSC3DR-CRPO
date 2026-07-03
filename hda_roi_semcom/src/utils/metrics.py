"""分区域评估指标：全图 / ROI / 背景 PSNR，以及 MS-SSIM。"""
import math
import torch


def _masked_sse_area(I, I_hat, mask):
    """返回 mask 区域内的平方误差和有效像素通道数。"""
    area = (mask.sum() * I.shape[1]).item()
    sse = (((I - I_hat) ** 2) * mask).sum().item()
    return sse, area


def _psnr_masked(I, I_hat, mask):
    """mask 加权区域的 PSNR（mask 全 1 时即全图）。"""
    sse, area = _masked_sse_area(I, I_hat, mask)
    if area <= 0:
        return float("nan")
    mse = max(sse / area, 1e-10)
    return -10.0 * math.log10(mse)


def region_psnr(I, I_hat, mask):
    """返回 (full, roi, bg) 三个 PSNR。mask 为软值时按阈值 0.5 二值化用于分区。"""
    hard = (mask > 0.5).float()
    full = _psnr_masked(I, I_hat, torch.ones_like(hard))
    roi = _psnr_masked(I, I_hat, hard)
    bg = _psnr_masked(I, I_hat, 1.0 - hard)
    return full, roi, bg


class RegionPSNRMeter:
    """跨 batch 累计区域 SSE 后计算 PSNR，避免空 ROI batch 传播 nan。"""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.stats = {
            "full": [0.0, 0.0],
            "roi": [0.0, 0.0],
            "bg": [0.0, 0.0],
        }

    def update(self, I, I_hat, mask):
        hard = (mask > self.threshold).float()
        masks = {
            "full": torch.ones_like(hard),
            "roi": hard,
            "bg": 1.0 - hard,
        }
        for name, region_mask in masks.items():
            sse, area = _masked_sse_area(I, I_hat, region_mask)
            self.stats[name][0] += sse
            self.stats[name][1] += area

    def compute(self):
        out = []
        for name in ("full", "roi", "bg"):
            sse, area = self.stats[name]
            if area <= 0:
                out.append(float("nan"))
                continue
            mse = max(sse / area, 1e-10)
            out.append(-10.0 * math.log10(mse))
        return tuple(out)


try:
    from pytorch_msssim import ms_ssim as _ms_ssim

    def ms_ssim_db(I, I_hat):
        # 输入需 (B,C,H,W) in [0,1]；用 fp64 计算避免 1-1e-8 舍入到 1.0 引发 inf
        v = _ms_ssim(I, I_hat, data_range=1.0).double()
        v = v.clamp(min=1e-8, max=1.0 - 1e-6)   # 1e-6 在 fp64 下安全
        return (-10.0 * torch.log10(1.0 - v)).item()
except Exception:  # pragma: no cover
    def ms_ssim_db(I, I_hat):
        return float("nan")
