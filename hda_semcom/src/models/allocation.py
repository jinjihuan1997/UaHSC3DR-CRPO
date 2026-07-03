"""
ROI 感知模数分配模块（替代原版超级编解码器的内容自适应分离）。

分配：
    Mf = downsample(mask)               # 像素掩码 -> 特征空间
    z_D = z * Mf                        # ROI 区域 -> 数字路（精确）
    z_A = z * (1 - Mf)                  # 背景区域 -> 模拟路（近似）
融合：
    ẑ  = ẑ_D + ẑ_A                       # 空间互补，相加还原
恒等保证：Mf + (1 - Mf) = 1 => z_D + z_A = z
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAwareAllocation(nn.Module):
    def __init__(self, feat_dim, mask_refine=True):
        super().__init__()
        self.mask_refine = mask_refine
        if mask_refine:
            # 在特征空间细化掩码边界，缓解下采样带来的锯齿
            self.refine = nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid(),
            )

    def mask_to_feat(self, mask_pixel, feat_hw):
        """像素掩码 (B,1,H,W) -> 特征空间掩码 (B,1,Hf,Wf)。"""
        mf = F.interpolate(mask_pixel, size=feat_hw, mode="bilinear",
                           align_corners=False)
        if self.mask_refine:
            mf = self.refine(mf)
        return mf.clamp(0, 1)

    def split(self, z, mask_pixel):
        mf = self.mask_to_feat(mask_pixel, (z.shape[2], z.shape[3]))
        z_D = z * mf
        z_A = z * (1.0 - mf)
        return z_D, z_A, mf

    @staticmethod
    def fuse(z_D_hat, z_A_hat):
        return z_D_hat + z_A_hat
