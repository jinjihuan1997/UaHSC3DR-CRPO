"""
ROI 感知损失函数。

语义失真 = ROI 加权 MSE + 傅里叶频域损失（捕获长程依赖）+ 可选 LPIPS
信道失真 = 特征空间 ROI 加权 L1（阶段2/3）
速率约束 = 数字路特征能量（简化版的比特率代理）
"""
import torch
import torch.nn as nn
import torch.fft as fft


class ROIAwareLoss(nn.Module):
    def __init__(self, lambda_roi=5.0, lambda_fourier=0.1, lambda_rate=5e-4,
                 lambda_lpips=0.0):
        super().__init__()
        self.lambda_roi = lambda_roi
        self.lambda_fourier = lambda_fourier
        self.lambda_rate = lambda_rate
        self.lambda_lpips = lambda_lpips
        self._lpips = None  # 延迟初始化，避免不需要时加载 alex 权重

    def _ensure_lpips(self, device):
        if self._lpips is None:
            import lpips
            self._lpips = lpips.LPIPS(net="alex").to(device).eval()
            for p in self._lpips.parameters():
                p.requires_grad_(False)
        return self._lpips

    def weighted_mse(self, I, I_hat, mask):
        # ROI 区域权重 = lambda_roi，背景 = 1（mask 软值时按比例插值）
        w = 1.0 + (self.lambda_roi - 1.0) * mask  # (B,1,H,W) 广播到 3 通道
        return (w * (I - I_hat) ** 2).mean()

    def fourier_loss(self, I, I_hat):
        FI = fft.fft2(I, norm="ortho").abs()
        FH = fft.fft2(I_hat, norm="ortho").abs()
        return (FI - FH).abs().mean()

    def lpips_loss(self, I, I_hat):
        # LPIPS 只定义在 RGB 图像上；多模态输入时仅取前 3 个通道。
        net = self._ensure_lpips(I.device)
        I_rgb = I[:, :3]
        I_hat_rgb = I_hat[:, :3]
        return net(I_rgb * 2.0 - 1.0, I_hat_rgb * 2.0 - 1.0).mean()

    def semantic_loss(self, I, I_hat, mask, use_lpips=False):
        loss = self.weighted_mse(I, I_hat, mask) + \
               self.lambda_fourier * self.fourier_loss(I, I_hat)
        if use_lpips and self.lambda_lpips > 0:
            loss = loss + self.lambda_lpips * self.lpips_loss(I, I_hat)
        return loss

    def channel_loss(self, z, z_hat, mask_feat):
        w = 1.0 + (self.lambda_roi - 1.0) * mask_feat
        return (w * (z - z_hat).abs()).mean()

    def rate_loss(self, z, mask_feat):
        # 数字路承载的特征能量越大，所需比特越多
        z_D = z * mask_feat
        return z_D.pow(2).mean()
