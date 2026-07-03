"""
模拟信道编解码器：卷积 + 自适应池化把背景特征压缩为复符号传输。

设计要点：
1. 用 1x1/3x3 卷积保留空间结构，参数远少于全连接。
2. 自适应池化把 (B,C,Hf,Wf) 调整到目标符号数对应的空间布局，避免 5200 个符号
   被夹在 240 维隐层瓶颈里。
3. 输出做单位平均功率归一化。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def _pick_symbol_grid(num_symbols):
    """把 num_symbols 拆成接近正方形的 (h, w)，每个空间位置承载 2 个实数（1 个复符号）。"""
    s = max(1, int(round(math.sqrt(num_symbols))))
    while num_symbols % s != 0 and s > 1:
        s -= 1
    h = s
    w = num_symbols // s
    return h, w


class AnalogChannelEncoder(nn.Module):
    def __init__(self, feat_dim, feat_hw, hidden=240, num_symbols=256):
        super().__init__()
        self.feat_dim = feat_dim
        self.feat_hw = feat_hw
        self.num_symbols = num_symbols
        self.sym_hw = _pick_symbol_grid(num_symbols)
        self.net = nn.Sequential(
            nn.Conv2d(feat_dim, hidden, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, 2, 1, 1, 0),   # 实/虚 双通道
        )

    def forward(self, z_A):
        x = self.net(z_A)                                # (B,2,Hf,Wf)
        x = F.adaptive_avg_pool2d(x, self.sym_hw)         # (B,2,h,w)
        re = x[:, 0].flatten(1)
        im = x[:, 1].flatten(1)
        xc = torch.complex(re, im)                       # (B, L)
        power = xc.abs().pow(2).mean(dim=1, keepdim=True).clamp_min(1e-8).sqrt()
        return xc / power


class AnalogChannelDecoder(nn.Module):
    def __init__(self, feat_dim, feat_hw, hidden=240, num_symbols=256):
        super().__init__()
        self.feat_dim = feat_dim
        self.feat_hw = feat_hw
        self.num_symbols = num_symbols
        self.sym_hw = _pick_symbol_grid(num_symbols)
        self.net = nn.Sequential(
            nn.Conv2d(2, hidden, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, feat_dim, 1, 1, 0),
        )

    def forward(self, x_hat):
        B = x_hat.shape[0]
        re = x_hat.real.view(B, 1, *self.sym_hw)
        im = x_hat.imag.view(B, 1, *self.sym_hw)
        x = torch.cat([re, im], dim=1)                    # (B,2,h,w)
        x = F.interpolate(x, size=self.feat_hw, mode="bilinear", align_corners=False)
        return self.net(x)
