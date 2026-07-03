"""
语义编解码器：基于残差窗口注意力的 Swin 风格块。
自包含实现，避免对 timm 不同版本内部 API 的依赖。

SemanticEncoder: 图像 (3,H,W) -> 特征 (C, H/stride, W/stride)
SemanticDecoder: 特征 -> 重建图像 (3,H,W)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def window_partition(x, ws):
    # x: (B, H, W, C) -> (num_windows*B, ws, ws, C)
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def window_reverse(windows, ws, H, W):
    B = int(windows.shape[0] / (H * W / ws / ws))
    x = windows.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (nW*B, ws*ws, C)
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class SwinBlock(nn.Module):
    """带可选移位窗口的 Transformer 块。"""
    def __init__(self, dim, num_heads, window_size, shift, mlp_ratio=4.0):
        super().__init__()
        self.ws = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))

    def forward(self, x, H, W):
        # x: (B, H*W, C)
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        # pad 到窗口整数倍
        pad_b = (self.ws - H % self.ws) % self.ws
        pad_r = (self.ws - W % self.ws) % self.ws
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        Hp, Wp = H + pad_b, W + pad_r

        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))

        win = window_partition(x, self.ws).view(-1, self.ws * self.ws, C)
        win = self.attn(win)
        win = win.view(-1, self.ws, self.ws, C)
        x = window_reverse(win, self.ws, Hp, Wp)

        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))

        if pad_b or pad_r:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)

        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualSwin(nn.Module):
    """若干 Swin 块 + 残差卷积，增强平移等变性。"""
    def __init__(self, dim, depth, heads, window):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, heads, window, shift=0 if i % 2 == 0 else window // 2)
            for i in range(depth)
        ])
        self.res_conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        seq = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        for blk in self.blocks:
            seq = blk(seq, H, W)
        x = seq.transpose(1, 2).view(B, C, H, W)
        return self.res_conv(x) + shortcut


class SemanticEncoder(nn.Module):
    def __init__(self, in_ch=3, feat_dim=120, stride=8, depth=6, heads=6, window=8):
        super().__init__()
        assert stride in (4, 8, 16)
        layers = [nn.Conv2d(in_ch, feat_dim, 3, 2, 1), nn.GELU()]
        cur = 2
        while cur < stride:
            layers += [nn.Conv2d(feat_dim, feat_dim, 3, 2, 1), nn.GELU()]
            cur *= 2
        self.stem = nn.Sequential(*layers)
        self.body = ResidualSwin(feat_dim, depth, heads, window)

    def forward(self, x):
        return self.body(self.stem(x))


class SemanticDecoder(nn.Module):
    def __init__(self, out_ch=3, feat_dim=120, stride=8, depth=6, heads=6, window=8):
        super().__init__()
        self.body = ResidualSwin(feat_dim, depth, heads, window)
        ups = []
        cur = stride
        while cur > 1:
            ups += [nn.ConvTranspose2d(feat_dim, feat_dim, 4, 2, 1), nn.GELU()]
            cur //= 2
        self.up = nn.Sequential(*ups)
        self.head = nn.Sequential(nn.Conv2d(feat_dim, out_ch, 3, 1, 1), nn.Sigmoid())

    def forward(self, z):
        return self.head(self.up(self.body(z)))
