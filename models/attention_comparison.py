"""Attention method comparison modules for SCI paper.

Implements standard attention mechanisms as drop-in replacements for MSCA,
enabling fair comparison on the same FasterNet-T0 backbone:
    - SE-Net (Hu et al., CVPR 2018): Squeeze-and-Excitation channel attention
    - CBAM (Woo et al., ECCV 2018): Channel + Spatial attention
    - ECA-Net (Wang et al., CVPR 2020): Efficient Channel Attention (1D conv)
    - SK-Net (Li et al., CVPR 2019): Selective Kernel (multi-scale + soft attention)

All modules follow the same interface:
    - __init__(dim, ...)
    - forward(x) -> attention output (NO internal residual; block handles residual)
    - self.dim attribute for compatibility

This design ensures:
    1. Fair comparison: identical backbone, training, and evaluation
    2. Consistent parameter overhead measurement
    3. Same insertion point (Stage3, last 2 blocks)
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SEAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention (Hu et al., CVPR 2018).

    Architecture:
        GAP -> FC(C->C/r) -> ReLU -> FC(C/r->C) -> Sigmoid -> scale

    Args:
        dim: Input channel dimension.
        reduction: Channel reduction ratio. Default: 16.
    """

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.dim = dim
        mid_channels = max(dim // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns calibrated features (no internal residual)."""
        return x * self.se(x)


class CBAMChannelAttention(nn.Module):
    """CBAM Channel Attention sub-module (Woo et al., ECCV 2018).

    Uses both AvgPool and MaxPool for richer channel descriptors.

    Architecture:
        AvgPool(C) + MaxPool(C) -> Shared FC(C->C/r->C) -> Sigmoid -> sum
    """

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(dim // reduction, 4)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(dim, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, dim, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.shared_mlp(nn.functional.adaptive_avg_pool2d(x, 1))
        max_out = self.shared_mlp(nn.functional.adaptive_max_pool2d(x, 1))
        return self.sigmoid(avg_out + max_out)


class CBAMSpatialAttention(nn.Module):
    """CBAM Spatial Attention sub-module (Woo et al., ECCV 2018).

    Architecture:
        Channel Avg + Channel Max -> Conv7x7 -> Sigmoid
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAMAttention(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018).

    Combines channel attention and spatial attention sequentially.

    Architecture:
        X -> Channel Attention -> scale -> Spatial Attention -> scale -> Output

    Args:
        dim: Input channel dimension.
        reduction: Channel reduction ratio. Default: 16.
        spatial_kernel: Kernel size for spatial attention. Default: 7.
    """

    def __init__(self, dim: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.dim = dim
        self.channel_attention = CBAMChannelAttention(dim, reduction)
        self.spatial_attention = CBAMSpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns calibrated features (no internal residual)."""
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class ECAAttention(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020).

    Replaces FC layers with a 1D adaptive convolution, capturing
    cross-channel interaction with minimal parameters.

    Architecture:
        GAP -> 1D Conv(kernel=k) -> Sigmoid -> scale

    The kernel size k is adaptively determined from channel dimension:
        k = |log2(C) / gamma + b/gamma|_odd

    Args:
        dim: Input channel dimension.
        gamma: Kernel size formula parameter. Default: 2.
        beta: Kernel size formula parameter. Default: 1.
    """

    def __init__(self, dim: int, gamma: int = 2, beta: int = 1):
        super().__init__()
        self.dim = dim

        # Adaptive kernel size
        k_size = max(int(abs(math.log2(dim) / gamma + beta / gamma)), 3)
        if k_size % 2 == 0:
            k_size += 1  # Ensure odd kernel for symmetric padding

        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns calibrated features (no internal residual)."""
        # GAP: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        y = nn.functional.adaptive_avg_pool2d(x, 1).squeeze(-1).transpose(-1, -2)
        # 1D conv: (B, 1, C) -> (B, 1, C)
        y = self.conv(y)
        # Sigmoid and reshape: (B, 1, C) -> (B, C, 1, 1)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y


class SKAttention(nn.Module):
    """Selective Kernel Attention (Li et al., CVPR 2019).

    Multi-scale feature extraction with per-channel soft attention.
    This is the standard SKNet implementation for comparison with our MSCA.

    Architecture:
        3x3 DWConv -> F3
        5x5 DWConv -> F5
        Fuse: F3 + F5
        GAP -> FC(C->C/r) -> FC(C/r->2C) -> reshape -> Softmax
        Output: a*F3 + b*F5 (per-channel weights)

    Key difference from our MSCA: SK-Net uses per-channel scale selection,
    while MSCA uses per-sample scale selection + SE channel calibration.

    Args:
        dim: Input channel dimension.
        reduction: Channel reduction ratio. Default: 16.
    """

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.dim = dim
        mid_channels = max(dim // reduction, 4)

        # Multi-scale branches
        self.dwconv_3x3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.dwconv_5x5 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

        # Per-channel scale selection (SKNet-style)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Conv2d(dim, mid_channels, 1, bias=False)
        self.fc_expand = nn.Conv2d(mid_channels, dim * 2, 1, bias=False)  # 2C for 2 branches
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns adaptively fused multi-scale features (no internal residual)."""
        B, C, H, W = x.shape

        # Multi-scale features
        f3 = self.dwconv_3x3(x)  # (B, C, H, W)
        f5 = self.dwconv_5x5(x)  # (B, C, H, W)

        # Fuse for attention computation
        fused = f3 + f5
        s = self.gap(fused)  # (B, C, 1, 1)
        s = self.fc_reduce(s)  # (B, mid, 1, 1)
        s = nn.functional.relu(s, inplace=True)
        s = self.fc_expand(s)  # (B, 2C, 1, 1)

        # Per-channel soft attention
        s = s.reshape(B, 2, C, 1, 1)  # (B, 2, C, 1, 1)
        s = self.softmax(s)  # Normalize across 2 branches, per-channel

        # Weighted sum
        a3 = s[:, 0]  # (B, C, 1, 1)
        a5 = s[:, 1]  # (B, C, 1, 1)
        output = a3 * f3 + a5 * f5

        return output


# ============================================================
# Attention factory for model builders
# ============================================================

ATTENTION_MODULES = {
    "se": SEAttention,
    "cbam": CBAMAttention,
    "eca": ECAAttention,
    "sk": SKAttention,
    "msca": None,  # MSCA is imported separately
}


def get_attention_module(name: str, dim: int, **kwargs) -> nn.Module:
    """Factory function to create attention modules by name.

    Args:
        name: Attention type ('se', 'cbam', 'eca', 'sk').
        dim: Input channel dimension.
        **kwargs: Additional arguments for the attention module.

    Returns:
        Attention module instance.
    """
    if name not in ATTENTION_MODULES:
        raise ValueError(f"Unknown attention: {name}. Available: {list(ATTENTION_MODULES.keys())}")
    cls = ATTENTION_MODULES[name]
    return cls(dim=dim, **kwargs)
