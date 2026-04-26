"""Multi-Scale Channel Attention (MSCA) Module.

Design motivation:
    Crop pest/disease lesions vary significantly in scale (3-5px spots to
    half-leaf patches). FasterNet's fixed 3x3 PConv cannot capture this
    multi-scale variation effectively.

MSCA addresses this by:
    1. Multi-scale spatial feature extraction via 3x3 + 5x5 Depthwise Conv
    2. Channel importance calibration via SE-style attention

Parameter overhead: ~9.3K per module (dim=160), only 0.48% of FasterNet-T0.
"""

import torch
import torch.nn as nn
from typing import Optional


class MSCA(nn.Module):
    """Multi-Scale Channel Attention Module.

    Structure:
        Input X (B, C, H, W)
            |
            +-- Branch A: Channel Attention (SE)
            |   AvgPool -> 1x1Conv(C->C/r) -> ReLU -> 1x1Conv(C/r->C) -> Sigmoid
            |   Output: channel_weights (B, C, 1, 1)
            |
            +-- Branch B: Multi-scale Spatial Features
            |   3x3 DWConv(C, groups=C) -> BN -> GELU -> F3
            |   5x5 DWConv(C, groups=C) -> BN -> GELU -> F5
            |   Fused: F3 + F5 (element-wise add)
            |
            +-- Calibration: Fused * channel_weights (broadcast multiply)
            |
            Output (B, C, H, W)

    Args:
        dim: Input channel dimension.
        reduction: Channel reduction ratio for SE branch. Default: 16.
        act_layer: Activation for DWConv branches. Default: nn.GELU.
        norm_layer: Normalization for DWConv branches. Default: nn.BatchNorm2d.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 16,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        self.dim = dim
        self.reduction = reduction

        # Ensure minimum intermediate dimension
        mid_channels = max(dim // reduction, 4)

        # === Branch A: Channel Attention (SE) ===
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # (B, C, 1, 1)
            nn.Conv2d(dim, mid_channels, 1, bias=False),  # Compress
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, dim, 1, bias=False),  # Expand
            nn.Sigmoid(),                       # (B, C, 1, 1) weights
        )

        # === Branch B: Multi-scale Spatial Features ===
        # 3x3 Depthwise Conv - captures small lesions (aphids, mites)
        self.dwconv_3x3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            norm_layer(dim),
            act_layer(),
        )

        # 5x5 Depthwise Conv - captures large patches (leaf spots, blight)
        self.dwconv_5x5 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False),
            norm_layer(dim),
            act_layer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature tensor (B, C, H, W).

        Returns:
            Attention-calibrated feature tensor (B, C, H, W).
        """
        # Multi-scale spatial feature extraction
        f3 = self.dwconv_3x3(x)       # Small-scale features
        f5 = self.dwconv_5x5(x)       # Large-scale features
        f_fused = f3 + f5              # Element-wise fusion (zero params)

        # Channel attention calibration
        channel_weights = self.channel_attention(x)  # (B, C, 1, 1)

        # Broadcast multiply: spatial features weighted by channel importance
        output = f_fused * channel_weights

        return output


class MSCALight(nn.Module):
    """Lightweight MSCA variant without SE branch (for ablation study).

    Only contains multi-scale DWConv branches without channel attention.
    Used in ablation experiments to validate the contribution of SE branch.
    """

    def __init__(
        self,
        dim: int,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        self.dwconv_3x3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False),
            norm_layer(dim),
            act_layer(),
        )
        self.dwconv_5x5 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False),
            norm_layer(dim),
            act_layer(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dwconv_3x3(x) + self.dwconv_5x5(x)


class SEOnly(nn.Module):
    """SE attention only (for ablation study - no multi-scale DWConv)."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        mid_channels = max(dim // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, dim, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)
