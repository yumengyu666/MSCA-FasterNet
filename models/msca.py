"""Multi-Scale Channel Attention (MSCA) Module.

Design motivation:
    Crop pest/disease lesions vary significantly in scale (3-5px spots to
    half-leaf patches). FasterNet's fixed 3x3 PConv cannot capture this
    multi-scale variation effectively.

MSCA addresses this by:
    1. Multi-scale spatial feature extraction via 3x3 + 5x5 Depthwise Conv
    2. Adaptive scale selection via learned soft attention (SKNet-style)
    3. Channel importance calibration via SE-style attention

The adaptive scale selection allows the network to dynamically weight
small-scale (3x3) vs large-scale (5x5) features based on input content,
rather than using a fixed 1:1 fusion ratio. This is particularly important
for pest/disease recognition where lesion sizes vary dramatically.

Reference:
    - Selective Kernel Networks (Li et al., CVPR 2019) for adaptive selection
    - Squeeze-and-Excitation Networks (Hu et al., CVPR 2018) for channel attention

Parameter overhead: ~10.3K per module (dim=160), only 0.53% of FasterNet-T0.
"""

import torch
import torch.nn as nn
from typing import Optional


class MSCA(nn.Module):
    """Multi-Scale Channel Attention Module with Adaptive Scale Selection.

    Structure:
        Input X (B, C, H, W)
            |
            +-- Branch A: Channel Attention (SE)
            |   AvgPool -> 1x1Conv(C->C/r) -> ReLU -> 1x1Conv(C/r->C) -> Sigmoid
            |   Output: channel_weights (B, C, 1, 1)
            |
            +-- Branch B: Adaptive Multi-scale Spatial Features
            |   3x3 DWConv(C, groups=C) -> BN -> GELU -> F3
            |   5x5 DWConv(C, groups=C) -> BN -> GELU -> F5
            |   Soft attention: GAP -> FC(C->max(C/r,4)) -> ReLU -> FC(max(C/r,4)->2) -> Softmax
            |   Fused: a*F3 + (1-a)*F5  (a is learned per-sample, per-channel)
            |
            +-- Calibration: Fused * channel_weights (broadcast multiply)
            |
            Output = X + Fused * channel_weights  (residual connection)

    Args:
        dim: Input channel dimension.
        reduction: Channel reduction ratio for SE branch. Default: 16.
        scale_reduction: Reduction ratio for scale attention MLP. Default: 8.
        act_layer: Activation for DWConv branches. Default: nn.GELU.
        norm_layer: Normalization for DWConv branches. Default: nn.BatchNorm2d.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 16,
        scale_reduction: int = 8,
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

        # === Branch B: Adaptive Multi-scale Spatial Features ===
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

        # === Adaptive Scale Selection (SKNet-style) ===
        # Learns per-sample weights for combining multi-scale features
        scale_mid = max(dim // scale_reduction, 4)
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                          # (B, C, 1, 1)
            nn.Flatten(1),                                    # (B, C)
            nn.Linear(dim, scale_mid),                        # Compress
            nn.ReLU(inplace=True),
            nn.Linear(scale_mid, 2),                          # 2 scale weights
            nn.Softmax(dim=1),                                # Normalize
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection and adaptive scale selection.

        Args:
            x: Input feature tensor (B, C, H, W).

        Returns:
            Attention-calibrated feature tensor (B, C, H, W).
        """
        # Multi-scale spatial feature extraction
        f3 = self.dwconv_3x3(x)       # Small-scale features
        f5 = self.dwconv_5x5(x)       # Large-scale features

        # Adaptive scale selection (learned per-sample weights)
        scale_weights = self.scale_attention(x)  # (B, 2)
        # Reshape for broadcast: (B, 2, 1, 1)
        scale_weights = scale_weights.unsqueeze(-1).unsqueeze(-1)
        # Adaptive fusion: a * F3 + (1-a) * F5
        f_fused = scale_weights[:, 0:1] * f3 + scale_weights[:, 1:2] * f5

        # Channel attention calibration
        channel_weights = self.channel_attention(x)  # (B, C, 1, 1)

        # Broadcast multiply: spatial features weighted by channel importance
        # Residual connection: output = input + calibrated features
        output = x + f_fused * channel_weights

        return output


class MSCALight(nn.Module):
    """Lightweight MSCA variant without SE branch (for ablation study).

    Only contains multi-scale DWConv branches without channel attention.
    Used in ablation experiments to validate the contribution of SE branch.
    Uses simple element-wise addition (no adaptive scale selection).
    """

    def __init__(
        self,
        dim: int,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        self.dim = dim  # Required attribute for compatibility with other modules
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
        return x + self.dwconv_3x3(x) + self.dwconv_5x5(x)  # Residual connection


class SEOnly(nn.Module):
    """SE attention only (for ablation study - no multi-scale DWConv)."""

    def __init__(self, dim: int, reduction: int = 16):
        super().__init__()
        self.dim = dim  # Required attribute for compatibility with other modules
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
