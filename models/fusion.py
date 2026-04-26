"""Cross-Layer Feature Fusion Module.

Design motivation:
    The original FasterNet only uses Stage4 output for classification.
    However, pest/disease recognition needs both:
    - Shallow texture details (lesion color, shape) from Stage2
    - Mid-level local patterns (lesion texture) from Stage3
    - Deep global semantics (disease category) from Stage4

This module aligns and fuses features from Stage2, Stage3, and Stage4,
then applies MSCA for channel calibration.

Parameter overhead: ~150K (including alignment convs + compression + MSCA).
"""

import torch
import torch.nn as nn
from typing import Optional

from .msca import MSCA


class CrossLayerFusion(nn.Module):
    """Cross-Layer Feature Fusion for multi-scale feature integration.

    Architecture:
        Stage2 (80, 28, 28)  -> 1x1Conv(80->160) -> AvgPool(2x) -> (160, 14, 14) -+
        Stage3 (160, 14, 14) -> identity                                          -> (160, 14, 14) -+-> Concat (480, 14, 14)
        Stage4 (320, 7, 7)   -> 1x1Conv(320->160) -> Upsample(2x) -> (160, 14, 14) -+
                                                                                       |
                                                                            1x1Conv(480->160) + BN + GELU
                                                                                       |
                                                                                  MSCA(160)
                                                                                       |
                                                                                  Output (160, 14, 14)

    Args:
        s2_channels: Stage2 output channels (default: 80).
        s3_channels: Stage3 output channels (default: 160).
        s4_channels: Stage4 output channels (default: 160).
        fusion_dim: Target fusion dimension (default: 160, aligned with Stage3).
        target_size: Target spatial size (default: 14, aligned with Stage3).
        use_msca: Whether to apply MSCA after fusion (default: True).
        act_layer: Activation layer. Default: nn.GELU.
        norm_layer: Normalization layer. Default: nn.BatchNorm2d.
    """

    def __init__(
        self,
        s2_channels: int = 80,
        s3_channels: int = 160,
        s4_channels: int = 320,
        fusion_dim: int = 160,
        target_size: int = 14,
        use_msca: bool = True,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        self.fusion_dim = fusion_dim
        self.target_size = target_size
        self.use_msca = use_msca

        # === Channel Alignment ===
        # Stage2: 80 -> 160 (channel expansion + cross-channel interaction)
        self.s2_align = nn.Sequential(
            nn.Conv2d(s2_channels, fusion_dim, 1, bias=False),
            norm_layer(fusion_dim),
        )

        # Stage3: already aligned (160 == fusion_dim), no transform needed

        # Stage4: 320 -> 160 (channel reduction + cross-channel interaction)
        self.s4_align = nn.Sequential(
            nn.Conv2d(s4_channels, fusion_dim, 1, bias=False),
            norm_layer(fusion_dim),
        )

        # === Spatial Alignment ===
        # Stage2: 28x28 -> 14x14 (downsample)
        self.s2_downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        # Stage4: 7x7 -> 14x14 (upsample)
        self.s4_upsample = nn.Upsample(
            size=target_size, mode="bilinear", align_corners=False
        )

        # === Fusion Compression ===
        concat_dim = fusion_dim * 3  # 480 = 160 * 3
        self.fusion_compress = nn.Sequential(
            nn.Conv2d(concat_dim, fusion_dim, 1, bias=False),
            norm_layer(fusion_dim),
            act_layer(),
        )

        # === Post-fusion MSCA Calibration ===
        if use_msca:
            self.fusion_msca = MSCA(
                dim=fusion_dim,
                reduction=16,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

    def forward(
        self,
        s2_feat: torch.Tensor,
        s3_feat: torch.Tensor,
        s4_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse features from Stage2, Stage3, and Stage4.

        Args:
            s2_feat: Stage2 feature (B, 80, 28, 28).
            s3_feat: Stage3 feature (B, 160, 14, 14).
            s4_feat: Stage4 feature (B, 320, 7, 7).

        Returns:
            Fused feature tensor (B, 160, 14, 14).
        """
        # Channel alignment
        s2 = self.s2_align(s2_feat)   # (B, 160, 28, 28)
        # s3: no alignment needed     # (B, 160, 14, 14)
        s4 = self.s4_align(s4_feat)   # (B, 160, 7, 7)

        # Spatial alignment
        s2 = self.s2_downsample(s2)   # (B, 160, 14, 14)
        # s3: already 14x14
        s4 = self.s4_upsample(s4)     # (B, 160, 14, 14)

        # Channel concatenation
        fused = torch.cat([s2, s3_feat, s4], dim=1)  # (B, 480, 14, 14)

        # Compression
        fused = self.fusion_compress(fused)  # (B, 160, 14, 14)

        # MSCA calibration
        if self.use_msca:
            fused = self.fusion_msca(fused)  # (B, 160, 14, 14)

        return fused


class TwoStageFusion(nn.Module):
    """Simplified fusion using only Stage3 + Stage4 (for ablation).

    Used in ablation experiments to test whether Stage2 is necessary.
    """

    def __init__(
        self,
        s3_channels: int = 160,
        s4_channels: int = 320,
        fusion_dim: int = 160,
        target_size: int = 14,
        use_msca: bool = True,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        self.s4_align = nn.Sequential(
            nn.Conv2d(s4_channels, fusion_dim, 1, bias=False),
            norm_layer(fusion_dim),
        )
        self.s4_upsample = nn.Upsample(
            size=target_size, mode="bilinear", align_corners=False
        )

        concat_dim = fusion_dim * 2  # 320
        self.fusion_compress = nn.Sequential(
            nn.Conv2d(concat_dim, fusion_dim, 1, bias=False),
            norm_layer(fusion_dim),
            act_layer(),
        )

        if use_msca:
            self.fusion_msca = MSCA(
                dim=fusion_dim, reduction=16,
                act_layer=act_layer, norm_layer=norm_layer,
            )
        self.use_msca = use_msca

    def forward(
        self, s3_feat: torch.Tensor, s4_feat: torch.Tensor
    ) -> torch.Tensor:
        s4 = self.s4_align(s4_feat)
        s4 = self.s4_upsample(s4)
        fused = torch.cat([s3_feat, s4], dim=1)
        fused = self.fusion_compress(fused)
        if self.use_msca:
            fused = self.fusion_msca(fused)
        return fused
