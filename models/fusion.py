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
        Stage2 (80, 28, 28)  -> 1x1Conv(80->160) -> BN -> AvgPool(2x) -> (160, 14, 14) -+
        Stage3 (160, 14, 14) -> BN normalization                                     -> (160, 14, 14) -+-> Concat (480, 14, 14)
        Stage4 (320, 7, 7)   -> 1x1Conv(320->160) -> BN -> Upsample(2x) -> (160, 14, 14) -+
                                                                                       |
                                                                            1x1Conv(480->160) + BN + GELU
                                                                                       |
                                                                                  MSCA(160) + Residual
                                                                                       |
                                                                                  Output (160, 14, 14)

    Args:
        s2_channels: Stage2 output channels (default: 80).
        s3_channels: Stage3 output channels (default: 160).
        s4_channels: Stage4 output channels (default: 160).
        fusion_dim: Target fusion dimension (default: 160, aligned with Stage3).
        target_size: Target spatial size (default: 14, aligned with Stage3).
            If 0, auto-detect from Stage3 spatial size at runtime.
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

        # Stage3: BN normalization (ensures consistent feature scale with s2/s4)
        self.s3_norm = norm_layer(s3_channels)

        # Stage4: 320 -> 160 (channel reduction + cross-channel interaction)
        self.s4_align = nn.Sequential(
            nn.Conv2d(s4_channels, fusion_dim, 1, bias=False),
            norm_layer(fusion_dim),
        )

        # === Spatial Alignment ===
        # Stage2: downsample via AvgPool (factor=2 relative to Stage3)
        self.s2_downsample = nn.AvgPool2d(kernel_size=2, stride=2)

        # Stage4: upsample (will be done dynamically in forward)
        # No longer use fixed nn.Upsample — compute target size at runtime

        # === Fusion Compression ===
        concat_dim = fusion_dim * 3  # 480 = 160 * 3
        self.fusion_compress = nn.Sequential(
            nn.Conv2d(concat_dim, fusion_dim, 1, bias=False),
            norm_layer(fusion_dim),
            act_layer(),
        )

        # === Post-fusion MSCA Calibration (with residual connection) ===
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
            s2_feat: Stage2 feature (B, 80, H2, W2).
            s3_feat: Stage3 feature (B, 160, H3, W3).
            s4_feat: Stage4 feature (B, 320, H4, W4).

        Returns:
            Fused feature tensor (B, 160, H3, W3).
        """
        # Determine target spatial size from Stage3
        target_h, target_w = s3_feat.shape[2], s3_feat.shape[3]

        # Channel alignment
        s2 = self.s2_align(s2_feat)   # (B, 160, H2, W2)
        s3 = self.s3_norm(s3_feat)    # (B, 160, H3, W3) — BN normalization for scale consistency
        s4 = self.s4_align(s4_feat)   # (B, 160, H4, W4)

        # Spatial alignment
        s2 = self.s2_downsample(s2)   # (B, 160, H2/2, W2/2)
        # If still not matching target, use adaptive pool
        if s2.shape[2] != target_h or s2.shape[3] != target_w:
            s2 = nn.functional.adaptive_avg_pool2d(s2, (target_h, target_w))

        # Stage4: upsample to match Stage3
        s4 = nn.functional.interpolate(s4, size=(target_h, target_w),
                                        mode="bilinear", align_corners=False)

        # Channel concatenation
        fused = torch.cat([s2, s3, s4], dim=1)  # (B, 480, H3, W3)

        # Compression
        fused = self.fusion_compress(fused)  # (B, 160, H3, W3)

        # MSCA calibration with residual connection
        if self.use_msca:
            fused = fused + self.fusion_msca(fused)  # Residual: preserves original features

        return fused


class TwoStageFusion(nn.Module):
    """Simplified fusion using only Stage3 + Stage4 (for ablation).

    Used in ablation experiments to test whether Stage2 is necessary.
    Spatial size is auto-detected from Stage3 at runtime.
    Includes BN normalization for Stage3 and residual connection for MSCA.
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
        # BN normalization for Stage3 (consistent scale with aligned Stage4)
        self.s3_norm = norm_layer(s3_channels)

        self.s4_align = nn.Sequential(
            nn.Conv2d(s4_channels, fusion_dim, 1, bias=False),
            norm_layer(fusion_dim),
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
        target_h, target_w = s3_feat.shape[2], s3_feat.shape[3]
        s3 = self.s3_norm(s3_feat)  # BN normalization
        s4 = self.s4_align(s4_feat)
        s4 = nn.functional.interpolate(s4, size=(target_h, target_w),
                                        mode="bilinear", align_corners=False)
        fused = torch.cat([s3, s4], dim=1)
        fused = self.fusion_compress(fused)
        if self.use_msca:
            fused = fused + self.fusion_msca(fused)  # Residual connection
        return fused
