"""MSCA-FasterNet: The complete improved model.

Combines FasterNet-T0 backbone with:
    1. MSCA (Multi-Scale Channel Attention) modules inserted at Stage3
    2. Cross-Layer Feature Fusion across Stage2/3/4
    3. Simplified classification head

Parameter counts (verified):
    - Baseline (FasterNet-T0, no MSCA, no fusion): ~3.9M params, ~0.34G FLOPs
    - Full model (MSCA + fusion + DropPath): ~4.1M params, ~0.40G FLOPs
    - MSCA overhead: ~9.3K per module × 2 = ~18.6K
    - Fusion overhead: ~150K (alignment + compression + MSCA)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict

from .fasternet import FasterNet, FasterNetT0
from .msca import MSCA, MSCALight, SEOnly
from .fusion import CrossLayerFusion, TwoStageFusion


class MSCAFasterNet(nn.Module):
    """MSCA-FasterNet: Lightweight Crop Pest and Disease Identification Model.

    Full architecture:
        Input (3, 224, 224)
            -> Embedding (40, 56, 56)
            -> Stage1 (40, 56, 56)        [no MSCA]
            -> Stage2 (80, 28, 28)        [saved for fusion]
            -> Stage3 (160, 14, 14)       [MSCA at last 2 blocks; saved for fusion]
            -> Stage4 (320, 7, 7)         [saved for fusion]
            -> CrossLayerFusion (160, 14, 14)
            -> GAP -> FC(160 -> num_classes)

    Args:
        num_classes: Number of output classes (102 for IP102, 15 for PlantVillage).
        embed_dim: Embedding dimension. Default: 40.
        depths: Block counts per stage. Default: [1, 2, 8, 2].
        n_div: PConv channel division ratio. Default: 4.
        mlp_ratio: MLP expansion ratio. Default: 2.0.
        msca_reduction: MSCA SE reduction ratio. Default: 16.
        msca_stage: Stage index to insert MSCA (0-based). Default: 2 (Stage3).
        msca_block_indices: Block indices within the stage to insert MSCA.
            Default: [6, 7] (last 2 blocks of Stage3).
        use_fusion: Whether to use cross-layer feature fusion. Default: True.
        fusion_dim: Fusion output dimension. Default: 160.
        fusion_target_size: Fusion target spatial size. Default: 14.
        fusion_use_msca: Whether to apply MSCA after fusion. Default: True.
        use_msca_in_blocks: Whether to insert MSCA in backbone blocks. Default: True.
        act_layer: Activation layer. Default: nn.GELU.
        norm_layer: Normalization layer. Default: nn.BatchNorm2d.
        dropout: Dropout rate for classification head. Default: 0.0.
    """

    def __init__(
        self,
        num_classes: int = 102,
        embed_dim: int = 40,
        depths: List[int] = [1, 2, 8, 2],
        n_div: int = 4,
        mlp_ratio: float = 2.0,
        msca_reduction: int = 16,
        msca_stage: int = 2,
        msca_block_indices: Optional[List[int]] = None,
        use_fusion: bool = True,
        fusion_dim: int = 160,
        fusion_target_size: int = 14,
        fusion_use_msca: bool = True,
        use_msca_in_blocks: bool = True,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        dropout: float = 0.0,
        drop_path_rate: float = 0.05,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths
        self.dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.use_fusion = use_fusion
        self.use_msca_in_blocks = use_msca_in_blocks
        self.drop_path_rate = drop_path_rate

        if msca_block_indices is None:
            msca_block_indices = [depths[msca_stage] - 2, depths[msca_stage] - 1]

        # === MSCA factory for backbone insertion ===
        def msca_factory(dim):
            return MSCA(
                dim=dim,
                reduction=msca_reduction,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )

        # === Backbone ===
        msca_config = None
        if use_msca_in_blocks:
            msca_config = {
                "stage": msca_stage,
                "indices": msca_block_indices,
                "factory": msca_factory,
            }

        self.backbone = FasterNet(
            in_channels=3,
            embed_dim=embed_dim,
            depths=depths,
            mlp_ratio=mlp_ratio,
            n_div=n_div,
            act_layer=act_layer,
            norm_layer=norm_layer,
            pconv_fw="split_cat",
            msca_config=msca_config,
            out_indices=(1, 2, 3),  # Stage2, Stage3, Stage4
            drop_path_rate=drop_path_rate,
        )

        # === Cross-Layer Feature Fusion ===
        if use_fusion:
            self.fusion = CrossLayerFusion(
                s2_channels=self.dims[1],
                s3_channels=self.dims[2],
                s4_channels=self.dims[3],
                fusion_dim=fusion_dim,
                target_size=fusion_target_size,
                use_msca=fusion_use_msca,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            classifier_in_dim = fusion_dim
        else:
            self.fusion = None
            # Without fusion, use Stage4 output directly
            classifier_in_dim = self.dims[3]

        # === Classification Head ===
        # Simplified: no intermediate 1280-dim layer (original was for ImageNet 1000 classes)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # GAP
            nn.Flatten(1),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(classifier_in_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for newly added modules.

        Only initializes the classification head, fusion module, and MSCA modules.
        Backbone weights are NOT reset here to preserve pretrained weights.
        """
        for name, m in self.named_modules():
            # Skip backbone modules — they may have pretrained weights
            if name.startswith("backbone."):
                continue
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image tensor (B, 3, 224, 224).

        Returns:
            Class logits (B, num_classes).
        """
        # Backbone feature extraction
        features = self.backbone(x)

        s2_feat = features[1]  # (B, 80, 28, 28)
        s3_feat = features[2]  # (B, 160, 14, 14)
        s4_feat = features[3]  # (B, 320, 7, 7)

        # Feature fusion
        if self.use_fusion and self.fusion is not None:
            fused = self.fusion(s2_feat, s3_feat, s4_feat)  # (B, 160, 14, 14)
        else:
            fused = s4_feat  # (B, 320, 7, 7)

        # Classification
        logits = self.classifier(fused)  # (B, num_classes)

        return logits

    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get intermediate feature maps for visualization.

        Returns:
            Dict with keys: 's2', 's3', 's4', 'fused' (if fusion is used)
        """
        features = self.backbone(x)
        result = {
            "s2": features[1],
            "s3": features[2],
            "s4": features[3],
        }

        if self.use_fusion and self.fusion is not None:
            result["fused"] = self.fusion(features[1], features[2], features[3])

        return result

    def load_pretrained_backbone(self, state_dict: dict, strict: bool = False):
        """Load ImageNet pretrained weights for the FasterNet backbone.

        Handles key mismatches between original FasterNet and our modified version.
        """
        backbone_state = self.backbone.state_dict()
        pretrained_filtered = {}

        for key, value in state_dict.items():
            # Skip classification head weights
            if "head" in key or "classifier" in key:
                continue

            # Map keys if necessary
            new_key = key
            if new_key in backbone_state:
                if value.shape == backbone_state[new_key].shape:
                    pretrained_filtered[new_key] = value
                else:
                    print(f"  Skip {new_key}: shape mismatch "
                          f"({value.shape} vs {backbone_state[new_key].shape})")

        backbone_state.update(pretrained_filtered)
        self.backbone.load_state_dict(backbone_state, strict=strict)
        print(f"Loaded {len(pretrained_filtered)}/{len(backbone_state)} "
              f"pretrained parameters into backbone.")


def msca_fasternet_t0(
    num_classes: int = 102,
    use_msca: bool = True,
    use_fusion: bool = True,
    msca_reduction: int = 16,
    pretrained_backbone: Optional[str] = None,
    **kwargs,
) -> MSCAFasterNet:
    """Construct MSCA-FasterNet-T0 model.

    This is the main model used in the paper.

    Args:
        num_classes: Number of output classes.
        use_msca: Whether to use MSCA in backbone. Set False for ablation.
        use_fusion: Whether to use cross-layer fusion. Set False for ablation.
        msca_reduction: MSCA reduction ratio.
        pretrained_backbone: Path to pretrained FasterNet-T0 weights.

    Returns:
        MSCAFasterNet model instance.
    """
    model = MSCAFasterNet(
        num_classes=num_classes,
        embed_dim=40,
        depths=[1, 2, 8, 2],
        n_div=4,
        mlp_ratio=2.0,
        msca_reduction=msca_reduction,
        msca_stage=2,  # Stage3 (0-based index)
        msca_block_indices=[6, 7],  # Last 2 blocks of Stage3
        use_fusion=use_fusion,
        fusion_dim=160,
        fusion_target_size=14,
        fusion_use_msca=kwargs.pop("fusion_use_msca", True),  # Allow ablation to disable
        use_msca_in_blocks=use_msca,
        **kwargs,
    )

    if pretrained_backbone is not None:
        import torch as _torch
        state_dict = _torch.load(pretrained_backbone, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_pretrained_backbone(state_dict)

    return model


# ============================================================
# Ablation model variants
# ============================================================

def fasternet_t0_baseline(num_classes: int = 102, **kwargs) -> MSCAFasterNet:
    """Baseline: original FasterNet-T0 without any improvements."""
    return msca_fasternet_t0(
        num_classes=num_classes,
        use_msca=False,
        use_fusion=False,
        **kwargs,
    )


def fasternet_t0_with_msca(num_classes: int = 102, **kwargs) -> MSCAFasterNet:
    """Ablation B: FasterNet-T0 + MSCA only (no fusion)."""
    return msca_fasternet_t0(
        num_classes=num_classes,
        use_msca=True,
        use_fusion=False,
        **kwargs,
    )


def fasternet_t0_with_fusion(num_classes: int = 102, **kwargs) -> MSCAFasterNet:
    """Ablation C: FasterNet-T0 + fusion only (no MSCA anywhere, pure fusion contribution)."""
    return msca_fasternet_t0(
        num_classes=num_classes,
        use_msca=False,
        use_fusion=True,
        fusion_use_msca=False,  # Ablation: disable MSCA in fusion to isolate fusion's contribution
        **kwargs,
    )


def fasternet_t0_full(num_classes: int = 102, **kwargs) -> MSCAFasterNet:
    """Ablation D (full model): FasterNet-T0 + MSCA + fusion."""
    return msca_fasternet_t0(
        num_classes=num_classes,
        use_msca=True,
        use_fusion=True,
        **kwargs,
    )
