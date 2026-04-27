"""Attention comparison model builders.

Creates FasterNet-T0 variants with different attention modules (SE, CBAM, ECA, SK)
inserted at the same position as MSCA (Stage3, last 2 blocks).
This ensures fair comparison: identical backbone, same insertion point, same training.

Usage in training:
    python scripts/train.py --model attention_se --dataset ip102 --seed 42
    python scripts/train.py --model attention_cbam --dataset ip102 --seed 42

Usage in evaluation:
    python scripts/evaluate.py --model attention_eca --checkpoint <path>
"""

import torch
import torch.nn as nn
from typing import Optional, List

from .fasternet import FasterNet
from .attention_comparison import SEAttention, CBAMAttention, ECAAttention, SKAttention


def _build_attention_fasternet(
    attention_type: str,
    num_classes: int = 102,
    attention_dim: int = 160,  # Stage3 channel dimension for FasterNet-T0
    pretrained_backbone: Optional[str] = None,
    **kwargs,
) -> FasterNet:
    """Build FasterNet-T0 with a specific attention module at Stage3.

    Uses the same architecture as MSCA-FasterNet but with a different attention:
        - Backbone: FasterNet-T0 (embed_dim=40, depths=[1,2,8,2])
        - Attention: inserted at Stage3, last 2 blocks (index 6, 7)
        - NO cross-layer fusion (to isolate attention's contribution)
        - Classification: GAP -> Linear(160, num_classes) from Stage4 output

    Args:
        attention_type: 'se', 'cbam', 'eca', or 'sk'.
        num_classes: Number of output classes.
        attention_dim: Channel dimension at the insertion stage.
        pretrained_backbone: Path to pretrained FasterNet-T0 weights.
    """
    from .msca_fasternet import MSCAFasterNet

    # Map attention type to module class
    attention_map = {
        "se": SEAttention,
        "cbam": CBAMAttention,
        "eca": ECAAttention,
        "sk": SKAttention,
    }

    if attention_type not in attention_map:
        raise ValueError(f"Unknown attention type: {attention_type}")

    attention_cls = attention_map[attention_type]

    # Attention factory for backbone insertion
    def attention_factory(dim):
        return attention_cls(dim=dim)

    # Build with MSCA-FasterNet framework but using alternative attention
    model = MSCAFasterNet(
        num_classes=num_classes,
        embed_dim=40,
        depths=[1, 2, 8, 2],
        n_div=4,
        mlp_ratio=2.0,
        msca_stage=2,  # Stage3 (0-based)
        msca_block_indices=[6, 7],  # Last 2 blocks
        use_fusion=False,  # No fusion — isolate attention contribution
        use_msca_in_blocks=True,  # Will use attention_factory instead of MSCA
        drop_path_rate=0.05,
    )

    # Replace MSCA modules with the target attention
    # The MSCAFasterNet constructor already created MSCA modules at the right positions
    # We need to swap them out
    for name, module in model.named_modules():
        if type(module).__name__ == "MSCA":
            # Navigate to parent and replace
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            dim = module.dim
            new_module = attention_cls(dim=dim)
            setattr(parent, parts[-1], new_module)

    # Load pretrained weights
    if pretrained_backbone is not None:
        state_dict = torch.load(pretrained_backbone, map_location="cpu", weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_pretrained_backbone(state_dict)

    return model


# ============================================================
# Public model builders (matching train.py interface)
# ============================================================

def fasternet_t0_with_se(num_classes: int = 102, **kwargs):
    """FasterNet-T0 + SE-Net attention at Stage3."""
    pretrained = kwargs.pop("pretrained_backbone", None)
    return _build_attention_fasternet(
        "se", num_classes, pretrained_backbone=pretrained, **kwargs
    )


def fasternet_t0_with_cbam(num_classes: int = 102, **kwargs):
    """FasterNet-T0 + CBAM attention at Stage3."""
    pretrained = kwargs.pop("pretrained_backbone", None)
    return _build_attention_fasternet(
        "cbam", num_classes, pretrained_backbone=pretrained, **kwargs
    )


def fasternet_t0_with_eca(num_classes: int = 102, **kwargs):
    """FasterNet-T0 + ECA-Net attention at Stage3."""
    pretrained = kwargs.pop("pretrained_backbone", None)
    return _build_attention_fasternet(
        "eca", num_classes, pretrained_backbone=pretrained, **kwargs
    )


def fasternet_t0_with_sk(num_classes: int = 102, **kwargs):
    """FasterNet-T0 + SK-Net attention at Stage3."""
    pretrained = kwargs.pop("pretrained_backbone", None)
    return _build_attention_fasternet(
        "sk", num_classes, pretrained_backbone=pretrained, **kwargs
    )


# Map for train.py integration
ATTENTION_MODEL_BUILDERS = {
    "attention_se": fasternet_t0_with_se,
    "attention_cbam": fasternet_t0_with_cbam,
    "attention_eca": fasternet_t0_with_eca,
    "attention_sk": fasternet_t0_with_sk,
}
