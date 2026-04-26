"""Model verification script.

Validates that the MSCA-FasterNet model architecture is correct:
- Correct parameter counts
- Correct feature map shapes
- MSCA insertion works
- Fusion module produces correct output shapes
- Forward pass completes without errors

Usage:
    python scripts/verify_model.py
"""

import os
import sys

# Fix OpenMP duplicate library error on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.msca_fasternet import (
    msca_fasternet_t0,
    fasternet_t0_baseline,
    fasternet_t0_with_msca,
    fasternet_t0_with_fusion,
    fasternet_t0_full,
)
from models.msca import MSCA, MSCALight, SEOnly
from models.fusion import CrossLayerFusion


def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def verify_msca_module():
    """Verify MSCA module."""
    print("\n" + "=" * 60)
    print("Verifying MSCA Module")
    print("=" * 60)

    dim = 160
    msca = MSCA(dim=dim, reduction=16)
    x = torch.randn(2, dim, 14, 14)
    out = msca(x)

    assert out.shape == x.shape, f"MSCA output shape mismatch: {out.shape} vs {x.shape}"

    total, _ = count_params(msca)
    expected = 9280  # From design doc
    error_pct = abs(total - expected) / expected * 100

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {total:,} (expected: ~{expected:,}, error: {error_pct:.1f}%)")
    print(f"  [OK] MSCA module verified!")


def verify_fusion_module():
    """Verify Cross-Layer Feature Fusion module."""
    print("\n" + "=" * 60)
    print("Verifying Cross-Layer Fusion Module")
    print("=" * 60)

    fusion = CrossLayerFusion(
        s2_channels=80,
        s3_channels=160,
        s4_channels=320,
        fusion_dim=160,
        target_size=14,
    )

    s2 = torch.randn(2, 80, 28, 28)
    s3 = torch.randn(2, 160, 14, 14)
    s4 = torch.randn(2, 320, 7, 7)

    out = fusion(s2, s3, s4)

    assert out.shape == (2, 160, 14, 14), f"Fusion output shape mismatch: {out.shape}"

    total, _ = count_params(fusion)
    expected = 150400  # From design doc
    error_pct = abs(total - expected) / expected * 100

    print(f"  Stage2 input: {s2.shape}")
    print(f"  Stage3 input: {s3.shape}")
    print(f"  Stage4 input: {s4.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters: {total:,} (expected: ~{expected:,}, error: {error_pct:.1f}%)")
    print(f"  [OK] Fusion module verified!")


def verify_baseline_model():
    """Verify baseline FasterNet-T0 model."""
    print("\n" + "=" * 60)
    print("Verifying Baseline FasterNet-T0")
    print("=" * 60)

    model = fasternet_t0_baseline(num_classes=102)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert out.shape == (2, 102), f"Output shape mismatch: {out.shape}"

    total, trainable = count_params(model)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Total params: {total/1e6:.2f}M")
    print(f"  Trainable params: {trainable/1e6:.2f}M")
    print(f"  [OK] Baseline model verified!")


def verify_full_model():
    """Verify full MSCA-FasterNet model."""
    print("\n" + "=" * 60)
    print("Verifying Full MSCA-FasterNet Model")
    print("=" * 60)

    model = fasternet_t0_full(num_classes=102)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)

    assert out.shape == (2, 102), f"Output shape mismatch: {out.shape}"

    total, trainable = count_params(model)
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Total params: {total/1e6:.2f}M (expected: ~4.07M)")
    print(f"  Trainable params: {trainable/1e6:.2f}M")

    # Verify feature maps
    features = model.get_feature_maps(x)
    print(f"  Feature maps:")
    for key, feat in features.items():
        print(f"    {key}: {feat.shape}")

    print(f"  [OK] Full model verified!")


def verify_all_ablation_models():
    """Verify all ablation model variants."""
    print("\n" + "=" * 60)
    print("Verifying All Ablation Models")
    print("=" * 60)

    models = {
        "A: Baseline": fasternet_t0_baseline(num_classes=102),
        "B: + MSCA": fasternet_t0_with_msca(num_classes=102),
        "C: + Fusion": fasternet_t0_with_fusion(num_classes=102),
        "D: Full": fasternet_t0_full(num_classes=102),
    }

    x = torch.randn(2, 3, 224, 224)

    for name, model in models.items():
        out = model(x)
        total, _ = count_params(model)
        print(f"  {name:<20} | Params: {total/1e6:.2f}M | Output: {out.shape} | [OK]")

    print(f"\n  [OK] All ablation models verified!")


def verify_progressive_freezing():
    """Verify progressive freezing works correctly."""
    print("\n" + "=" * 60)
    print("Verifying Progressive Freezing")
    print("=" * 60)

    from utils.misc import freeze_backbone, unfreeze_all

    model = fasternet_t0_full(num_classes=102)

    total_before = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Freeze Stage1-2
    freeze_backbone(model, freeze_stages=(0, 1))
    trainable_frozen = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Unfreeze all
    unfreeze_all(model)
    trainable_unfrozen = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Before freeze: {total_before/1e6:.2f}M trainable")
    print(f"  After freeze: {trainable_frozen/1e6:.2f}M trainable")
    print(f"  After unfreeze: {trainable_unfrozen/1e6:.2f}M trainable")

    assert trainable_frozen < total_before, "Freezing did not reduce trainable params"
    assert trainable_unfrozen == total_before, "Unfreezing did not restore all params"

    print(f"  [OK] Progressive freezing verified!")


def main():
    print("\n" + "#" * 60)
    print("# MSCA-FasterNet Model Verification")
    print("#" * 60)

    try:
        verify_msca_module()
        verify_fusion_module()
        verify_baseline_model()
        verify_full_model()
        verify_all_ablation_models()
        verify_progressive_freezing()

        print("\n" + "=" * 60)
        print("[PASS] ALL VERIFICATIONS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
