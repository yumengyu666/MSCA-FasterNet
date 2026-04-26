"""Grad-CAM visualization for MSCA-FasterNet.

Generates comparison heatmaps between baseline FasterNet and MSCA-FasterNet
to demonstrate that MSCA focuses attention on pest/disease lesion regions.

Usage:
    python scripts/visualize.py --mode gradcam --checkpoint path/to/model.pth
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_layer: nn.Module,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Generate Grad-CAM heatmap.

    Args:
        model: Classification model.
        input_tensor: Input image tensor (1, 3, H, W).
        target_layer: Target convolutional layer for gradient computation.
        target_class: Target class index. If None, uses predicted class.

    Returns:
        Grad-CAM heatmap (H, W), normalized to [0, 1].
    """
    model.eval()

    # Hooks
    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    # Register hooks
    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward pass
        output = model(input_tensor)

        # Target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute Grad-CAM
        act = activations["value"]  # (1, C, h, w)
        grad = gradients["value"]   # (1, C, h, w)

        # Global average pooling of gradients
        weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination
        cam = (weights * act).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)  # ReLU to keep only positive contributions

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam = np.uint8(cam * 255)
        from PIL import Image as PILImage
        cam_img = PILImage.fromarray(cam)
        cam_img = cam_img.resize(
            (input_tensor.shape[3], input_tensor.shape[2]),
            PILImage.BILINEAR,
        )
        cam = np.array(cam_img).astype(np.float32) / 255.0

        return cam

    finally:
        fwd_handle.remove()
        bwd_handle.remove()


def overlay_cam_on_image(
    image: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = plt.cm.jet,
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on original image.

    Args:
        image: Original image (H, W, 3), normalized [0, 1].
        cam: Grad-CAM heatmap (H, W), normalized [0, 1].
        alpha: Blending factor.
        colormap: Matplotlib colormap.

    Returns:
        Blended image (H, W, 3).
    """
    heatmap = colormap(cam)[:, :, :3]  # Apply colormap, drop alpha
    blended = (1 - alpha) * image + alpha * heatmap
    blended = np.clip(blended, 0, 1)
    return blended


def generate_gradcam_comparison(
    baseline_model: nn.Module,
    improved_model: nn.Module,
    image_tensor: torch.Tensor,
    original_image: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = "",
) -> plt.Figure:
    """Generate side-by-side Grad-CAM comparison.

    Args:
        baseline_model: Original FasterNet-T0 model.
        improved_model: MSCA-FasterNet model.
        image_tensor: Preprocessed input tensor (1, 3, H, W).
        original_image: Original unnormalized image (H, W, 3) for display.
        class_names: List of class names.
        save_path: Path to save the figure.
        title: Figure title.

    Returns:
        matplotlib Figure.
    """
    # Get target layers
    baseline_target = _find_last_conv(baseline_model)
    improved_target = _find_stage3_last_conv(improved_model)

    # Generate Grad-CAMs
    baseline_cam = get_gradcam(baseline_model, image_tensor.clone(), baseline_target)
    improved_cam = get_gradcam(improved_model, image_tensor.clone(), improved_target)

    # Get predictions
    with torch.no_grad():
        baseline_pred = baseline_model(image_tensor).argmax(1).item()
        improved_pred = improved_model(image_tensor).argmax(1).item()

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original", fontsize=14)
    axes[0].axis("off")

    # Baseline Grad-CAM
    baseline_overlay = overlay_cam_on_image(original_image, baseline_cam)
    axes[1].imshow(baseline_overlay)
    pred_name = class_names[baseline_pred] if class_names else str(baseline_pred)
    axes[1].set_title(f"FasterNet-T0 (pred: {pred_name})", fontsize=14)
    axes[1].axis("off")

    # Improved Grad-CAM
    improved_overlay = overlay_cam_on_image(original_image, improved_cam)
    pred_name = class_names[improved_pred] if class_names else str(improved_pred)
    axes[2].set_title(f"MSCA-FasterNet (pred: {pred_name})", fontsize=14)
    axes[2].imshow(improved_overlay)
    axes[2].axis("off")

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Grad-CAM comparison saved to {save_path}")

    return fig


def _find_last_conv(model: nn.Module) -> nn.Module:
    """Find the last Conv2d layer in the model."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def _find_stage3_last_conv(model: nn.Module) -> nn.Module:
    """Find the last Conv2d in Stage3 of MSCA-FasterNet."""
    if hasattr(model, "backbone") and hasattr(model.backbone, "stages"):
        stage3 = model.backbone.stages[2]
        last_conv = None
        for module in stage3.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv
    return _find_last_conv(model)
