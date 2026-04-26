"""Miscellaneous utilities: seed, checkpoint, meter, scheduler."""

import os
import random
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: dict,
    filepath: str,
    is_best: bool = False,
):
    """Save model checkpoint.

    Args:
        state: Checkpoint state dict (model, optimizer, epoch, etc.).
        filepath: Path to save checkpoint.
        is_best: If True, also save as best_model.pth.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(os.path.dirname(filepath), "best_model.pth")
        torch.save(state, best_path)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:
    """Load model checkpoint.

    Args:
        filepath: Path to checkpoint file.
        model: Model to load weights into.
        optimizer: Optimizer to load state into (optional).
        device: Device to map tensors to.

    Returns:
        Checkpoint metadata dict (epoch, best_acc, etc.).
    """
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_acc": checkpoint.get("best_acc", 0.0),
        "config": checkpoint.get("config", {}),
    }


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self, name: str = "", fmt: str = ":.4f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = f"{self.name} {self.fmt} ({self.fmt})"
        return fmtstr.format(self.val, self.avg)


class ProgressMeter:
    """Display training progress."""

    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> str:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return "\t".join(entries)

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def warmup_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    base_lr: float,
    current_epoch: int,
):
    """Linear warmup for learning rate.

    Args:
        optimizer: Model optimizer.
        warmup_epochs: Number of warmup epochs.
        base_lr: Target learning rate after warmup.
        current_epoch: Current epoch number.
    """
    if current_epoch < warmup_epochs:
        lr = base_lr * (current_epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


def get_gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Get the target layer for Grad-CAM visualization.

    For MSCA-FasterNet, we target the last block of Stage3
    (before the feature fusion).

    Args:
        model: MSCAFasterNet model.

    Returns:
        Target nn.Module for Grad-CAM.
    """
    # Try to find the last Stage3 block
    if hasattr(model, "backbone") and hasattr(model.backbone, "stages"):
        stage3 = model.backbone.stages[2]  # Stage3 is index 2
        # Last block in Stage3
        last_block = stage3.blocks[-1]
        # Target the pwconv2 (last conv before residual addition)
        return last_block.pwconv2

    # Fallback: return the last conv layer
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def freeze_backbone(model: nn.Module, freeze_stages: tuple = (0, 1)):
    """Freeze specified backbone stages for transfer learning.

    Args:
        model: MSCAFasterNet model.
        freeze_stages: Stage indices to freeze (0-based).
    """
    if hasattr(model, "backbone") and hasattr(model.backbone, "stages"):
        for stage_idx in freeze_stages:
            if stage_idx < len(model.backbone.stages):
                for param in model.backbone.stages[stage_idx].parameters():
                    param.requires_grad = False

        # Also freeze embedding
        if 0 in freeze_stages:
            for param in model.backbone.embedding.parameters():
                param.requires_grad = False

        # Freeze merging layers
        for param in model.backbone.mergings.parameters():
            param.requires_grad = False

    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Frozen stages {freeze_stages}: {trainable/1e6:.2f}M / {total/1e6:.2f}M params trainable")


def unfreeze_all(model: nn.Module):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    print("Unfrozen all parameters.")
