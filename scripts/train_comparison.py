"""Training script for comparison models (from timm library).

Supports: MobileNetV3-Small, ShuffleNetV2-x0.5, GhostNetV2, EfficientNet-Lite0
All models use ImageNet pretrained weights and identical training settings
for fair comparison with MSCA-FasterNet.

Uses the SAME training pipeline as the main train.py:
    - MixUp / CutMix data augmentation
    - Gradient clipping
    - Progressive unfreezing
    - AMP with proper scaler

Usage:
    python scripts/train_comparison.py --model mobilenetv3_small_100 --dataset ip102
"""

import os
import sys
import argparse
import json
import time

# Fix OpenMP duplicate library error on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.ip102 import build_ip102_dataloader
from datasets.plantvillage import build_plantvillage_dataloader
from utils import (
    setup_logger,
    set_seed,
    save_checkpoint,
    AverageMeter,
    ProgressMeter,
    warmup_lr_scheduler,
    freeze_backbone,
    unfreeze_all,
)


# timm model name mapping
TIMM_MODEL_MAP = {
    "mobilenetv3_small_100": "mobilenetv3_small_100",
    "shufflenetv2_x0.5": "shufflenetv2_x0_5",
    "ghostnetv2_100": "ghostnetv2_100",
    "efficientnet_lite0": "efficientnet_lite0",
}


def build_comparison_model(model_name, num_classes, pretrained=True):
    """Build comparison model from timm library."""
    try:
        import timm
    except ImportError:
        raise ImportError("timm library is required. Install with: pip install timm")

    timm_name = TIMM_MODEL_MAP.get(model_name, model_name)

    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Train Comparison Models")

    parser.add_argument("--model", type=str, required=True,
                        choices=list(TIMM_MODEL_MAP.keys()))
    parser.add_argument("--dataset", type=str, default="ip102",
                        choices=["ip102", "plantvillage"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.005)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints/comparison")
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--no-pretrained", action="store_true")

    # Same augmentation args as train.py for fair comparison
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="MixUp alpha (0 to disable)")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0,
                        help="CutMix alpha (0 to disable)")
    parser.add_argument("--mix-prob", type=float, default=0.5,
                        help="Probability of applying MixUp/CutMix")
    parser.add_argument("--clip-grad", type=float, default=5.0,
                        help="Gradient clipping max norm")
    parser.add_argument("--freeze-epochs", type=int, default=50,
                        help="Epochs to freeze backbone (0 to disable)")

    return parser.parse_args()


# === MixUp / CutMix (same implementation as train.py) ===

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    W, H = x.size(2), x.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    mixed_x = x.clone()
    mixed_x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def main():
    args = parse_args()
    set_seed(args.seed)

    num_classes = args.num_classes
    if num_classes is None:
        num_classes = 102 if args.dataset == "ip102" else 15

    # Logger (must be created before any logger calls)
    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(name=args.model, log_dir=os.path.join(output_dir, "logs"))

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logger.warning("CUDA not available! Running on CPU!")

    # Model
    model = build_comparison_model(
        args.model, num_classes,
        pretrained=not args.no_pretrained,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {args.model} | Params: {total_params/1e6:.2f}M")

    # Data
    data_dir = args.data_dir or f"data/{args.dataset.upper()}"
    if args.dataset == "ip102":
        train_loader = build_ip102_dataloader(data_dir, "train", args.batch_size, args.workers,
                                               use_weighted_sampler=True)
        val_loader = build_ip102_dataloader(data_dir, "val", args.batch_size, args.workers,
                                             use_weighted_sampler=False)
        test_loader = build_ip102_dataloader(data_dir, "test", args.batch_size, args.workers,
                                              use_weighted_sampler=False)
    else:
        train_loader = build_plantvillage_dataloader(data_dir, "train", args.batch_size, args.workers)
        val_loader = build_plantvillage_dataloader(data_dir, "val", args.batch_size, args.workers)
        test_loader = build_plantvillage_dataloader(data_dir, "test", args.batch_size, args.workers)

    # Training setup (identical to main train.py)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr
    )
    use_cuda = device.type == "cuda"
    scaler = GradScaler("cuda") if use_cuda else None

    # Progressive freezing (safe for any model architecture)
    if args.freeze_epochs > 0:
        # Only freeze parameters, don't assume specific architecture
        frozen_count = 0
        total_count = 0
        for name, param in model.named_parameters():
            total_count += 1
            # Freeze early feature extraction layers
            # For timm models: features.0, features.1, etc.
            # For our models: backbone.stages.0, backbone.stages.1
            if any(name.startswith(prefix) for prefix in [
                "features.0", "features.1", "features.2",
                "conv_stem",
                "backbone.stages.0", "backbone.stages.1",
                "backbone.embedding",
            ]):
                param.requires_grad = False
                frozen_count += 1
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"Frozen {frozen_count}/{total_count} layers for first {args.freeze_epochs} epochs")
        logger.info(f"Trainable: {trainable/1e6:.2f}M / {total/1e6:.2f}M params")

    best_acc = 0.0

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warmup_lr_scheduler(optimizer, args.warmup_epochs, args.lr, epoch)

        # Progressive unfreezing (safe for any model)
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            for param in model.parameters():
                param.requires_grad = True
            for param_group in optimizer.param_groups:
                param_group["lr"] *= 0.1
            logger.info(f"Unfrozen all layers at epoch {epoch}")

        # Train
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Apply MixUp/CutMix (same as main train.py)
            use_mix = False
            if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
                if np.random.rand() < args.mix_prob:
                    use_mix = True
                    if np.random.rand() < 0.5 and args.mixup_alpha > 0:
                        images, labels_a, labels_b, lam = mixup_data(images, labels, args.mixup_alpha)
                    elif args.cutmix_alpha > 0:
                        images, labels_a, labels_b, lam = cutmix_data(images, labels, args.cutmix_alpha)
                    else:
                        use_mix = False

            with autocast(device_type="cuda", enabled=use_cuda):
                outputs = model(images)
                if use_mix:
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    loss = criterion(outputs, labels)

            optimizer.zero_grad()
            if use_cuda and scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = 100.0 * correct / total

        if epoch >= args.warmup_epochs:
            scheduler.step()

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        if (epoch + 1) % 10 == 0 or is_best:
            logger.info(f"Epoch [{epoch}/{args.epochs}] Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%")
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
            }
            save_checkpoint(state, os.path.join(output_dir, f"ckpt_e{epoch}.pth"), is_best)

    # Final test
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = 100.0 * correct / total
    logger.info(f"Test Acc: {test_acc:.2f}%")

    results = {
        "model": args.model,
        "dataset": args.dataset,
        "test_acc1": test_acc,
        "best_val_acc1": best_acc,
        "total_params_M": total_params / 1e6,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
