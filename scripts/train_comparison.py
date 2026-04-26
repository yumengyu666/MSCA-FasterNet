"""Training script for comparison models (from timm library).

Supports: MobileNetV3-Small, ShuffleNetV2-x0.5, GhostNetV2, EfficientNet-Lite0
All models use ImageNet pretrained weights and identical training settings
for fair comparison with MSCA-FasterNet.

Usage:
    python scripts/train_comparison.py --model mobilenetv3_small_100 --dataset ip102
"""

import os
import sys
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

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

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    num_classes = args.num_classes
    if num_classes is None:
        num_classes = 102 if args.dataset == "ip102" else 38

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Logger
    output_dir = os.path.join(args.output_dir, args.model)
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(name=args.model, log_dir=os.path.join(output_dir, "logs"))

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
        train_loader = build_ip102_dataloader(data_dir, "train", args.batch_size, args.workers)
        val_loader = build_ip102_dataloader(data_dir, "val", args.batch_size, args.workers, use_weighted_sampler=False)
        test_loader = build_ip102_dataloader(data_dir, "test", args.batch_size, args.workers, use_weighted_sampler=False)
    else:
        train_loader = build_plantvillage_dataloader(data_dir, "train", args.batch_size, args.workers)
        val_loader = build_plantvillage_dataloader(data_dir, "val", args.batch_size, args.workers)
        test_loader = build_plantvillage_dataloader(data_dir, "test", args.batch_size, args.workers)

    # Training setup
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=args.min_lr
    )
    scaler = GradScaler()

    best_acc = 0.0

    for epoch in range(args.epochs):
        if epoch < args.warmup_epochs:
            warmup_lr_scheduler(optimizer, args.warmup_epochs, args.lr, epoch)

        # Train
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validate
        model.eval()
        correct = total = 0
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
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
