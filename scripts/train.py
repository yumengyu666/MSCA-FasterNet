"""Training script for MSCA-FasterNet.

Supports:
    - Mixed precision training (FP16)
    - Progressive unfreezing
    - Label smoothing
    - Cosine annealing with warmup
    - MixUp / CutMix data augmentation
    - Gradient clipping
    - Checkpoint saving & resuming
    - TensorBoard logging

Usage:
    python scripts/train.py --dataset ip102 --model full --epochs 150
    python scripts/train.py --dataset plantvillage --model baseline --epochs 100
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
try:
    from torch.amp import GradScaler, autocast
    _AMP_NEW_API = True
except ImportError:
    # PyTorch < 2.0 / 2.1 compatibility
    from torch.cuda.amp import GradScaler, autocast
    _AMP_NEW_API = False

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.msca_fasternet import (
    msca_fasternet_t0,
    fasternet_t0_baseline,
    fasternet_t0_with_msca,
    fasternet_t0_with_fusion,
    fasternet_t0_full,
)
from models.attention_models import ATTENTION_MODEL_BUILDERS
from datasets.ip102 import build_ip102_dataloader
from datasets.plantvillage import build_plantvillage_dataloader
from utils import (
    setup_logger,
    compute_metrics,
    set_seed,
    save_checkpoint,
    load_checkpoint,
    AverageMeter,
    ProgressMeter,
    warmup_lr_scheduler,
    freeze_backbone,
    unfreeze_all,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MSCA-FasterNet Training")

    # Dataset
    parser.add_argument("--dataset", type=str, default="ip102",
                        choices=["ip102", "plantvillage"],
                        help="Dataset name")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Dataset root directory")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HDF5 cache file path (enables ultra-fast loading)")

    # Model
    parser.add_argument("--model", type=str, default="full",
                        choices=["baseline", "msca", "fusion", "full",
                                 "attention_se", "attention_cbam", "attention_eca", "attention_sk"],
                        help="Model variant: baseline/msca/fusion/full or attention_* for comparison")
    parser.add_argument("--num-classes", type=int, default=None,
                        help="Number of classes (auto-detected if None)")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Path to pretrained FasterNet-T0 weights")
    parser.add_argument("--msca-reduction", type=int, default=16,
                        help="MSCA reduction ratio")

    # Training
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=4e-4,
                        help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=1e-6,
                        help="Minimum learning rate for cosine annealing")
    parser.add_argument("--weight-decay", type=float, default=0.005,
                        help="Weight decay")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--drop-path", type=float, default=0.05,
                        help="DropPath rate")

    # Scheduler
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Number of warmup epochs")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step"],
                        help="Learning rate scheduler")

    # Augmentation
    parser.add_argument("--mixup-alpha", type=float, default=0.2,
                        help="MixUp alpha (0 to disable)")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0,
                        help="CutMix alpha (0 to disable)")
    parser.add_argument("--mix-prob", type=float, default=0.5,
                        help="Probability of applying MixUp/CutMix")

    # Progressive unfreezing
    parser.add_argument("--freeze-epochs", type=int, default=50,
                        help="Epochs to freeze backbone (0 to disable)")
    parser.add_argument("--unfreeze-lr-factor", type=float, default=0.1,
                        help="LR multiplier when unfreezing backbone")

    # System
    parser.add_argument("--workers", type=int, default=4,
                        help="Data loading workers")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device ID")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable mixed precision")
    parser.add_argument("--clip-grad", type=float, default=5.0,
                        help="Gradient clipping max norm")

    # Output
    parser.add_argument("--output-dir", type=str, default="checkpoints",
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--save-freq", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--print-freq", type=int, default=50,
                        help="Print training stats every N batches")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--use-tensorboard", action="store_true", default=False,
                        help="Enable TensorBoard logging")

    return parser.parse_args()


def build_model(args):
    """Build model based on args."""
    num_classes = args.num_classes
    if num_classes is None:
        num_classes = 102 if args.dataset == "ip102" else 15

    # Standard ablation models
    model_builders = {
        "baseline": fasternet_t0_baseline,
        "msca": fasternet_t0_with_msca,
        "fusion": fasternet_t0_with_fusion,
        "full": fasternet_t0_full,
    }

    # Merge attention comparison models
    model_builders.update(ATTENTION_MODEL_BUILDERS)

    builder = model_builders[args.model]

    # Attention comparison models use different interface
    if args.model.startswith("attention_"):
        model = builder(
            num_classes=num_classes,
            pretrained_backbone=args.pretrained,
        )
    else:
        model = builder(
            num_classes=num_classes,
            msca_reduction=args.msca_reduction,
            pretrained_backbone=args.pretrained,
            dropout=0.0,
        )

    return model, num_classes


def build_dataloaders(args, num_classes):
    """Build train/val/test dataloaders."""
    # 如果指定了缓存目录，使用HDF5高速缓存
    if args.cache_dir:
        from datasets.hdf5_dataset import build_cached_dataloader

        print(f"[Cache] 📦 使用HDF5缓存模式: {args.cache_dir}")

        train_loader = build_cached_dataloader(
            hdf5_path=args.cache_dir,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.workers,
            input_size=224,
            use_weighted_sampler=True,
        )
        val_loader = build_cached_dataloader(
            hdf5_path=args.cache_dir,
            split="val",
            batch_size=args.batch_size,
            num_workers=args.workers,
            input_size=224,
        )
        test_loader = build_cached_dataloader(
            hdf5_path=args.cache_dir,
            split="test",
            batch_size=args.batch_size,
            num_workers=args.workers,
            input_size=224,
        )
        return train_loader, val_loader, test_loader

    # 原始数据加载方式（无缓存）
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = f"data/{args.dataset.upper()}"

    if args.dataset == "ip102":
        train_loader = build_ip102_dataloader(
            root_dir=data_dir, split="train",
            batch_size=args.batch_size, num_workers=args.workers,
            use_weighted_sampler=True,
        )
        val_loader = build_ip102_dataloader(
            root_dir=data_dir, split="val",
            batch_size=args.batch_size, num_workers=args.workers,
            use_weighted_sampler=False,
        )
        test_loader = build_ip102_dataloader(
            root_dir=data_dir, split="test",
            batch_size=args.batch_size, num_workers=args.workers,
            use_weighted_sampler=False,
        )
    else:
        train_loader = build_plantvillage_dataloader(
            root_dir=data_dir, split="train",
            batch_size=args.batch_size, num_workers=args.workers,
        )
        val_loader = build_plantvillage_dataloader(
            root_dir=data_dir, split="val",
            batch_size=args.batch_size, num_workers=args.workers,
        )
        test_loader = build_plantvillage_dataloader(
            root_dir=data_dir, split="test",
            batch_size=args.batch_size, num_workers=args.workers,
        )

    return train_loader, val_loader, test_loader


def mixup_data(x, y, alpha=0.2):
    """Apply MixUp augmentation.

    Returns mixed inputs, pairs of targets, and lambda value.
    """
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
    """Apply CutMix augmentation.

    Returns mixed inputs, pairs of targets, and lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Generate random bounding box
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

    # Adjust lambda based on actual cut area
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute MixUp/CutMix loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(
    model, criterion, optimizer, dataloader, device, epoch, args, scaler=None
):
    """Train for one epoch."""
    model.train()

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch [{epoch}/{args.epochs}]",
    )

    end = time.time()

    for i, (images, labels) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Measure data loading time
        data_time.update(time.time() - end)

        # Apply MixUp/CutMix
        use_mix = False
        if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
            if np.random.rand() < args.mix_prob:
                use_mix = True
                if np.random.rand() < 0.5 and args.mixup_alpha > 0:
                    images, labels_a, labels_b, lam = mixup_data(
                        images, labels, args.mixup_alpha
                    )
                elif args.cutmix_alpha > 0:
                    images, labels_a, labels_b, lam = cutmix_data(
                        images, labels, args.cutmix_alpha
                    )
                else:
                    use_mix = False

        # Forward + backward
        use_amp = args.amp and not args.no_amp

        if use_amp:
            if _AMP_NEW_API:
                with autocast(device_type="cuda"):
                    outputs = model(images)
            else:
                with autocast():
                    outputs = model(images)
            if use_mix:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if use_mix:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

        # Metrics
        if use_mix:
            # For mixed samples, compute weighted accuracy from both label sets
            acc1_a, acc5_a = _accuracy(outputs, labels_a, topk=(1, 5))
            acc1_b, acc5_b = _accuracy(outputs, labels_b, topk=(1, 5))
            acc1 = lam * acc1_a + (1 - lam) * acc1_b
            acc5 = lam * acc5_a + (1 - lam) * acc5_b
        else:
            acc1, acc5 = _accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(progress.display(i))

    return losses.avg, top1.avg, top5.avg


@torch.no_grad()
def validate(model, criterion, dataloader, device):
    """Validate the model."""
    model.eval()

    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        acc1, acc5 = _accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

    print(f"  * Val Acc@1 {top1.avg:.3f}  Acc@5 {top5.avg:.3f}  Loss {losses.avg:.4e}")

    return losses.avg, top1.avg, top5.avg


def _accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    args = parse_args()

    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Logger (must be created before any logger calls)
    logger = setup_logger(
        name=f"{args.dataset}_{args.model}",
        log_dir=os.path.join(args.output_dir, "logs"),
    )
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")

    # Device
    use_amp = args.amp and not args.no_amp
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)} | "
                    f"CUDA {torch.version.cuda} | "
                    f"Memory: {torch.cuda.get_device_properties(device).total_memory/1024/1024:.0f}MB")
    else:
        device = torch.device("cpu")
        use_amp = False
        logger.warning("CUDA not available! Training on CPU - this will be very slow!")

    # Model
    model, num_classes = build_model(args)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {args.model} | Params: {total_params/1e6:.2f}M | "
                f"Trainable: {trainable_params/1e6:.2f}M")

    # Dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(args, num_classes)
    logger.info(f"Dataset: {args.dataset} | Train: {len(train_loader.dataset)} | "
                f"Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr,
    )

    # AMP scaler
    scaler = GradScaler("cuda" if _AMP_NEW_API else 1) if use_amp else None

    # Resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume:
        if os.path.isfile(args.resume):
            meta = load_checkpoint(args.resume, model, optimizer, device)
            start_epoch = meta["epoch"] + 1
            best_acc = meta.get("best_acc", 0.0)
            logger.info(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.3f}")

    # Progressive freezing
    if args.freeze_epochs > 0:
        freeze_backbone(model, freeze_stages=(0, 1))
        logger.info(f"Backbone frozen for first {args.freeze_epochs} epochs")

    # TensorBoard
    writer = None
    if args.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.output_dir, "tensorboard"))
        except ImportError:
            logger.warning("TensorBoard not available. Skipping TB logging.")

    # === Training Loop ===
    logger.info("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        # Warmup
        if epoch < args.warmup_epochs:
            warmup_lr_scheduler(optimizer, args.warmup_epochs, args.lr, epoch)

        # Progressive unfreezing
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            unfreeze_all(model)
            # Reduce LR when unfreezing
            for param_group in optimizer.param_groups:
                param_group["lr"] *= args.unfreeze_lr_factor
            logger.info(f"Unfrozen backbone at epoch {epoch}, LR reduced by {args.unfreeze_lr_factor}")

        # Train
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch, args, scaler
        )

        # Validate
        val_loss, val_acc1, val_acc5 = validate(model, criterion, val_loader, device)

        # Update LR (after warmup)
        if epoch >= args.warmup_epochs:
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch [{epoch}/{args.epochs}] "
            f"Train Loss: {train_loss:.4e} Acc@1: {train_acc1:.2f} Acc@5: {train_acc5:.2f} | "
            f"Val Loss: {val_loss:.4e} Acc@1: {val_acc1:.2f} Acc@5: {val_acc5:.2f} | "
            f"LR: {current_lr:.2e}"
        )

        # TensorBoard
        if writer is not None:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("acc1", {"train": train_acc1, "val": val_acc1}, epoch)
            writer.add_scalars("acc5", {"train": train_acc5, "val": val_acc5}, epoch)
            writer.add_scalar("lr", current_lr, epoch)

        # Save checkpoint
        is_best = val_acc1 > best_acc
        best_acc = max(val_acc1, best_acc)

        if (epoch + 1) % args.save_freq == 0 or is_best:
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler else None,
                "best_acc": best_acc,
                "config": vars(args),
            }
            save_path = os.path.join(
                args.output_dir,
                f"{args.dataset}_{args.model}",
                f"checkpoint_epoch{epoch:03d}.pth",
            )
            save_checkpoint(state, save_path, is_best)

    # Final test
    logger.info("=" * 60)
    logger.info("Training complete. Running final test evaluation...")

    # Load best model
    best_model_path = os.path.join(
        args.output_dir, f"{args.dataset}_{args.model}", "best_model.pth"
    )
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model, device=device)
        logger.info(f"Loaded best model from {best_model_path}")

    test_loss, test_acc1, test_acc5 = validate(model, criterion, test_loader, device)
    logger.info(f"Final Test - Acc@1: {test_acc1:.3f} | Acc@5: {test_acc5:.3f} | Loss: {test_loss:.4e}")
    logger.info(f"Best Val Acc@1: {best_acc:.3f}")

    # Save final results
    results = {
        "dataset": args.dataset,
        "model": args.model,
        "best_val_acc1": best_acc,
        "test_acc1": test_acc1,
        "test_acc5": test_acc5,
        "test_loss": test_loss,
        "total_params_M": total_params / 1e6,
        "epochs": args.epochs,
    }
    results_path = os.path.join(
        args.output_dir, f"{args.dataset}_{args.model}", "results.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    if writer is not None:
        writer.close()

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
