"""Evaluation and inference script for MSCA-FasterNet.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/ip102_full/best_model.pth --dataset ip102
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --dataset plantvillage --compute-flops
"""

import os
import sys
import argparse
import json
import numpy as np

# Fix OpenMP duplicate library error on Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.msca_fasternet import (
    msca_fasternet_t0,
    fasternet_t0_baseline,
    fasternet_t0_with_msca,
    fasternet_t0_with_fusion,
    fasternet_t0_full,
)
from datasets.ip102 import build_ip102_dataloader
from datasets.plantvillage import build_plantvillage_dataloader
from utils import (
    setup_logger,
    compute_metrics,
    compute_flops,
    measure_fps,
    compute_confusion_matrix,
    compute_f1_score,
    load_checkpoint,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MSCA-FasterNet Evaluation")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="ip102",
                        choices=["ip102", "plantvillage"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model", type=str, default="full",
                        choices=["baseline", "msca", "fusion", "full"])
    parser.add_argument("--compute-flops", action="store_true", default=True,
                        help="Compute FLOPs and parameter count")
    parser.add_argument("--no-compute-flops", action="store_true",
                        help="Skip FLOPs computation")
    parser.add_argument("--measure-fps", action="store_true", default=True,
                        help="Measure inference FPS")
    parser.add_argument("--no-measure-fps", action="store_true",
                        help="Skip FPS measurement")
    parser.add_argument("--save-predictions", action="store_true",
                        help="Save all predictions for visualization")
    parser.add_argument("--output-dir", type=str, default="results")

    return parser.parse_args()


def build_model_from_checkpoint(args, device):
    """Reconstruct model from checkpoint config."""
    # Load checkpoint to get config
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # Try to get config from checkpoint
    if "config" in ckpt:
        config = ckpt["config"]
        num_classes = config.get("num_classes",
                                102 if config.get("dataset") == "ip102" else 15)
        model_name = config.get("model", args.model)
    else:
        num_classes = 102 if args.dataset == "ip102" else 15
        model_name = args.model

    model_builders = {
        "baseline": fasternet_t0_baseline,
        "msca": fasternet_t0_with_msca,
        "fusion": fasternet_t0_with_fusion,
        "full": fasternet_t0_full,
    }

    model = model_builders[model_name](num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, num_classes, model_name


@torch.no_grad()
def evaluate(model, dataloader, device, num_classes, model_name="full"):
    """Full evaluation with per-class metrics."""
    model.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    for images, labels in tqdm(dataloader, desc="Evaluating"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_logits.append(logits.cpu())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Overall accuracy
    top1_acc = (all_preds == all_labels).mean() * 100

    # Top-5 accuracy
    all_logits = torch.cat(all_logits, dim=0)
    all_labels_t = torch.from_numpy(all_labels)
    _, top5_preds = all_logits.topk(5, dim=1)
    top5_correct = top5_preds.eq(all_labels_t.unsqueeze(1)).any(dim=1)
    top5_acc = top5_correct.float().mean().item() * 100

    # Per-class accuracy
    per_class_acc = {}
    for cls in range(num_classes):
        mask = all_labels == cls
        if mask.sum() > 0:
            cls_acc = (all_preds[mask] == all_labels[mask]).mean() * 100
            per_class_acc[cls] = cls_acc

    # Confusion matrix
    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)

    # F1-scores (critical for imbalanced datasets)
    f1_macro = compute_f1_score(all_preds, all_labels, num_classes, average="macro")
    f1_weighted = compute_f1_score(all_preds, all_labels, num_classes, average="weighted")

    # Find most confused pairs
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((i, j, cm[i, j]))
    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    results = {
        "model": model_name,
        "dataset": args.dataset,
        "top1_acc": top1_acc,
        "top5_acc": top5_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_acc_mean": np.mean(list(per_class_acc.values())),
        "per_class_acc_std": np.std(list(per_class_acc.values())),
        "most_confused_pairs": [
            {"true": int(i), "pred": int(j), "count": int(c)}
            for i, j, c in confused_pairs[:10]
        ],
    }

    return results, all_preds, all_labels, cm


def main():
    args = parse_args()

    # Logger (must be created before any logger calls)
    logger = setup_logger(name="evaluate", log_dir=args.output_dir)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        logger.warning("CUDA not available! Running on CPU!")

    # Load model
    model, num_classes, model_name = build_model_from_checkpoint(args, device)
    logger.info(f"Model: {model_name} | Classes: {num_classes}")

    # Build test dataloader
    data_dir = args.data_dir or f"data/{args.dataset.upper()}"

    if args.dataset == "ip102":
        test_loader = build_ip102_dataloader(
            root_dir=data_dir, split="test",
            batch_size=args.batch_size, num_workers=args.workers,
            use_weighted_sampler=False,
        )
    else:
        test_loader = build_plantvillage_dataloader(
            root_dir=data_dir, split="test",
            batch_size=args.batch_size, num_workers=args.workers,
        )

    # Evaluate
    results, all_preds, all_labels, cm = evaluate(model, test_loader, device, num_classes, model_name)

    logger.info(f"Top-1 Accuracy: {results['top1_acc']:.2f}%")
    logger.info(f"Top-5 Accuracy: {results['top5_acc']:.2f}%")
    logger.info(f"F1 (Macro): {results['f1_macro']:.2f}%")
    logger.info(f"F1 (Weighted): {results['f1_weighted']:.2f}%")
    logger.info(f"Per-class Acc Mean: {results['per_class_acc_mean']:.2f}%")
    logger.info(f"Per-class Acc Std: {results['per_class_acc_std']:.2f}%")

    # Compute FLOPs
    if args.compute_flops and not args.no_compute_flops:
        flop_results = compute_flops(model)
        logger.info(f"Parameters: {flop_results['params_M']:.2f}M")
        if flop_results["flops_G"] > 0:
            logger.info(f"FLOPs: {flop_results['flops_G']:.3f}G")
            if flop_results.get("macs_G", -1) > 0:
                logger.info(f"MACs: {flop_results['macs_G']:.3f}G")
        results.update(flop_results)

    # Measure FPS
    if args.measure_fps and not args.no_measure_fps:
        fps_results = measure_fps(model, device=str(device))
        logger.info(f"FPS: {fps_results['fps']:.1f} | Latency: {fps_results['latency_ms']:.2f}ms")
        results.update(fps_results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(
        args.output_dir,
        f"eval_{args.dataset}_{model_name}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save predictions and confusion matrix
    if args.save_predictions:
        np.savez(
            os.path.join(args.output_dir, f"predictions_{args.dataset}_{model_name}.npz"),
            preds=all_preds,
            labels=all_labels,
            confusion_matrix=cm,
        )

    logger.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
