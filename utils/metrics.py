"""Evaluation metrics for model comparison."""

import torch
import time
import numpy as np
from typing import Dict, Optional


def compute_f1_score(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    num_classes: int,
    average: str = "macro",
) -> float:
    """Compute F1-score.

    Args:
        all_preds: Predicted labels (N,).
        all_labels: True labels (N,).
        num_classes: Number of classes.
        average: Averaging method - 'macro', 'weighted', or 'micro'.

    Returns:
        F1-score (0-100 scale).
    """
    # Compute per-class precision, recall, F1
    f1_per_class = []
    support_per_class = []

    for cls in range(num_classes):
        tp = np.sum((all_preds == cls) & (all_labels == cls))
        fp = np.sum((all_preds == cls) & (all_labels != cls))
        fn = np.sum((all_preds != cls) & (all_labels == cls))
        support = np.sum(all_labels == cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        f1_per_class.append(f1)
        support_per_class.append(support)

    f1_per_class = np.array(f1_per_class)
    support_per_class = np.array(support_per_class)

    if average == "macro":
        return float(np.mean(f1_per_class) * 100)
    elif average == "weighted":
        total = support_per_class.sum()
        if total == 0:
            return 0.0
        return float(np.sum(f1_per_class * support_per_class / total) * 100)
    elif average == "micro":
        # Micro F1 = accuracy for single-label classification
        return float(np.mean(all_preds == all_labels) * 100)
    else:
        raise ValueError(f"Unknown average mode: {average}")


def compute_metrics(
    preds: torch.Tensor,
    labels: torch.Tensor,
    topk: tuple = (1, 5),
) -> Dict[str, float]:
    """Compute classification accuracy metrics.

    Args:
        preds: Model output logits (B, C).
        labels: Ground truth labels (B,).
        topk: Tuple of k values for top-k accuracy.

    Returns:
        Dictionary with accuracy metrics.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred_topk = preds.topk(maxk, 1, True, True)
        pred_topk = pred_topk.t()
        correct = pred_topk.eq(labels.view(1, -1).expand_as(pred_topk))

        results = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            results[f"top{k}_acc"] = (correct_k.item() / batch_size) * 100.0

        # Per-class accuracy
        results["per_class_correct"] = {}
        results["per_class_total"] = {}
        for cls_idx in labels.unique():
            mask = labels == cls_idx
            cls_correct = (preds[mask].argmax(1) == labels[mask]).sum().item()
            results["per_class_correct"][cls_idx.item()] = cls_correct
            results["per_class_total"][cls_idx.item()] = mask.sum().item()

    return results


def compute_flops(model: torch.nn.Module, input_size: tuple = (1, 3, 224, 224)) -> Dict[str, float]:
    """Compute model FLOPs, MACs, and parameter count.

    Uses fvcore if available, falls back to manual parameter counting.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.

    Returns:
        Dictionary with params (M), FLOPs (G), and MACs (G).
    """
    # Parameter count (always available)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    results = {
        "params_M": total_params / 1e6,
        "trainable_params_M": trainable_params / 1e6,
    }

    # FLOPs calculation
    try:
        from fvcore.nn import FlopCountAnalysis
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_size).to(device)
        model.eval()
        with torch.no_grad():
            flop_analysis = FlopCountAnalysis(model, dummy_input)
            flops = flop_analysis.total()
            # fvcore reports total FLOPs; MACs ≈ FLOPs / 2 for most conv ops
            macs = flop_analysis.total().item() if hasattr(flop_analysis.total(), 'item') else flop_analysis.total()
            # fvcore uses total flops (multiply-adds counted as 2 ops)
            macs_g = macs / 1e9  # This is actually total FLOPs from fvcore
        results["flops_G"] = flops / 1e9
        results["macs_G"] = macs_g
    except ImportError:
        try:
            from thop import profile
            device = next(model.parameters()).device
            dummy_input = torch.randn(*input_size).to(device)
            model.eval()
            with torch.no_grad():
                flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            results["flops_G"] = flops / 1e9
            results["macs_G"] = flops / 2 / 1e9  # thop returns MACs
        except ImportError:
            results["flops_G"] = -1  # Not available
            results["macs_G"] = -1
            print("Warning: Neither fvcore nor thop available for FLOPs calculation.")

    return results


def measure_fps(
    model: torch.nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    num_warmup: int = 50,
    num_iterations: int = 100,
    device: str = "cuda",
) -> Dict[str, float]:
    """Measure model inference speed (FPS).

    Args:
        model: PyTorch model.
        input_size: Input tensor shape (batch=1 for FPS measurement).
        num_warmup: Number of warmup iterations.
        num_iterations: Number of measurement iterations.
        device: Device for inference.

    Returns:
        Dictionary with FPS and latency metrics.
    """
    model.eval()
    model.to(device)
    dummy_input = torch.randn(*input_size).to(device)

    # GPU warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()

    # Measure
    latencies = []
    with torch.no_grad():
        for _ in range(num_iterations):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(dummy_input)

            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append(end - start)

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    fps = 1.0 / mean_latency

    return {
        "fps": fps,
        "latency_ms": mean_latency * 1000,
        "latency_std_ms": std_latency * 1000,
        "num_iterations": num_iterations,
    }


def compute_confusion_matrix(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        all_preds: Predicted labels (N,).
        all_labels: True labels (N,).
        num_classes: Number of classes.

    Returns:
        Confusion matrix (num_classes, num_classes).
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(all_preds, all_labels):
        cm[label, pred] += 1
    return cm
