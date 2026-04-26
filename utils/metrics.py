"""Evaluation metrics for model comparison."""

import torch
import time
import numpy as np
from typing import Dict, Optional


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
    """Compute model FLOPs and parameter count.

    Uses fvcore if available, falls back to manual parameter counting.

    Args:
        model: PyTorch model.
        input_size: Input tensor shape.

    Returns:
        Dictionary with params (M) and FLOPs (G).
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
        results["flops_G"] = flops / 1e9
    except ImportError:
        try:
            from thop import profile
            device = next(model.parameters()).device
            dummy_input = torch.randn(*input_size).to(device)
            model.eval()
            with torch.no_grad():
                flops, _ = profile(model, inputs=(dummy_input,), verbose=False)
            results["flops_G"] = flops / 1e9
        except ImportError:
            results["flops_G"] = -1  # Not available
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
