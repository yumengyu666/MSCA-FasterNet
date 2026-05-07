"""MSCA Attention Weight Visualization.

Extracts and visualizes:
    1. Scale attention weights (3x3 vs 5x5 selection probability per sample)
    2. Channel attention weights (per-channel importance from SE branch)
    3. Scale preference distribution across classes
    4. Top-K channel importance heatmap

These visualizations demonstrate:
    - MSCA's adaptive scale selection behavior on different pest/disease types
    - Channel-wise feature importance patterns
    - Correlation between lesion size and scale attention

Usage:
    python scripts/visualize.py --checkpoint <path> --dataset ip102 --vis-msca
    # Or standalone:
    python -m visualization.msca_weights --checkpoint <path> --dataset ip102
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.msca_fasternet import (
    msca_fasternet_t0,
    fasternet_t0_baseline,
    fasternet_t0_with_msca,
    fasternet_t0_with_fusion,
    fasternet_t0_full,
)


class MSCAWeightExtractor:
    """Extract scale and channel attention weights from MSCA modules via hooks.

    Registers forward hooks on all MSCA modules inside the model to capture
    intermediate attention weights during inference.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = []
        self.scale_weights = {}  # {module_name: tensor}
        self.channel_weights = {}  # {module_name: tensor}
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on all MSCA modules."""
        for name, module in self.model.named_modules():
            module_type = type(module).__name__
            if module_type == "MSCA":
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, name: str):
        """Create a hook that captures scale and channel attention weights."""
        def hook_fn(module, input, output):
            # We need to capture the weights BEFORE the forward computation
            # Since MSCA.forward computes them internally, we re-extract from
            # the module's sub-components
            x = input[0]

            # Extract scale attention weights
            with torch.no_grad():
                scale_w = module.scale_attention(x)  # (B, 2)
                channel_w = module.channel_attention(x)  # (B, C, 1, 1)

            self.scale_weights[name] = scale_w.detach().cpu()
            self.channel_weights[name] = channel_w.detach().cpu()

        return hook_fn

    def extract(self, dataloader, device, max_batches: int = None) -> Dict:
        """Extract attention weights for an entire dataset.

        Returns:
            Dict with keys:
                'scale_weights': {module_name: np.ndarray (N, 2)}
                'channel_weights': {module_name: np.ndarray (N, C)}
                'labels': np.ndarray (N,)
        """
        self.model.eval()
        all_scale = {k: [] for k in self.scale_weights} if self.scale_weights else None
        all_channel = {k: [] for k in self.channel_weights} if self.channel_weights else None
        all_labels = []

        # First pass to initialize keys
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting MSCA weights")):
                if max_batches and batch_idx >= max_batches:
                    break

                images = images.to(device, non_blocking=True)
                _ = self.model(images)

                # Collect weights from this batch
                if all_scale is None:
                    all_scale = {k: [] for k in self.scale_weights}
                    all_channel = {k: [] for k in self.channel_weights}

                for name in self.scale_weights:
                    all_scale[name].append(self.scale_weights[name].numpy())
                for name in self.channel_weights:
                    all_channel[name].append(self.channel_weights[name].squeeze(-1).squeeze(-1).numpy())

                all_labels.append(labels.numpy())

        # Concatenate
        result = {
            "scale_weights": {},
            "channel_weights": {},
            "labels": np.concatenate(all_labels),
        }

        for name in all_scale:
            result["scale_weights"][name] = np.concatenate(all_scale[name], axis=0)
        for name in all_channel:
            result["channel_weights"][name] = np.concatenate(all_channel[name], axis=0)

        return result

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def plot_scale_attention_distribution(
    scale_weights: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[Dict[int, str]] = None,
    save_path: str = "msca_scale_attention.png",
    top_k_classes: int = 20,
):
    """Visualize scale attention weight distribution.

    Creates a 2-panel figure:
        Left:  Overall distribution of 3x3 vs 5x5 weights across all samples
        Right: Per-class mean scale weight (top-K classes by sample count)

    Args:
        scale_weights: (N, 2) array of [w_3x3, w_5x5] per sample.
        labels: (N,) class labels.
        class_names: Optional mapping from class index to name.
        save_path: Output file path.
        top_k_classes: Number of classes to show in per-class plot.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel 1: Overall distribution ---
    ax = axes[0]
    ax.hist(scale_weights[:, 0], bins=50, alpha=0.6, label="3x3 weight", color="#2196F3")
    ax.hist(scale_weights[:, 1], bins=50, alpha=0.6, label="5x5 weight", color="#FF5722")
    ax.set_xlabel("Attention Weight", fontsize=12)
    ax.set_ylabel("Sample Count", fontsize=12)
    ax.set_title("MSCA Scale Attention Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Per-class mean scale weight ---
    ax = axes[1]
    unique_labels = np.unique(labels)
    class_means = []
    class_counts = []

    for cls in unique_labels:
        mask = labels == cls
        cls_weights = scale_weights[mask]
        class_means.append(cls_weights.mean(axis=0))  # [w_3x3, w_5x5]
        class_counts.append(mask.sum())

    class_means = np.array(class_means)  # (num_classes, 2)
    class_counts = np.array(class_counts)

    # Sort by sample count, take top-K
    sorted_idx = np.argsort(-class_counts)[:top_k_classes]
    sorted_means = class_means[sorted_idx]
    sorted_counts = class_counts[sorted_idx]

    x = np.arange(len(sorted_idx))
    width = 0.35

    if class_names:
        tick_labels = [class_names.get(int(unique_labels[i]), f"C{int(unique_labels[i])}")
                       for i in sorted_idx]
    else:
        tick_labels = [f"C{int(unique_labels[i])}" for i in sorted_idx]

    ax.bar(x - width/2, sorted_means[:, 0], width, label="3x3", color="#2196F3", alpha=0.8)
    ax.bar(x + width/2, sorted_means[:, 1], width, label="5x5", color="#FF5722", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Mean Attention Weight", fontsize=12)
    ax.set_title(f"Per-Class Scale Preference (Top {top_k_classes})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scale attention plot saved to {save_path}")


def plot_channel_attention_heatmap(
    channel_weights: np.ndarray,
    labels: np.ndarray,
    save_path: str = "msca_channel_attention.png",
    top_k_channels: int = 40,
    top_k_classes: int = 15,
):
    """Visualize channel attention as a class-vs-channel heatmap.

    Creates a 2-panel figure:
        Left:  Per-class mean channel attention (top-K channels by variance)
        Right: Channel attention variance across classes (shows discriminative channels)

    Args:
        channel_weights: (N, C) array of channel attention weights per sample.
        labels: (N,) class labels.
        save_path: Output file path.
        top_k_channels: Number of channels to show.
        top_k_classes: Number of classes to show.
    """
    C = channel_weights.shape[1]
    unique_labels = np.unique(labels)

    # Select top-K classes by sample count
    class_counts = {cls: (labels == cls).sum() for cls in unique_labels}
    top_classes = sorted(class_counts, key=class_counts.get, reverse=True)[:top_k_classes]

    # Compute per-class mean channel weights
    class_channel_means = np.zeros((len(top_classes), C))
    for i, cls in enumerate(top_classes):
        mask = labels == cls
        class_channel_means[i] = channel_weights[mask].mean(axis=0)

    # Select top-K channels by variance across classes (most discriminative)
    channel_variance = class_channel_means.var(axis=0)
    top_ch_idx = np.argsort(-channel_variance)[:top_k_channels]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel 1: Heatmap ---
    ax = axes[0]
    heatmap_data = class_channel_means[:, top_ch_idx]
    im = ax.imshow(heatmap_data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel(f"Channel Index (top {top_k_channels} by variance)", fontsize=11)
    ax.set_ylabel("Class", fontsize=11)
    ax.set_yticks(range(len(top_classes)))
    ax.set_yticklabels([f"C{int(c)}" for c in top_classes], fontsize=8)
    ax.set_xticks(range(0, top_k_channels, 5))
    ax.set_xticklabels([str(int(top_ch_idx[i])) for i in range(0, top_k_channels, 5)], fontsize=8)
    ax.set_title("Channel Attention Heatmap", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Mean Channel Weight")

    # --- Panel 2: Channel variance ---
    ax = axes[1]
    sorted_var = np.sort(channel_variance)[::-1][:top_k_channels]
    ax.bar(range(len(sorted_var)), sorted_var, color="#4CAF50", alpha=0.8)
    ax.set_xlabel("Channel Rank", fontsize=12)
    ax.set_ylabel("Inter-Class Variance", fontsize=12)
    ax.set_title("Channel Discriminability", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Channel attention plot saved to {save_path}")


def plot_scale_vs_class_scatter(
    scale_weights: np.ndarray,
    labels: np.ndarray,
    save_path: str = "msca_scale_vs_class.png",
):
    """Scatter plot of 3x3 vs 5x5 weight colored by class.

    Each point is one sample. Color represents class.
    This shows whether MSCA forms class-dependent scale preference clusters.

    Args:
        scale_weights: (N, 2) array.
        labels: (N,) class labels.
        save_path: Output file path.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    unique_labels = np.unique(labels)
    colors = cm.get_cmap("tab20", len(unique_labels))

    for i, cls in enumerate(unique_labels):
        mask = labels == cls
        ax.scatter(
            scale_weights[mask, 0],
            scale_weights[mask, 1],
            c=[colors(i)],
            label=f"C{int(cls)}",
            alpha=0.5,
            s=10,
        )

    ax.plot([0, 1], [1, 0], "k--", alpha=0.3, label="Equal preference")
    ax.set_xlabel("3x3 Scale Weight", fontsize=12)
    ax.set_ylabel("5x5 Scale Weight", fontsize=12)
    ax.set_title("MSCA Scale Preference by Class", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Show legend for first 10 classes only (avoid clutter)
    handles, lbls = ax.get_legend_handles_labels()
    n_show = min(10, len(handles))
    ax.legend(handles[:n_show+1], lbls[:n_show+1], fontsize=8, loc="upper right",
              markerscale=2, ncol=2)

    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scale vs class scatter saved to {save_path}")


def visualize_msca_weights(
    checkpoint_path: str,
    dataset: str = "ip102",
    data_dir: str = None,
    model_name: str = "full",
    output_dir: str = "results/msca_weights",
    batch_size: int = 64,
    gpu: str = "0",
    max_batches: int = None,
):
    """Main function: extract MSCA weights and generate all visualizations."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    num_classes = 102 if dataset == "ip102" else 15
    model_builders = {
        "baseline": fasternet_t0_baseline,
        "msca": fasternet_t0_with_msca,
        "fusion": fasternet_t0_with_fusion,
        "full": fasternet_t0_full,
    }

    if model_name not in model_builders:
        print(f"Warning: model_name={model_name} has no MSCA. Use 'msca' or 'full'.")
        return

    model = model_builders[model_name](num_classes=num_classes)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()

    # Build dataloader (test split)
    if data_dir is None:
        data_dir = f"data/{dataset.upper()}"

    if dataset == "ip102":
        from datasets.ip102 import build_ip102_dataloader
        loader = build_ip102_dataloader(data_dir, "test", batch_size, 4, False)
    else:
        from datasets.plantvillage import build_plantvillage_dataloader
        loader = build_plantvillage_dataloader(data_dir, "test", batch_size, 4)

    # Extract weights
    extractor = MSCAWeightExtractor(model)
    weights = extractor.extract(loader, device, max_batches=max_batches)
    extractor.remove_hooks()

    os.makedirs(output_dir, exist_ok=True)

    # Generate all plots for each MSCA module
    for name in weights["scale_weights"]:
        safe_name = name.replace(".", "_")
        scale_w = weights["scale_weights"][name]
        labels = weights["labels"]

        print(f"\nGenerating plots for {name}...")

        plot_scale_attention_distribution(
            scale_w, labels,
            save_path=os.path.join(output_dir, f"{safe_name}_scale_distribution.png"),
        )

        plot_scale_vs_class_scatter(
            scale_w, labels,
            save_path=os.path.join(output_dir, f"{safe_name}_scale_scatter.png"),
        )

    for name in weights["channel_weights"]:
        safe_name = name.replace(".", "_")
        channel_w = weights["channel_weights"][name]
        labels = weights["labels"]

        plot_channel_attention_heatmap(
            channel_w, labels,
            save_path=os.path.join(output_dir, f"{safe_name}_channel_heatmap.png"),
        )

    # Save raw weights for further analysis
    np.savez(
        os.path.join(output_dir, "msca_weights.npz"),
        scale_weights=weights["scale_weights"],
        channel_weights=weights["channel_weights"],
        labels=weights["labels"],
    )
    print(f"\nAll MSCA weight visualizations saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MSCA Weight Visualization")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="ip102", choices=["ip102", "plantvillage"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="full", choices=["msca", "fusion", "full"])
    parser.add_argument("--output-dir", type=str, default="results/msca_weights")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--max-batches", type=int, default=None)

    args = parser.parse_args()
    visualize_msca_weights(
        checkpoint_path=args.checkpoint,
        dataset=args.dataset,
        data_dir=args.data_dir,
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gpu=args.gpu,
        max_batches=args.max_batches,
    )
