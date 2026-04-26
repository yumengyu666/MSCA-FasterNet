"""t-SNE feature distribution visualization.

Compares feature distributions between baseline FasterNet and MSCA-FasterNet
to show that MSCA produces better class-discriminative features.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional, List


def extract_features(model, dataloader, device, max_samples: int = 5000):
    """Extract features from the model before classification head.

    Args:
        model: MSCAFasterNet model.
        dataloader: Data loader.
        device: Torch device.
        max_samples: Maximum number of samples for t-SNE.

    Returns:
        features: (N, D) numpy array.
        labels: (N,) numpy array.
    """
    model.eval()
    all_features = []
    all_labels = []
    count = 0

    with torch.no_grad():
        import torch
        for images, labels in dataloader:
            if count >= max_samples:
                break
            images = images.to(device)

            # Get feature maps (before GAP)
            feat_maps = model.get_feature_maps(images)

            if "fused" in feat_maps:
                feat = feat_maps["fused"]  # (B, 160, 14, 14)
            else:
                feat = feat_maps["s4"]  # (B, 320, 7, 7)

            # Global average pool
            feat = feat.mean(dim=[2, 3])  # (B, D)

            all_features.append(feat.cpu().numpy())
            all_labels.append(labels.numpy())
            count += len(labels)

    features = np.concatenate(all_features, axis=0)[:max_samples]
    labels = np.concatenate(all_labels, axis=0)[:max_samples]

    return features, labels


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "t-SNE Feature Distribution",
    save_path: Optional[str] = None,
    perplexity: int = 30,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """Plot t-SNE visualization of feature distribution.

    Args:
        features: (N, D) feature array.
        labels: (N,) label array.
        class_names: Optional class name list.
        title: Plot title.
        save_path: Path to save figure.
        perplexity: t-SNE perplexity parameter.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    print(f"Running t-SNE on {features.shape[0]} samples, {features.shape[1]} dims...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    embedded = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)

    # Use a colormap with enough distinct colors
    cmap = plt.cm.get_cmap("tab20" if num_classes <= 20 else "nipy_spectral",
                            num_classes)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = class_names[label] if class_names and label < len(class_names) else str(label)
        ax.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=[cmap(i)],
            label=name,
            s=8,
            alpha=0.6,
        )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("t-SNE Dim 1", fontsize=12)
    ax.set_ylabel("t-SNE Dim 2", fontsize=12)

    # Legend (outside plot for many classes)
    if num_classes <= 20:
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8,
                  markerscale=2)
    else:
        # For many classes, skip legend
        ax.text(0.02, 0.98, f"{num_classes} classes",
                transform=ax.transAxes, fontsize=10, va="top")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"t-SNE plot saved to {save_path}")

    return fig


def plot_tsne_comparison(
    features_baseline: np.ndarray,
    features_improved: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Side-by-side t-SNE comparison of baseline vs improved model features.

    Args:
        features_baseline: Baseline model features (N, D).
        features_improved: Improved model features (N, D).
        labels: Ground truth labels (N,).
        class_names: Class name list.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    print("Computing t-SNE for baseline features...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    emb_baseline = tsne.fit_transform(features_baseline)

    print("Computing t-SNE for improved features...")
    tsne2 = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    emb_improved = tsne2.fit_transform(features_improved)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    cmap = plt.cm.get_cmap("tab20" if num_classes <= 20 else "nipy_spectral",
                            num_classes)

    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = cmap(i)
        name = class_names[label] if class_names and label < len(class_names) else str(label)

        ax1.scatter(emb_baseline[mask, 0], emb_baseline[mask, 1],
                    c=[color], label=name, s=8, alpha=0.6)
        ax2.scatter(emb_improved[mask, 0], emb_improved[mask, 1],
                    c=[color], label=name, s=8, alpha=0.6)

    ax1.set_title("FasterNet-T0 Baseline", fontsize=14, fontweight="bold")
    ax2.set_title("MSCA-FasterNet (Ours)", fontsize=14, fontweight="bold")

    for ax in [ax1, ax2]:
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"t-SNE comparison saved to {save_path}")

    return fig
