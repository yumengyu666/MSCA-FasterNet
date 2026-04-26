"""Confusion matrix visualization for IP102 (102 classes) and PlantVillage (38 classes).

Generates:
    - Full confusion matrix heatmap
    - Zoomed-in view of most confused class pairs
    - Per-class accuracy bar chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: tuple = (20, 18),
    normalize: bool = True,
    top_k_confused: int = 10,
) -> plt.Figure:
    """Plot confusion matrix with highlighted most-confused pairs.

    Args:
        cm: Confusion matrix (num_classes, num_classes).
        class_names: List of class names.
        title: Plot title.
        save_path: Path to save figure.
        figsize: Figure size.
        normalize: Whether to normalize by row (true class).
        top_k_confused: Number of most confused pairs to annotate.

    Returns:
        matplotlib Figure.
    """
    num_classes = cm.shape[0]

    if normalize:
        cm_norm = cm.astype(np.float64)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums
    else:
        cm_norm = cm.astype(np.float64)

    # Figure 1: Full confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    sns.heatmap(
        cm_norm,
        annot=False,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        ax=ax,
        xticklabels=class_names or range(num_classes),
        yticklabels=class_names or range(num_classes),
        vmin=0,
        vmax=1 if normalize else None,
    )

    ax.set_xlabel("Predicted Label", fontsize=14)
    ax.set_ylabel("True Label", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")

    if class_names and num_classes > 20:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=6)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    # Figure 2: Most confused pairs
    confused_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                confused_pairs.append((i, j, cm_norm[i, j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = confused_pairs[:top_k_confused]

    if top_pairs and class_names:
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        labels = [f"{class_names[i]} → {class_names[j]}" for i, j, _ in top_pairs]
        values = [v for _, _, v in top_pairs]

        bars = ax2.barh(range(len(labels)), values, color="salmon", edgecolor="darkred")
        ax2.set_yticks(range(len(labels)))
        ax2.set_yticklabels(labels, fontsize=10)
        ax2.set_xlabel("Confusion Rate", fontsize=12)
        ax2.set_title(f"Top {top_k_confused} Most Confused Class Pairs",
                       fontsize=14, fontweight="bold")
        ax2.invert_yaxis()

        for bar, val in zip(bars, values):
            ax2.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=9)

        plt.tight_layout()

        confused_path = save_path.replace(".png", "_confused.png") if save_path else None
        if confused_path:
            fig2.savefig(confused_path, dpi=300, bbox_inches="tight")

    # Figure 3: Per-class accuracy
    if class_names:
        fig3, ax3 = plt.subplots(figsize=(max(12, num_classes * 0.3), 6))
        per_class_acc = np.diag(cm_norm) * 100

        colors = ["green" if acc >= 80 else "orange" if acc >= 50 else "red"
                  for acc in per_class_acc]

        ax3.bar(range(num_classes), per_class_acc, color=colors, edgecolor="gray", alpha=0.8)
        ax3.set_xticks(range(num_classes))
        ax3.set_xticklabels(class_names, rotation=90, fontsize=6)
        ax3.set_ylabel("Accuracy (%)", fontsize=12)
        ax3.set_title("Per-Class Accuracy", fontsize=14, fontweight="bold")
        ax3.axhline(y=80, color="green", linestyle="--", alpha=0.5, label="80%")
        ax3.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="50%")
        ax3.legend()

        plt.tight_layout()

        acc_path = save_path.replace(".png", "_perclass.png") if save_path else None
        if acc_path:
            fig3.savefig(acc_path, dpi=300, bbox_inches="tight")

    return fig
