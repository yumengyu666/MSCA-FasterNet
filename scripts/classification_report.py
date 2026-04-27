"""Comprehensive Precision/Recall/F1 classification report.

Generates publication-ready per-class and overall metrics tables for SCI papers.
Includes:
    - Per-class Precision, Recall, F1-score with support counts
    - Macro/Weighted/Micro averages
    - LaTeX table export for paper
    - CSV export for data analysis
    - Publication-ready figure

Usage:
    python scripts/classification_report.py --checkpoint <path> --dataset ip102
    python scripts/classification_report.py --checkpoint <path> --dataset plantvillage --output-dir results/report
"""

import os
import sys
import argparse
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from tqdm import tqdm

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
from utils import compute_confusion_matrix


def compute_classification_report(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    num_classes: int,
    class_names: Optional[Dict[int, str]] = None,
) -> Dict:
    """Compute comprehensive per-class and overall classification metrics.

    This is the core function that produces all P/R/F1 data for the paper.

    Args:
        all_preds: Predicted labels (N,).
        all_labels: True labels (N,).
        num_classes: Number of classes.
        class_names: Optional mapping from class index to name.

    Returns:
        Dict with keys:
            'per_class': List of dicts per class with precision, recall, f1, support
            'macro_avg': Dict with macro-averaged metrics
            'weighted_avg': Dict with weighted-averaged metrics
            'micro_avg': Dict with micro-averaged metrics (= accuracy)
            'overall_accuracy': float
            'confusion_matrix': np.ndarray
    """
    # Per-class metrics
    per_class = []
    tp_total = fp_total = fn_total = 0
    f1_list = []
    prec_list = []
    rec_list = []
    support_list = []

    for cls in range(num_classes):
        tp = int(np.sum((all_preds == cls) & (all_labels == cls)))
        fp = int(np.sum((all_preds == cls) & (all_labels != cls)))
        fn = int(np.sum((all_preds != cls) & (all_labels == cls)))
        support = int(np.sum(all_labels == cls))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class.append({
            "class_id": cls,
            "class_name": class_names.get(cls, f"class_{cls}") if class_names else f"class_{cls}",
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

        tp_total += tp
        fp_total += fp
        fn_total += fn
        f1_list.append(f1)
        prec_list.append(precision)
        rec_list.append(recall)
        support_list.append(support)

    # Macro average (unweighted mean of per-class metrics)
    macro_avg = {
        "precision": round(float(np.mean(prec_list)), 4),
        "recall": round(float(np.mean(rec_list)), 4),
        "f1_score": round(float(np.mean(f1_list)), 4),
        "support": int(np.sum(support_list)),
    }

    # Weighted average (weighted by support)
    total_support = sum(support_list)
    if total_support > 0:
        weights = np.array(support_list) / total_support
        weighted_avg = {
            "precision": round(float(np.sum(np.array(prec_list) * weights)), 4),
            "recall": round(float(np.sum(np.array(rec_list) * weights)), 4),
            "f1_score": round(float(np.sum(np.array(f1_list) * weights)), 4),
            "support": total_support,
        }
    else:
        weighted_avg = {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "support": 0}

    # Micro average (= overall accuracy for single-label classification)
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0.0

    micro_avg = {
        "precision": round(micro_precision, 4),
        "recall": round(micro_recall, 4),
        "f1_score": round(micro_f1, 4),
        "support": total_support,
    }

    # Overall accuracy
    accuracy = float(np.mean(all_preds == all_labels))

    # Confusion matrix
    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)

    return {
        "per_class": per_class,
        "macro_avg": macro_avg,
        "weighted_avg": weighted_avg,
        "micro_avg": micro_avg,
        "overall_accuracy": round(accuracy, 4),
        "confusion_matrix": cm.tolist(),
    }


def report_to_latex(
    report: Dict,
    caption: str = "Classification Results",
    label: str = "tab:classification",
    top_k: Optional[int] = None,
    show_all: bool = False,
) -> str:
    """Convert classification report to LaTeX table for paper.

    Args:
        report: Output from compute_classification_report.
        caption: Table caption.
        label: LaTeX label.
        top_k: Only show top-K classes by F1 (None = all).
        show_all: If True, show all classes; otherwise show summary only.

    Returns:
        LaTeX table string.
    """
    per_class = report["per_class"]

    if top_k is not None and not show_all:
        per_class = sorted(per_class, key=lambda x: -x["f1_score"])[:top_k]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{" + caption + "}")
    lines.append(r"  \label{" + label + "}")
    lines.append(r"  \begin{tabular}{lcccc}")
    lines.append(r"    \toprule")
    lines.append(r"    Class & Precision & Recall & F1-Score & Support \\")
    lines.append(r"    \midrule")

    for cls in per_class:
        lines.append(
            f"    {cls['class_name']} & {cls['precision']:.2f} & {cls['recall']:.2f} "
            f"& {cls['f1_score']:.2f} & {cls['support']} \\\\"
        )

    lines.append(r"    \midrule")
    for avg_name, avg_data in [("Macro", report["macro_avg"]),
                               ("Weighted", report["weighted_avg"])]:
        lines.append(
            f"    {avg_name} Avg & {avg_data['precision']:.2f} & {avg_data['recall']:.2f} "
            f"& {avg_data['f1_score']:.2f} & {avg_data['support']} \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def report_to_csv(report: Dict, include_per_class: bool = True) -> str:
    """Convert classification report to CSV string."""
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)

    # Summary header
    writer.writerow(["Metric", "Precision", "Recall", "F1-Score", "Support"])
    writer.writerow([
        "Overall Accuracy", "-", "-", f"{report['overall_accuracy']:.4f}",
        report["macro_avg"]["support"],
    ])
    writer.writerow(["Macro Avg", report["macro_avg"]["precision"],
                     report["macro_avg"]["recall"], report["macro_avg"]["f1_score"],
                     report["macro_avg"]["support"]])
    writer.writerow(["Weighted Avg", report["weighted_avg"]["precision"],
                     report["weighted_avg"]["recall"], report["weighted_avg"]["f1_score"],
                     report["weighted_avg"]["support"]])
    writer.writerow(["Micro Avg", report["micro_avg"]["precision"],
                     report["micro_avg"]["recall"], report["micro_avg"]["f1_score"],
                     report["micro_avg"]["support"]])

    if include_per_class:
        writer.writerow([])
        writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support", "TP", "FP", "FN"])
        for cls in report["per_class"]:
            writer.writerow([
                cls["class_name"], cls["precision"], cls["recall"],
                cls["f1_score"], cls["support"], cls["tp"], cls["fp"], cls["fn"],
            ])

    return output.getvalue()


def plot_classification_report(
    report: Dict,
    save_path: str = "classification_report.png",
    top_k_classes: int = 20,
):
    """Generate publication-ready per-class P/R/F1 bar chart.

    Args:
        report: Output from compute_classification_report.
        save_path: Output file path.
        top_k_classes: Number of classes to show (sorted by F1).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_class = sorted(report["per_class"], key=lambda x: -x["f1_score"])[:top_k_classes]

    names = [c["class_name"] for c in per_class]
    precision = [c["precision"] * 100 for c in per_class]
    recall = [c["recall"] * 100 for c in per_class]
    f1 = [c["f1_score"] * 100 for c in per_class]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(names) * 0.5), 6))

    ax.bar(x - width, precision, width, label="Precision", color="#2196F3", alpha=0.85)
    ax.bar(x, recall, width, label="Recall", color="#4CAF50", alpha=0.85)
    ax.bar(x + width, f1, width, label="F1-Score", color="#FF9800", alpha=0.85)

    # Add average lines
    ax.axhline(y=report["macro_avg"]["f1_score"] * 100, color="red", linestyle="--",
               alpha=0.5, label=f"Macro F1 = {report['macro_avg']['f1_score']*100:.1f}%")

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Per-Class Classification Metrics", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Classification report plot saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Classification Report Generator")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, default="ip102",
                        choices=["ip102", "plantvillage"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default="full",
                        choices=["baseline", "msca", "fusion", "full",
                                 "attention_se", "attention_cbam", "attention_eca", "attention_sk"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--output-dir", type=str, default="results/classification_report")
    parser.add_argument("--top-k", type=int, default=20,
                        help="Top-K classes to show in LaTeX table and plot")
    parser.add_argument("--latex", action="store_true",
                        help="Export LaTeX table")
    parser.add_argument("--class-names", type=str, default=None,
                        help="JSON file mapping class IDs to names")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Load model
    num_classes = 102 if args.dataset == "ip102" else 15

    model_builders = {
        "baseline": fasternet_t0_baseline,
        "msca": fasternet_t0_with_msca,
        "fusion": fasternet_t0_with_fusion,
        "full": fasternet_t0_full,
    }
    model_builders.update(ATTENTION_MODEL_BUILDERS)

    model = model_builders[args.model](num_classes=num_classes)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model = model.to(device)
    model.eval()

    # Load class names
    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names) as f:
            class_names = json.load(f)
            class_names = {int(k): v for k, v in class_names.items()}

    # Build dataloader
    data_dir = args.data_dir or f"data/{args.dataset.upper()}"
    if args.dataset == "ip102":
        loader = build_ip102_dataloader(data_dir, "test", args.batch_size, args.workers, False)
    else:
        loader = build_plantvillage_dataloader(data_dir, "test", args.batch_size, args.workers)

    # Run inference
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Compute report
    report = compute_classification_report(all_preds, all_labels, num_classes, class_names)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # JSON
    json_report = {k: v for k, v in report.items() if k != "confusion_matrix"}
    json_report["model"] = args.model
    json_report["dataset"] = args.dataset
    with open(os.path.join(args.output_dir, f"report_{args.dataset}_{args.model}.json"), "w") as f:
        json.dump(json_report, f, indent=2)

    # CSV
    csv_str = report_to_csv(report)
    with open(os.path.join(args.output_dir, f"report_{args.dataset}_{args.model}.csv"), "w") as f:
        f.write(csv_str)

    # LaTeX
    if args.latex:
        latex_str = report_to_latex(
            report,
            caption=f"Classification results on {args.dataset.upper()}",
            label=f"tab:{args.dataset}_results",
            top_k=args.top_k,
        )
        with open(os.path.join(args.output_dir, f"report_{args.dataset}_{args.model}.tex"), "w") as f:
            f.write(latex_str)

    # Plot
    plot_classification_report(
        report,
        save_path=os.path.join(args.output_dir, f"report_{args.dataset}_{args.model}.png"),
        top_k_classes=args.top_k,
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"Classification Report: {args.model} on {args.dataset}")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {report['overall_accuracy']*100:.2f}%")
    print(f"Macro Avg  — P: {report['macro_avg']['precision']*100:.2f}%  "
          f"R: {report['macro_avg']['recall']*100:.2f}%  "
          f"F1: {report['macro_avg']['f1_score']*100:.2f}%")
    print(f"Weighted Avg — P: {report['weighted_avg']['precision']*100:.2f}%  "
          f"R: {report['weighted_avg']['recall']*100:.2f}%  "
          f"F1: {report['weighted_avg']['f1_score']*100:.2f}%")
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
