"""Unified visualization script.

Usage:
    python scripts/visualize.py --mode gradcam --checkpoint path/to/model.pth --dataset ip102
    python scripts/visualize.py --mode confusion --checkpoint path/to/model.pth --dataset ip102
    python scripts/visualize.py --mode tsne --checkpoint-baseline path/to/baseline.pth --checkpoint-improved path/to/improved.pth
    python scripts/visualize.py --mode all --checkpoint path/to/model.pth --dataset ip102
"""

import os
import sys
import argparse

import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.msca_fasternet import (
    msca_fasternet_t0,
    fasternet_t0_baseline,
    fasternet_t0_full,
)
from datasets.ip102 import IP102Dataset, build_ip102_dataloader
from datasets.plantvillage import PlantVillageDataset, build_plantvillage_dataloader
from visualization.gradcam import generate_gradcam_comparison, get_gradcam, overlay_cam_on_image
from visualization.confusion_matrix import plot_confusion_matrix
from visualization.tsne_vis import extract_features, plot_tsne, plot_tsne_comparison
from utils import load_checkpoint, compute_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="MSCA-FasterNet Visualization")

    parser.add_argument("--mode", type=str, default="all",
                        choices=["gradcam", "confusion", "tsne", "all"],
                        help="Visualization mode")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to improved model checkpoint")
    parser.add_argument("--checkpoint-baseline", type=str, default=None,
                        help="Path to baseline model checkpoint (for comparison)")
    parser.add_argument("--dataset", type=str, default="ip102",
                        choices=["ip102", "plantvillage"])
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--output-dir", type=str, default="results/visualization")
    parser.add_argument("--num-images", type=int, default=8,
                        help="Number of images for Grad-CAM")

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, model_type, num_classes, device):
    """Load model from checkpoint."""
    if model_type == "baseline":
        model = fasternet_t0_baseline(num_classes=num_classes)
    else:
        model = fasternet_t0_full(num_classes=num_classes)

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def run_gradcam(args, device):
    """Generate Grad-CAM visualizations."""
    num_classes = 102 if args.dataset == "ip102" else 38

    # Load improved model
    improved_model = load_model_from_checkpoint(args.checkpoint, "full", num_classes, device)

    # Load baseline model
    if args.checkpoint_baseline:
        baseline_model = load_model_from_checkpoint(
            args.checkpoint_baseline, "baseline", num_classes, device
        )
    else:
        print("No baseline checkpoint provided. Using untrained baseline for comparison.")
        baseline_model = fasternet_t0_baseline(num_classes=num_classes).to(device)
        baseline_model.eval()

    # Load test dataset
    data_dir = args.data_dir or f"data/{args.dataset.upper()}"
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

    if args.dataset == "ip102":
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = IP102Dataset(data_dir, "test", test_transform, return_path=True)
        class_names = IP102Dataset.CLASS_NAMES
    else:
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        dataset = PlantVillageDataset(data_dir, "test", test_transform, return_path=True)
        class_names = dataset.classes

    # Select images (first N from test set)
    n = min(args.num_images, len(dataset))
    save_dir = os.path.join(args.output_dir, "gradcam")
    os.makedirs(save_dir, exist_ok=True)

    for idx in range(n):
        image_tensor, label, img_path = dataset[idx]
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Load original image for display
        orig_image = Image.open(img_path).convert("RGB")
        orig_image = orig_image.resize((224, 224))
        orig_np = np.array(orig_image).astype(np.float32) / 255.0

        try:
            fig = generate_gradcam_comparison(
                baseline_model, improved_model,
                image_tensor, orig_np,
                class_names=class_names,
                save_path=os.path.join(save_dir, f"gradcam_{idx:03d}.png"),
                title=f"True: {class_names[label] if label < len(class_names) else label}",
            )
            plt.close(fig)
            print(f"  Generated Grad-CAM for image {idx+1}/{n}")
        except Exception as e:
            print(f"  Error on image {idx}: {e}")


def run_confusion(args, device):
    """Generate confusion matrix visualization."""
    num_classes = 102 if args.dataset == "ip102" else 38
    model = load_model_from_checkpoint(args.checkpoint, "full", num_classes, device)

    data_dir = args.data_dir or f"data/{args.dataset.upper()}"

    if args.dataset == "ip102":
        test_loader = build_ip102_dataloader(
            root_dir=data_dir, split="test",
            batch_size=args.batch_size, num_workers=args.workers,
            use_weighted_sampler=False,
        )
        class_names = IP102Dataset.CLASS_NAMES[:num_classes]
    else:
        test_loader = build_plantvillage_dataloader(
            root_dir=data_dir, split="test",
            batch_size=args.batch_size, num_workers=args.workers,
        )
        pv_dataset = PlantVillageDataset(data_dir, "test")
        class_names = pv_dataset.classes

    # Collect predictions
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)

    save_dir = os.path.join(args.output_dir, "confusion_matrix")
    os.makedirs(save_dir, exist_ok=True)

    plot_confusion_matrix(
        cm,
        class_names=class_names,
        title=f"Confusion Matrix - MSCA-FasterNet ({args.dataset.upper()})",
        save_path=os.path.join(save_dir, f"confusion_{args.dataset}.png"),
    )


def run_tsne(args, device):
    """Generate t-SNE visualization."""
    num_classes = 102 if args.dataset == "ip102" else 38

    # Load improved model
    improved_model = load_model_from_checkpoint(args.checkpoint, "full", num_classes, device)

    # Load baseline
    if args.checkpoint_baseline:
        baseline_model = load_model_from_checkpoint(
            args.checkpoint_baseline, "baseline", num_classes, device
        )
    else:
        baseline_model = fasternet_t0_baseline(num_classes=num_classes).to(device)
        baseline_model.eval()

    data_dir = args.data_dir or f"data/{args.dataset.upper()}"

    if args.dataset == "ip102":
        test_loader = build_ip102_dataloader(
            root_dir=data_dir, split="test",
            batch_size=args.batch_size, num_workers=args.workers,
            use_weighted_sampler=False,
        )
        class_names = IP102Dataset.CLASS_NAMES[:num_classes]
    else:
        test_loader = build_plantvillage_dataloader(
            root_dir=data_dir, split="test",
            batch_size=args.batch_size, num_workers=args.workers,
        )
        pv_dataset = PlantVillageDataset(data_dir, "test")
        class_names = pv_dataset.classes

    save_dir = os.path.join(args.output_dir, "tsne")
    os.makedirs(save_dir, exist_ok=True)

    # Extract features
    print("Extracting features from improved model...")
    features_improved, labels = extract_features(improved_model, test_loader, device)

    print("Extracting features from baseline model...")
    features_baseline, _ = extract_features(baseline_model, test_loader, device)

    # Comparison plot
    plot_tsne_comparison(
        features_baseline, features_improved, labels,
        class_names=class_names,
        save_path=os.path.join(save_dir, f"tsne_comparison_{args.dataset}.png"),
    )


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    if args.mode in ["gradcam", "all"]:
        print("Generating Grad-CAM visualizations...")
        run_gradcam(args, device)

    if args.mode in ["confusion", "all"]:
        print("Generating confusion matrix...")
        run_confusion(args, device)

    if args.mode in ["tsne", "all"]:
        print("Generating t-SNE visualization...")
        run_tsne(args, device)

    print(f"All visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()
