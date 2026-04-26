"""Dataset download and preparation helper.

Usage:
    python scripts/download_data.py --dataset ip102
    python scripts/download_data.py --dataset plantvillage
"""

import os
import sys
import argparse
import zipfile
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def download_ip102(data_dir="data/IP102"):
    """Download IP102 dataset."""
    os.makedirs(data_dir, exist_ok=True)
    target_dir = os.path.join(data_dir, "ip102_v1.1")

    if os.path.exists(target_dir) and os.path.exists(os.path.join(target_dir, "images")):
        print(f"IP102 dataset already exists at {target_dir}")
        return

    print("=" * 60)
    print("IP102 Dataset Download")
    print("=" * 60)
    print()
    print("IP102 is hosted on GitHub. Please download manually:")
    print()
    print("  Repository: https://github.com/xpwu95/IP102")
    print()
    print("Steps:")
    print("  1. Clone or download the repository")
    print("  2. Extract to data/IP102/ip102_v1.1/")
    print("  3. Ensure the following structure exists:")
    print("     data/IP102/ip102_v1.1/images/001/*.jpg")
    print("     data/IP102/ip102_v1.1/list/train.txt")
    print()

    # Alternative: try direct download
    alt_url = "https://cloud.189.cn/t/3qU7Jv AFnIFn2"  # Chinese cloud
    print(f"  Alternative download (Chinese cloud): {alt_url}")
    print()

    os.makedirs(target_dir, exist_ok=True)


def download_plantvillage(data_dir="data/PlantVillage"):
    """Download PlantVillage dataset."""
    os.makedirs(data_dir, exist_ok=True)
    target_dir = os.path.join(data_dir, "PlantVillage")

    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 10:
        print(f"PlantVillage dataset already exists at {target_dir}")
        return

    print("=" * 60)
    print("PlantVillage Dataset Download")
    print("=" * 60)
    print()
    print("PlantVillage is available on Kaggle. Please download manually:")
    print()
    print("  Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease")
    print()
    print("Steps:")
    print("  1. Download the dataset from Kaggle")
    print("  2. Extract to data/PlantVillage/PlantVillage/")
    print("  3. Ensure class folders exist:")
    print("     data/PlantVillage/PlantVillage/Tomato___Bacterial_spot/")
    print()


def download_fasternet_pretrained(weights_dir="pretrained"):
    """Download FasterNet-T0 ImageNet pretrained weights."""
    os.makedirs(weights_dir, exist_ok=True)
    weight_path = os.path.join(weights_dir, "fasternet_t0.pth")

    if os.path.exists(weight_path):
        print(f"Pretrained weights already exist at {weight_path}")
        return

    print("=" * 60)
    print("FasterNet-T0 Pretrained Weights")
    print("=" * 60)
    print()
    print("Download from the official FasterNet repository:")
    print()
    print("  Repository: https://github.com/JierunChen/FasterNet")
    print("  Direct link: https://drive.google.com/file/d/1tKg7RJ2N2YkpMg6vB8s7z6uMA6b3JGvK/view")
    print()
    print(f"Save to: {weight_path}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Download datasets and pretrained weights")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["ip102", "plantvillage", "pretrained", "all"])
    parser.add_argument("--data-dir", type=str, default="data")

    args = parser.parse_args()

    if args.dataset in ["ip102", "all"]:
        download_ip102(os.path.join(args.data_dir, "IP102"))

    if args.dataset in ["plantvillage", "all"]:
        download_plantvillage(os.path.join(args.data_dir, "PlantVillage"))

    if args.dataset in ["pretrained", "all"]:
        download_fasternet_pretrained()


if __name__ == "__main__":
    main()
