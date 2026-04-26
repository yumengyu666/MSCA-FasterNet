"""Download datasets for the project.

Supports:
  - IP102 from Kaggle (recommended)
  - IP102 from OpenDataLab (alternative)
  - Agricultural Pests Image Dataset from Kaggle (fallback, 12 classes)

Usage:
  # Download IP102 via Kaggle (need kaggle.json in ~/.kaggle/)
  py -3 scripts/download_datasets.py --dataset ip102 --source kaggle

  # Download IP102 via OpenDataLab
  py -3 scripts/download_datasets.py --dataset ip102 --source opendatalab

  # Download Agricultural Pests (12 classes, smaller, fallback)
  py -3 scripts/download_datasets.py --dataset pests12 --source kaggle

  # Download PlantVillage (if not already present)
  py -3 scripts/download_datasets.py --dataset plantvillage --source kaggle
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def check_kaggle_cli():
    """Check if kaggle CLI is available and authenticated."""
    try:
        result = subprocess.run(
            ["py", "-3", "-m", "kaggle", "config", "view"],
            capture_output=True, text=True, timeout=10
        )
        if "error" in result.stdout.lower() or "error" in result.stderr.lower():
            return False
        return True
    except Exception:
        return False


def setup_kaggle_instructions():
    """Print instructions for setting up Kaggle API."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           Kaggle API 配置指南                                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. 登录 https://www.kaggle.com                             ║
║  2. 点击右上角头像 -> Settings                               ║
║  3. 滚动到 API 部分，点击 "Create New Token"                 ║
║  4. 下载 kaggle.json 文件                                    ║
║  5. 将文件放到:                                              ║
║     Windows: C:\\Users\\<你的用户名>\\.kaggle\\kaggle.json       ║
║     Linux:   ~/.kaggle/kaggle.json                           ║
║  6. 重新运行此脚本                                           ║
║                                                              ║
║  或者：直接在浏览器下载后解压到 data/ 目录                    ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")


def download_ip102_kaggle():
    """Download IP102 from Kaggle."""
    target_dir = DATA_DIR / "IP102"
    target_dir.mkdir(parents=True, exist_ok=True)

    if not check_kaggle_cli():
        print("❌ Kaggle CLI 未配置或未认证")
        setup_kaggle_instructions()

        # Fallback: open browser
        print("正在打开 Kaggle 下载页面，请手动下载...")
        url = "https://www.kaggle.com/datasets/rahimanshu/pest-classification-ip102-dataset"
        print(f"\n📥 下载链接: {url}")
        print(f"📂 解压到: {target_dir}")
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
        return False

    print(f"正在从 Kaggle 下载 IP102 到 {target_dir}...")
    try:
        subprocess.run(
            [
                "py", "-3", "-m", "kaggle",
                "datasets", "download",
                "-d", "rahimanshu/pest-classification-ip102-dataset",
                "-p", str(target_dir),
                "--unzip"
            ],
            check=True,
        )
        print(f"✅ IP102 下载完成: {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        setup_kaggle_instructions()
        return False


def download_pests12_kaggle():
    """Download Agricultural Pests Image Dataset (12 classes) from Kaggle."""
    target_dir = DATA_DIR / "AgriculturalPests"
    target_dir.mkdir(parents=True, exist_ok=True)

    if not check_kaggle_cli():
        print("❌ Kaggle CLI 未配置")
        url = "https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset"
        print(f"\n📥 下载链接: {url}")
        print(f"📂 解压到: {target_dir}")
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
        return False

    print(f"正在从 Kaggle 下载 Agricultural Pests (12类) 到 {target_dir}...")
    try:
        subprocess.run(
            [
                "py", "-3", "-m", "kaggle",
                "datasets", "download",
                "-d", "vencerlanz09/agricultural-pests-image-dataset",
                "-p", str(target_dir),
                "--unzip"
            ],
            check=True,
        )
        print(f"✅ Agricultural Pests 下载完成: {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        return False


def download_plantvillage_kaggle():
    """Download PlantVillage from Kaggle (full 38-class version)."""
    target_dir = DATA_DIR / "PlantVillage_full"
    target_dir.mkdir(parents=True, exist_ok=True)

    if not check_kaggle_cli():
        print("❌ Kaggle CLI 未配置")
        url = "https://www.kaggle.com/datasets/emmarex/plantdisease"
        print(f"\n📥 下载链接: {url}")
        print(f"📂 解压到: {target_dir}")
        try:
            import webbrowser
            webbrowser.open(url)
        except Exception:
            pass
        return False

    print(f"正在从 Kaggle 下载 PlantVillage (完整38类) 到 {target_dir}...")
    try:
        subprocess.run(
            [
                "py", "-3", "-m", "kaggle",
                "datasets", "download",
                "-d", "emmarex/plantdisease",
                "-p", str(target_dir),
                "--unzip"
            ],
            check=True,
        )
        print(f"✅ PlantVillage 完整版下载完成: {target_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 下载失败: {e}")
        return False


def check_existing_data():
    """Check what datasets already exist."""
    print("=" * 50)
    print("当前数据集状态")
    print("=" * 50)

    datasets = {
        "IP102": DATA_DIR / "IP102",
        "PlantVillage": DATA_DIR / "PlantVillage",
        "AgriculturalPests": DATA_DIR / "AgriculturalPests",
    }

    for name, path in datasets.items():
        if path.exists():
            # Count images
            img_count = 0
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
                img_count += len(list(path.rglob(ext)))
            print(f"  ✅ {name}: {path} ({img_count} images)")
        else:
            print(f"  ❌ {name}: 未下载")

    print()


def main():
    parser = argparse.ArgumentParser(description="Download datasets for MSCA-FasterNet")
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=["ip102", "pests12", "plantvillage", "all", "check"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--source", type=str, default="kaggle",
        choices=["kaggle", "opendatalab"],
        help="Download source"
    )

    args = parser.parse_args()

    print("\n🌿 MSCA-FasterNet 数据集下载工具\n")

    if args.dataset == "check":
        check_existing_data()
        return

    # Check existing data first
    check_existing_data()

    if args.dataset == "ip102" or args.dataset == "all":
        print("\n--- 下载 IP102 (102类害虫, ~75K张) ---")
        download_ip102_kaggle()

    if args.dataset == "pests12" or args.dataset == "all":
        print("\n--- 下载 Agricultural Pests (12类, ~6K张) ---")
        download_pests12_kaggle()

    if args.dataset == "plantvillage" or args.dataset == "all":
        print("\n--- 下载 PlantVillage 完整版 (38类, ~54K张) ---")
        download_plantvillage_kaggle()

    # Final status
    print("\n")
    check_existing_data()
    print("🎉 下载任务完成！")


if __name__ == "__main__":
    main()
