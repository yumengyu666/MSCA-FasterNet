# MSCA-FasterNet: Lightweight Crop Pest and Disease Identification

## A Lightweight Crop Pest and Disease Identification Method Based on Improved FasterNet with Multi-Scale Channel Attention

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

### 🎯 Overview

This project proposes a lightweight crop pest and disease identification method based on improved FasterNet with Multi-Scale Channel Attention (MSCA). The key innovations include:

1. **First application of FasterNet** to crop pest and disease identification domain
2. **MSCA module**: Multi-scale depthwise convolution (3×3 + 5×5) with SE channel attention for capturing pest/disease lesions of varying scales
3. **Cross-layer feature fusion**: Integrating shallow texture, mid-level local patterns, and deep semantic features from Stage2/3/4

### 📊 Model Architecture

```
Input (3, 224, 224)
    → Embedding (40, 56, 56)
    → Stage1 (40, 56, 56)           [no modification]
    → Stage2 (80, 28, 28)           [saved for fusion]
    → Stage3 (160, 14, 14)          [MSCA at last 2 blocks; saved for fusion]
    → Stage4 (320, 7, 7)            [saved for fusion]
    → Cross-Layer Fusion (160, 14, 14)
    → GAP → FC(160 → num_classes)
```

| Model | Params (M) | FLOPs (G) | Top-1 Acc (IP102) |
|-------|-----------|----------|-------------------|
| FasterNet-T0 (baseline) | 3.9 | 0.34 | - |
| MSCA-FasterNet (ours) | ~4.07 | ~0.40 | - |

> Results will be updated after experiments are completed.

---

### 🚀 Quick Start

#### 1. Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/MSCA-FasterNet.git
cd MSCA-FasterNet

# Create conda environment
conda create -n msca python=3.10 -y
conda activate msca

# Install dependencies
pip install -r requirements.txt
```

#### 2. Prepare Datasets

**IP102** (Insect Pest Recognition, 102 classes, ~75K images):
```bash
# Download from GitHub
git clone https://github.com/xpwu95/IP102.git data/IP102
# Or download manually from: https://github.com/xpwu95/IP102
```

**PlantVillage** (Plant Disease, 38 classes, ~54K images):
```bash
# Download from Kaggle
# https://www.kaggle.com/datasets/emmarex/plantdisease
# Extract to data/PlantVillage/
```

Expected directory structure:
```
data/
├── IP102/
│   └── ip102_v1.1/
│       ├── images/
│       │   ├── 001/
│       │   ├── 002/
│       │   └── ...
│       └── list/
│           ├── train.txt
│           ├── val.txt
│           └── test.txt
└── PlantVillage/
    └── PlantVillage/
        ├── Tomato___Bacterial_spot/
        ├── Apple___Apple_scab/
        └── ...
```

#### 3. Training

**Train the full MSCA-FasterNet model:**
```bash
python scripts/train.py --dataset ip102 --model full --epochs 150 --gpu 0
```

**Train baseline for comparison:**
```bash
python scripts/train.py --dataset ip102 --model baseline --epochs 150 --gpu 0
```

**Run all ablation experiments:**
```bash
python scripts/ablation.py --dataset ip102 --gpu 0
```

**Train comparison models:**
```bash
python scripts/train_comparison.py --model mobilenetv3_small_100 --dataset ip102
python scripts/train_comparison.py --model shufflenetv2_x0.5 --dataset ip102
python scripts/train_comparison.py --model ghostnetv2_100 --dataset ip102
python scripts/train_comparison.py --model efficientnet_lite0 --dataset ip102
```

#### 4. Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --dataset ip102 \
    --compute-flops \
    --measure-fps
```

#### 5. Visualization

```bash
# Grad-CAM heatmaps
python scripts/visualize.py --mode gradcam \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --checkpoint-baseline checkpoints/ip102_baseline/best_model.pth \
    --dataset ip102

# Confusion matrix
python scripts/visualize.py --mode confusion \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --dataset ip102

# t-SNE feature distribution
python scripts/visualize.py --mode tsne \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --checkpoint-baseline checkpoints/ip102_baseline/best_model.pth \
    --dataset ip102

# All visualizations
python scripts/visualize.py --mode all \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --checkpoint-baseline checkpoints/ip102_baseline/best_model.pth \
    --dataset ip102
```

---

### 📁 Project Structure

```
MSCA-FasterNet/
├── configs/                    # Configuration files
│   ├── base.yaml
│   ├── ip102.yaml
│   └── plantvillage.yaml
├── data/                       # Dataset directory (gitignored)
├── datasets/                   # Dataset loaders
│   ├── __init__.py
│   ├── ip102.py               # IP102 dataset loader
│   └── plantvillage.py        # PlantVillage dataset loader
├── models/                     # Model implementations
│   ├── __init__.py
│   ├── fasternet.py           # FasterNet-T0 backbone
│   ├── msca.py                # MSCA attention module
│   ├── fusion.py              # Cross-layer feature fusion
│   └── msca_fasternet.py      # Complete improved model
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── logger.py              # Logging
│   ├── metrics.py             # Evaluation metrics
│   ├── sampler.py             # Weighted sampler
│   └── misc.py                # Seed, checkpoint, etc.
├── visualization/              # Visualization tools
│   ├── __init__.py
│   ├── gradcam.py             # Grad-CAM heatmap
│   ├── confusion_matrix.py    # Confusion matrix plot
│   └── tsne_vis.py            # t-SNE feature plot
├── scripts/                    # Executable scripts
│   ├── train.py               # Main training script
│   ├── evaluate.py            # Evaluation script
│   ├── train_comparison.py    # Comparison model training
│   ├── ablation.py            # Ablation experiment runner
│   └── visualize.py           # Visualization entry point
├── checkpoints/                # Model weights (gitignored)
├── logs/                       # Training logs (gitignored)
├── results/                    # Experiment results (gitignored)
├── requirements.txt
├── README.md
└── .gitignore
```

---

### 🔬 Ablation Study Design

| ID | Configuration | Validation Target |
|----|--------------|-------------------|
| A | FasterNet-T0 baseline | Baseline performance |
| B | A + MSCA | Attention module contribution |
| C | A + Cross-layer fusion | Fusion strategy contribution |
| D | A + MSCA + fusion | Combined effect (full model) |

### 📊 Comparison Models

| Model | Year | Params (M) |
|-------|------|-----------|
| MobileNetV3-Small | 2019 | 2.5 |
| ShuffleNetV2-x0.5 | 2018 | 1.4 |
| GhostNetV2 | 2022 | 6.2 |
| EfficientNet-Lite0 | 2020 | 4.7 |
| FasterNet-T0 | 2023 | 3.9 |
| **MSCA-FasterNet (Ours)** | **2026** | **~4.07** |

---

### 🔑 Key Design Choices

**Why FasterNet-T0?**
- Only ~5 SCI papers use FasterNet for pest recognition as of 2026 → genuine innovation
- PConv (Partial Convolution) provides fast actual inference speed
- Clear improvement opportunities: fixed 3×3 kernel, no multi-scale perception, single-layer classification

**Why MSCA?**
- Pest/disease lesions vary dramatically in scale (3-5px spots to half-leaf patches)
- 3×3 DWConv captures small lesions, 5×5 DWConv captures large patches
- SE attention calibrates channel importance after multi-scale fusion

**Why Cross-Layer Fusion?**
- Stage2: texture details (lesion color, shape)
- Stage3: local patterns (lesion texture, insect body parts)
- Stage4: global semantics (disease/insect category)
- Single-layer classification misses complementary information

---

### 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@article{msca_fasternet2026,
  title={A Lightweight Crop Pest and Disease Identification Method Based on Improved FasterNet with Multi-Scale Channel Attention},
  author={Your Name},
  journal={To be determined},
  year={2026}
}
```

### 📄 References

1. Chen J, et al. "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks." CVPR 2023.
2. Wu X, et al. "IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition." CVPR 2019.
3. Hu J, et al. "Squeeze-and-Excitation Networks." CVPR 2018.

---

### 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
