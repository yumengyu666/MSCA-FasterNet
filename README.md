# MSCA-FasterNet: Lightweight Crop Pest and Disease Identification

## A Lightweight Crop Pest and Disease Identification Method Based on Improved FasterNet with Multi-Scale Channel Attention

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

### 🎯 Overview

This project proposes a lightweight crop pest and disease identification method based on improved FasterNet with Multi-Scale Channel Attention (MSCA). The key innovations include:

1. **First application of FasterNet** to crop pest and disease identification domain
2. **MSCA module**: Adaptive multi-scale depthwise convolution (3×3 + 5×5) with **learned scale selection** (SKNet-style soft attention) and SE channel attention for capturing pest/disease lesions of varying scales
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

| Model | Params (M) | FLOPs (G) | MACs (G) | Top-1 Acc (IP102) |
|-------|-----------|-----------|----------|-------------------|
| FasterNet-T0 (baseline, 102-class) | 2.25 | ~0.34 | ~0.17 | - |
| + MSCA only | 2.27 | ~0.37 | ~0.18 | - |
| + Fusion only | 2.37 | ~0.38 | ~0.19 | - |
| **MSCA-FasterNet (ours, full)** | **2.41** | **~0.40** | **~0.20** | - |

> Results will be updated after experiments are completed.

---

### 🔑 Key Innovation: Adaptive Scale Selection in MSCA

Unlike simple fixed-weight fusion (`F3 + F5`), our MSCA uses **SKNet-style soft attention** to dynamically weight 3×3 vs 5×5 features:

```
F3 = DWConv_3x3(X)         # Small lesion features
F5 = DWConv_5x5(X)         # Large lesion features

# Scale attention: network learns which scale matters more per sample
a = Softmax(MLP(GAP(F3 + F5)))   # a ∈ [0, 1] per sample

F_fused = a * F3 + (1-a) * F5    # Adaptive fusion
Output = F_fused ⊗ SE(X)         # Channel calibration
```

This is critical for pest/disease recognition where lesion sizes vary dramatically across classes.

---

### 🚀 Quick Start

#### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yumengyu666/MSCA-FasterNet.git
cd MSCA-FasterNet

# Install dependencies
pip install -r requirements.txt
```

#### 2. Prepare Datasets

**IP102** (Insect Pest Recognition, 102 classes, ~45K images):
```bash
# Download from GitHub
git clone https://github.com/xpwu95/IP102.git data/IP102
# Or download manually from: https://github.com/xpwu95/IP102
```

**PlantVillage** (Plant Disease, 15 classes selected, ~20K images):
```bash
# Download from Kaggle
# https://www.kaggle.com/datasets/emmarex/plantdisease
# Extract to data/PlantVillage/
# Note: We use 15 classes (Pepper, Potato, Tomato) for focused disease recognition
```

Expected directory structure:
```
data/
├── IP102/
│   └── classes.txt
│   └── 001/
│   └── 002/
│   └── ...
└── PlantVillage/
    └── PlantVillage/
        ├── Pepper___bell___Bacterial_spot/
        ├── Potato___Early_blight/
        ├── Tomato___Target_Spot/
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

**Train attention method comparison models (SE/CBAM/ECA/SK):**
```bash
python scripts/train.py --dataset ip102 --model attention_se --epochs 150 --gpu 0
python scripts/train.py --dataset ip102 --model attention_cbam --epochs 150 --gpu 0
python scripts/train.py --dataset ip102 --model attention_eca --epochs 150 --gpu 0
python scripts/train.py --dataset ip102 --model attention_sk --epochs 150 --gpu 0
```

**Run 3-seed repeated experiments with statistical tests:**
```bash
python scripts/run_multiseed.py --dataset ip102 --seeds 42 123 456 --gpu 0
```

**Run all ablation experiments:**
```bash
python scripts/ablation.py --dataset ip102 --gpu 0
```

**Train lightweight comparison models:**
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

**Generate detailed Precision/Recall/F1 classification report:**
```bash
python scripts/classification_report.py \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --dataset ip102 \
    --output results/classification_report
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

# MSCA attention weight visualization (scale weights + channel weights)
python visualization/msca_weights.py \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --dataset ip102 \
    --output_dir results/msca_weights
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
│   ├── fasternet.py           # FasterNet-T0 backbone (with linear DropPath)
│   ├── msca.py                # MSCA attention module (adaptive scale selection, no internal residual)
│   ├── fusion.py              # Cross-layer feature fusion
│   ├── msca_fasternet.py      # Complete improved model (with pretrained weight mapping)
│   ├── attention_comparison.py # SE/CBAM/ECA/SK attention modules for comparison
│   └── attention_models.py    # Attention comparison model builders
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── logger.py              # Logging
│   ├── metrics.py             # Evaluation metrics (FLOPs, MACs=FLOPs/2, F1, etc.)
│   ├── sampler.py             # Weighted sampler
│   └── misc.py                # Seed, checkpoint, etc.
├── visualization/              # Visualization tools
│   ├── __init__.py
│   ├── gradcam.py             # Grad-CAM heatmap
│   ├── confusion_matrix.py    # Confusion matrix plot
│   ├── tsne_vis.py            # t-SNE feature plot
│   └── msca_weights.py        # MSCA attention weight visualization
├── scripts/                    # Executable scripts
│   ├── train.py               # Main training script (supports attention_* models)
│   ├── evaluate.py            # Evaluation script (supports attention_* models)
│   ├── train_comparison.py    # Lightweight model comparison training
│   ├── ablation.py            # Ablation experiment runner
│   ├── run_multiseed.py       # Multi-seed experiment + statistical tests
│   ├── classification_report.py # Precision/Recall/F1 detailed report
│   ├── verify_model.py        # Model architecture verification
│   ├── test_e2e.py            # End-to-end test suite
│   ├── download_weights.py    # Download pretrained weights
│   └── visualize.py           # Visualization entry point
├── checkpoints/                # Model weights (gitignored)
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
| C | A + Cross-layer fusion (no MSCA) | Fusion strategy contribution |
| D | A + MSCA + fusion | Combined effect (full model) |

### 📊 Comparison Models

| Model | Type | Year | Params (M) |
|-------|------|------|-----------|
| MobileNetV3-Small | Lightweight | 2019 | 2.5 |
| ShuffleNetV2-x0.5 | Lightweight | 2018 | 1.4 |
| GhostNetV2 | Lightweight | 2022 | 6.2 |
| EfficientNet-Lite0 | Lightweight | 2020 | 4.7 |
| FasterNet-T0 + SE | Attention comparison | 2018 | ~2.27 |
| FasterNet-T0 + CBAM | Attention comparison | 2018 | ~2.28 |
| FasterNet-T0 + ECA | Attention comparison | 2020 | ~2.26 |
| FasterNet-T0 + SK | Attention comparison | 2019 | ~2.30 |
| FasterNet-T0 (baseline) | Ours (baseline) | 2023 | 2.25 (102-class) |
| **MSCA-FasterNet (Ours)** | **Ours (full)** | **2026** | **2.41 (102-class)** |

---

### 🔧 Bug Fixes (2026-04-27)

| # | Bug | Fix | Impact |
|---|-----|-----|--------|
| 1 | MSCA double residual (internal + block) | Removed internal residual in MSCA/MSCALight; block handles it | Correct gradient flow |
| 2 | Pretrained weight mapping (mlp.4→pwconv2.1) | timm has no mlp.4 (no second BN); added safety mapping | 128/128=100% backbone load |
| 3 | evaluate.py NameError (model_name, dataset_name) | Added as function parameters | Evaluation runs without error |
| 4 | MACs calculation error (fvcore) | `macs_G = flops / 2 / 1e9` (was `flops / 1e9`) | MACs/FLOPs ratio: 1.0→0.50 ✓ |

---

### 🔑 Key Design Choices

**Why FasterNet-T0?**
- Only ~5 SCI papers use FasterNet for pest recognition as of 2026 → genuine innovation
- PConv (Partial Convolution) provides fast actual inference speed
- Clear improvement opportunities: fixed 3×3 kernel, no multi-scale perception, single-layer classification

**Why Adaptive Scale Selection in MSCA?**
- Pest/disease lesions vary dramatically in scale (3-5px spots to half-leaf patches)
- 3×3 DWConv captures small lesions, 5×5 DWConv captures large patches
- **SKNet-style soft attention** lets the network learn per-sample optimal scale weights, rather than fixed 1:1 fusion
- SE attention calibrates channel importance after adaptive multi-scale fusion
- Only +3.2K params per module (0.13% of total model) for significant innovation gain

**Why Cross-Layer Fusion?**
- Stage2: texture details (lesion color, shape)
- Stage3: local patterns (lesion texture, insect body parts)
- Stage4: global semantics (disease/insect category)
- Single-layer classification misses complementary information

**Why Linear Stochastic Depth (DropPath)?**
- Standard practice following Huang et al. (ECCV 2016)
- First block: drop rate = 0, last block: drop rate = max
- Progressive regularization: early layers learn stable features, later layers get more regularization

---

### 📝 Citation

If you find this work useful, please consider citing:

```bibtex
@article{msca_fasternet2026,
  title={A Lightweight Crop Pest and Disease Identification Method Based on Improved FasterNet with Multi-Scale Channel Attention},
  author={Yu Mengyu},
  journal={To be determined},
  year={2026}
}
```

### 📄 References

1. Chen J, et al. "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks." CVPR 2023.
2. Wu X, et al. "IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition." CVPR 2019.
3. Hu J, et al. "Squeeze-and-Excitation Networks." CVPR 2018.
4. Li X, et al. "Selective Kernel Networks." CVPR 2019. (SKNet - basis for adaptive scale selection)
5. Huang G, et al. "Deep Networks with Stochastic Depth." ECCV 2016. (DropPath)

---

### 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
