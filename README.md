# MSCA-FasterNet

## A Lightweight Crop Pest and Disease Identification Method Based on Improved FasterNet with Multi-Scale Channel Attention

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/CUDA-12.4-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experiments%20In%20Progress-orange.svg)]()

---

## Overview

Accurate crop pest and disease identification is critical for agricultural productivity. While deep learning has made significant progress, most existing approaches rely on heavy models unsuitable for edge deployment. We propose **MSCA-FasterNet**, a lightweight method that achieves competitive accuracy with only **2.41M parameters**.

**Three innovations:**

| # | Innovation | Description |
|---|-----------|-------------|
| 1 | **First FasterNet in agriculture** | FasterNet-T0 backbone applied to pest/disease recognition for the first time |
| 2 | **MSCA: Multi-Scale Channel Attention** | Adaptive scale selection (SKNet-style) + SE channel calibration → handles lesions from 3px spots to half-leaf patches |
| 3 | **Cross-layer feature fusion** | Stage2/3/4 features aligned and fused — texture, patterns, and semantics combined |

---

## Architecture

```
Input (3, 224, 224)
  │
  ├─ Embedding (40, 56, 56)
  ├─ Stage1   (40, 56, 56)            ← unmodified
  ├─ Stage2   (80, 28, 28)            ← saved for fusion
  ├─ Stage3   (160, 14, 14)           ← MSCA at last 2 blocks, saved for fusion
  ├─ Stage4   (320, 7, 7)             ← saved for fusion
  │
  └─ Cross-Layer Fusion → Align(Stage2, Stage3, Stage4) → 160×14×14
       │
       └─ GAP → FC(160 → num_classes)
```

| Model Variant | Params | FLOPs | MACs |
|--------------|--------|-------|------|
| Baseline (FasterNet-T0, 102-class) | 2.25M | ~0.34G | ~0.17G |
| + MSCA only | 2.27M (+0.9%) | ~0.37G | ~0.18G |
| + Fusion only | 2.37M (+5.3%) | ~0.38G | ~0.19G |
| **MSCA-FasterNet (full)** | **2.41M (+7.1%)** | **~0.40G** | **~0.20G** |

---

## MSCA Module: Adaptive Scale Selection

Unlike simple fixed-weight multi-scale fusion, MSCA uses **SKNet-style learned soft attention** to dynamically weight 3×3 vs 5×5 depthwise features per sample:

```
F₃ = DWConv₃ₓ₃(X)              # captures small lesions
F₅ = DWConv₅ₓ₅(X)              # captures large patches

# Per-sample scale weight (learned)
α = Softmax(MLP(GAP(F₃ + F₅)))   # α ∈ (0, 1)

F_fused = α · F₃ + (1-α) · F₅     # adaptive fusion
Output  = F_fused ⊗ SE(X)          # channel calibration
```

**Why this matters:** Pest lesions range from tiny spots (3-5px) to half-leaf blight patches. A fixed kernel cannot cover this scale diversity. The network learns when to trust fine vs. coarse features — critically, it can make *different choices for different images*.

Parameter cost: only **~12.5K per module** (0.5% of total), deployed at 2 blocks in Stage3.

---

## Experimental Results

All experiments on RTX 4060 Laptop (8GB), PyTorch 2.6.0+cu124, AdamW (lr=4e-4), CosineAnnealing, 150 epochs, batch=64.

### IP102 (102-class insect pest, ~45K images)

| Configuration | Test Acc@1 | Test Acc@5 | Params | Notes |
|--------------|-----------|-----------|--------|-------|
| Baseline | **64.00%** | 86.07% | 2.25M | Solid baseline |
| + MSCA | 63.17% | 86.11% | 2.27M | MSCA alone: neutral, room for tuning |
| + Fusion | 44.73% | 75.03% | 2.37M | ⚠️ Fusion w/o MSCA destabilizes training |
| **Full (MSCA+Fusion)** | 63.50% | 86.46% | 2.41M | Comparable to baseline, fusion needs rework |

### PlantVillage (15-class disease, ~20K images)

| Configuration | Test Acc@1 | Test Acc@5 | Params |
|--------------|-----------|-----------|--------|
| Baseline | **98.79%** | 100.00% | 2.22M |

### Key Findings

1. **Fusion module is currently broken** — when used without MSCA, accuracy collapses to 44.73%. This suggests the cross-layer alignment strategy needs redesign (possibly better upsampling or attention-guided fusion).
2. **MSCA alone** performs at baseline level (63.17% vs 64.00%), indicating the attention mechanism works correctly but doesn't yet provide gains on this challenging fine-grained dataset.
3. **PlantVillage** is a simpler task — 98.79% baseline confirms the backbone is capable; the bottleneck is IP102's extreme class granularity.

**Next steps:** Redesign fusion module → rerun ablation → then comparison experiments.

---

## Quick Start

### Installation

```bash
git clone https://github.com/yumengyu666/MSCA-FasterNet.git
cd MSCA-FasterNet
pip install -r requirements.txt
```

### Datasets

**IP102** (102 insect pest classes, ~45K images):
```bash
git clone https://github.com/xpwu95/IP102.git data/IP102
```

**PlantVillage** (15 disease classes — Pepper, Potato, Tomato):
```bash
# Download from Kaggle: https://www.kaggle.com/datasets/emmarex/plantdisease
# Extract to data/PlantVillage/
```

Expected structure:
```
data/
├── IP102/
│   ├── classes.txt
│   └── 001/  002/  ...  102/
└── PlantVillage/
    └── PlantVillage/
        ├── Pepper___bell___Bacterial_spot/
        ├── Potato___Early_blight/
        └── Tomato___Target_Spot/
```

### Training

```bash
# Ablation experiments
python scripts/train.py --dataset ip102 --model baseline  --epochs 150 --gpu 0 --amp
python scripts/train.py --dataset ip102 --model msca      --epochs 150 --gpu 0 --amp
python scripts/train.py --dataset ip102 --model fusion    --epochs 150 --gpu 0 --amp
python scripts/train.py --dataset ip102 --model full       --epochs 150 --gpu 0 --amp

# Attention comparison (SE / CBAM / ECA / SK)
python scripts/train.py --dataset ip102 --model attention_se   --epochs 150 --gpu 0 --amp
python scripts/train.py --dataset ip102 --model attention_cbam --epochs 150 --gpu 0 --amp
python scripts/train.py --dataset ip102 --model attention_eca  --epochs 150 --gpu 0 --amp
python scripts/train.py --dataset ip102 --model attention_sk   --epochs 150 --gpu 0 --amp

# Lightweight model comparison
python scripts/train_comparison.py --model mobilenetv3_small_100 --dataset ip102 --gpu 0
python scripts/train_comparison.py --model shufflenetv2_x0.5    --dataset ip102 --gpu 0
python scripts/train_comparison.py --model ghostnetv2_100       --dataset ip102 --gpu 0
python scripts/train_comparison.py --model efficientnet_lite0   --dataset ip102 --gpu 0

# PlantVillage generalization
python scripts/train.py --dataset plantvillage --model baseline --epochs 100 --gpu 0 --amp
python scripts/train.py --dataset plantvillage --model full      --epochs 100 --gpu 0 --amp

# Multi-seed statistical validation (3 seeds)
python scripts/run_multiseed.py --dataset ip102 --seeds 42 123 456 --gpu 0

# Run all ablations sequentially
python scripts/ablation.py --dataset ip102 --gpu 0
```

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --dataset ip102 \
    --compute-flops --measure-fps

# Detailed per-class report
python scripts/classification_report.py \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --dataset ip102 \
    --output results/classification_report
```

### Visualization

```bash
# Grad-CAM
python scripts/visualize.py --mode gradcam \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --dataset ip102

# Confusion matrix
python scripts/visualize.py --mode confusion \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --dataset ip102

# t-SNE features
python scripts/visualize.py --mode tsne \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --checkpoint-baseline checkpoints/ip102_baseline/best_model.pth \
    --dataset ip102

# MSCA attention weights
python visualization/msca_weights.py \
    --checkpoint checkpoints/ip102_full/best_model.pth \
    --dataset ip102 --output_dir results/msca_weights
```

---

## Project Structure

```
MSCA-FasterNet/
├── models/                         # Model implementations
│   ├── fasternet.py               # FasterNet-T0 backbone (linear DropPath)
│   ├── msca.py                    # MSCA: adaptive scale selection + SE
│   ├── fusion.py                  # Cross-layer fusion (Stage2+3+4)
│   ├── msca_fasternet.py          # Full model with pretrained weight mapping
│   ├── attention_comparison.py    # SE / CBAM / ECA / SK reference modules
│   └── attention_models.py        # Comparison model builders
├── scripts/                        # Executables
│   ├── train.py                   # Main training (supports all model variants)
│   ├── evaluate.py                # Evaluation with FLOPs/MACs/FPS
│   ├── train_comparison.py        # Lightweight model comparison training
│   ├── ablation.py                # Ablation experiment runner
│   ├── run_multiseed.py           # Multi-seed + statistical tests
│   ├── classification_report.py   # Per-class Precision/Recall/F1
│   ├── verify_model.py            # Architecture verification
│   ├── download_weights.py        # Pretrained weight downloader
│   └── visualize.py               # Visualization entry point
├── visualization/                  # Visualization tools
│   ├── gradcam.py                 # Grad-CAM heatmaps
│   ├── confusion_matrix.py        # Confusion matrix
│   ├── tsne_vis.py                # t-SNE embeddings
│   └── msca_weights.py            # MSCA attention weight analysis
├── datasets/                       # IP102 & PlantVillage loaders
├── utils/                          # Logger, metrics, sampler, misc
├── configs/                        # YAML configs
├── results_v6/                     # V6 experiment outputs (best_model + logs + results.json)
├── checkpoints/                    # Pretrained weights (git-ignored)
├── data/                           # Datasets (git-ignored)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Design Rationale

**Why FasterNet-T0?**
As of 2026, fewer than 5 SCI papers apply FasterNet to agricultural vision — genuine novelty. PConv (Partial Convolution) reduces redundant computation, making it ideal for edge deployment. The fixed 3×3 kernel and single-layer classification head are clear targets for improvement.

**Why adaptive scale selection?**
Crop lesions span orders of magnitude in scale. A 3×3 kernel misses large blight patches; a 5×5 kernel loses fine insect details. Fixed fusion forces a compromise. SKNet-style soft attention learns per-sample scale preferences — only +3.2K params for a meaningful architectural innovation.

**Why cross-layer fusion?**
Stage2 carries texture (color, lesion boundaries), Stage3 encodes local patterns (texture, body parts), Stage4 captures semantics (class-level features). Feeding only the final layer discards complementary signals from earlier stages.

**Why linear stochastic depth?**
Standard practice following Huang et al. (ECCV 2016): first block drop rate = 0, last block = max. Progressive regularization — early layers learn stable features, later layers benefit from stronger regularization.

---

## Ablation Design

| ID | Name | MSCA | Fusion | Purpose |
|----|------|:----:|:------:|---------|
| A | baseline | ✗ | ✗ | Establish lower bound |
| B | msca | ✓ | ✗ | Isolate attention contribution |
| C | fusion | ✗ | ✓ | Isolate fusion contribution |
| D | full | ✓ | ✓ | Combined effect |

## Comparison Models

| Model | Type | Params |
|-------|------|--------|
| MobileNetV3-Small | Lightweight CNN | 2.5M |
| ShuffleNetV2-x0.5 | Lightweight CNN | 1.4M |
| GhostNetV2-100 | Lightweight CNN | 6.2M |
| EfficientNet-Lite0 | Lightweight CNN | 4.7M |
| FasterNet-T0 + SE | Attention baseline | 2.27M |
| FasterNet-T0 + CBAM | Attention baseline | 2.28M |
| FasterNet-T0 + ECA | Attention baseline | 2.26M |
| FasterNet-T0 + SK | Attention baseline | 2.30M |
| **MSCA-FasterNet (Ours)** | **Proposed** | **2.41M** |

---

## Citation

```bibtex
@article{msca_fasternet2026,
  title   = {A Lightweight Crop Pest and Disease Identification Method
             Based on Improved FasterNet with Multi-Scale Channel Attention},
  author  = {Yu Mengyu},
  journal = {To be determined},
  year    = {2026}
}
```

## References

1. Chen J, et al. *"Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks."* CVPR 2023.
2. Wu X, et al. *"IP102: A Large-Scale Benchmark Dataset for Insect Pest Recognition."* CVPR 2019.
3. Hu J, et al. *"Squeeze-and-Excitation Networks."* CVPR 2018.
4. Li X, et al. *"Selective Kernel Networks."* CVPR 2019.
5. Huang G, et al. *"Deep Networks with Stochastic Depth."* ECCV 2016.

---

## License

MIT — see [LICENSE](LICENSE).
