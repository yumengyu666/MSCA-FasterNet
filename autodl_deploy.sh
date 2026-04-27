#!/bin/bash
# ============================================================
#  MSCA-FasterNet AutoDL 远程训练一键部署脚本
#  使用方法：bash autodl_deploy.sh
# ============================================================

set -e

echo "========================================="
echo "  MSCA-FasterNet AutoDL 部署脚本"
echo "========================================="

# ---------- 1. 环境配置 ----------
echo ""
echo "[1/6] 配置环境..."

# AutoDL 默认 conda 基础环境已有 PyTorch
# 检查 PyTorch 是否可用
python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# 安装项目依赖
echo "安装项目依赖..."
pip install timm fvcore scikit-learn opencv-python matplotlib seaborn PyYAML tqdm -q

echo "✓ 环境配置完成"

# ---------- 2. 数据集检查 ----------
echo ""
echo "[2/6] 检查数据集..."

PROJECT_ROOT="/root/autodl-tmp/MSCA-FasterNet"

if [ ! -d "$PROJECT_ROOT/data/IP102/classes.txt" ]; then
    echo "⚠ IP102 数据集未找到，请确保已上传到 $PROJECT_ROOT/data/IP102/"
    echo "  预期结构：data/IP102/classes.txt + data/IP102/001/ ~ data/IP102/102/"
fi

if [ ! -d "$PROJECT_ROOT/data/PlantVillage" ]; then
    echo "⚠ PlantVillage 数据集未找到，请确保已上传到 $PROJECT_ROOT/data/PlantVillage/"
fi

# 检查预训练权重
if [ ! -f "$PROJECT_ROOT/checkpoints/fasternet_t0.pth" ]; then
    echo "⚠ 预训练权重未找到，正在下载..."
    mkdir -p "$PROJECT_ROOT/checkpoints"
    pip install huggingface_hub -q
    python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download('timm/fasternet_t0.in1k', 'model.safetensors', local_dir='$PROJECT_ROOT/checkpoints')
print(f'Downloaded to {path}')
" 2>/dev/null || echo "下载失败，请手动上传 fasternet_t0.pth 到 checkpoints/"
fi

echo "✓ 数据检查完成"

# ---------- 3. 训练参数适配（32GB 显存优化）----------
echo ""
echo "[3/6] 适配 AutoDL 环境..."

# 32GB 显存可以加大 batch_size，加速训练
# 原始: batch_size=64, workers=4
# AutoDL 32GB: batch_size=128, workers=8
echo "✓ 32GB 显存建议 batch_size=128, workers=8"

# ---------- 4. 验证环境 ----------
echo ""
echo "[4/6] 验证训练环境..."

cd "$PROJECT_ROOT"

python -c "
import sys
sys.path.insert(0, '.')
import torch
from models.msca_fasternet import fasternet_t0_full, fasternet_t0_baseline

# 测试模型构建
model = fasternet_t0_full(num_classes=102, pretrained_backbone=None)
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    y = model(x)
print(f'✓ 模型前向传播正常: input {x.shape} → output {y.shape}')

# 检查 GPU
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
    print(f'✓ GPU: {torch.cuda.get_device_name(0)} ({mem:.1f} GB)')
else:
    print('✗ GPU 不可用!')
"

echo "✓ 环境验证完成"

# ---------- 5. 训练计划 ----------
echo ""
echo "[5/6] 训练计划概览"
echo ""
echo "========================================="
echo "  阶段1: 消融实验 (IP102, 各150 epochs)"
echo "========================================="
echo "  A) baseline        → bash run_train.sh baseline"
echo "  B) msca_only       → bash run_train.sh msca"
echo "  C) fusion_only     → bash run_train.sh fusion"
echo "  D) full_model      → bash run_train.sh full"
echo ""
echo "========================================="
echo "  阶段2: 注意力对比 (IP102, 各150 epochs)"
echo "========================================="
echo "  E) attention_se    → bash run_train.sh attention_se"
echo "  F) attention_cbam  → bash run_train.sh attention_cbam"
echo "  G) attention_eca   → bash run_train.sh attention_eca"
echo "  H) attention_sk    → bash run_train.sh attention_sk"
echo ""
echo "========================================="
echo "  阶段3: 轻量模型对比 (IP102, 各150 epochs)"
echo "========================================="
echo "  I) mobilenetv3     → python scripts/train_comparison.py ..."
echo "  J) shufflenetv2    → python scripts/train_comparison.py ..."
echo "  K) ghostnetv2      → python scripts/train_comparison.py ..."
echo "  L) efficientnet    → python scripts/train_comparison.py ..."
echo ""
echo "========================================="
echo "  阶段4: 泛化验证 (PlantVillage, 各100 epochs)"
echo "========================================="
echo "  M) baseline_pv     → bash run_train.sh baseline plantvillage"
echo "  N) full_pv         → bash run_train.sh full plantvillage"
echo ""
echo "========================================="
echo "  阶段5: 多种子统计 (3 seeds × full model)"
echo "========================================="
echo "  O) multiseed       → bash run_multiseed.sh"
echo ""

# ---------- 6. 准备训练脚本 ----------
echo "[6/6] 生成训练启动脚本..."

cat > "$PROJECT_ROOT/run_train.sh" << 'TRAIN_EOF'
#!/bin/bash
# 单次训练启动脚本 (支持HDF5缓存加速)
# 用法: bash run_train.sh [model] [dataset]
# 示例: bash run_train.sh full ip102
#        bash run_train.sh baseline plantvillage

MODEL=${1:-full}
DATASET=${2:-ip102}
EPOCHS=150
BATCH_SIZE=128
WORKERS=4        # 缓存模式降低workers(数据读取已极快)
SEED=42
CACHE_DIR="/root/autodl-tmp/cache"

# PlantVillage epoch 少一些
if [ "$DATASET" = "plantvillage" ]; then
    EPOCHS=100
fi

# 缓存路径
if [ "$DATASET" = "ip102" ]; then
    CACHE_PATH="$CACHE_DIR/ip102.h5"
else
    CACHE_PATH="$CACHE_DIR/plantvillage.h5"
fi

echo "========================================="
echo "  训练: ${MODEL} | 数据集: ${DATASET}"
echo "  Epochs: ${EPOCHS} | Batch: ${BATCH_SIZE}"
echo "  Cache: ${CACHE_PATH}"
echo "========================================="

python scripts/train.py \
    --dataset ${DATASET} \
    --model ${MODEL} \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --workers ${WORKERS} \
    --seed ${SEED} \
    --gpu 0 \
    --amp \
    --use-tensorboard \
    --cache-dir ${CACHE_PATH} \
    --output-dir /root/autodl-tmp/checkpoints \
    --pretrained /root/autodl-tmp/MSCA-FasterNet/checkpoints/fasternet_t0.pth \
    2>&1 | tee /root/autodl-tmp/logs/${DATASET}_${MODEL}_train.log

echo "✓ ${MODEL} on ${DATASET} 训练完成"
TRAIN_EOF

cat > "$PROJECT_ROOT/run_multiseed.sh" << 'SEED_EOF'
#!/bin/bash
# 多种子统计实验 (HDF5缓存加速版)
# 3 seeds × full model × IP102

SEEDS="42 123 456"
MODEL=full
DATASET=ip102
CACHE_PATH="/root/autodl-tmp/cache/ip102.h5"

for SEED in $SEEDS; do
    echo "========================================="
    echo "  Seed ${SEED} | ${MODEL} | ${DATASET}"
    echo "========================================="

    python scripts/train.py \
        --dataset ${DATASET} \
        --model ${MODEL} \
        --epochs 150 \
        --batch-size 128 \
        --workers 4 \
        --seed ${SEED} \
        --gpu 0 \
        --amp \
        --use-tensorboard \
        --cache-dir ${CACHE_PATH} \
        --output-dir /root/autodl-tmp/checkpoints \
        --pretrained /root/autodl-tmp/MSCA-FasterNet/checkpoints/fasternet_t0.pth \
        2>&1 | tee /root/autodl-tmp/logs/${DATASET}_${MODEL}_seed${SEED}.log
done

# 运行统计检验
echo "运行多种子统计检验..."
python scripts/run_multiseed.py \
    --dataset ip102 \
    --seeds 42 123 456 \
    --gpu 0 \
    --output-dir /root/autodl-tmp/results/multiseed

echo "✓ 多种子实验完成"
SEED_EOF

cat > "$PROJECT_ROOT/run_all.sh" << 'ALL_EOF'
#!/bin/bash
# ========== 全量训练流水线 (HDF5加速版) ==========
# 预估总时间（32GB GPU, batch=128, HDF5缓存）:
#   预处理缓存: ~10min (一次性)
#   消融 4个 × ~1.5h = ~6h     (原3h, 加速2x)
#   注意力对比 4个 × ~1.5h = ~6h
#   轻量对比 4个 × ~2h = ~8h    (timm模型加载较慢)
#   泛化验证 2个 × ~0.5h = ~1h
#   多种子 3个 × ~1.5h = ~4.5h
#   总计: ~25.5h (原47h, 节省46%!)
#   费用: 25.5h × ¥1.58/h ≈ ¥40

CACHE_DIR="/root/autodl-tmp/cache"
IP102_CACHE="$CACHE_DIR/ip102.h5"
PV_CACHE="$CACHE_DIR/plantvillage.h5"

mkdir -p /root/autodl-tmp/logs

echo "======== 阶段0: HDF5预处理缓存 (~10min) ========"
if [ ! -f "$IP102_CACHE" ]; then
    python scripts/preprocess_to_hdf5.py --dataset ip102 --data-dir data/IP102 --output "$IP102_CACHE"
else
    echo "✓ IP102缓存已存在"
fi
if [ ! -f "$PV_CACHE" ]; then
    python scripts/preprocess_to_hdf5.py --dataset plantvillage --data-dir data/PlantVillage --output "$PV_CACHE"
else
    echo "✓ PlantVillage缓存已存在"
fi

echo "======== 阶段1: 消融实验 (IP102) ========"
for MODEL in baseline msca fusion full; do
    bash run_train.sh ${MODEL} ip102
done

echo "======== 阶段2: 注意力对比 (IP102) ========"
for MODEL in attention_se attention_cbam attention_eca attention_sk; do
    bash run_train.sh ${MODEL} ip102
done

echo "======== 阶段3: 轻量模型对比 (IP102) ========"
for TIMM_MODEL in mobilenetv3_small_100 shufflenetv2_x0_5 ghostnetv2_100 efficientnet_lite0; do
    python scripts/train_comparison.py \
        --model ${TIMM_MODEL} \
        --dataset ip102 \
        --epochs 150 \
        --batch-size 128 \
        --workers 4 \
        --gpu 0 \
        --output-dir /root/autodl-tmp/checkpoints \
        2>&1 | tee /root/autodl-tmp/logs/ip102_${TIMM_MODEL}_train.log
done

echo "======== 阶段4: 泛化验证 (PlantVillage) ========"
for MODEL in baseline full; do
    bash run_train.sh ${MODEL} plantvillage
done

echo "======== 阶段5: 多种子统计 ========"
bash run_multiseed.sh

echo "========================================="
echo "  🎉 全部实验训练完成！"
echo "========================================="
ALL_EOF

chmod +x "$PROJECT_ROOT"/run_*.sh

echo ""
echo "✓ 全部部署脚本生成完成！"
echo ""
echo "========================================="
echo "  🚀 快速开始"
echo "========================================="
echo ""
echo "  方式1: 单次训练"
echo "    cd /root/autodl-tmp/MSCA-FasterNet"
echo "    bash run_train.sh full ip102"
echo ""
echo "  方式2: 全量流水线（约47小时）"
echo "    cd /root/autodl-tmp/MSCA-FasterNet"
echo "    nohup bash run_all.sh > /root/autodl-tmp/full_pipeline.log 2>&1 &"
echo ""
echo "  方式3: 后台单任务（断开SSH不断训练）"
echo "    nohup bash run_train.sh full ip102 > /root/autodl-tmp/logs/full_train.log 2>&1 &"
echo "    tail -f /root/autodl-tmp/logs/full_train.log  # 查看实时日志"
echo ""
