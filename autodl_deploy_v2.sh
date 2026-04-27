#!/bin/bash
# ============================================================
#  MSCA-FasterNet AutoDL 全自动部署 v2.0
#  含: 诊断清理 → HDF5预缓存 → 高速训练
#
#  用法:
#    bash autodl_deploy_v2.sh              # 完整流程
#    bash autodl_deploy_v2.sh clean        # 仅诊断+清理
#    bash autodl_deploy_v2.sh cache        # 仅预处理缓存
#    bash autodl_deploy_v2.sh train full   # 跳过缓存,直接训练
# ============================================================

set -e

# ===================== 配置区 =====================
PROJECT_ROOT="/root/autodl-tmp/MSCA-FasterNet"
CACHE_DIR="/root/autodl-tmp/cache"
DATA_DIR_IP102="$PROJECT_ROOT/data/IP102"
DATA_DIR_PV="$PROJECT_ROOT/data/PlantVillage"
CHECKPOINT="$PROJECT_ROOT/checkpoints/fasternet_t0.pth"
LOG_DIR="/root/autodl-tmp/logs"
OUTPUT_DIR="/root/autodl-tmp/checkpoints"

BATCH_SIZE=128       # 32GB显存优化值
WORKERS=4            # 缓存模式下workers可降低 (数据读取已极快)
SEED=42
EPOCHS=150
# ===================================================

MODE="${1:-full}"
MODEL="${2:-full}"
DATASET="${3:-ip102}"

echo "╔══════════════════════════════════════════════════════╗"
echo "║   MSCA-FasterNet AutoDL 部署 v2.0 (HDF5加速版)      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo "  模式: $MODE | 模型: $MODEL | 数据集: $DATASET"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

mkdir -p "$CACHE_DIR" "$LOG_DIR" "$OUTPUT_DIR"

# ==================== Step 0: 系统诊断 ====================
step_0_diagnose() {
    echo ""
    echo "━━━ Step 0: 系统诊断 ━━━"
    echo ""

    # GPU状态
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "🎮 GPU信息:"
        nvidia-smi --query-gpu=name,memory.total,utilization.gpu \
            --format=csv,noheader 2>/dev/null || nvidia-smi
        echo ""
    fi

    # CPU & 内存
    echo "💻 系统资源:"
    free -h | head -2
    echo ""

    # 残留进程检查
    PY_COUNT=$(ps aux 2>/dev/null | grep -E 'train\.py|python' | grep -v grep | wc -l)
    if [ "$PY_COUNT" -gt 3 ]; then
        echo "⚠️  检测到 ${PY_COUNT} 个Python进程，建议先清理:"
        echo "   运行: bash scripts/autodl_diagnose.sh clean"
        echo ""
    else
        echo "✅ 进程正常 (${PY_COUNT}个)"
    fi

    # 磁盘空间
    echo ""
    echo "💾 磁盘空间:"
    df -h /root/autodl-tmp 2>/dev/null || df -h /
}

# ==================== Step 1: 清理残留进程 ====================
step_1_clean() {
    echo ""
    echo "━━━ Step 1: 清理残留进程 ━━━"
    echo ""

    # 杀train.py
    for pid in $(ps aux 2>/dev/null | grep 'train\.py' | grep -v grep | awk '{print $2}'); do
        kill -9 "$pid" 2>/dev/null && echo "  🛑 已杀 train.py PID=$pid"
    done

    # 杀孤立worker
    for pid in $(ps aux 2>/dev/null | grep 'DataLoader.*worker' | grep -v grep | awk '{print $2}'); do
        kill -9 "$pid" 2>/dev/null && echo "  🛑 已杀 Worker PID=$pid"
    done

    # 释放显存
    python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('✅ 显存已释放')
" 2>/dev/null || true

    echo "  ✅ 清理完成"
}

# ==================== Step 2: 环境准备 ====================
step_2_env() {
    echo ""
    echo "━━━ Step 2: 环境准备 ━━━"
    echo ""

    cd "$PROJECT_ROOT"

    # PyTorch检测
    python -c "
import torch
print(f'PyTorch {torch.__version__} | CUDA {torch.version.cuda}')
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'GPU: {p.name} ({p.total_mem/1024**3:.1f} GB)')
" || { echo "❌ PyTorch不可用!"; exit 1; }

    # 安装依赖(含h5py用于HDF5)
    echo ""
    echo "安装依赖..."
    pip install h5py timm fvcore scikit-learn opencv-python matplotlib seaborn tqdm PyYAML -q 2>/dev/null
    echo "  ✅ 依赖就绪"
}

# ==================== Step 3: HDF5预缓存 (核心!) ================
step_3_cache() {
    echo ""
    echo "━━━ Step 3: HDF5 预处理缓存 ━━━"
    echo "  这是关键步骤! 将JPEG图片预解码+resize存为HDF5"
    echo "  训练时从HDF5 mmap读取, GPU利用率可达85%+"
    echo ""

    cd "$PROJECT_ROOT"

    IP102_CACHE="$CACHE_DIR/ip102.h5"
    PV_CACHE="$CACHE_DIR/plantvillage.h5"

    CACHE_NEEDED=0

    if [ "$DATASET" = "ip102" ] && [ ! -f "$IP102_CACHE" ]; then
        echo "📦 预处理 IP102 → HDF5 ..."
        python scripts/preprocess_to_hdf5.py \
            --dataset ip102 \
            --data-dir "$DATA_DIR_IP102" \
            --output "$IP102_CACHE" \
            --target-size 256
        CACHE_NEEDED=1
    elif [ "$DATASET" = "ip102" ]; then
        echo "✅ IP102缓存已存在: $IP102_CACHE ($(du -sh $IP102_CACHE | cut -f1))"
    fi

    if [ "$DATASET" = "plantvillage" ] && [ ! -f "$PV_CACHE" ]; then
        echo "📦 预处理 PlantVillage → HDF5 ..."
        python scripts/preprocess_to_hdf5.py \
            --dataset plantvillage \
            --data-dir "$DATA_DIR_PV" \
            --output "$PV_CACHE" \
            --target-size 256
        CACHE_NEEDED=1
    elif [ "$DATASET" = "plantvillage" ]; then
        echo "✅ PlantVillage缓存已存在: $PV_CACHE ($(du -sh $PV_CACHE | cut -f1))"
    fi

    # 同时生成另一个数据集的缓存(可选,为后续实验省时间)
    if [ "$CACHE_NEEDED" = "1" ] || [ "$MODE" = "cache" ]; then
        if [ "$DATASET" != "ip102" ] && [ ! -f "$IP102_CACHE" ] && [ -d "$DATA_DIR_IP102" ]; then
            echo ""
            echo "(顺便预处理IP102缓存供后续使用...)"
            python scripts/preprocess_to_hdf5.py \
                --dataset ip102 --data-dir "$DATA_DIR_IP102" \
                --output "$IP102_CACHE" --target-size 256 2>/dev/null || true
        fi
        if [ "$DATASET" != "plantvillage" ] && [ ! -f "$PV_CACHE" ] && [ -d "$DATA_DIR_PV" ]; then
            echo "(顺便预处理PlantVillage缓存供后续使用...)"
            python scripts/preprocess_to_hdf5.py \
                --dataset plantvillage --data-dir "$DATA_DIR_PV" \
                --output "$PV_CACHE" --target-size 256 2>/dev/null || true
        fi
    fi

    echo ""
    echo "  📦 缓存目录总大小:"
    du -sh "$CACHE_DIR/" 2>/dev/null || echo "  (空)"
}

# ==================== Step 4: 验证模型 + 缓存 ================
step_4_verify() {
    echo ""
    echo "━━━ Step 4: 模型验证 ━━━"
    echo ""

    cd "$PROJECT_ROOT"

    # 确定缓存路径
    if [ "$DATASET" = "ip102" ]; then
        CACHE_PATH="$CACHE_DIR/ip102.h5"
    else
        CACHE_PATH="$CACHE_DIR/plantvillage.h5"
    fi

    python -c "
import sys
sys.path.insert(0, '.')
import torch
from models.msca_fasternet import fasternet_t0_full

# 模型构建测试
num_classes = 102 if '$DATASET' == 'ip102' else 15
model = fasternet_t0_full(num_classes=num_classes, pretrained_backbone=None)
model = model.cuda()
x = torch.randn(2, 3, 224, 224).cuda()
with torch.no_grad():
    y = model(x)
print(f'✅ 模型前向正常: {x.shape} → {y.shape}')
print(f'   参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')

# 缓存读取测试
from datasets.hdf5_dataset import HDF5CachedDataset
ds = HDF5CachedDataset('$CACHE_PATH', 'train', input_size=224)
img, label = ds[0]
print(f'✅ 缓存读取正常: img={img.shape}, label={label}')
print(f'   数据集长度: {len(ds)}, 类别数: {ds.num_classes}')
ds.close()
print('✅ 全部验证通过!')
"
}

# ==================== Step 5: 开始训练 ====================
step_5_train() {
    echo ""
    echo "━━━ Step 5: 开始训练 ━━━"
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║           🚀 训练参数                              ║"
    echo "╠══════════════════════════════════════════════════╣"
    echo "║  模型:     $MODEL                                  ║"
    echo "║  数据集:   $DATASET                                ║"
    echo "║  Epochs:   $EPOCHS                                 ║"
    echo "║  Batch:    $BATCH_SIZE                             ║"
    echo "║  Workers:  $WORKERS                                ║"
    echo "║  Seed:     $SEED                                   ║"
    echo "║  AMP:      ✅ 开启                                 ║"
    echo "║  Cache:    ✅ HDF5 加速                            ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""

    cd "$PROJECT_ROOT"

    # 缓存路径
    if [ "$DATASET" = "ip102" ]; then
        CACHE_PATH="$CACHE_DIR/ip102.h5"
        EPOCHS=150
    else
        CACHE_PATH="$CACHE_DIR/plantvillage.h5"
        EPOCHS=100
    fi

    LOG_FILE="$LOG_DIR/${DATASET}_${MODEL}_$(date '+%m%d_%H%M').log"

    echo "📝 日志: $LOG_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    exec python scripts/train.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --workers "$WORKERS" \
        --seed "$SEED" \
        --gpu 0 \
        --amp \
        --use-tensorboard \
        --cache-dir "$CACHE_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --pretrained "$CHECKPOINT" \
        --print-freq 50 \
        --save-freq 10 \
        2>&1 | tee "$LOG_FILE"
}

# ==================== 主流程调度 ====================
case "$MODE" in
    diagnose|diag)
        step_0_diagnose
        ;;
    clean)
        step_0_diagnose
        step_1_clean
        ;;
    cache)
        step_2_env
        step_3_cache
        ;;
    train)
        step_0_diagnose
        step_2_env
        step_4_verify
        step_5_train
        ;;
    full|"")
        step_0_diagnose
        step_1_clean
        step_2_env
        step_3_cache
        step_4_verify
        step_5_train
        ;;
    *)
        echo "用法: $0 [clean|cache|train|full] [model] [dataset]"
        echo ""
        echo "  clean   - 诊断系统并清理残留进程"
        echo "  cache   - 只做HDF5预处理缓存"
        echo "  train   - 跳过缓存和清理, 直接训练"
        echo "  full    - 完整流程 (默认): 诊断→清理→环境→缓存→训练"
        echo ""
        echo "示例:"
        echo "  $0                          # 完整流程, 默认full模型+ip102"
        echo "  $0 full baseline ip102      # 基线消融实验"
        echo "  $0 clean                    # 只做诊断清理"
        echo "  $0 cache                    # 只做预缓存"
        exit 1
        ;;
esac
