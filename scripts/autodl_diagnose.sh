#!/bin/bash
# ============================================================
#  AutoDL 一键诊断 & 清理脚本
#  用法: bash scripts/autodl_diagnose.sh [clean]
#  参数:
#    (无参数)  → 仅诊断，不杀进程
#    clean     → 诊断 + 清理残留进程
# ============================================================

set -e

echo "╔══════════════════════════════════════════════════╗"
echo "║     AutoDL 系统诊断 & 进程清理工具 v1.0          ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

ACTION=${1:-"diagnose"}
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# ---------- 1. 系统资源概览 ----------
echo "━━━ [1/6] 系统资源概览 ━━━"
echo ""
free -h
echo ""
echo "磁盘使用:"
df -h / /root/autodl-tmp 2>/dev/null || df -h /
echo ""

# ---------- 2. CPU 占用 Top 15 ----------
echo "━━━ [2/6] CPU 占用 TOP 15 进程 ━━━"
echo ""
if command -v ps >/dev/null 2>&1; then
    ps aux --sort=-%cpu | head -16
else
    echo "ps 命令不可用"
fi
echo ""

# ---------- 3. GPU 状态 ----------
echo "━━━ [3/6] GPU 状态 ━━━"
echo ""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
    echo ""
    echo "GPU 进程详情:"
    nvidia-smi pmon -c 1 2>/dev/null || echo "(pmon不可用)"
else
    echo "⚠ nvidia-smi 不可用，可能无GPU或驱动未加载"
fi
echo ""

# ---------- 4. Python 进程扫描 ----------
echo "━━━ [4/6] Python 相关进程 ━━━"
echo ""
PYTHON_PROCS=$(ps aux | grep -E 'python|train\.py|DataLoader' | grep -v grep | wc -l)
echo "发现 ${PYTHON_PROCS} 个 Python 相关进程:"
echo ""
ps aux | grep -E 'python|train\.py|DataLoader' | grep -v grep | \
    awk '{printf "  PID=%-6s CPU=%-6s MEM=%-6s CMD=%s\n", $2, $3"%", $4"%", $11" "$12" "$13" "$14}'
echo ""

# ---------- 5. DataLoader worker 泄漏检测 ----------
echo "━━━ [5/6] DataLoader Worker 泄漏检测 ━━━"
echo ""
WORKER_COUNT=$(ps aux | grep -E 'DataLoader|multiprocessing' | grep -v grep | wc -l)
if [ "$WORKER_COUNT" -gt 20 ]; then
    echo "⚠️  检测到 ${WORKER_COUNT} 个 DataLoader/Multiprocessing 进程"
    echo "   可能存在 worker 泄漏！建议执行: bash autodl_diagnose.sh clean"
else
    echo "✅ DataLoader Worker 数量正常 (${WORKER_COUNT})"
fi
echo ""

# ---------- 6. Jupyter 内核检测 ----------
echo "━━━ [6/6] Jupyter Lab 内核状态 ━━━"
echo ""
if command -v jupyter >/dev/null 2>&1; then
    jupyter kernelspec list 2>/dev/null || echo "(jupyter命令可用但无法列出内核)"
    # 检查运行中的内核
    RUNNING_KERNELS=$(ps aux | grep 'ipykernel|jupyter' | grep -v grep | wc -l)
    echo "运行中的 Jupyter/IPython 内核: ${RUNNING_KERNELS}"
else
    echo "(未安装jupyter或不在PATH中)"
fi
echo ""

# ========== 诊断汇总 ==========
echo "╔══════════════════════════════════════════════════╗"
echo "║                 诊断汇总                          ║"
echo "╚══════════════════════════════════════════════════╝"

TOTAL_CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEM_USED=$(free -m | grep Mem | awk '{printf "%.0f/%.0fMB (%.0f%%)", $3, $2, $3/$2*100}')
GPU_USAGE="N/A"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk '{print $1"/"$2"MB"}')
fi

echo ""
echo "  📊 时间: ${TIMESTAMP}"
echo "  💻 CPU 总占用: ${TOTAL_CPU_USAGE}%"
echo "  🧠 内存使用: ${MEM_USED}"
if [ "${GPU_USAGE}" != "N/A" ]; then
    echo "  🎮 GPU 利用率: ${GPU_USAGE}%"
    echo "  🎮 GPU 显存:   ${GPU_MEM}"
fi
echo "  🐍 Python 进程: ${PYTHON_PROCS} 个"
echo "  🔧 DL Workers:  ${WORKER_COUNT} 个"
echo ""

# 判断是否异常
IS_ABNORMAL=0
if [ "${PYTHON_PROCS}" -gt 5 ]; then
    echo "  ⚠️  Python 进程过多 (>5)，可能有残留训练任务"
    IS_ABNORMAL=1
fi
if [ "${WORKER_COUNT}" -gt 20 ]; then
    echo "  ⚠️  DataLoader Worker 可能泄漏"
    IS_ABNORMAL=1
fi
if [ "${GPU_USAGE}" != "N/A" ] && [ "${GPU_USAGE}" -lt 10 ] && [ "${PYTHON_PROCS}" -gt 0 ]; then
    echo "  ⚠️  GPU 空闲但有Python进程在跑 → 可能在做数据预处理或死循环"
    IS_ABNORMAL=1
fi

if [ "$IS_ABNORMAL" -eq 0 ]; then
    echo "  ✅ 系统状态正常"
else
    echo "  🔴 检测到异常！"
fi

# ========== 清理操作 ==========
if [ "$ACTION" = "clean" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║              执行清理操作                        ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""
    
    # 杀掉 train.py 进程
    TRAIN_PIDS=$(ps aux | grep 'train\.py' | grep -v grep | awk '{print $2}')
    if [ -n "$TRAIN_PIDS" ]; then
        echo "🛑 停止训练进程 (train.py):"
        echo "$TRAIN_PIDS" | while read pid; do
            kill -9 "$pid" 2>/dev/null && echo "   ✓ 已杀 PID=$pid" || echo "   ✗ PID=$pid 杀失败"
        done
    else
        echo "✅ 无运行中的 train.py 进程"
    fi
    
    # 杀掉孤立的数据加载worker
    ORPHAN_WORKERS=$(ps aux | grep -E 'DataLoader.*worker|multiprocessing.*fork' | grep -v grep | awk '{print $2}')
    if [ -n "$ORPHAN_WORKERS" ]; then
        echo "🛑 清理孤立的 DataLoader Worker:"
        echo "$ORPHAN_WORKERS" | while read pid; do
            kill -9 "$pid" 2>/dev/null && echo "   ✓ 已杀 Worker PID=$pid"
        done
    else
        echo "✅ 无孤立Worker"
    fi
    
    # 杀掉所有用户Python进程（可选，谨慎）
    ALL_PYTHON=$(ps aux | grep 'python' | grep -v grep | grep -v 'autodl_diagnose' | awk '{print $2}')
    if [ -n "$ALL_PYTHON" ]; then
        echo ""
        echo "⚠️  发现其他Python进程:"
        ps aux | grep python | grep -v grep | grep -v 'autodl_diagnose'
        echo ""
        read -p "  是否杀掉所有Python进程? (y/N) " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "$ALL_PYTHON" | while read pid; do
                kill -9 "$pid" 2>/dev/null && echo "   ✓ 已杀 PID=$pid"
            done
        else
            echo "   跳过"
        fi
    fi
    
    # 清理显存
    echo ""
    if command -v nvidia-smi >/dev/null 2>&1; then
        # 使用python释放所有GPU显存
        python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('✅ GPU 显存已释放')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free = torch.cuda.mem_get_info(i)[0] / 1024**3
        total = props.total_mem / 1024**3
        print(f'   GPU {i}: {free:.1f}/{total:.1f} GB 可用')
" 2>/dev/null
    fi
    
    echo ""
    echo "━━━ 清理完成，验证系统状态 ━━━"
    echo ""
    REMAINING=$(ps aux | grep python | grep -v grep | wc -l)
    NEW_GPU="N/A"
    if command -v nvidia-smi >/dev/null 2>&1; then
        NEW_GPU=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
    fi
    echo "  剩余Python进程: ${REMAINING}"
    echo "  GPU利用率:      ${NEW_GPU}%"
    echo ""
    echo "✅ 清理完成！现在可以安全启动新任务了。"
fi

echo ""
echo "提示: 运行清理请使用: bash autodl_diagnose.sh clean"
