# MSCA-FasterNet AutoDL 远程训练指南

## 📋 AutoDL 镜像选择

推荐选择：
- **基础镜像**: PyTorch 2.0+ / Python 3.10+ / CUDA 12.x
- **GPU**: vGPU-32GB (32G显存, ¥1.58/时) ← 你已选这个 ✅

## 📤 数据上传（3种方式）

### 方式1：AutoDL 文件管理器上传（适合小文件）
1. 打开 AutoDL 控制台 → 容器实例 → 自定义服务 → JupyterLab
2. 直接拖拽上传（有大小限制，适合代码）

### 方式2：SCP 命令上传（推荐，适合大文件）
```powershell
# 先获取 AutoDL SSH 连接信息（在实例详情页查看）
# 格式: ssh -p <端口> root@connect.westb.seetacloud.com

# 上传代码（从本地 PowerShell 执行）
scp -P <端口> -r D:\Project\MSCA-FasterNet root@connect.westb.seetacloud.com:/root/autodl-tmp/

# 上传数据集
scp -P <端口> -r D:\Project\data\IP102 root@connect.westb.seetacloud.com:/root/autodl-tmp/MSCA-FasterNet/data/
scp -P <端口> -r D:\Project\data\PlantVillage root@connect.westb.seetacloud.com:/root/autodl-tmp/MSCA-FasterNet/data/

# 上传预训练权重
scp -P <端口> D:\Project\checkpoints\fasternet_t0.pth root@connect.westb.seetacloud.com:/root/autodl-tmp/MSCA-FasterNet/checkpoints/
```

### 方式3：Git 拉取（最快，适合代码）
```bash
# SSH 到 AutoDL 后执行
cd /root/autodl-tmp
git clone https://github.com/yumengyu666/MSCA-FasterNet.git
```
> ⚠️ 数据集和权重在 .gitignore 中，仍需单独上传

### 方式4：AutoDL 学术加速下载（适合有公开链接的数据集）
```bash
# AutoDL 自带学术加速
source /etc/network_turbo  # 开启加速
# 下载后关闭
unset http_proxy https_proxy
```

## 🔧 环境部署

SSH 连接到实例后：

```bash
cd /root/autodl-tmp/MSCA-FasterNet

# 一键部署
bash autodl_deploy.sh
```

或者手动执行：

```bash
# 1. 安装依赖
pip install timm fvcore scikit-learn opencv-python matplotlib seaborn PyYAML tqdm

# 2. 验证环境
python scripts/verify_model.py

# 3. 开始训练
bash run_train.sh full ip102
```

## 🚀 训练命令速查

### 后台运行（断开SSH不会中断）
```bash
# 单个实验
nohup bash run_train.sh full ip102 > logs/full_train.log 2>&1 &

# 查看实时日志
tail -f logs/full_train.log

# 全量流水线（约47小时）
nohup bash run_all.sh > full_pipeline.log 2>&1 &
```

### 单次训练
```bash
# IP102 消融实验
bash run_train.sh baseline ip102    # A: 基线
bash run_train.sh msca ip102        # B: 仅MSCA
bash run_train.sh fusion ip102      # C: 仅融合
bash run_train.sh full ip102        # D: 完整模型

# 注意力对比
bash run_train.sh attention_se ip102
bash run_train.sh attention_cbam ip102
bash run_train.sh attention_eca ip102
bash run_train.sh attention_sk ip102

# 泛化验证 (PlantVillage)
bash run_train.sh baseline plantvillage
bash run_train.sh full plantvillage
```

## 📊 32GB 显存优化建议

| 参数 | 本地 8GB | AutoDL 32GB | 说明 |
|------|---------|-------------|------|
| batch_size | 64 | **128** | 加倍，训练更快更稳 |
| workers | 4 | **8** | 更多数据加载线程 |
| epochs | 150 | 150 | 不变 |
| 预估单次时间 | ~6h | ~3h | 约2倍加速 |

## ⏱️ 训练时间与费用估算

| 阶段 | 实验数 | 单次时间 | 总时间 |
|------|--------|---------|--------|
| 消融实验 | 4 | ~3h | ~12h |
| 注意力对比 | 4 | ~3h | ~12h |
| 轻量模型对比 | 4 | ~3h | ~12h |
| 泛化验证 | 2 | ~1h | ~2h |
| 多种子统计 | 3 | ~3h | ~9h |
| **总计** | **17** | - | **~47h** |

💰 **费用**: 47h × ¥1.58/h ≈ **¥74**

> 如果只跑核心实验（消融4 + 注意力对比4 + 泛化2 = 10个），约 **30h / ¥47**

## 📥 结果下载

训练完成后，从 AutoDL 下载结果到本地：

```powershell
# 下载 checkpoints
scp -P <端口> -r root@connect.westb.seetacloud.com:/root/autodl-tmp/checkpoints D:\Project\autodl_checkpoints

# 下载结果日志
scp -P <端口> -r root@connect.westb.seetacloud.com:/root/autodl-tmp/results D:\Project\autodl_results

# 下载训练日志
scp -P <端口> -r root@connect.westb.seetacloud.com:/root/autodl-tmp/logs D:\Project\autodl_logs
```

## ⚠️ 注意事项

1. **AutoDL 数据持久化**: `/root/autodl-tmp/` 是持久目录，实例关机后数据不丢失；`/root/` 下其他目录关机会清空
2. **定时快照**: 建议每完成一个阶段就下载一次结果到本地
3. **TensorBoard**: 可通过 AutoDL 的自定义服务端口映射查看
4. **OOM 处理**: 如果 batch_size=128 显存不足，降到 96 或 64
5. **断点续训**: 如果中断，用 `--resume checkpoints/xxx/checkpoint_epochXXX.pth` 继续

## 🔗 AutoDL 常用操作

```bash
# 查看 GPU 使用情况
nvidia-smi

# 查看后台任务
jobs -l

# 杀掉后台训练
kill %1  # 或 kill <PID>

# 查看磁盘空间
df -h /root/autodl-tmp
```
