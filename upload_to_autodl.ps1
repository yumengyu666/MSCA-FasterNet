# MSCA-FasterNet AutoDL 上传脚本
# 在本地 PowerShell 中执行
# 用法: .\upload_to_autodl.ps1 -Port <SSH端口> -Host <SSH地址>

param(
    [Parameter(Mandatory=$true)]
    [string]$Port,
    
    [Parameter(Mandatory=$false)]
    [string]$Host = "connect.westb.seetacloud.com",
    
    [Parameter(Mandatory=$false)]
    [string]$LocalProject = "D:\Project",
    
    [Parameter(Mandatory=$false)]
    [string]$RemoteBase = "/root/autodl-tmp"
)

$Remote = "root@${Host}"
$ScpBase = "scp -P $Port -r"

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  MSCA-FasterNet → AutoDL 上传工具" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "目标: ${Remote}:${RemoteBase}" -ForegroundColor Yellow
Write-Host ""

# 1. 上传代码（通过 Git 更轻量）
Write-Host "[1/4] 上传项目代码..." -ForegroundColor Green
& ssh -p $Port "$Remote" "mkdir -p ${RemoteBase}/MSCA-FasterNet"
& scp -P $Port -r "$LocalProject\models" "${Remote}:${RemoteBase}/MSCA-FasterNet/"
& scp -P $Port -r "$LocalProject\scripts" "${Remote}:${RemoteBase}/MSCA-FasterNet/"
& scp -P $Port -r "$LocalProject\configs" "${Remote}:${RemoteBase}/MSCA-FasterNet/"
& scp -P $Port -r "$LocalProject\datasets" "${Remote}:${RemoteBase}/MSCA-FasterNet/"
& scp -P $Port -r "$LocalProject\utils" "${Remote}:${RemoteBase}/MSCA-FasterNet/"
& scp -P $Port -r "$LocalProject\visualization" "${Remote}:${RemoteBase}/MSCA-FasterNet/"
& scp -P $Port "$LocalProject\requirements.txt" "${Remote}:${RemoteBase}/MSCA-FasterNet/"
& scp -P $Port "$LocalProject\autodl_deploy.sh" "${Remote}:${RemoteBase}/MSCA-FasterNet/"
Write-Host "✓ 代码上传完成" -ForegroundColor Green

# 2. 上传数据集
Write-Host ""
Write-Host "[2/4] 上传 IP102 数据集 (~1.8GB)..." -ForegroundColor Green
& ssh -p $Port "$Remote" "mkdir -p ${RemoteBase}/MSCA-FasterNet/data"
& scp -P $Port -r "$LocalProject\data\IP102" "${Remote}:${RemoteBase}/MSCA-FasterNet/data/"
Write-Host "✓ IP102 上传完成" -ForegroundColor Green

Write-Host ""
Write-Host "[3/4] 上传 PlantVillage 数据集 (~0.3GB)..." -ForegroundColor Green
& scp -P $Port -r "$LocalProject\data\PlantVillage" "${Remote}:${RemoteBase}/MSCA-FasterNet/data/"
Write-Host "✓ PlantVillage 上传完成" -ForegroundColor Green

# 3. 上传预训练权重
Write-Host ""
Write-Host "[4/4] 上传预训练权重 (~238MB)..." -ForegroundColor Green
& ssh -p $Port "$Remote" "mkdir -p ${RemoteBase}/MSCA-FasterNet/checkpoints"
& scp -P $Port "$LocalProject\checkpoints\fasternet_t0.pth" "${Remote}:${RemoteBase}/MSCA-FasterNet/checkpoints/"
Write-Host "✓ 权重上传完成" -ForegroundColor Green

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  ✅ 全部上传完成！" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步：SSH 连接到 AutoDL 后执行：" -ForegroundColor Yellow
Write-Host "  cd ${RemoteBase}/MSCA-FasterNet" -ForegroundColor White
Write-Host "  bash autodl_deploy.sh" -ForegroundColor White
Write-Host ""
