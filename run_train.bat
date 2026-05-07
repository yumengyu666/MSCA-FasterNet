@echo off
REM MSCA-FasterNet Training Launcher

set KMP_DUPLICATE_LIB_OK=TRUE

echo ============================================
echo  MSCA-FasterNet Training
echo ============================================

python scripts\train.py %*

pause
