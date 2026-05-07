@echo off
REM MSCA-FasterNet Evaluation Launcher

set KMP_DUPLICATE_LIB_OK=TRUE

echo ============================================
echo  MSCA-FasterNet Evaluation
echo ============================================

python scripts\evaluate.py %*

pause
