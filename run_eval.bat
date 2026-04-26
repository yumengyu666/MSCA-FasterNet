@echo off
REM MSCA-FasterNet Evaluation Launcher
REM Uses the Project conda environment with CUDA support

set PYTHON=C:\Users\23065\miniconda3\envs\Project\python.exe
set KMP_DUPLICATE_LIB_OK=TRUE

echo ============================================
echo  MSCA-FasterNet Evaluation Launcher
echo  Python: %PYTHON%
echo ============================================

%PYTHON% scripts\evaluate.py %*

pause
