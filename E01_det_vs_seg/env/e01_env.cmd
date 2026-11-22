@echo off
REM E01 session environment. Usage:  call G:\BrainMetShare\experiments\E01_det_vs_seg\env\e01_env.cmd
REM Points nnU-Net at E01\data only. code\e01\common.py sets the same values in-process and asserts them.
set "E01=G:\BrainMetShare\experiments\E01_det_vs_seg"
set "E01_PY=G:\BrainMetShare\1.60_artifacts\venv-1.60\Scripts\python.exe"
set "nnUNet_raw=%E01%\data\nnUNet_raw"
set "nnUNet_preprocessed=%E01%\data\nnUNet_preprocessed"
set "nnUNet_results=%E01%\data\nnUNet_results"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=%E01%\code"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "FOR_DISABLE_CONSOLE_CTRL_HANDLER=1"
REM GPU is selected by name inside launch_train.py (select_gpu), not here.
