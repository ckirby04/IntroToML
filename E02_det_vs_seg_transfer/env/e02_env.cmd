@echo off
REM E02 session environment. Usage:  call G:\BrainMetShare\experiments\E02_det_vs_seg_transfer\env\e02_env.cmd
REM Points nnU-Net at E02\data only. code\e02\common.py sets the same values in-process and asserts them.
set "E02=G:\BrainMetShare\experiments\E02_det_vs_seg_transfer"
set "E02_PY=G:\BrainMetShare\1.60_artifacts\venv-1.60\Scripts\python.exe"
set "nnUNet_raw=%E02%\data\nnUNet_raw"
set "nnUNet_preprocessed=%E02%\data\nnUNet_preprocessed"
set "nnUNet_results=%E02%\data\nnUNet_results"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONPATH=%E02%\code"
set "nnUNet_n_proc_DA=6"
set "CUDA_DEVICE_ORDER=PCI_BUS_ID"
set "FOR_DISABLE_CONSOLE_CTRL_HANDLER=1"
