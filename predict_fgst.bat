@echo off
setlocal

set PY=D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe
set RADAR_DIR=.\2s
set CKPT_DIR=checkpoints_fgst
set SAMPLE=%~1

if "%SAMPLE%"=="" (
  set SAMPLE=.\2s\p_1\0.npy
)

if not exist "%PY%" (
  echo Python env not found: %PY%
  exit /b 1
)

if not exist "%CKPT_DIR%\best_model_fgst.pth" (
  echo Checkpoint not found: %CKPT_DIR%\best_model_fgst.pth
  echo Please run train_fgst.bat first.
  exit /b 1
)

if not exist "%SAMPLE%" (
  echo Sample not found: %SAMPLE%
  exit /b 1
)

"%PY%" .\python\fgst_reid.py predict "%SAMPLE%" --radar_base_dir "%RADAR_DIR%" --split_ratio 0.7 --seed 42 --checkpoint_dir "%CKPT_DIR%"
exit /b %ERRORLEVEL%
