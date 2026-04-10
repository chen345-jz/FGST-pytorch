@echo off
setlocal

set PY=D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe
set RADAR_DIR=.\2s

if not exist "%PY%" (
  echo Python env not found: %PY%
  exit /b 1
)

"%PY%" .\python\radar_baseline.py --radar_base_dir "%RADAR_DIR%" --split_ratio 0.7 --num_epochs 50 --eval_interval 1 --early_stopping_patience 20 --batch_size 16 --accumulation_steps 2 --learning_rate 0.0003 --seed 42
exit /b %ERRORLEVEL%
