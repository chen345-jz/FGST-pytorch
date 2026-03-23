@echo off
setlocal

set PY=D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe
set CFG=.\config\fgst_2s_pytorch.cfg

if not exist "%PY%" (
  echo Python env not found: %PY%
  exit /b 1
)

if "%1"=="" (
  echo Usage:
  echo   run_fgst_pytorch.bat train
  echo   run_fgst_pytorch.bat eval
  echo   run_fgst_pytorch.bat predict sample.npy
  exit /b 1
)

if /I "%1"=="train" (
  "%PY%" .\python\mmwave_pt.py train "%CFG%"
  exit /b %ERRORLEVEL%
)

if /I "%1"=="eval" (
  "%PY%" .\python\mmwave_pt.py eval "%CFG%"
  exit /b %ERRORLEVEL%
)

if /I "%1"=="predict" (
  if "%2"=="" (
    echo predict mode requires sample npy path
    exit /b 1
  )
  "%PY%" .\python\mmwave_pt.py predict "%CFG%" "%2"
  exit /b %ERRORLEVEL%
)

echo Unknown mode: %1
exit /b 1
