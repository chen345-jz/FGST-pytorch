@echo off
setlocal

cd /d "%~dp0"

set "LIBTORCH_ROOT=D:\cppsoft\libtorch"
set "PATH=%LIBTORCH_ROOT%\lib;%LIBTORCH_ROOT%\bin;%PATH%"

if "%~1"=="" goto usage

if /I "%~1"=="train" (
  .\build\Release\mmwave_app.exe train .\config\example.cfg
  goto end
)

if /I "%~1"=="eval" (
  .\build\Release\mmwave_app.exe eval .\config\example.cfg
  goto end
)

if /I "%~1"=="predict" (
  if "%~2"=="" (
    echo Please provide sample csv path for predict.
    echo Example: run_fgst.bat predict .\data\sequences\seq_0001.csv
    goto end
  )
  .\build\Release\mmwave_app.exe predict .\config\example.cfg "%~2"
  goto end
)

:usage
echo Usage:
echo   run_fgst.bat train
echo   run_fgst.bat eval
echo   run_fgst.bat predict ^<sample_csv_path^>

:end
endlocal
