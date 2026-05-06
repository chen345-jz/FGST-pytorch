# PyTorch Usage (DSFE/PGBP/LGTE)

## Main Script
- `python/fgst_reid.py`
- Input point attributes are 4-D: `x, y, z, velocity`.

## One-Click Run
```powershell
run_fgst_reid.bat
```

## Manual Run
Train:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\fgst_reid.py train --radar_base_dir ".\2s" --split_ratio 0.7 --num_epochs 50
```

Eval:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\fgst_reid.py eval --radar_base_dir ".\2s" --split_ratio 0.7 --seed 42
```

Predict:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\fgst_reid.py predict .\2s\p_1\0.npy --radar_base_dir ".\2s" --split_ratio 0.7 --seed 42
```
