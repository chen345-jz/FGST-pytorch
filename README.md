# mmWave Gait Recognition (PyTorch)

Current model:
- `python/fgst_reid.py` (DSFE -> PGBP -> LGTE, following the fine-grained spatial-temporal gait recognition paper)
- Input point attributes are 4-D: `x, y, z, velocity`.

## Project Layout
- `python/fgst_reid.py`: DSFE/PGBP/LGTE training + ReID evaluation script
- `run_fgst_reid.bat`: one-click Windows runner for DSFE/PGBP/LGTE reproduction
- `2s/p_*/`: dataset samples (`.npy`)
- `checkpoints_fgst/`: current checkpoints and training plots

## Environment
Recommended Python environment:
- `D:\cppsoft\venvs\mmwave_pt`

Verify:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Run
```powershell
run_fgst_reid.bat
```

Or run directly:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\fgst_reid.py train --radar_base_dir ".\2s" --split_ratio 0.7 --num_epochs 50
```

Evaluate only:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\fgst_reid.py eval --radar_base_dir ".\2s" --split_ratio 0.7 --seed 42
```

Predict one sample:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\fgst_reid.py predict .\2s\p_1\0.npy --radar_base_dir ".\2s" --split_ratio 0.7 --seed 42
```

## Main Metrics
- Rank-1 (Top-1 in retrieval setting)
- mAP
