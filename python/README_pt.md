# PyTorch Mainline Usage (RadarFeatureExtractor)

## Mainline
- Main script: `python/radar_baseline.py`

## Environment
- Python venv: `D:\cppsoft\venvs\mmwave_pt`
- Use direct python path:
  - `D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe`

## Dataset
- Root directory: `2s`
- Person folders: `p_1 ... p_30`
- Sample files: `*.npy`

## One-click run
- `run_radar_baseline.bat`

## Manual run
Train:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\radar_baseline.py train --radar_base_dir ".\2s" --split_ratio 0.7 --num_epochs 50 --eval_interval 1 --early_stopping_patience 20 --batch_size 16 --accumulation_steps 2 --learning_rate 0.0003 --seed 42
```

Eval:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\radar_baseline.py eval --radar_base_dir ".\2s" --split_ratio 0.7 --seed 42
```

Predict:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\radar_baseline.py predict .\2s\p_1\0.npy --radar_base_dir ".\2s" --split_ratio 0.7 --seed 42
```

## Metrics
- Rank-1 (Top-1 under retrieval protocol)
- mAP
