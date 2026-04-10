# mmWave Gait Recognition (PyTorch)

Current mainline model is:
- `python/radar_baseline.py` (RadarFeatureExtractor: PointNet -> LSTM -> Mean Pooling)

## Project Layout
- `python/radar_baseline.py`: main training + ReID evaluation script
- `run_radar_baseline.bat`: one-click Windows runner for mainline
- `2s/p_*/`: dataset samples (`.npy`)
- `checkpoints_radar_only_baselines/`: baseline checkpoints and plots
- `model/`: other experiment outputs

## Environment
Recommended Python environment:
- `D:\cppsoft\venvs\mmwave_pt`

Verify:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Run (Mainline)
```powershell
run_radar_baseline.bat
```

Or run directly:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\radar_baseline.py --radar_base_dir ".\2s" --split_ratio 0.7 --num_epochs 50
```

## Main Metrics
- Rank-1 (Top-1 in retrieval setting)
- mAP
