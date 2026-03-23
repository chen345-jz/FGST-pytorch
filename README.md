# mmWave Gait Recognition (PyTorch)

This repository is now PyTorch-only and supports:
- train
- eval
- predict

for the 30-subject 2-second mmWave point-cloud dataset.

## Project Layout
- `python/mmwave_pt.py`: main script (`train/eval/predict`)
- `config/fgst_2s_pytorch.cfg`: runtime config
- `run_fgst_pytorch.bat`: one-click Windows runner
- `2s/2s/p_*/`: dataset samples (`.npy`)
- `model/`: output models and reports

## Environment
Recommended Python environment:
- `D:\cppsoft\venvs\mmwave_pt`

Verify:
```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Run
### Train
```powershell
run_fgst_pytorch.bat train
```

### Eval
```powershell
run_fgst_pytorch.bat eval
```

### Predict
```powershell
run_fgst_pytorch.bat predict 2s\2s\p_1\0.npy
```

## Output Files
Configured in `config/fgst_2s_pytorch.cfg`:
- `model/mmwave_fgst_2s_pytorch.pt`
- `model/metrics_fgst_2s_pytorch.csv`
- `model/predict_result_2s_pytorch.csv`
