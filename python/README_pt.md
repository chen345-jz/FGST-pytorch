# PyTorch Version Usage

## Environment
- Python venv: `D:\cppsoft\venvs\mmwave_pt`
- Start:
  - `D:\cppsoft\venvs\mmwave_pt\Scripts\activate`
  - or run with full path:
    `D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe`

## Config
- Main config: `config/fgst_2s_pytorch.cfg`
- Outputs:
  - model: `model/mmwave_fgst_2s_pytorch.pt`
  - metrics: `model/metrics_fgst_2s_pytorch.csv`
  - predict csv: `model/predict_result_2s_pytorch.csv`

## Commands
- Train:
  - `D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe python/mmwave_pt.py train config/fgst_2s_pytorch.cfg`
- Eval:
  - `D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe python/mmwave_pt.py eval config/fgst_2s_pytorch.cfg`
- Predict:
  - `D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe python/mmwave_pt.py predict config/fgst_2s_pytorch.cfg 2s/2s/p_1/0.npy`

## One-click script
- `run_fgst_pytorch.bat train`
- `run_fgst_pytorch.bat eval`
- `run_fgst_pytorch.bat predict 2s/2s/p_1/0.npy`
