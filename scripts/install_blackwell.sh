#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements_train.txt
python - <<'PY'
import torch
print('torch',torch.__version__,'CUDA',torch.version.cuda)
if not torch.cuda.is_available(): raise SystemExit('CUDA not available')
print('GPU',torch.cuda.get_device_name(0),'compute',torch.cuda.get_device_capability(0))
if torch.cuda.get_device_capability(0)[0] < 12:
    print('Notice: device is not sm_120; validate hardware.')
x=torch.randn(1024,1024,device='cuda')
(x@x).sum().item()
print('CUDA matmul OK')
PY
