#!/usr/bin/env bash
# One-time server setup (run via `scripts/remote.sh setup`). No sudo needed.
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv -e .
# Pin day 1: record the exact resolved versions the results were produced with.
uv pip freeze --python .venv > requirements.lock
command -v ffmpeg >/dev/null || echo "WARN: no system ffmpeg; LeRobot falls back to PyAV video decoding (slower dataloading)"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv
# Fail loudly: if the torch wheel's CUDA is newer than the driver supports, training silently runs on CPU.
# Fix by reinstalling torch from the matching index, e.g. --index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -c "
import torch
assert torch.cuda.is_available(), f'CUDA unavailable to torch {torch.__version__} (built for CUDA {torch.version.cuda}); check driver vs nvidia-smi'
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
.venv/bin/python -m pytest tests -q
