#!/usr/bin/env bash
# User-space environment bootstrap (no sudo needed).
# Installs Miniconda, creates a py3.12 env 'dapt', installs torch cu128 for Blackwell.
set -euo pipefail

MINICONDA="$HOME/miniconda3"
INSTALLER="/tmp/miniconda.sh"

echo "===== [1/4] Install Miniconda (user-space) ====="
if [ ! -d "$MINICONDA" ]; then
  wget -nv -O "$INSTALLER" \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$INSTALLER" -b -p "$MINICONDA"
  rm -f "$INSTALLER"
else
  echo "Miniconda already present at $MINICONDA"
fi

# make conda available in this non-interactive shell
source "$MINICONDA/etc/profile.d/conda.sh"

echo "===== [2/4] Create conda env 'dapt' (python 3.12, conda-forge only) ====="
# Use conda-forge exclusively to avoid the Anaconda 'defaults' channel ToS gate.
if ! conda env list | grep -q "/dapt$"; then
  conda create -y -n dapt -c conda-forge --override-channels python=3.12
fi
conda activate dapt

echo "===== [3/4] Install PyTorch (CUDA 12.8, Blackwell sm_120) ====="
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "===== [4/4] Verify ====="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(2000, 2000, device="cuda"); y = x @ x; torch.cuda.synchronize()
    print("cuda_matmul_ok:", bool(y.sum().item() != 0))
    free, total = torch.cuda.mem_get_info()
    print(f"vram_free_gb: {free/1e9:.2f} / {total/1e9:.2f}")
PY

echo "===== DONE ====="
echo "To use later:  source ~/miniconda3/etc/profile.d/conda.sh && conda activate dapt"
