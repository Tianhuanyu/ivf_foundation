#!/usr/bin/env bash
# Install remaining Python deps into env 'dapt', clone repos, fetch free weights.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dapt

ROOT=/mnt/d/Video/domain_transfer
REPOS="$ROOT/repos"
WEIGHTS="$ROOT/weights"
mkdir -p "$REPOS" "$WEIGHTS"

echo "===== [1/4] pip deps ====="
python -m pip install \
  timm einops transformers huggingface_hub safetensors \
  scikit-learn umap-learn pandas pyyaml tqdm \
  pillow opencv-python-headless av

echo "===== [2/4] video reader (decord -> eva-decord fallback) ====="
python -m pip install decord || python -m pip install eva-decord || \
  echo "WARN: decord unavailable; will use torchvision/av reader instead"

echo "===== [3/4] clone repos ====="
[ -d "$REPOS/dinov3/.git" ] || git clone --depth 1 https://github.com/facebookresearch/dinov3 "$REPOS/dinov3"
[ -d "$REPOS/vjepa2/.git" ] || git clone --depth 1 https://github.com/facebookresearch/vjepa2 "$REPOS/vjepa2"

echo "===== [4/4] download V-JEPA2 ViT-L weights (free, ~1.1GB) ====="
if [ ! -f "$WEIGHTS/vjepa2_vitl.pt" ]; then
  wget -nv -O "$WEIGHTS/vjepa2_vitl.pt" https://dl.fbaipublicfiles.com/vjepa2/vitl.pt
fi
ls -lh "$WEIGHTS"
echo "===== DONE ====="
