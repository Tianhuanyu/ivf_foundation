#!/usr/bin/env bash
# Install remaining Python deps into env 'dapt', clone repos, fetch free weights.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dapt

ROOT="${DT_ROOT:-/mnt/d/Video/domain_transfer}"
REPOS="$ROOT/repos"
WEIGHTS="$ROOT/weights"
mkdir -p "$REPOS" "$WEIGHTS"

echo "===== [1/5] pip deps ====="
python -m pip install \
  timm einops transformers huggingface_hub safetensors \
  scikit-learn umap-learn pandas pyyaml tqdm \
  pillow opencv-python-headless av \
  omegaconf iopath submitit fvcore torchmetrics termcolor ftfy regex  # dinov3 official trainer deps

echo "===== [2/5] video reader (decord -> eva-decord fallback) ====="
python -m pip install decord || python -m pip install eva-decord || \
  echo "WARN: decord unavailable; will use torchvision/av reader instead"

echo "===== [3/5] clone repos ====="
[ -d "$REPOS/dinov3/.git" ] || git clone --depth 1 https://github.com/facebookresearch/dinov3 "$REPOS/dinov3"
[ -d "$REPOS/vjepa2/.git" ] || git clone --depth 1 https://github.com/facebookresearch/vjepa2 "$REPOS/vjepa2"

echo "===== [4/5] apply our DINOv3 DAPT patch (Frames dataset + pretrained-backbone hook + configs) ====="
# repos/dinov3 is gitignored, so our edits live only in this patch. Idempotent: skip if already applied.
if [ -f "$ROOT/patches/dinov3_dapt.patch" ]; then
  if git -C "$REPOS/dinov3" apply --check "$ROOT/patches/dinov3_dapt.patch" 2>/dev/null; then
    git -C "$REPOS/dinov3" apply "$ROOT/patches/dinov3_dapt.patch" && echo "patch applied"
  else
    echo "patch already applied or not applicable (skipping)"
  fi
else
  echo "WARN: $ROOT/patches/dinov3_dapt.patch not found — DINOv3 DAPT customizations missing"
fi

echo "===== [5/5] download V-JEPA2 ViT-L weights (free, ~4.8GB) ====="
if [ ! -f "$WEIGHTS/vjepa2_vitl.pt" ]; then
  wget -nv -O "$WEIGHTS/vjepa2_vitl.pt" https://dl.fbaipublicfiles.com/vjepa2/vitl.pt
fi
ls -lh "$WEIGHTS"
echo "===== DONE ====="
echo "NOTE: DINOv3 repo weights are gated (no public URL). Convert from HF weights:"
echo "  cd $REPOS/dinov3 && python _convert_hf_to_repo.py   # needs weights/dinov3_vits16 (HF)"
