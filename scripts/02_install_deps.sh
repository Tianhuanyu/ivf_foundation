#!/usr/bin/env bash
# Install Python deps into env 'dapt', clone the official DINOv3 repo and apply our patch.
# Used both locally and on the GCP VM (scripts/_gcp_dinov3_dapt.sh calls this). Idempotent.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate dapt

ROOT="${DT_ROOT:-/mnt/d/Video/domain_transfer}"
DINO="$ROOT/repos/dinov3"
mkdir -p "$ROOT/repos" "$ROOT/weights"

echo "===== [1/3] pip deps (requirements.txt) ====="
python -m pip install --quiet -r "$ROOT/requirements.txt"

echo "===== [2/3] clone official DINOv3 ====="
[ -d "$DINO/.git" ] || git clone --depth 1 https://github.com/facebookresearch/dinov3 "$DINO"

echo "===== [3/3] apply patches/dinov3_dapt.patch ====="
# repos/dinov3 is gitignored, so our edits live only in this patch.
if git -C "$DINO" apply --check "$ROOT/patches/dinov3_dapt.patch" 2>/dev/null; then
  git -C "$DINO" apply "$ROOT/patches/dinov3_dapt.patch" && echo "patch applied"
elif git -C "$DINO" apply --check -R "$ROOT/patches/dinov3_dapt.patch" 2>/dev/null; then
  echo "patch already applied"
else
  echo "!! patch neither applies nor is already applied -- repos/dinov3 has diverged from patches/dinov3_dapt.patch"
  exit 1
fi

echo "===== DONE ====="
echo "DINOv3 weights are gated: hf download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir $ROOT/weights/dinov3_vitb16"
echo "then: cd $DINO && python _convert_hf_to_repo.py --hub dinov3_vitb16 --hf-dir $ROOT/weights/dinov3_vitb16 --out $ROOT/weights/dinov3_vitb16_repo.pth"
