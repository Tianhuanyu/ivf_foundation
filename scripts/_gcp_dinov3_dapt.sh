#!/usr/bin/env bash
# =============================================================================
# _gcp_dinov3_dapt.sh — VM 端:官方 DINOv3 DAPT 长程(ViT-B, 两分辨率) on A100.
# 由 dapt_train.sh 通过 tmux 启动(需环境变量 BUCKET)。不要手动直接跑。
# 流程: 拉代码/权重/帧缓存 -> 装环境 -> 官方 DINOv3 DAPT -> 抽 backbone -> 回传 -> 关机
# =============================================================================
set -euo pipefail
BUCKET="${BUCKET:?need BUCKET env}"
export DT_ROOT="$HOME/domain_transfer"

# ── 训练超参(A100-40GB 起步值,可 dapt_train.sh 里改) ──────────────────────────
CONFIG="${CONFIG:-dapt_vitb16.yaml}"                 # 官方 ViT-B 配置(local_crops_native=true)
REPO_WEIGHTS="${REPO_WEIGHTS:-dinov3_vitb16_repo.pth}"
DINO_BATCH="${DINO_BATCH:-48}"                        # A100-40GB;可试 64
DINO_EPOCH_LEN="${DINO_EPOCH_LEN:-2500}"
DINO_EPOCHS="${DINO_EPOCHS:-8}"                       # 8*2500 = 20000 iters(长程)
OUT="$DT_ROOT/dinov3_dapt_gcp"

echo "===== [1/6] 拉权重 + 帧缓存(代码已由 cmd_pushcode 用 scp 送到 $DT_ROOT) ====="
mkdir -p "$DT_ROOT"
# 源码(scripts/patches/manifests)已 scp 到 $DT_ROOT;bucket 只拉权重(数据类)。
[ -d "$DT_ROOT/scripts" ] || { echo "!! $DT_ROOT/scripts 不存在——请先本地 ./scripts/dapt_train.sh pushcode"; exit 1; }
gcloud storage rsync -r "$BUCKET/dapt/domain_transfer/weights" "$DT_ROOT/weights"
if gcloud storage ls "$BUCKET/dapt/cache/frames_hires.tar" >/dev/null 2>&1; then
  gcloud storage cp "$BUCKET/dapt/cache/frames_hires.tar" /tmp/f.tar
  tar -C "$DT_ROOT" -xf /tmp/f.tar && rm -f /tmp/f.tar
else
  echo "!! 没有帧缓存,请先本地跑 ./scripts/dapt_prep.sh run"; exit 1
fi
echo "帧数: $(find "$DT_ROOT/frames_hires/train" -name '*.jpg' | wc -l)"

echo "===== [2/6] conda 环境 + torch(cu128 兼容 A100) ====="
bash "$DT_ROOT/scripts/01_bootstrap_env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate dapt

echo "===== [3/6] 依赖(含官方 DINOv3 训练器依赖) ====="
python -m pip install --quiet timm transformers safetensors pillow opencv-python-headless \
  omegaconf iopath submitit fvcore torchmetrics termcolor ftfy regex \
  scikit-learn umap-learn pandas

echo "===== [4/6] 确保 DINOv3 已打补丁(若 repos 从桶拉来已含改动则跳过) ====="
DINO="$DT_ROOT/repos/dinov3"
[ -d "$DINO/.git" ] || git clone --depth 1 https://github.com/facebookresearch/dinov3 "$DINO"
if [ -f "$DT_ROOT/patches/dinov3_dapt.patch" ]; then
  if git -C "$DINO" apply --check "$DT_ROOT/patches/dinov3_dapt.patch" 2>/dev/null; then
    git -C "$DINO" apply "$DT_ROOT/patches/dinov3_dapt.patch" && echo "patch applied"
  else echo "patch already applied (skip)"; fi
fi
# 若 repo 权重没同步来,尝试从 HF 权重转换
if [ ! -f "$DT_ROOT/weights/$REPO_WEIGHTS" ]; then
  echo "转换 HF->repo 权重..."; cd "$DINO"
  python _convert_hf_to_repo.py --hub dinov3_vitb16 \
    --hf-dir "$DT_ROOT/weights/dinov3_vitb16" --out "$DT_ROOT/weights/$REPO_WEIGHTS"
fi

echo "===== [5/6] 官方 DINOv3 DAPT 长程(ViT-B, 两分辨率 resize+native) ====="
cd "$DINO"; mkdir -p "$OUT"
# 下面要把 dataset_path 的 root 改成 VM 路径,但必须保留 config 里的 :extra=...(如 weighted),
# 否则 motion_weighted 这类 config 会被静默降级成均匀采样帧。
EXTRA=$(grep -o 'dataset_path:.*' "dinov3/configs/train/$CONFIG" | grep -o ':extra=[A-Za-z_]*' | head -1 || true)
if [ "$EXTRA" = ":extra=weighted" ] && [ ! -f "$DT_ROOT/frames_hires/train/frame_weights.npy" ]; then
  echo "!! $CONFIG 要求 extra=weighted,但帧缓存里没有 frame_weights.npy——重新打包带 13_frame_weights.py 产物的缓存"; exit 1
fi
if grep -q 'crop_sampler: motion' "dinov3/configs/train/$CONFIG"; then
  echo "motion sidecar 覆盖: $(find "$DT_ROOT/frames_hires/train" -name '*.me.png' | wc -l) / $(find "$DT_ROOT/frames_hires/train" -name '*.jpg' | wc -l) 帧"
fi
PYTHONPATH=. torchrun --nproc_per_node=1 dinov3/train/train.py \
  --config-file "dinov3/configs/train/$CONFIG" --output-dir "$OUT" --no-resume \
  train.dataset_path="Frames:root=$DT_ROOT/frames_hires/train$EXTRA" \
  student.pretrained_weights="$DT_ROOT/weights/$REPO_WEIGHTS" \
  train.batch_size_per_gpu=$DINO_BATCH \
  train.OFFICIAL_EPOCH_LENGTH=$DINO_EPOCH_LEN optim.epochs=$DINO_EPOCHS \
  2>&1 | tee "$DT_ROOT/dinov3_dapt_gcp.log"

echo "===== [6/6] 抽 backbone + 回传 ====="
LAST=$(ls -d "$OUT"/ckpt/* 2>/dev/null | sort -t/ -k100 -n | tail -1)
python - <<PY
import torch, glob, os
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
ck = sorted(glob.glob("$OUT/ckpt/*"), key=lambda p:int(os.path.basename(p)))[-1]
dcp_to_torch_save(ck, "/tmp/full.pth")
sd = torch.load("/tmp/full.pth", map_location="cpu", weights_only=False)["model"]
bb = {k[len("teacher.backbone."):]:v for k,v in sd.items() if k.startswith("teacher.backbone.")}
torch.save(bb, "$DT_ROOT/weights/dinov3_vitb16_dapt_backbone.pth")
print("saved backbone:", len(bb), "tensors from", ck)
PY
gcloud storage cp "$DT_ROOT/weights/dinov3_vitb16_dapt_backbone.pth" "$BUCKET/dapt/outputs/"
gcloud storage cp "$DT_ROOT/dinov3_dapt_gcp.log" "$BUCKET/dapt/outputs/"
echo "DINOV3 DAPT DONE" | gcloud storage cp - "$BUCKET/dapt/outputs/DINOV3_DONE.txt"
echo "===== FINISHED ====="
if [ "${AUTO_POWEROFF:-1}" = "1" ]; then sudo shutdown -h +10 || true; fi
