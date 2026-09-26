#!/usr/bin/env bash
# =============================================================================
# dapt_run.sh — 本地(WSL, conda env dapt)跑一个 DAPT 实验臂,跑完自动抽 backbone。
#
#   ARM=motion_weighted BUDGET=e1 ./scripts/dapt_run.sh
#
# ARM/BUDGET 的定义只在 scripts/dapt_arms.sh。可选环境变量:
#   DINO_BATCH(默认 16,12GB 卡)  RUN_TAG(输出目录名后缀,比如 seed43)  EXTRA_OPTS(额外 dotlist)
#   OUT_DIR / BB_OUT(覆盖下面两个输出路径)
# 输出: dinov3_dapt_<arm>_<budget>[_<tag>]_out/  和  weights/dinov3_vitb16_dapt_<同名>_backbone.pth
# =============================================================================
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/dapt_arms.sh"

ARM="${ARM:?need ARM (one of: $DAPT_ARMS)}"
BUDGET="${BUDGET:-e1}"
DINO_BATCH="${DINO_BATCH:-16}"
RUN="${ARM}_${BUDGET}${RUN_TAG:+_$RUN_TAG}"
OUT="${OUT_DIR:-$ROOT/dinov3_dapt_${RUN}_out}"
BB="${BB_OUT:-$ROOT/weights/dinov3_vitb16_dapt_${RUN}_backbone.pth}"
FRAMES="$ROOT/frames_hires/train"
WEIGHTS="$ROOT/weights/dinov3_vitb16_repo.pth"

dapt_overrides "$ARM" "$BUDGET" "$FRAMES" "$WEIGHTS" >/dev/null   # validate ARM/BUDGET (set -e exits)
mapfile -t OVR < <(dapt_overrides "$ARM" "$BUDGET" "$FRAMES" "$WEIGHTS")
OVR+=("train.batch_size_per_gpu=$DINO_BATCH")
# shellcheck disable=SC2206
[ -n "${EXTRA_OPTS:-}" ] && OVR+=($EXTRA_OPTS)

dapt_arm_needs_weights "$ARM" && [ ! -f "$FRAMES/frame_weights.npy" ] && {
  echo "!! ARM=$ARM 需要 $FRAMES/frame_weights.npy(先跑 scripts/13_frame_weights.py)"; exit 1; }

echo "[dapt_run] ARM=$ARM BUDGET=$BUDGET -> $OUT"; printf '  %s\n' "${OVR[@]}"
cd "$ROOT/repos/dinov3"
PYTHONPATH=. torchrun --nproc_per_node=1 dinov3/train/train.py \
  --config-file dinov3/configs/train/dapt_vitb16.yaml --output-dir "$OUT" "${OVR[@]}"

python _extract_dapt_backbone_param.py --ckpt-root "$OUT/ckpt" --out "$BB"
