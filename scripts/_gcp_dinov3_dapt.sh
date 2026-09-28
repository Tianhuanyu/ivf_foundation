#!/usr/bin/env bash
# =============================================================================
# _gcp_dinov3_dapt.sh — VM 端:官方 DINOv3 DAPT(ViT-B, 两分辨率) on A100.
# 由 dapt_train.sh 通过 tmux 启动(需环境变量 BUCKET)。不要手动直接跑。
# 流程: 拉权重/帧缓存 -> 装环境+打补丁 -> DAPT -> 抽 backbone -> 回传 -> 关机
#
# 实验臂/训练量由 ARM/BUDGET 选(定义只在 scripts/dapt_arms.sh 一处):
#   ARM    = uniform | content | motion | motion_weighted | motion_ibotlocal_high | motion_ibotlocal_low
#   BUDGET = e1 (3000 iters) | long (20000 iters)
# DINO_BATCH/DINO_EPOCH_LEN/DINO_EPOCHS 可再覆盖 budget 的默认值。
#
# 防崩溃(断点续训):
#   - 训练中每写完一个 checkpoint(long 预算每 1000 iter),就上传到桶 dapt/runs/<ARM>_<BUDGET>/ckpt/<iter>/,
#     上传完写 <iter>.complete 标记,桶里只保留最新 2 个;
#   - 同一个 ARM/BUDGET 再次启动时(新 VM 也行),先从桶里恢复最新的完整 checkpoint,官方 trainer 从那里接着训
#     (恢复模型、优化器和迭代数);没有 checkpoint 才从预训练权重开始;
#   - 无论成功还是失败,退出时都会上传最新 checkpoint 和日志、写标记(DINOV3_DONE_<RUN>.txt / DINOV3_FAILED_<RUN>.txt)、
#     然后自动关机,不会空跑计费。
# =============================================================================
set -euo pipefail
BUCKET="${BUCKET:?need BUCKET env}"
export DT_ROOT="$HOME/domain_transfer"
source "$DT_ROOT/scripts/dapt_arms.sh"

ARM="${ARM:?need ARM (see scripts/dapt_arms.sh)}"
BUDGET="${BUDGET:-long}"
REPO_WEIGHTS="${REPO_WEIGHTS:-dinov3_vitb16_repo.pth}"
DINO_BATCH="${DINO_BATCH:-48}"                 # A100-40GB
MIN_SIDECAR_COVERAGE="${MIN_SIDECAR_COVERAGE:-0.65}"   # 本地 frames_hires 实测约 0.71
CKPT_SYNC_MIN="${CKPT_SYNC_MIN:-5}"            # 每几分钟检查一次有没有新写完的 checkpoint 要上传
RUN="${ARM}_${BUDGET}"
OUT="$DT_ROOT/dinov3_dapt_gcp_$RUN"
FRAMES="$DT_ROOT/frames_hires/train"
CKPT_URL="$BUCKET/dapt/runs/$RUN/ckpt"
LOG="$DT_ROOT/dinov3_dapt_gcp_$RUN.log"
dapt_overrides "$ARM" "$BUDGET" "$FRAMES" "$DT_ROOT/weights/$REPO_WEIGHTS" >/dev/null   # 尽早校验 ARM/BUDGET

# ── checkpoint 上传/恢复 ───────────────────────────────────────────────────────
upload_ckpts(){   # 上传所有已写完、还没上传过的 checkpoint;桶里只留最新 2 个
  local d it
  [ -d "$OUT/ckpt" ] || return 0
  for d in "$OUT"/ckpt/*/; do
    d="${d%/}"; it="$(basename "$d")"
    [[ "$it" =~ ^[0-9]+$ ]] || continue
    [ -f "$d/.metadata" ] || continue                                   # DCP 最后才写 .metadata
    [ -z "$(find "$d" -newermt '-60 seconds' -print -quit)" ] || continue  # 一分钟内还在写,下次再传
    gcloud storage ls "$CKPT_URL/$it.complete" >/dev/null 2>&1 && continue
    echo "[ckpt] 上传 checkpoint $it -> $CKPT_URL/$it/"
    gcloud storage rsync -r "$d" "$CKPT_URL/$it" >/dev/null 2>&1 \
      && echo "ok $(date '+%F %T')" | gcloud storage cp - "$CKPT_URL/$it.complete" >/dev/null 2>&1 || true
  done
  local done_list; done_list="$(gcloud storage ls "$CKPT_URL/*.complete" 2>/dev/null | grep -o '[0-9]*\.complete' | sort -n | sed 's/\.complete//')" || true
  for it in $(echo "$done_list" | head -n -2); do                      # 删掉比最新 2 个更旧的
    gcloud storage rm -r "$CKPT_URL/$it" "$CKPT_URL/$it.complete" >/dev/null 2>&1 || true
  done
}
restore_ckpt(){   # 从桶里恢复最新的完整 checkpoint 到 $OUT/ckpt/<iter>;有就返回 0
  local it
  it="$(gcloud storage ls "$CKPT_URL/*.complete" 2>/dev/null | grep -o '[0-9]*\.complete' | sed 's/\.complete//' | sort -n | tail -1)" || true
  [ -n "$it" ] || return 1
  if [ ! -f "$OUT/ckpt/$it/.metadata" ]; then
    echo "[ckpt] 从桶恢复 checkpoint $it"
    mkdir -p "$OUT/ckpt/$it"
    gcloud storage rsync -r "$CKPT_URL/$it" "$OUT/ckpt/$it"
  fi
  return 0
}

SYNC_PID=""
on_exit(){
  local rc=$?
  [ -n "$SYNC_PID" ] && kill "$SYNC_PID" 2>/dev/null || true
  echo "===== 退出(rc=$rc):上传最新 checkpoint 和日志 ====="
  upload_ckpts
  [ -f "$LOG" ] && gcloud storage cp "$LOG" "$BUCKET/dapt/outputs/" >/dev/null 2>&1 || true
  if [ "$rc" -eq 0 ]; then
    echo "DINOV3 DAPT DONE ($RUN) $(date '+%F %T')" | gcloud storage cp - "$BUCKET/dapt/outputs/DINOV3_DONE_$RUN.txt"
  else
    echo "DINOV3 DAPT FAILED ($RUN) rc=$rc $(date '+%F %T')" | gcloud storage cp - "$BUCKET/dapt/outputs/DINOV3_FAILED_$RUN.txt" || true
  fi
  if [ "${AUTO_POWEROFF:-1}" = "1" ]; then sudo shutdown -h +10 || true; fi
}
trap on_exit EXIT
gcloud storage rm "$BUCKET/dapt/outputs/DINOV3_FAILED_$RUN.txt" >/dev/null 2>&1 || true

echo "===== [1/5] 拉权重 + 帧缓存(代码已由 cmd_pushcode 用 scp 送到 $DT_ROOT) ====="
[ -d "$DT_ROOT/scripts" ] || { echo "!! $DT_ROOT/scripts 不存在——请先本地 ./scripts/dapt_train.sh pushcode"; exit 1; }
gcloud storage rsync -r "$BUCKET/dapt/domain_transfer/weights" "$DT_ROOT/weights"
# 帧缓存,按优先级:
#   dapt/cache/frames_hires_parts/   upload_frame_cache.sh 按阶段打的 tar + frame_weights.npy + DONE.txt(清单)
#   dapt/cache/frames_hires/train/   本地 frames_hires/train 直接 rsync 上去的目录
#   dapt/cache/frames_hires.tar      dapt_prep.sh 在 CPU 机上从视频重新制作的打包缓存(备用)
PARTS_URL="$BUCKET/dapt/cache/frames_hires_parts"
EXPECT_JPG=""; EXPECT_ME=""
if [ -f "$FRAMES/.cache_complete" ]; then
  echo "帧缓存已在本机(同一台 VM 重跑),跳过下载"
elif gcloud storage ls "$PARTS_URL/DONE.txt" >/dev/null 2>&1; then
  mkdir -p "$FRAMES" /tmp/parts
  gcloud storage cp "$PARTS_URL/*.tar" /tmp/parts/
  for t in /tmp/parts/*.tar; do tar -C "$FRAMES" -xf "$t" && rm -f "$t"; done
  gcloud storage cp "$PARTS_URL/frame_weights.npy" "$PARTS_URL/frame_weights_meta.json" "$FRAMES/"
  read -r _ EXPECT_JPG EXPECT_ME < <(gcloud storage cat "$PARTS_URL/DONE.txt" | grep '^TOTAL')
elif gcloud storage ls "$BUCKET/dapt/cache/frames_hires/train/frame_weights.npy" >/dev/null 2>&1; then
  mkdir -p "$FRAMES"
  gcloud storage rsync -r "$BUCKET/dapt/cache/frames_hires/train" "$FRAMES"
elif gcloud storage ls "$BUCKET/dapt/cache/frames_hires.tar" >/dev/null 2>&1; then
  gcloud storage cp "$BUCKET/dapt/cache/frames_hires.tar" /tmp/f.tar
  tar -C "$DT_ROOT" -xf /tmp/f.tar && rm -f /tmp/f.tar
else
  echo "!! 桶里没有帧缓存:先跑 ./scripts/upload_frame_cache.sh(GCP_WORKFLOW.md)"; exit 1
fi

N_JPG=$(find "$FRAMES" -name '*.jpg' | wc -l)
echo "帧数: $N_JPG"
if [ -n "$EXPECT_JPG" ]; then          # 按阶段打包的缓存:解包结果必须与清单一致
  N_ME_ALL=$(find "$FRAMES" -name '*.me.png' | wc -l)
  [ "$N_JPG" = "$EXPECT_JPG" ] && [ "$N_ME_ALL" = "$EXPECT_ME" ] \
    || { echo "!! 帧缓存不完整:帧 $N_JPG/$EXPECT_JPG,运动图 $N_ME_ALL/$EXPECT_ME"; exit 1; }
  echo "帧缓存与清单一致:$N_JPG 帧,$N_ME_ALL 运动图"
  touch "$FRAMES/.cache_complete"
fi
if dapt_arm_needs_motion "$ARM"; then
  N_ME=$(find "$FRAMES" -name '*.me.png' | wc -l)
  echo "motion sidecar 覆盖: $N_ME / $N_JPG 帧"
  awk -v a="$N_ME" -v b="$N_JPG" -v m="$MIN_SIDECAR_COVERAGE" 'BEGIN{exit !(b>0 && a/b>=m)}' \
    || { echo "!! ARM=$ARM 需要 motion sidecar,覆盖率低于 $MIN_SIDECAR_COVERAGE"; exit 1; }
fi
if dapt_arm_needs_weights "$ARM" && [ ! -f "$FRAMES/frame_weights.npy" ]; then
  echo "!! ARM=$ARM 需要 frame_weights.npy,帧缓存里没有"; exit 1
fi

echo "===== [2/5] conda 环境 + torch / 依赖 / DINOv3 + 补丁 ====="
bash "$DT_ROOT/scripts/01_bootstrap_env.sh"
bash "$DT_ROOT/scripts/02_install_deps.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate dapt
DINO="$DT_ROOT/repos/dinov3"
if [ ! -f "$DT_ROOT/weights/$REPO_WEIGHTS" ]; then
  echo "转换 HF->repo 权重..."
  (cd "$DINO" && python _convert_hf_to_repo.py --hub dinov3_vitb16 \
    --hf-dir "$DT_ROOT/weights/dinov3_vitb16" --out "$DT_ROOT/weights/$REPO_WEIGHTS")
fi

echo "===== [3/5] DAPT: ARM=$ARM BUDGET=$BUDGET batch=$DINO_BATCH ====="
mapfile -t OVR < <(dapt_overrides "$ARM" "$BUDGET" "$FRAMES" "$DT_ROOT/weights/$REPO_WEIGHTS")
OVR+=("train.batch_size_per_gpu=$DINO_BATCH")
[ -n "${DINO_EPOCH_LEN:-}" ] && OVR+=("train.OFFICIAL_EPOCH_LENGTH=$DINO_EPOCH_LEN")
[ -n "${DINO_EPOCHS:-}" ] && OVR+=("optim.epochs=$DINO_EPOCHS")
printf '  %s\n' "${OVR[@]}"
cd "$DINO"; mkdir -p "$OUT"
RESUME_ARGS=(--no-resume)
if restore_ckpt; then
  RESUME_ARGS=()
  echo "断点续训:从 checkpoint $(ls "$OUT/ckpt" | sort -n | tail -1) 继续"
fi
( while true; do sleep $((CKPT_SYNC_MIN * 60)); upload_ckpts; done ) &
SYNC_PID=$!
PYTHONPATH=. torchrun --nproc_per_node=1 dinov3/train/train.py \
  --config-file dinov3/configs/train/dapt_vitb16.yaml --output-dir "$OUT" "${RESUME_ARGS[@]}" \
  "${OVR[@]}" 2>&1 | tee -a "$LOG"

echo "===== [4/5] 抽 backbone ====="
BB="$DT_ROOT/weights/dinov3_vitb16_dapt_${RUN}_backbone.pth"
python _extract_dapt_backbone_param.py --ckpt-root "$OUT/ckpt" --out "$BB"

echo "===== [5/5] 回传 backbone(checkpoint、日志和完成标记在退出时处理)====="
gcloud storage cp "$BB" "$BUCKET/dapt/outputs/"
echo "===== FINISHED ====="
