#!/usr/bin/env bash
# =============================================================================
# dapt_prep.sh — 在一台便宜的纯 CPU 机上制作训练用帧缓存,打包上传到桶。一次性运行。
#   缓存 = frames_hires/train 的抽帧 + motion-energy sidecar(.me.png)+ frame_weights.npy,
#   GPU 训练(dapt_train.sh)直接下载解包,不再自己抽帧。
#
# 用法(WSL, 项目根目录):
#   ./scripts/dapt_prep.sh run     # 上传视频/代码 -> 建 CPU 机 -> 制作缓存 -> 传桶 -> 删机
#   ./scripts/dapt_prep.sh down    # 手动删除残留的 prep 实例
#
# 配对的 VM 端脚本:_gcp_prep.sh
# =============================================================================
set -euo pipefail
LOG_TAG="prep"
source "$(dirname "$0")/_gcp_common.sh"

INSTANCE="dapt-prep"
MACHINE="c2d-standard-32"                 # 32 vCPU 纯 CPU(~$1.3/小时)
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
VIDEO_SRC="/mnt/d/Video/Video"            # 原始视频根(manifest 引用的路径)

pssh(){ local z="$1"; shift; gcloud compute ssh "$INSTANCE" --zone="$z" --project="$PROJECT" --tunnel-through-iap "$@"; }

cmd_run(){
  preflight
  if gcloud storage ls "$BUCKET/dapt/cache/frames_hires.tar" >/dev/null 2>&1; then
    log "帧缓存已存在:$BUCKET/dapt/cache/frames_hires.tar(如需重做,先手动删掉它)"; return
  fi
  log "同步视频/代码/清单到桶(视频可断点续传,首次较久)"
  gcloud storage rsync -r "$VIDEO_SRC" "$BUCKET/dapt/data/Video/Video"
  gcloud storage rsync -r "$DT_LOCAL/scripts"   "$BUCKET/dapt/domain_transfer/scripts"
  gcloud storage rsync -r "$DT_LOCAL/manifests" "$BUCKET/dapt/domain_transfer/manifests"
  gcloud storage cp "$DT_LOCAL/requirements.txt" "$BUCKET/dapt/domain_transfer/requirements.txt"

  confirm "创建 $MACHINE(纯 CPU,约 \$1.3/小时)制作帧缓存(抽帧 + motion sidecar,耗时数小时)。继续?"
  for zz in $ZONES; do gcloud compute instances delete "$INSTANCE" --zone="$zz" --quiet 2>/dev/null && break; done || true
  local z
  z=$(create_in_zones "$INSTANCE" --machine-type="$MACHINE" \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size=250GB --boot-disk-type=pd-ssd --scopes=cloud-platform) \
    || { echo "无法创建 prep 实例(所有 zone 失败)"; exit 1; }
  wait_ssh pssh "$z" || { echo "SSH 未就绪"; exit 1; }

  log "在 CPU 机上制作缓存(同步运行,输出如下)..."
  pssh "$z" --command="gcloud storage cp '$BUCKET/dapt/domain_transfer/scripts/_gcp_prep.sh' /tmp/p.sh && sed -i 's/\r\$//' /tmp/p.sh && BUCKET='$BUCKET' bash /tmp/p.sh" \
    -- -o ServerAliveInterval=30 -o ServerAliveCountMax=30

  log "完成,删除 CPU 实例(停止计费)..."
  gcloud compute instances delete "$INSTANCE" --zone="$z" --quiet
  log "✅ 帧缓存:$BUCKET/dapt/cache/frames_hires.tar —— 现在可跑 ARM=... ./scripts/dapt_train.sh start"
}

cmd_down(){
  preflight
  for zz in $ZONES; do
    gcloud compute instances delete "$INSTANCE" --zone="$zz" --quiet 2>/dev/null && { log "已删 $INSTANCE ($zz)"; return; }
  done
  log "没有残留的 prep 实例"
}

case "${1:-}" in
  run)  cmd_run ;;
  down) cmd_down ;;
  *) echo "用法: $0 {run|down}"; exit 1 ;;
esac
