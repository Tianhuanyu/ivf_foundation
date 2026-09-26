#!/usr/bin/env bash
# =============================================================================
# dapt_prep.sh — CPU 侧:在一台便宜纯 CPU 机上抽高分帧,打包缓存到桶。
#   一次性运行。帧缓存后,GPU 训练(dapt_train.sh)直接下载解包,不再自己抽帧。
#
# 用法:
#   ./scripts/dapt_prep.sh run     # 抽帧+缓存(建 CPU 机 -> 抽 -> 传桶 -> 删机)
#   ./scripts/dapt_prep.sh down    # 手动删除残留的 prep 实例
#
# 配对的 VM 端脚本:_gcp_prep.sh
# =============================================================================
set -euo pipefail

PROJECT="hidden-outrider-390502"
BUCKET="gs://mlflow-artifacts-ai-a100"
ZONES="us-central1-b us-central1-c us-central1-f us-central1-a"
INSTANCE="dapt-prep"
MACHINE="c2d-standard-32"                 # 32 vCPU 纯 CPU,快且便宜(~$1.3/小时;抽帧约 15 分钟)
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DT_LOCAL="/mnt/d/Video/domain_transfer"

log(){ echo -e "\n\033[1;35m[prep] $*\033[0m"; }
confirm(){ read -r -p "$1 [y/N] " a; [[ "$a" == "y" || "$a" == "Y" ]] || { echo "已取消"; exit 1; }; }
preflight(){ gcloud config set project "$PROJECT" >/dev/null 2>&1; }

cmd_run(){
  preflight
  if gcloud storage ls "$BUCKET/dapt/cache/frames_hires.tar" >/dev/null 2>&1; then
    log "帧缓存已存在,无需再抽:$BUCKET/dapt/cache/frames_hires.tar(如需重抽先删它)"; return
  fi
  log "同步代码/清单到桶(含 _gcp_prep.sh、11_extract_frames.py)"
  gcloud storage rsync -r "$DT_LOCAL/scripts"    "$BUCKET/dapt/domain_transfer/scripts"
  gcloud storage rsync -r "$DT_LOCAL/manifests"  "$BUCKET/dapt/domain_transfer/manifests"

  confirm "创建 $MACHINE(纯 CPU,约 \$1.3/小时)抽帧+缓存(约 15 分钟,不到 \$0.5)。继续?"
  # 若有残留同名实例先删,保证干净
  for zz in $ZONES; do gcloud compute instances delete "$INSTANCE" --zone="$zz" --quiet 2>/dev/null && break; done || true

  local z=""
  for zz in $ZONES; do
    log "尝试在 $zz 创建 $INSTANCE ..."
    if gcloud compute instances create "$INSTANCE" --project="$PROJECT" --zone="$zz" \
        --machine-type="$MACHINE" \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size=250GB --boot-disk-type=pd-ssd \
        --scopes=cloud-platform; then
      z="$zz"; break
    fi
    log "$zz 无容量或失败,换下一个 ..."
  done
  [ -z "$z" ] && { echo "无法创建 prep 实例(所有 zone 失败)"; exit 1; }

  log "等待 SSH 就绪(IAP)..."
  for i in $(seq 1 30); do
    gcloud compute ssh "$INSTANCE" --zone="$z" --project="$PROJECT" --tunnel-through-iap \
      --command="echo ok" >/dev/null 2>&1 && break
    sleep 10
  done

  log "在 CPU 机上抽帧 + 缓存(同步运行,约 15 分钟,输出如下)..."
  gcloud compute ssh "$INSTANCE" --zone="$z" --project="$PROJECT" --tunnel-through-iap \
    --command="gcloud storage cp '$BUCKET/dapt/domain_transfer/scripts/_gcp_prep.sh' /tmp/p.sh && sed -i 's/\r\$//' /tmp/p.sh && BUCKET='$BUCKET' bash /tmp/p.sh" \
    -- -o ServerAliveInterval=30 -o ServerAliveCountMax=30

  log "抽帧+缓存完成,删除 CPU 实例(停止计费)..."
  gcloud compute instances delete "$INSTANCE" --zone="$z" --quiet
  log "✅ 帧已缓存到 $BUCKET/dapt/cache/frames_hires.tar —— 现在可跑 ./scripts/dapt_train.sh up + train"
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
