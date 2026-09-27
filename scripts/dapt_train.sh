#!/usr/bin/env bash
# =============================================================================
# dapt_train.sh — 在 GCP A100(40GB) 上跑 DINOv3 DAPT(VM 端脚本: _gcp_dinov3_dapt.sh)
#
# 用法(在 WSL、项目根目录 /mnt/d/Video/domain_transfer 下运行):
#   ARM=<arm> BUDGET=<e1|long> ./scripts/dapt_train.sh start   # 上传权重 -> 建实例 -> 传代码 -> 后台开跑
#   ./scripts/dapt_train.sh status                             # 看训练日志/完成标记
#   ./scripts/dapt_train.sh finish                             # 下载结果 -> 删实例(停计费)
#
# 细分子命令(排障用): upload / up / pushcode / train / fetch / down / ssh
#
# 说明:
#   - ARM 没有默认值,必须显式指定(可选值见 scripts/dapt_arms.sh)。这是故意的:以前远端
#     脚本/配置有默认值,忘记设变量时静默跑错实验、白烧过两次 GCP 账单。
#   - BUDGET 默认 long(20000 iters);DINO_BATCH/DINO_EPOCH_LEN/DINO_EPOCHS/REPO_WEIGHTS 可再覆盖。
#   - 训练在 VM 的 tmux 会话里跑,关掉本地终端不影响。"建实例"和"删实例"都有二次确认。
#   - 帧缓存(含 motion sidecar 和帧权重)由 dapt_prep.sh 单独制作。
# =============================================================================
set -euo pipefail
LOG_TAG="dapt_gcp"
source "$(dirname "$0")/_gcp_common.sh"
source "$(dirname "$0")/dapt_arms.sh"

INSTANCE="dapt-a100"
MACHINE="a2-highgpu-1g"                     # 1x A100 40GB(us-central1 配额 limit=16)
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_GB=300
SPOT="false"                                # true => 抢占式(省~65%,可能被抢占)
ZONE_STATE="$DT_LOCAL/.dapt_a100_zone"      # 实例实际所在 zone(缺货轮换后)
REMOTE_SCRIPT="_gcp_dinov3_dapt.sh"

get_zone(){ if [ -f "$ZONE_STATE" ]; then cat "$ZONE_STATE"; else echo "${ZONES%% *}"; fi; }
# 显式指定用户名 thy。不指定时 gcloud 可能用 OS Login 从 htian@conceivable.life 现推一个
# htian_conceivable_life 用户,那个账户的 $HOME 下没有环境。(OS Login 有时仍会强制覆盖,
# 那种情况下远端会自己重装环境,多花几分钟,不是错误。)
gssh(){ gcloud compute ssh "thy@$INSTANCE" --zone="$(get_zone)" --project="$PROJECT" --tunnel-through-iap "$@"; }

require_arm(){
  [ -n "${ARM:-}" ] || { echo "!! 没有指定 ARM,拒绝执行。可选: $DAPT_ARMS"; echo "   例如: ARM=motion_weighted BUDGET=long $0 $1"; exit 1; }
  dapt_overrides "$ARM" "${BUDGET:-long}" x x >/dev/null
}

cmd_upload(){
  preflight
  # bucket 只放【权重+数据】;源码走 scp(cmd_pushcode)。原始视频只有制作帧缓存时需要(见 dapt_prep.sh)。
  log "上传权重 -> $BUCKET/dapt/domain_transfer/weights"
  gcloud storage rsync -r "$DT_LOCAL/weights" "$BUCKET/dapt/domain_transfer/weights"
}

cmd_pushcode(){
  preflight
  gcloud compute instances describe "$INSTANCE" --zone="$(get_zone)" >/dev/null 2>&1 \
    || { echo "实例不存在,请先 up"; exit 1; }
  log "scp 源码直传 VM: scripts patches requirements.txt"
  gssh --command="mkdir -p ~/domain_transfer"
  gcloud compute scp --recurse --tunnel-through-iap --zone="$(get_zone)" --project="$PROJECT" \
    "$DT_LOCAL/scripts" "$DT_LOCAL/patches" "$DT_LOCAL/requirements.txt" "$INSTANCE:~/domain_transfer/"
  log "源码已到 VM:~/domain_transfer(repos/dinov3 由远端 clone + 打补丁)"
}

cmd_up(){
  preflight
  # 按实际状态处理:RUNNING 跳过;STOPPED 等则 start;不存在则创建。
  local st
  st=$(gcloud compute instances describe "$INSTANCE" --zone="$(get_zone)" --format="value(status)" 2>/dev/null || echo "")
  if [ "$st" = "RUNNING" ]; then
    log "实例 $INSTANCE 已在运行(zone=$(get_zone)),跳过"; return
  elif [ -n "$st" ]; then
    log "实例 $INSTANCE 状态是 $st,执行 start ..."
    gcloud compute instances start "$INSTANCE" --project="$PROJECT" --zone="$(get_zone)" || {
      echo ""
      echo "!! 已关机的实例只能在原来的 zone($(get_zone))重新开机,那里现在没有 A100。"
      echo "   先 ./scripts/dapt_train.sh fetch 取回结果,再删掉旧实例、在有货的 zone 新建:"
      echo "     ./scripts/dapt_train.sh down"
      echo "     RETRY_MIN=10 ARM=<臂> BUDGET=<e1|long> ./scripts/dapt_train.sh start"
      exit 1; }
  else
    confirm "即将创建 $MACHINE (A100 40GB) 实例,会开始计费。继续?"
    local spot_args=(); [[ "$SPOT" == "true" ]] && spot_args=(--provisioning-model=SPOT --instance-termination-action=STOP)
    local z
    z=$(create_in_zones "$INSTANCE" --machine-type="$MACHINE" --maintenance-policy=TERMINATE \
          --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
          --boot-disk-size="${BOOT_DISK_GB}GB" --boot-disk-type=pd-ssd \
          --metadata="install-nvidia-driver=True" --scopes=cloud-platform "${spot_args[@]}") \
      || { echo "所有候选 zone 都无 A100 容量。稍后再试,或改 _gcp_common.sh 的 ZONES。"; exit 1; }
    echo "$z" > "$ZONE_STATE"
    log "实例已创建于 $z"
  fi
  wait_ssh gssh || echo "SSH 迟迟未就绪,请稍后手动: ./scripts/dapt_train.sh train"
}

cmd_train(){
  preflight; require_arm "train"
  local env="BUCKET='$BUCKET' ARM='$ARM' BUDGET='${BUDGET:-long}' DINO_BATCH='${DINO_BATCH:-48}' DINO_EPOCH_LEN='${DINO_EPOCH_LEN:-}' DINO_EPOCHS='${DINO_EPOCHS:-}' REPO_WEIGHTS='${REPO_WEIGHTS:-dinov3_vitb16_repo.pth}'"
  log "在 VM 上后台启动(tmux 'dapt'): $env"
  # 直接跑 pushcode 刚 scp 上去的脚本(不从 bucket 取,避免跑到 bucket 里残留的旧版本)。
  gssh --command="
    set -e
    test -f ~/domain_transfer/scripts/$REMOTE_SCRIPT || { echo '!! VM 上没有代码,先 pushcode'; exit 1; }
    sed -i 's/\r\$//' ~/domain_transfer/scripts/*.sh
    tmux kill-session -t dapt 2>/dev/null || true
    tmux new-session -d -s dapt \"$env bash ~/domain_transfer/scripts/$REMOTE_SCRIPT > \$HOME/dapt.log 2>&1\"
    echo '已在 tmux 会话 dapt 启动。'
  "
  log "已启动。用 ./scripts/dapt_train.sh status 查看进度"
}

cmd_status(){
  preflight
  local st
  st=$(gcloud compute instances describe "$INSTANCE" --zone="$(get_zone)" --format="value(status)" 2>/dev/null || echo "")
  echo "实例状态: ${st:-不存在}"
  if [ "$st" = "RUNNING" ]; then
    log "最近 40 行日志(~/dapt.log):"
    gssh --command="tail -n 40 \$HOME/dapt.log 2>/dev/null || echo '日志还没生成'"
  elif [ "$st" = "TERMINATED" ]; then
    log ">>> VM 已关机(训练完成或已停,GPU 计费已停)。运行 finish 取回结果并删除。"
  fi
  # 每个 ARM_BUDGET 写自己的完成标记,不会把上一次别的实验的标记误判成这次完成。
  log "bucket 里的完成标记:"
  gcloud storage ls -l "$BUCKET/dapt/outputs/DINOV3_DONE_*.txt" 2>/dev/null || echo "  (无)"
}

cmd_fetch(){
  preflight
  if gcloud storage ls "$BUCKET/dapt/outputs/" >/dev/null 2>&1; then
    log "下载 DAPT 结果 -> $DT_LOCAL/gcp_outputs/"
    gcloud storage rsync -r "$BUCKET/dapt/outputs" "$DT_LOCAL/gcp_outputs" || true
  else
    log "还没有 outputs(训练未完成/未产出),跳过下载"
  fi
}

cmd_down(){
  preflight
  gcloud compute instances describe "$INSTANCE" --zone="$(get_zone)" >/dev/null 2>&1 \
    || { log "实例不存在,无需删除"; rm -f "$ZONE_STATE"; return; }
  confirm "即将删除实例 $INSTANCE(不可逆,停止计费)。确认已 fetch 结果?"
  gcloud compute instances delete "$INSTANCE" --zone="$(get_zone)" --quiet
  rm -f "$ZONE_STATE"
  log "实例已删除,计费停止"
}

cmd_ssh(){ preflight; gssh; }
cmd_start(){ require_arm "start"; cmd_upload; cmd_up; cmd_pushcode; cmd_train; }
cmd_finish(){ cmd_fetch || true; cmd_down; }

case "${1:-}" in
  upload)   cmd_upload ;;
  up)       cmd_up ;;
  pushcode) cmd_pushcode ;;
  train)    cmd_train ;;
  status)   cmd_status ;;
  fetch)    cmd_fetch ;;
  down)     cmd_down ;;
  ssh)      cmd_ssh ;;
  start)    cmd_start ;;
  finish)   cmd_finish ;;
  *) echo "用法: ARM=<arm> [BUDGET=e1|long] $0 {start|status|finish | upload|up|pushcode|train|fetch|down|ssh}"; exit 1 ;;
esac
