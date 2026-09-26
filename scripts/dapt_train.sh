#!/usr/bin/env bash
# =============================================================================
# dapt_train.sh — 在 GCP A100(40GB) 上跑 V-JEPA2 + DINOv3 的 DAPT
#
# 用法(在 WSL、项目根目录 /mnt/d/Video/domain_transfer 下运行):
#   ./scripts/dapt_train.sh start     # 上传数据 -> 建实例 -> 装环境 -> 串行开跑(后台)
#   ./scripts/dapt_train.sh status    # 查看训练日志/进度
#   ./scripts/dapt_train.sh finish    # 下载 *_dapt 结果 -> 删实例(停计费)
#
# 细分子命令(排障用): upload / up / pushcode / train / fetch / down / ssh
#
# 说明:
#   - 训练在 VM 的 tmux 会话里跑,关掉本地终端不影响。
#   - "建实例"和"删实例"都有二次确认(涉及计费/不可逆)。
#   - 唯一的远端脚本是 _gcp_dinov3_dapt.sh(REMOTE_SCRIPT=_gcp_dinov3_dapt.sh),
#     超参用 CONFIG/DINO_BATCH/DINO_EPOCH_LEN/DINO_EPOCHS/REPO_WEIGHTS 环境变量覆盖。
# =============================================================================
set -euo pipefail

# ── 配置(按需修改) ─────────────────────────────────────────────────────────
PROJECT="hidden-outrider-390502"          # Conceivable Cloud(已开结算)
REGION="us-central1"                        # A100-40GB 配额所在区(已确认 limit=16)
ZONE="us-central1-b"                        # 首选;缺货时按 ZONES 顺序自动轮换
ZONES="us-central1-b us-central1-c us-central1-f us-central1-a"  # A100 缺货时依次尝试
INSTANCE="dapt-a100"
MACHINE="a2-highgpu-1g"                     # 1x A100 40GB
IMAGE_FAMILY="common-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
BOOT_DISK_GB=300                           # 视频62G+权重5G+环境+输出
BUCKET="gs://mlflow-artifacts-ai-a100"     # 管理员建的专用桶(us-east1);数据放其 dapt/ 前缀下
# 注:桶在 us-east1,VM 在 us-central1(A100 配额所在)——跨区拉取可行,仅少量流量费
SPOT="false"                               # true => 抢占式(省~65%,可能被抢占)
REMOTE_SCRIPT="${REMOTE_SCRIPT:-}"    # 故意不给默认值——已经因为"忘记设这个环境变量导致
                                       # 静默跑成旧脚本、白烧两次GCP账单"这个问题吃过两次亏,
                                       # 宁可现在直接报错,不要再让 train 静默退回某个默认脚本。

# 本地数据源
DT_LOCAL="/mnt/d/Video/domain_transfer"    # 项目目录
VIDEO_SRC="/mnt/d/Video/Video"             # V-JEPA2 原始视频根(清单引用的)
ZONE_STATE="$DT_LOCAL/.dapt_a100_zone"     # 记录实例实际所在 zone(缺货轮换后)

# ── 小工具 ─────────────────────────────────────────────────────────────────
log(){ echo -e "\n\033[1;36m[dapt_gcp] $*\033[0m"; }
confirm(){ read -r -p "$1 [y/N] " a; [[ "$a" == "y" || "$a" == "Y" ]] || { echo "已取消"; exit 1; }; }
get_zone(){ if [ -f "$ZONE_STATE" ]; then cat "$ZONE_STATE"; else echo "$ZONE"; fi; }
# 显式指定用户名 thy(而不是让 gcloud 用 OS Login 从当前登录的 Google 账号
# htian@conceivable.life 现推一个 htian_conceivable_life 用户——那个账户在VM上
# 从来没建过环境,conda/domain_transfer全都不在它的$HOME下)。之前一直"恰好"
# 解析成thy,直到这次多次stop/start/换zone后才暴露出这个不稳定性。
gssh(){ gcloud compute ssh "thy@$INSTANCE" --zone="$(get_zone)" --project="$PROJECT" --tunnel-through-iap "$@"; }

require_remote_script(){
  [ -n "$REMOTE_SCRIPT" ] || {
    echo "!! 没有指定 REMOTE_SCRIPT,拒绝执行(以前默认回退到 _gcp_dinov3_dapt.sh,"
    echo "   导致忘记设这个变量时静默跑错脚本、白烧GCP账单——这个坑已经踩过两次)。"
    echo "   显式指定要跑哪个,例如:"
    echo "     REMOTE_SCRIPT=_gcp_dinov3_dapt.sh CONFIG=<yaml> $0 $1"
    exit 1
  }
}

preflight(){
  gcloud config set project "$PROJECT" >/dev/null 2>&1
  gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q . \
    || { echo "未登录,请先: gcloud auth login"; exit 1; }
}

ensure_bucket(){
  # 使用管理员预建的桶(us-east1),本账号仅有对象级权限、不能建桶,故不尝试创建。
  # 用一次对象级 ls 验证可访问;失败只警告不中断。
  if gcloud storage ls "$BUCKET/" >/dev/null; then
    log "桶可访问:$BUCKET"
  else
    log "警告:$BUCKET ls 失败(可能仅有写权限无列权限),继续尝试上传"
  fi
}

# ── 子命令 ─────────────────────────────────────────────────────────────────
cmd_upload(){
  preflight; ensure_bucket
  # 代码/数据分离:bucket 只放【权重+数据】,源码(scripts/repos/patches/manifests)走 scp(见 cmd_pushcode)。
  log "上传权重(数据类产物)-> $BUCKET/dapt/domain_transfer/weights"
  [ -d "$DT_LOCAL/weights" ] && gcloud storage rsync -r "$DT_LOCAL/weights" "$BUCKET/dapt/domain_transfer/weights"
  # 注:不上传 frames_hires(~60G,可从视频推导)。官方 DINOv3 DAPT 只用帧缓存 -> SKIP_VIDEO=1 跳过 62G 视频。
  if [ "${SKIP_VIDEO:-0}" = "1" ]; then
    log "SKIP_VIDEO=1 -> 跳过原始视频上传(用帧缓存即可)"
  else
    log "上传原始视频 -> $BUCKET/dapt/data/Video/Video  (~62G, 首次较久, 可断点续传)"
    gcloud storage rsync -r "$VIDEO_SRC" "$BUCKET/dapt/data/Video/Video"
  fi
  log "上传完成(仅权重/数据;源码不进 bucket,走 cmd_pushcode 的 scp)"
}

cmd_pushcode(){
  preflight
  gcloud compute instances describe "$INSTANCE" --zone="$(get_zone)" >/dev/null 2>&1 \
    || { echo "实例不存在,请先 up"; exit 1; }
  log "scp 源码直传 VM(不经过 bucket): scripts patches manifests"
  gssh --command="mkdir -p ~/domain_transfer"
  for d in scripts patches manifests; do
    [ -d "$DT_LOCAL/$d" ] && gcloud compute scp --recurse --tunnel-through-iap \
      --zone="$(get_zone)" --project="$PROJECT" "$DT_LOCAL/$d" "$INSTANCE:~/domain_transfer/"
  done
  log "源码已 scp 到 VM:~/domain_transfer (repos 由远端脚本 git clone + 打补丁)"

  # _gcp_dinov3_dapt.sh 自己的第 [1/6] 步会做
  # `gcloud storage rsync -r "$BUCKET/dapt/domain_transfer" "$DT_ROOT"`,
  # 把整个 bucket 前缀盖回 VM 本地目录——如果 bucket 里的 scripts/patches 是旧的
  # (之前从没在这里同步过),就会把刚 scp
  # 上去的新代码/新 patch 悄悄覆盖回旧版本,训练用的还是过时代码而不报错。
  # 这里把 scripts/patches/manifests 也同步进 bucket,保证两边一致、不会被覆盖。
  log "同步 scripts/patches/manifests -> bucket(防止远端脚本自己的 rsync 步骤覆盖回旧版本)"
  for d in scripts patches manifests; do
    [ -d "$DT_LOCAL/$d" ] && gcloud storage rsync -r "$DT_LOCAL/$d" "$BUCKET/dapt/domain_transfer/$d"
  done
}

cmd_up(){
  preflight
  # 之前这里只检查实例存不存在,存在就直接跳过——没检查实际是不是RUNNING状态。
  # 我们这次会话自己手动 stop 过这台实例(诊断卡死期间省钱),之后再跑 up 就踩到了
  # 这个坑:明明实例是STOPPED,却被当成"已就绪"直接跳过,导致后面 pushcode/train
  # 尝试SSH到一台关着的机器,报"Failed to connect to port 22"——不是网络抖动,
  # 是机器压根没开。现在按实际状态处理:不存在则创建,STOPPED则start,RUNNING才跳过。
  local st
  st=$(gcloud compute instances describe "$INSTANCE" --zone="$(get_zone)" --format="value(status)" 2>/dev/null || echo "")
  if [ "$st" = "RUNNING" ]; then
    log "实例 $INSTANCE 已在运行(zone=$(get_zone)),跳过创建/启动"; return
  elif [ -n "$st" ]; then
    log "实例 $INSTANCE 已存在但状态是 $st,执行 start ..."
    gcloud compute instances start "$INSTANCE" --project="$PROJECT" --zone="$(get_zone)"
    log "等待 SSH 就绪 ..."
    for i in $(seq 1 30); do
      gssh --command="echo ok" >/dev/null 2>&1 && { log "SSH 就绪"; return; }
      sleep 10
    done
    echo "SSH 迟迟未就绪,请稍后手动: ./scripts/dapt_train.sh train"; return
  fi
  confirm "即将创建 $MACHINE (A100 40GB) 实例,会开始计费。继续?"
  local spot_args=(); [[ "$SPOT" == "true" ]] && spot_args=(--provisioning-model=SPOT --instance-termination-action=STOP)
  local z
  for z in $ZONES; do
    log "尝试在 $z 创建 $INSTANCE ..."
    if gcloud compute instances create "$INSTANCE" \
        --project="$PROJECT" --zone="$z" \
        --machine-type="$MACHINE" \
        --maintenance-policy=TERMINATE \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size="${BOOT_DISK_GB}GB" --boot-disk-type=pd-ssd \
        --metadata="install-nvidia-driver=True" \
        --scopes=cloud-platform \
        "${spot_args[@]}"; then
      echo "$z" > "$ZONE_STATE"
      log "实例已创建于 $z"
      log "等待 SSH 就绪 ..."
      for i in $(seq 1 30); do
        gssh --command="echo ok" >/dev/null 2>&1 && { log "SSH 就绪"; return; }
        sleep 10
      done
      echo "SSH 迟迟未就绪,请稍后手动: ./scripts/dapt_train.sh train"; return
    fi
    log "$z 无 A100 容量或创建失败,换下一个 zone ..."
  done
  echo "所有候选 zone 都无 A100 容量。稍后再试,或编辑脚本 ZONES 换区(如 us-east1-b)。"; exit 1
}

cmd_train(){
  preflight; require_remote_script "train"
  # 本地设的环境变量不会自动带到 SSH 远端,这里显式转发 _gcp_dinov3_dapt.sh 认的那几个;
  # 留空时远端用自己的默认值。
  log "在 VM 上启动环境搭建 + 串行训练(tmux 会话 'dapt', 后台) [REMOTE_SCRIPT=$REMOTE_SCRIPT CONFIG=${CONFIG:-默认} DINO_BATCH=${DINO_BATCH:-默认} DINO_EPOCH_LEN=${DINO_EPOCH_LEN:-默认} DINO_EPOCHS=${DINO_EPOCHS:-默认}]"
  gssh --command="
    set -e
    gcloud storage cp '$BUCKET/dapt/domain_transfer/scripts/$REMOTE_SCRIPT' /tmp/remote.sh
    sed -i 's/\r\$//' /tmp/remote.sh
    chmod +x /tmp/remote.sh
    tmux kill-session -t dapt 2>/dev/null || true
    tmux new-session -d -s dapt \"BUCKET='$BUCKET' CONFIG='${CONFIG:-}' DINO_BATCH='${DINO_BATCH:-}' DINO_EPOCH_LEN='${DINO_EPOCH_LEN:-}' DINO_EPOCHS='${DINO_EPOCHS:-}' REPO_WEIGHTS='${REPO_WEIGHTS:-}' bash /tmp/remote.sh > \$HOME/dapt.log 2>&1\"
    echo '已在 tmux 会话 dapt 启动 ($REMOTE_SCRIPT)。'
  "
  log "已启动。用 ./scripts/dapt_train.sh status 查看进度"
}

cmd_status(){
  preflight; require_remote_script "status"
  local st
  st=$(gcloud compute instances describe "$INSTANCE" --zone="$(get_zone)" --format="value(status)" 2>/dev/null || echo "")
  echo "实例状态: ${st:-不存在}"
  if [ "$st" = "TERMINATED" ]; then
    log ">>> VM 已自动关机(训练完成或已停,GPU 计费已停)。运行 finish 取回结果并删除。"
    return
  fi
  log "最近 40 行日志(~/dapt.log):"
  gssh --command="tail -n 40 \$HOME/dapt.log 2>/dev/null || echo '日志还没生成'"
  echo
  # 完成标记文件名跟着 REMOTE_SCRIPT 走,不同脚本写不同的标记——避免读到旧脚本
  # 上一次跑遗留在 bucket 里的标记文件(教训:曾经把 DINOV3_DONE.txt 误判成"这次也done了")。
  local done_marker
  case "$REMOTE_SCRIPT" in
    _gcp_dinov3_dapt.sh)             done_marker="DINOV3_DONE.txt" ;;
    *)                                done_marker="" ;;
  esac
  if [ -n "$done_marker" ]; then
    gssh --command="gcloud storage ls '$BUCKET/dapt/outputs/$done_marker' >/dev/null 2>&1 && echo '>>> $REMOTE_SCRIPT 已完成 ($done_marker)' || echo '>>> 仍在进行中 (看上面日志)'"
  else
    log "未知 REMOTE_SCRIPT=$REMOTE_SCRIPT,不检查完成标记文件,只看上面的日志判断进度"
  fi
}

cmd_fetch(){
  preflight
  if gcloud storage ls "$BUCKET/dapt/outputs/" >/dev/null 2>&1; then
    log "下载 DAPT 结果 -> $DT_LOCAL/gcp_outputs/"
    gcloud storage rsync -r "$BUCKET/dapt/outputs" "$DT_LOCAL/gcp_outputs" || true
    log "已下载到 $DT_LOCAL/gcp_outputs (含 *_dapt 权重与日志)"
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

# ── 组合命令 ────────────────────────────────────────────────────────────────
cmd_start(){ cmd_upload; cmd_up; cmd_pushcode; cmd_train; }
cmd_finish(){ cmd_fetch || true; cmd_down; }

case "${1:-}" in
  upload)    cmd_upload ;;
  up)        cmd_up ;;
  pushcode)  cmd_pushcode ;;
  train)     cmd_train ;;
  status)    cmd_status ;;
  fetch)     cmd_fetch ;;
  down)      cmd_down ;;
  ssh)       cmd_ssh ;;
  start)     cmd_start ;;
  finish)    cmd_finish ;;
  *) echo "用法: $0 {start|status|finish | upload|up|pushcode|train|fetch|down|ssh}"; exit 1 ;;
esac
