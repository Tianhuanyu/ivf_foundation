#!/usr/bin/env bash
# =============================================================================
# _gcp_common.sh — shared config + helpers for the LOCAL GCP drivers (dapt_train.sh, dapt_prep.sh).
# Sourced, not run. Everything GCP-account-specific lives here and only here.
# =============================================================================
PROJECT="hidden-outrider-390502"            # Conceivable Cloud
# A100 zones tried in order when out of capacity (STOCKOUT). us-east1-b first = same region as the bucket
# (fastest data, no cross-region egress). Override e.g. ZONES="europe-west4-a asia-northeast1-a" ...
ZONES="${ZONES:-us-east1-b us-central1-a us-central1-b us-central1-c us-central1-f us-west1-b us-west3-b us-west4-b}"
RETRY_MIN="${RETRY_MIN:-0}"   # >0: if every zone is sold out, wait this many minutes and try all zones again
# FLEX=1: Dynamic Workload Scheduler "flex-start" -- instead of failing on STOCKOUT, GCP QUEUES the request
#   and creates the VM once an A100 frees up in that zone (waits up to FLEX_WAIT, default 2h = the GCP maximum; with RETRY_MIN it re-queues). The VM runs at
#   most FLEX_RUN (default 24h) and is then DELETED automatically -> set FLEX_RUN above the job's runtime.
#   Results are uploaded to the bucket as soon as a job ends, so the forced deletion loses nothing.
FLEX="${FLEX:-0}"; FLEX_WAIT="${FLEX_WAIT:-2h}"; FLEX_RUN="${FLEX_RUN:-24h}"
flex_args(){ [ "$FLEX" = "1" ] && printf '%s\n' --provisioning-model=FLEX_START --request-valid-for-duration="$FLEX_WAIT" \
               --max-run-duration="$FLEX_RUN" --instance-termination-action=DELETE --reservation-affinity=none; }
BUCKET="gs://mlflow-artifacts-ai-a100"       # admin-created bucket (us-east1); our data lives under dapt/
DT_LOCAL="/mnt/d/Video/domain_transfer"

LOG_TAG="${LOG_TAG:-gcp}"
log(){ echo -e "\n\033[1;36m[$LOG_TAG] $*\033[0m"; }
confirm(){   # YES=1: non-interactive (experiments/run_all.sh) -- answers yes to create/delete prompts
  if [ "${YES:-0}" = "1" ]; then echo "$1 [YES=1 -> y]"; return 0; fi
  read -r -p "$1 [y/N] " a; [[ "$a" == "y" || "$a" == "Y" ]] || { echo "已取消"; exit 1; }; }

preflight(){
  gcloud config set project "$PROJECT" >/dev/null 2>&1
  gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q . \
    || { echo "未登录,请先: gcloud auth login"; exit 1; }
}

# create_in_zones <instance> <gcloud create args...>
#   Tries each zone in $ZONES; prints the zone that worked on stdout, returns 1 if none did.
create_in_zones(){
  local inst="$1"; shift
  local z round=1
  while true; do
    for z in $ZONES; do
      log "尝试在 $z 创建 $inst ..." >&2
      [ "$FLEX" = "1" ] && log "flex-start:在 $z 排队等 A100(最多 $FLEX_WAIT;实例最长运行 $FLEX_RUN,到时自动删除)..." >&2
      if gcloud compute instances create "$inst" --project="$PROJECT" --zone="$z" "$@" $(flex_args) >&2; then
        echo "$z"; return 0
      fi
      log "$z 无容量或创建失败,换下一个 zone ..." >&2
    done
    [ "$RETRY_MIN" -gt 0 ] || return 1
    log "第 $round 轮所有 zone 都没有容量,${RETRY_MIN} 分钟后重试(Ctrl+C 停止)..." >&2
    sleep $((RETRY_MIN * 60)); round=$((round + 1))
  done
}

# wait_ssh <ssh command...>   e.g. wait_ssh gssh
wait_ssh(){
  log "等待 SSH 就绪 ..."
  local i
  for i in $(seq 1 30); do
    "$@" --command="echo ok" >/dev/null 2>&1 && { log "SSH 就绪"; return 0; }
    sleep 10
  done
  return 1
}
