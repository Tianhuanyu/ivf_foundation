#!/usr/bin/env bash
# =============================================================================
# run_all.sh — the whole GCP pipeline in ONE script, crash-safe and resumable (WSL).
#
#   nohup setsid bash experiments/run_all.sh >> run_state/run_all.log 2>&1 &     # start / resume (detached!)
#   bash experiments/run_all.sh status                                         # what's done / running
#   bash experiments/run_all.sh stop                                           # stop it (and its child processes)
#   tail -f run_state/run_all.log                                              # live log
#
# Stages, in order (each one: start VM -> wait -> fetch -> delete VM -> verify):
#   e0_smoke  e0                                      ViT vs CNN on held-out acquisition batches
#   ── GATE ── stops here until you approve E1:  touch run_state/APPROVE_E1   (then run the start command again)
#   e1_smoke  e1_uniform  e1_content  e1_motion  e1_motion_weighted   (each long arm is shipped to the benchmark)
#   e234_smoke  e234                                  main table + sampling ablation + 320-px ablation
#   tables                                            experiments/paper.py tables -> experiments/out/tables.md
#
# Crash safety, three levels:
#   1. this script: every finished stage is recorded in run_state/<stage>.done; a launched stage in
#      run_state/<stage>.launched. Re-running continues where it stopped and never relaunches a job that is
#      still running on its VM -- it just waits for it again.
#   2. the VM job: results (benchmark) / checkpoints (training) are uploaded to the bucket while it runs; a
#      relaunched job restores them and continues (--skip-done / checkpoint resume).
#   3. VM loss (flex-start time limit, preemption, crash, STOCKOUT on restart): the stage is relaunched on a new
#      VM, up to MAX_ATTEMPTS times; each job writes DONE or FAILED to the bucket and powers the VM off either way.
#
# Knobs (env): RETRY_MIN (default 10), FLEX=1 [ZONES=us-east1-b], MAX_ATTEMPTS (6), POLL_MIN (5).
#   With FLEX=1 each stage gets its own max VM runtime (see the pipeline below); when it expires the VM is deleted
#   and the stage is simply relaunched and resumes.
# =============================================================================
set -uo pipefail
DT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${BENCH_ROOT:-/mnt/d/Conceivable-SharedData01-23Jun2026}"
STATE="${RUN_STATE:-$DT/run_state}"; mkdir -p "$STATE"
export PATH="$HOME/google-cloud-sdk/bin:$PATH"          # the WSL-native gcloud (the /mnt/c Windows one stalls)
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate dapt
export YES=1 RETRY_MIN="${RETRY_MIN:-10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"      # successful launches per stage (VM loss / job failure); a resumed job keeps its progress
MAX_START_FAILS="${MAX_START_FAILS:-10}"   # failed launches in a row before giving up
POLL_MIN="${POLL_MIN:-5}"
BUCKET="gs://mlflow-artifacts-ai-a100"
PROJECT="hidden-outrider-390502"
BENCH_OUT="$BUCKET/bench/outputs_a1"
DAPT_OUT="$BUCKET/dapt/outputs"
ARMS="uniform content motion motion_weighted"

log(){ echo "[run_all $(date '+%F %T')] $*"; }
is_done(){ [ -f "$STATE/$1.done" ]; }
mark(){ echo "$(date '+%F %T') ${2:-}" > "$STATE/$1.$3"; }
# gcloud login can expire mid-run (org re-auth policy). Every gcloud error must NOT be read as "VM gone" /
# "job failed": we pause until `gcloud auth login` is redone, without consuming attempts.
gcloud_ok(){ gcloud auth print-access-token >/dev/null 2>&1; }
wait_auth(){
  gcloud_ok && return 0
  log "!! gcloud 登录已过期 -> 请在 WSL 里执行: gcloud auth login   (脚本在等你,不消耗重试次数,VM 上的任务不受影响)"
  until gcloud_ok; do sleep 300; done
  log "gcloud 登录已恢复,继续"
}
inst_status(){   # inst_status <instance> -> RUNNING / TERMINATED / ... / ABSENT / UNKNOWN (gcloud error: never "lost")
  local out
  out="$(gcloud compute instances list --project="$PROJECT" --filter="name=$1" --format='value(status)' 2>/dev/null)" \
    || { echo UNKNOWN; return; }
  echo "${out:-ABSENT}"
}

# ── generic stage runner ───────────────────────────────────────────────────────────────────────
# run_stage <stage> <start-fn> <finish-fn> <done-url> <failed-url> <instance> <zone-file> [verify-fn]
# attempts = successful launches only; failed launches (STOCKOUT beyond RETRY_MIN, transient errors) retry
# separately, up to MAX_START_FAILS in a row.
run_stage(){
  local st="$1" start_fn="$2" finish_fn="$3" done_url="$4" failed_url="$5" inst="$6" zfile="$7" verify_fn="${8:-true}"
  is_done "$st" && { log "$st: 已完成,跳过"; return 0; }
  local attempt fails=0; attempt="$(cat "$STATE/$st.attempts" 2>/dev/null || echo 0)"
  while true; do
    if [ ! -f "$STATE/$st.launched" ]; then
      [ "$attempt" -lt "$MAX_ATTEMPTS" ] || { log "!! $st: 已启动 $MAX_ATTEMPTS 次仍没成功,停止。看 run_state/run_all.log 和 VM 日志"; return 1; }
      wait_auth
      [ "$attempt" -eq 0 ] && gcloud storage rm "$done_url" "$failed_url" >/dev/null 2>&1
      log "$st: 启动(第 $((attempt + 1)) 次)"
      if ! $start_fn; then
        fails=$((fails + 1))
        [ "$fails" -lt "$MAX_START_FAILS" ] || { log "!! $st: 连续 $fails 次启动失败,停止。看 run_state/run_all.log"; return 1; }
        log "$st: 启动失败($fails/$MAX_START_FAILS)-> 删除可能残留的旧实例后重试"
        wait_auth; $finish_fn down >/dev/null 2>&1 || true
        sleep 60; continue
      fi
      fails=0; attempt=$((attempt + 1)); echo "$attempt" > "$STATE/$st.attempts"
      mark "$st" "attempt $attempt" launched
    else
      log "$st: 之前已启动,继续等待(第 $attempt 次)"
    fi
    # wait for DONE / FAILED / VM loss
    local outcome=""
    while [ -z "$outcome" ]; do
      wait_auth
      if gcloud storage ls "$done_url" >/dev/null 2>&1; then outcome=done
      elif gcloud storage ls "$failed_url" >/dev/null 2>&1; then outcome=failed
      else
        case "$(inst_status "$inst")" in
          ABSENT)                      gcloud_ok && outcome=lost ;;   # e.g. flex-start time limit deleted it
          TERMINATED|STOPPED|SUSPENDED) sleep 120            # job may be writing its marker right before poweroff
                                       gcloud_ok && ! gcloud storage ls "$done_url" >/dev/null 2>&1 && outcome=lost ;;
          *) sleep $((POLL_MIN * 60)) ;;                      # RUNNING / PROVISIONING / UNKNOWN (gcloud error)
        esac
      fi
    done
    log "$st: 结果 = $outcome"
    if [ "$outcome" = done ]; then
      $finish_fn finish || { log "!! $st: finish(取回+删机)失败,重跑本脚本会重试"; return 1; }
      if $verify_fn; then mark "$st" "" done; rm -f "$STATE/$st.launched"; log "$st: ✅ 完成"; return 0; fi
      log "!! $st: 结果校验没通过,停止(不自动重跑,需要人工看)"; return 1
    fi
    log "$st: 没有成功($outcome)-> 取回已有结果、删机,然后续跑"
    wait_auth
    $finish_fn fetch >/dev/null 2>&1 || true
    $finish_fn down  >/dev/null 2>&1 || true
    gcloud storage rm "$failed_url" >/dev/null 2>&1
    rm -f "$STATE/$st.launched"
  done
}

# ── benchmark stages (bench VM, benchmark repo) ──────────────────────────────────────────────
bench_do(){ (cd "$BENCH" && ./bench_gcp.sh "$@"); }
bench_start(){ (cd "$BENCH" && BENCH_CMD="$BENCH_JOB" FLEX_RUN="$STAGE_FLEX_RUN" ./bench_gcp.sh start); }
bench_verify(){ grep -q "\[PASS\]" "$BENCH/stage1_out/gcp_bench_logs/bench_run.log" 2>/dev/null \
                && ! grep -q "\[FAIL\]" "$BENCH/stage1_out/gcp_bench_logs/bench_run.log"; }
bench_stage(){   # bench_stage <stage> <BENCH_CMD> <flex max runtime>
  BENCH_JOB="$2" STAGE_FLEX_RUN="$3" run_stage "$1" bench_start bench_do "$BENCH_OUT/DONE.txt" "$BENCH_OUT/FAILED.txt" \
    bench-a100 "$BENCH/.bench_zone" bench_verify
}

# ── training stages (dapt VM, training repo) ─────────────────────────────────────────────────
dapt_do(){ (cd "$DT" && ./scripts/dapt_train.sh "$@"); }
dapt_start(){ (cd "$DT" && env $DAPT_ENV FLEX_RUN="$STAGE_FLEX_RUN" ./scripts/dapt_train.sh start); }
dapt_stage(){    # dapt_stage <stage> <RUN name> "<ENV for dapt_train.sh start>" <flex max runtime>
  DAPT_ENV="$3" STAGE_FLEX_RUN="$4" run_stage "$1" dapt_start dapt_do "$DAPT_OUT/DINOV3_DONE_$2.txt" "$DAPT_OUT/DINOV3_FAILED_$2.txt" \
    dapt-a100 "$DT/.dapt_a100_zone" "test -f $DT/gcp_outputs/dinov3_vitb16_dapt_$2_backbone.pth"
}
ship_arm(){      # hand one long arm to the benchmark (idempotent)
  local src="$DT/gcp_outputs/dinov3_vitb16_dapt_$1_long_backbone.pth"
  local dst="$BENCH/stage1_out/dinov3_ckpt/dinov3_vitb16_dapt_$1_long.pth"
  [ -f "$dst" ] && { log "ship $1: 已交接,跳过"; return 0; }
  (cd "$DT" && ./scripts/ship_weights.sh "$src" "$dst" \
     "domain_transfer@$(git rev-parse --short HEAD), ARM=$1 BUDGET=long" "ViT-B/16, 20000 it, batch 48, A100")
}

# test hook: RUN_ALL_MOCK=<file> replaces gcloud / VM operations with local fakes (experiments/test_run_all.sh)
[ -n "${RUN_ALL_MOCK:-}" ] && source "$RUN_ALL_MOCK"

# ── the pipeline ──────────────────────────────────────────────────────────────────────────────
STAGES="e0_smoke e0 GATE e1_smoke $(for a in $ARMS; do printf 'e1_%s ' "$a"; done)e234_smoke e234 tables"
status(){
  local s
  for s in $STAGES; do
    if [ "$s" = GATE ]; then
      [ -f "$STATE/APPROVE_E1" ] && echo "  ── GATE: 已批准 E1" || echo "  ── GATE: 等待批准(看完 T0 后 touch run_state/APPROVE_E1)"
    elif is_done "$s"; then echo "  ✅ $s  $(cat "$STATE/$s.done")"
    elif [ -f "$STATE/$s.launched" ]; then echo "  … $s  运行中(第 $(cat "$STATE/$s.attempts") 次,$(cat "$STATE/$s.launched" | cut -c1-19) 启动)"
    else echo "  ·  $s"; fi
  done
}
[ "${1:-}" = status ] && { status; exit 0; }
if [ "${1:-}" = stop ]; then   # kill run_all AND its children (bench_gcp/dapt_train retry loops, gcloud) -- pkill -f run_all.sh alone leaves them
  pid="$(cat "$STATE/run_all.pid" 2>/dev/null)"
  ps -o args= -p "${pid:-0}" 2>/dev/null | grep -q run_all.sh || pid=""     # stale pid file / reused pid
  pgid="$(ps -o pgid= -p "${pid:-0}" 2>/dev/null | tr -d ' ')"
  if [ -z "$pgid" ]; then echo "run_all.sh 没有在运行"; exit 0; fi
  kill -TERM -- "-$pgid" && echo "已停止 run_all.sh(进程组 $pgid)。VM 上已经在跑的任务不受影响;重新启动后会接着等它。"
  exit 0
fi
exec 9> "$STATE/.lock"
flock -n 9 || { echo "!! run_all.sh 已经在运行(pgrep -af run_all.sh 查看),不要同时开两个"; exit 1; }
echo $$ > "$STATE/run_all.pid"

log "===== run_all 开始 / 续跑(状态目录 $STATE)====="
status
bench_stage e0_smoke "bash paper_jobs.sh E0 smoke" 12h || exit 1
bench_stage e0       "bash paper_jobs.sh E0 run"   72h || exit 1
if [ ! -f "$STATE/APPROVE_E1" ]; then
  (cd "$DT" && python experiments/paper.py tables >/dev/null 2>&1) || true
  log "===== GATE:E0 已完成。看 experiments/out/tables.md 的 T0,按 PAPER.md 判定。"
  log "      决定继续就执行:touch $STATE/APPROVE_E1  然后重新启动本脚本。====="
  exit 0
fi
dapt_stage e1_smoke motion_weighted_e1 "ARM=motion_weighted BUDGET=e1 DINO_EPOCH_LEN=5 DINO_EPOCHS=1 DINO_BATCH=8" 6h || exit 1
for a in $ARMS; do
  dapt_stage "e1_$a" "${a}_long" "ARM=$a BUDGET=long" 36h || exit 1
  ship_arm "$a" || { log "!! ship $a 失败"; exit 1; }
done
bench_stage e234_smoke "bash paper_jobs.sh E2 smoke && bash paper_jobs.sh E3 smoke && bash paper_jobs.sh E4 smoke" 24h  || exit 1
bench_stage e234       "bash paper_jobs.sh E2 run && bash paper_jobs.sh E3 run && bash paper_jobs.sh E4 run"       168h || exit 1
if ! is_done tables; then
  (cd "$DT" && python experiments/paper.py status && python experiments/paper.py tables) && mark tables "" done
fi
log "===== 全部完成:experiments/out/tables.md ====="
status
