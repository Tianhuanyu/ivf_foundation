#!/usr/bin/env bash
# test_run_all.sh — tests experiments/run_all.sh's resume/relaunch logic with a fake GCP (no VM, no cost).
#   bash experiments/test_run_all.sh          (WSL, conda env dapt)
# Each fake VM launch takes its outcome from a queue: done | fail | lost | slow (RUNNING for one poll, then done).
set -uo pipefail
DT="$(cd "$(dirname "$0")/.." && pwd)"
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
export FAKE="$T/fake" RUN_STATE="$T/state" RUN_ALL_MOCK="$T/mock.sh"
mkdir -p "$FAKE/bucket" "$FAKE/vm"

cat > "$RUN_ALL_MOCK" <<'MOCK'
sleep(){ :; }
python(){ echo "python $*" >> "$FAKE/ops.log"; }
_b(){ echo "$FAKE/bucket/${1//\//_}"; }
gcloud(){
  case "$1 $2" in
    "storage ls") [ -f "$(_b "$3")" ] ;;
    "storage rm") shift 2; local u; for u in "$@"; do rm -f "$(_b "$u")"; done; return 0 ;;
    *) return 0 ;;
  esac
}
inst_status(){
  local f="$FAKE/vm/$1"; [ -f "$f" ] || { echo ABSENT; return; }
  local s; read -r s url < "$f"
  case "$s" in
    SLOW) echo "TERMINATED_NEXT $url" > "$f"; echo RUNNING ;;         # one more poll, then the job finishes
    TERMINATED_NEXT) : > "$(_b "$url")"; echo TERMINATED > "$f"; echo TERMINATED ;;
    *) echo "$s" ;;
  esac
}
# fake launch: consume one outcome from the queue; run_stage's locals ($st $done_url $failed_url) are visible here
_launch(){
  local inst="$1" o; o="$(head -1 "$FAKE/queue" 2>/dev/null)"; sed -i 1d "$FAKE/queue" 2>/dev/null; o="${o:-done}"
  echo "launch $st $o" >> "$FAKE/ops.log"
  case "$o" in
    done) : > "$(_b "$done_url")";   echo TERMINATED > "$FAKE/vm/$inst" ;;
    fail) : > "$(_b "$failed_url")"; echo TERMINATED > "$FAKE/vm/$inst" ;;
    lost) rm -f "$FAKE/vm/$inst" ;;
    slow) echo "SLOW $done_url" > "$FAKE/vm/$inst" ;;
    stockout) return 1 ;;
  esac
}
bench_start(){ _launch bench-a100; }
dapt_start(){  _launch dapt-a100; }
_do(){ echo "$1 $2 ${st:-}" >> "$FAKE/ops.log"; case "$2" in down|finish) rm -f "$FAKE/vm/$1" ;; esac; return 0; }
bench_do(){ _do bench-a100 "$1"; }
dapt_do(){  _do dapt-a100 "$1"; }
bench_verify(){ return 0; }
dapt_stage(){ DAPT_ENV="$3" STAGE_FLEX_RUN="$4" run_stage "$1" dapt_start dapt_do "$DAPT_OUT/DINOV3_DONE_$2.txt" \
                "$DAPT_OUT/DINOV3_FAILED_$2.txt" dapt-a100 /dev/null true; }
ship_arm(){ echo "ship $1" >> "$FAKE/ops.log"; }
MOCK

RA(){ bash "$DT/experiments/run_all.sh" "$@" > "$T/out.log" 2>&1; echo $?; }
reset(){ rm -rf "$RUN_STATE" "$FAKE/bucket"/* "$FAKE/vm"/*; : > "$FAKE/ops.log"; printf '%s\n' "$@" > "$FAKE/queue"; }
n_launch(){ grep -c "^launch $1 " "$FAKE/ops.log"; }
pass=0; fail=0
check(){ if eval "$2"; then echo "  PASS  $1"; pass=$((pass+1)); else echo "  FAIL  $1"; fail=$((fail+1)); sed 's/^/        /' "$T/out.log" | tail -12; fi; }

echo "== 1. fresh start: runs e0_smoke, e0, stops at the GATE"
reset
rc=$(RA)
check "exit 0 at GATE" '[ "$rc" = 0 ] && grep -q "GATE" "$T/out.log"'
check "e0_smoke + e0 done, E1 not started" '[ -f "$RUN_STATE/e0.done" ] && [ ! -f "$RUN_STATE/e1_smoke.attempts" ]'
check "rerun before approval launches nothing" 'RA >/dev/null; [ "$(grep -c ^launch "$FAKE/ops.log")" = 2 ]'

echo "== 2. approved: failure, VM loss and STOCKOUT are all retried; full pipeline completes"
rm -f "$FAKE/bucket"/*; : > "$FAKE/ops.log"; touch "$RUN_STATE/APPROVE_E1"
printf '%s\n' done fail done done lost slow done stockout done done done > "$FAKE/queue"
#            e1_smoke | e1_uniform: fail -> done | content | motion: lost -> slow | motion_weighted | e234_smoke: stockout -> done | e234
rc=$(RA)
check "exit 0, everything done" '[ "$rc" = 0 ] && [ -f "$RUN_STATE/tables.done" ] && [ -f "$RUN_STATE/e234.done" ]'
check "e1_uniform relaunched after FAILED" '[ "$(n_launch e1_uniform)" = 2 ] && [ "$(cat "$RUN_STATE/e1_uniform.attempts")" = 2 ]'
check "e1_motion relaunched after VM loss" '[ "$(n_launch e1_motion)" = 2 ] && grep -q "e1_motion: 结果 = lost" "$T/out.log"'
check "STOCKOUT -> down + retry" '[ "$(n_launch e234_smoke)" = 2 ] && grep -q "dapt-a100 down\|bench-a100 down e234_smoke" "$FAKE/ops.log"'
check "all 4 arms shipped, in order" '[ "$(grep ^ship "$FAKE/ops.log" | tr "\n" " ")" = "ship uniform ship content ship motion ship motion_weighted " ]'
check "tables built" 'grep -q "python experiments/paper.py tables" "$FAKE/ops.log"'

echo "== 3. orchestrator dies while a job runs -> rerun waits for it, does NOT relaunch"
rm -f "$RUN_STATE"/e234.*; rm -f "$FAKE/bucket"/*; : > "$FAKE/ops.log"; : > "$FAKE/queue"
echo "2026-01-01 00:00:00 attempt 1" > "$RUN_STATE/e234.launched"; echo 1 > "$RUN_STATE/e234.attempts"
echo "SLOW gs://mlflow-artifacts-ai-a100/bench/outputs_a1/DONE.txt" > "$FAKE/vm/bench-a100"
rc=$(RA)
check "no new launch, waited, done" '[ "$rc" = 0 ] && [ "$(n_launch e234)" = 0 ] && [ -f "$RUN_STATE/e234.done" ] && grep -q "之前已启动" "$T/out.log"'

echo "== 4. gives up after MAX_ATTEMPTS"
rm -f "$RUN_STATE"/e234.*; rm -f "$FAKE/bucket"/*; : > "$FAKE/ops.log"; printf '%s\n' fail fail fail > "$FAKE/queue"
rc=$(MAX_ATTEMPTS=2 RA)
check "exit 1 after 2 attempts" '[ "$rc" = 1 ] && [ "$(n_launch e234)" = 2 ] && [ ! -f "$RUN_STATE/e234.done" ]'

echo "== 5. status"
RA status >/dev/null
check "status lists stages" 'grep -q "e1_motion_weighted" "$T/out.log" && grep -q "GATE: 已批准" "$T/out.log"'

echo "== 6. lock: a second copy refuses to run"
( exec 9> "$RUN_STATE/.lock"; flock 9; rc=$(RA); echo "$rc" > "$T/lockrc" )
check "second instance exits 1" '[ "$(cat "$T/lockrc")" = 1 ] && grep -q "已经在运行" "$T/out.log"'

echo "== 7. stop kills run_all and its children (the bench_gcp retry loop)"
rm -f "$RUN_STATE"/e234.*; echo slowstart > "$FAKE/queue"
cat >> "$RUN_ALL_MOCK" <<'MOCK'
bench_start(){ bash -c 'sleep 300' ; }    # stands in for the zone-retry loop; a child process that must die too
MOCK
setsid bash "$DT/experiments/run_all.sh" > "$T/bg.log" 2>&1 &
until [ -f "$RUN_STATE/run_all.pid" ] && pgrep -f "sleep 300" >/dev/null; do command sleep 0.2; done
RA stop >/dev/null; command sleep 1
check "stop reports and kills children" 'grep -q "已停止" "$T/out.log" && ! pgrep -f "sleep 300" >/dev/null && ! kill -0 "$(cat "$RUN_STATE/run_all.pid")" 2>/dev/null'
RA stop >/dev/null
check "stop when not running" 'grep -q "没有在运行" "$T/out.log"'

echo "RESULT: $pass passed, $fail failed"; [ "$fail" = 0 ]
