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
    "auth print-access-token")                     # $FAKE/authfail = n: the login is expired for the next n checks
      local n; n="$(cat "$FAKE/authfail" 2>/dev/null || echo 0)"
      [ "$n" -gt 0 ] && { echo $((n - 1)) > "$FAKE/authfail"; return 1; }; return 0 ;;
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
# fake launch: consume one outcome from queue_<stage> (parallel arms) or the shared queue;
# run_stage's locals ($st $done_url $failed_url) are visible here
_launch(){
  local inst="$1" q="$FAKE/queue_$st" o; [ -f "$q" ] || q="$FAKE/queue"
  o="$(head -1 "$q" 2>/dev/null)"; sed -i 1d "$q" 2>/dev/null; o="${o:-done}"
  echo "launch $st $o $inst" >> "$FAKE/ops.log"
  case "$o" in
    done) : > "$(_b "$done_url")";   echo TERMINATED > "$FAKE/vm/$inst" ;;
    fail) : > "$(_b "$failed_url")"; echo TERMINATED > "$FAKE/vm/$inst" ;;
    lost) rm -f "$FAKE/vm/$inst" ;;
    slow) echo "SLOW $done_url" > "$FAKE/vm/$inst" ;;
    stockout) return 1 ;;
  esac
}
_dinst(){ local i; i="$(echo "${DAPT_ENV:-}" | grep -o 'INSTANCE=[^ ]*' | cut -d= -f2)"; echo "${i:-dapt-a100}"; }
bench_start(){ _launch bench-a100; }
dapt_start(){  _launch "$(_dinst)"; }
_do(){ echo "$1 $2 ${st:-}" >> "$FAKE/ops.log"; case "$2" in down|finish) rm -f "$FAKE/vm/$1" ;; esac; return 0; }
bench_do(){ _do bench-a100 "$1"; }
dapt_do(){  _do "$(_dinst)" "$1"; }
bench_verify(){ return 0; }
dapt_verify(){ return 0; }
ship_arm(){ echo "ship $1" >> "$FAKE/ops.log"; }
MOCK

RA_SCRIPT="${RA_SCRIPT:-$DT/experiments/run_all.sh}"     # RA_SCRIPT=experiments/run_all.next.sh tests a staged version
RA(){ bash "$RA_SCRIPT" "$@" > "$T/out.log" 2>&1; echo $?; }
reset(){ rm -rf "$RUN_STATE" "$FAKE/bucket"/* "$FAKE/vm"/* "$FAKE"/queue_*; : > "$FAKE/ops.log"; printf '%s\n' "$@" > "$FAKE/queue"; }
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
printf '%s\n' done stockout done done > "$FAKE/queue"     # e1_smoke | e234_smoke: stockout -> done | e234
printf '%s\n' fail done > "$FAKE/queue_e1_uniform"; printf '%s\n' lost slow > "$FAKE/queue_e1_motion"
echo slow > "$FAKE/queue_e1_content"; echo done > "$FAKE/queue_e1_motion_weighted"
rc=$(RA)
check "exit 0, everything done" '[ "$rc" = 0 ] && [ -f "$RUN_STATE/tables.done" ] && [ -f "$RUN_STATE/e234.done" ]'
check "e1_uniform relaunched after FAILED" '[ "$(n_launch e1_uniform)" = 2 ] && [ "$(cat "$RUN_STATE/e1_uniform.attempts")" = 2 ]'
check "e1_motion relaunched after VM loss" '[ "$(n_launch e1_motion)" = 2 ] && grep -q "e1_motion: 结果 = lost" "$T/out.log"'
check "failed launch (STOCKOUT) does not count as an attempt" '[ "$(cat "$RUN_STATE/e234_smoke.attempts")" = 1 ]'
check "STOCKOUT -> down + retry" '[ "$(n_launch e234_smoke)" = 2 ] && grep -q "dapt-a100 down\|bench-a100 down e234_smoke" "$FAKE/ops.log"'
check "all 4 arms shipped, in order" '[ "$(grep ^ship "$FAKE/ops.log" | tr "\n" " ")" = "ship uniform ship content ship motion ship motion_weighted " ]'
check "tables built" 'grep -q "python experiments/paper.py tables" "$FAKE/ops.log"'
if grep -q "arm_inst" "$RA_SCRIPT"; then   # parallel-arms version
  check "4 arms in parallel, one VM each" 'grep -q "4 个训练臂并行运行中" "$T/out.log" && [ "$(grep "^launch e1_[a-z_]* " "$FAKE/ops.log" | grep -v e1_smoke | awk "{print \$4}" | sort -u | tr "\n" " ")" = "dapt-a100-content dapt-a100-motion dapt-a100-motion-weighted dapt-a100-uniform " ]'
  check "e1_smoke still on dapt-a100" 'grep -q "^launch e1_smoke done dapt-a100$" "$FAKE/ops.log"'
  check "each arm cleaned up its own VM" 'grep -q "^dapt-a100-uniform down e1_uniform" "$FAKE/ops.log" && grep -q "^dapt-a100-motion-weighted finish e1_motion_weighted" "$FAKE/ops.log"'

  echo "== 2b. one arm gives up -> the other 3 still finish and ship; rerun only redoes that arm"
  rm -rf "$RUN_STATE"/e1_* "$RUN_STATE"/e234* "$RUN_STATE"/tables.*; rm -f "$FAKE/bucket"/* "$FAKE"/queue_*; : > "$FAKE/ops.log"; : > "$FAKE/queue"
  touch "$RUN_STATE/e1_smoke.done"; printf '%s\n' fail fail > "$FAKE/queue_e1_motion"
  rc=$(MAX_ATTEMPTS=2 RA)
  check "exit 1, 3 arms done + shipped, motion not" '[ "$rc" = 1 ] && [ -f "$RUN_STATE/e1_uniform.done" ] && [ -f "$RUN_STATE/e1_motion_weighted.done" ] && [ ! -f "$RUN_STATE/e1_motion.done" ] && [ "$(grep -c ^ship "$FAKE/ops.log")" = 3 ] && grep -q "e1_motion 没有完成" "$T/out.log" && [ ! -f "$RUN_STATE/e234_smoke.attempts" ]'
  : > "$FAKE/ops.log"; rm -f "$RUN_STATE/e1_motion.attempts"; echo done > "$FAKE/queue_e1_motion"
  rc=$(RA)
  check "rerun: only e1_motion launched, then the pipeline goes on" '[ "$rc" = 0 ] && [ "$(grep -c "^launch e1_" "$FAKE/ops.log")" = 1 ] && [ "$(n_launch e1_motion)" = 1 ] && [ -f "$RUN_STATE/e234.done" ]'
fi

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

echo "== 4b. gcloud login expires while a job runs (2026-09-28 incident) -> pause, no false 'lost', no attempt used"
rm -f "$RUN_STATE"/e234.*; rm -f "$FAKE/bucket"/*; : > "$FAKE/ops.log"; : > "$FAKE/queue"
echo "2026-01-01 00:00:00 attempt 1" > "$RUN_STATE/e234.launched"; echo 1 > "$RUN_STATE/e234.attempts"
echo "SLOW gs://mlflow-artifacts-ai-a100/bench/outputs_a1/DONE.txt" > "$FAKE/vm/bench-a100"; echo 5 > "$FAKE/authfail"
rc=$(RA)
check "waited for re-login, then done without relaunch" '[ "$rc" = 0 ] && [ -f "$RUN_STATE/e234.done" ] && [ "$(n_launch e234)" = 0 ] && ! grep -q "结果 = lost" "$T/out.log" && grep -q "登录已过期" "$T/out.log" && grep -q "登录已恢复" "$T/out.log"'

echo "== 4c. login expired at launch time -> waits, then launches as attempt 1"
rm -f "$RUN_STATE"/e234.*; rm -f "$FAKE/bucket"/*; : > "$FAKE/ops.log"; echo done > "$FAKE/queue"; echo 4 > "$FAKE/authfail"
rc=$(RA)
check "one launch, attempts = 1" '[ "$rc" = 0 ] && [ "$(n_launch e234)" = 1 ] && [ "$(cat "$RUN_STATE/e234.attempts")" = 1 ]'

echo "== 4d. launches keep failing -> gives up after MAX_START_FAILS, attempts untouched"
rm -f "$RUN_STATE"/e234.*; rm -f "$FAKE/bucket"/*; : > "$FAKE/ops.log"; printf '%s\n' stockout stockout stockout > "$FAKE/queue"
rc=$(MAX_START_FAILS=3 RA)
check "exit 1 after 3 failed launches, no attempt recorded" '[ "$rc" = 1 ] && [ "$(n_launch e234)" = 3 ] && [ ! -f "$RUN_STATE/e234.attempts" ]'

echo "== 5. status"
RA status >/dev/null
check "status lists stages" 'grep -q "e1_motion_weighted" "$T/out.log" && grep -q "GATE: 已批准" "$T/out.log"'

echo "== 6. lock: a second copy refuses to run"
( exec 9> "$RUN_STATE/.lock"; flock 9; rc=$(RA); echo "$rc" > "$T/lockrc" )
check "second instance exits 1" '[ "$(cat "$T/lockrc")" = 1 ] && grep -q "已经在运行" "$T/out.log"'

echo "== 7. stop kills run_all and its children (the bench_gcp retry loop)"
rm -f "$RUN_STATE"/e234.*; echo slowstart > "$FAKE/queue"
cat >> "$RUN_ALL_MOCK" <<'MOCK'
bench_start(){ bash -c 'sleep 777' ; }    # stands in for the zone-retry loop; a child process that must die too
MOCK
rm -f "$RUN_STATE/run_all.pid"
setsid bash "$RA_SCRIPT" > "$T/bg.log" 2>&1 &
until [ -f "$RUN_STATE/run_all.pid" ] && pgrep -f "sleep 777" >/dev/null; do command sleep 0.2; done
RA stop >/dev/null; command sleep 1
for _ in $(seq 10); do pgrep -f "sleep 777" >/dev/null || kill -0 "$(cat "$RUN_STATE/run_all.pid")" 2>/dev/null || break; command sleep 0.5; done   # SIGTERM takes ~1-2 s
check "stop reports and kills children" 'grep -q "已停止" "$T/out.log" && ! pgrep -f "sleep 777" >/dev/null && ! kill -0 "$(cat "$RUN_STATE/run_all.pid")" 2>/dev/null'
RA stop >/dev/null
check "stop when not running" 'grep -q "没有在运行" "$T/out.log"'

echo "RESULT: $pass passed, $fail failed"; [ "$fail" = 0 ]
