#!/usr/bin/env bash
# =============================================================================
# smoke_wsl.sh — smoke-test every WSL-side script of the paper pipeline before real training/uploading.
#
#   bash experiments/smoke_wsl.sh            # run all steps (sequential, ~1-2 h, mostly benchmark smoke)
#   STEPS="data dapt ship" bash experiments/smoke_wsl.sh      # only some groups
#
# Run it DETACHED so a dropped terminal / tool timeout can't kill it or wedge WSL:
#   nohup setsid bash experiments/smoke_wsl.sh > smoke_out/smoke_wsl.log 2>&1 &
#   cat smoke_out/SUMMARY_wsl.txt            # PASS/FAIL per step, updated as it goes
#
# Nothing here touches real artifacts: outputs go to smoke_out/ (training repo) and
# stage1_out/benchmark_results_smoke/ (benchmark repo); ship_weights writes to a COPY of the registry.
# Windows-side steps (12/13, reports, paper.py) are in experiments/smoke_windows.py.
# =============================================================================
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate dapt
DT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${BENCH_ROOT:-/mnt/d/Conceivable-SharedData01-23Jun2026}"
S="$DT/smoke_out"; mkdir -p "$S/logs"
SUM="$S/SUMMARY_wsl.txt"
STEPS="${STEPS:-data dapt ship bench}"
echo "smoke_wsl started $(date '+%F %T')  steps: $STEPS" > "$SUM"

run(){   # run <name> <command...>: log to smoke_out/logs/<name>.log, record PASS/FAIL + duration
  local name="$1"; shift
  local t0=$SECONDS
  if ( "$@" ) > "$S/logs/$name.log" 2>&1; then r=PASS; else r="FAIL(rc=$?)"; fi
  printf '%-44s %-10s %5ss  logs/%s.log\n' "$name" "$r" "$((SECONDS - t0))" "$name" | tee -a "$SUM"
}
has(){ [[ " $STEPS " == *" $1 "* ]]; }

E0_DS="cellasp holding_pip routine2_coc cvit_incubator cvit_workstation"
DET4="holding_pip routine2_coc cvit_incubator cvit_workstation"

if has data; then
  run data_10_build_manifest    python3 "$DT/scripts/10_build_manifest.py" --out "$S/manifests"
  run data_11_extract_frames    python "$DT/scripts/11_extract_frames.py" --manifest "$DT/manifests/train_videos.txt" \
                                  --split train --limit 1 --short 1536 --fps 5 --out "$S/frames"
fi

if has dapt; then
  cd "$DT"
  for arm in uniform content motion motion_weighted motion_ibotlocal_high motion_ibotlocal_low; do
    run "dapt_run_${arm}_e1"    env SMOKE=1 ARM=$arm BUDGET=e1 bash scripts/dapt_run.sh
  done
  run dapt_run_motion_weighted_long  env SMOKE=1 ARM=motion_weighted BUDGET=long bash scripts/dapt_run.sh
  run dapt_arms_rejects_bad_arm  bash -c '! (source scripts/dapt_arms.sh && dapt_overrides nope e1 x y)'
  run dapt_train_requires_arm    bash -c '! env -u ARM bash scripts/dapt_train.sh start'
  run gcp_scripts_syntax         bash -c 'for f in scripts/dapt_train.sh scripts/dapt_prep.sh scripts/_gcp_dinov3_dapt.sh scripts/_gcp_prep.sh scripts/_gcp_common.sh scripts/02_install_deps.sh; do bash -n "$f" || exit 1; done'
  run patch_reproduces_worktree  bash experiments/check_patch.sh    # fresh upstream clone + patch == worktree (CRLF-insensitive)
fi

if has ship; then
  cp /mnt/d/Conceivable-ML/WEIGHTS_REGISTRY.md "$S/registry_copy.md"
  run ship_weights               env WEIGHTS_REGISTRY="$S/registry_copy.md" bash "$DT/scripts/ship_weights.sh" \
                                   "$S/dinov3_vitb16_dapt_motion_weighted_e1_backbone.pth" "$S/shipped/dapt_smoke.pth" "smoke" "smoke"
  run ship_weights_row_in_table  bash -c "tail -n 3 '$S/registry_copy.md' | grep -q 'dinov3_vitb16_dapt_motion_weighted_e1_backbone.pth'"
fi

if has bench; then
  cd "$BENCH"
  SMOKE_BB="$S/dinov3_vitb16_dapt_motion_weighted_e1_backbone.pth"
  # E0 (next real step): all 30 configs on the acquisition split
  for bb in resnet50_fpn dinov3_b_fpn; do
    run "bench_E0_$bb"          python run_benchmark_server.py --all-seeds --smoke --split-profile acquisition --dataset $E0_DS --backbone $bb
  done
  run bench_E0_dinov3_dapt_b_fpn python run_benchmark_server.py --all-seeds --smoke --split-profile acquisition --dataset $E0_DS --backbone dinov3_dapt_b_fpn
  # E2: every backbone x every dataset (frozen) + finetune on one dataset per task family
  for bb in resnet50_fpn dinov2_s_fpn dinov2_b_fpn biomedclip_fpn dinov3_b_fpn dinov3_dapt_b_fpn; do
    run "bench_E2_frozen_$bb"   python run_benchmark_server.py --all-seeds --smoke --protocol frozen --backbone $bb
    run "bench_E2_finetune_$bb" python run_benchmark_server.py --all-seeds --smoke --protocol finetune --dataset cellasp icsi_seg holding_pip --backbone $bb
  done
  # E3: a different DAPT checkpoint via env -> separate run_id (weights tag)
  run bench_E3_weights_override  env DINOV3_DAPT_B_WEIGHTS="$SMOKE_BB" python run_benchmark_server.py --all-seeds --smoke --protocol frozen --dataset holding_pip --backbone dinov3_dapt_b_fpn
  # E4: 320-px detection ablation
  for bb in resnet50_fpn dinov2_b_fpn dinov3_b_fpn dinov3_dapt_b_fpn; do
    run "bench_E4_$bb"          python run_benchmark_hires_ablation.py --all-seeds --smoke --imgsz 320 --protocol frozen --dataset $DET4 --backbone $bb
  done
  RUN_DIR=$(ls -d stage1_out/benchmark_results_smoke/resnet50_fpn__detect__holding_pip__frozen__seed42 2>/dev/null | head -1)
  run bench_verify_v2_perclass_ap python verify_v2_perclass_ap.py --run-dir "$RUN_DIR"
fi
echo "smoke_wsl finished $(date '+%F %T')" >> "$SUM"
