#!/usr/bin/env bash
# =============================================================================
# smoke.sh — ONE smoke test for every script of the paper pipeline (WSL, conda env `dapt`).
# Run it before real training / before submitting anything to GCP.
#
#   nohup setsid bash experiments/smoke.sh > smoke_out/smoke.log 2>&1 &     # always detached (see below)
#   cat smoke_out/SUMMARY.txt                                              # PASS/FAIL per step, live
#   STEPS="dapt bench" nohup setsid bash experiments/smoke.sh > smoke_out/smoke.log 2>&1 &   # some groups only
#
# Groups (default: all, in this order; ~3.5 h, almost all of it `bench`):
#   data    10 manifest, 11 frames (1 video), 12 motion sidecars, 13 frame weights  -> smoke_out/
#   dapt    dapt_run.sh SMOKE=1 for all 6 arms (e1) + motion_weighted (long); ARM guards; GCP script
#           syntax; check_patch.sh (patch reproduces repos/dinov3 on a fresh upstream clone)
#   ship    ship_weights.sh into a COPY of the weights registry
#   splits  make_splits.py --check for both split profiles (read-only)
#   bench   run_benchmark_server.py / run_benchmark_hires_ablation.py --smoke for E0, E2, E3, E4;
#           verify_v2_perclass_ap.py                                    -> stage1_out/benchmark_results*_smoke/
#   reports summarize_bench / where_ours / plot_bench / make_ppt_figs / make_report on the smoke results
#   paper   experiments/paper.py status / commands / tokens / tables on the smoke results
#
# Why detached: this tool environment kills long-running `wsl.exe` commands (SESSION_NOTES §2.3), which can
# also wedge WSL. Started with nohup+setsid the run survives; check progress with short commands only.
# Nothing touches real artifacts (weights/, real results, the real registry, locked split lists).
# Last full result: experiments/SMOKE_RESULTS.md
# =============================================================================
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate dapt
DT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${BENCH_ROOT:-/mnt/d/Conceivable-SharedData01-23Jun2026}"
REGISTRY="${WEIGHTS_REGISTRY:-/mnt/d/Conceivable-ML/WEIGHTS_REGISTRY.md}"
S="$DT/smoke_out"; mkdir -p "$S/logs"
SUM="$S/SUMMARY.txt"
STEPS="${STEPS:-data dapt ship splits bench reports paper}"
SMOKE_RES="$BENCH/stage1_out/benchmark_results_smoke"
SMOKE_ABL="$BENCH/stage1_out/benchmark_results_imgsz320_smoke"
SMOKE_BB="$S/dinov3_vitb16_dapt_motion_weighted_e1_backbone.pth"
E0_DS="cellasp holding_pip routine2_coc cvit_incubator cvit_workstation"
DET4="holding_pip routine2_coc cvit_incubator cvit_workstation"
OURS=(--ours dinov3_dapt_b_fpn@d7282330)   # smoke results hold 2 DAPT checkpoints (default d7282330 + E3 override); sha form works without the registry
echo "smoke started $(date '+%F %T')  steps: $STEPS" >> "$SUM"

run(){   # run <name> <command...>   -> PASS/FAIL + duration into SUMMARY.txt, output into logs/<name>.log
  local name="$1"; shift
  local t0=$SECONDS r
  if ( "$@" ) > "$S/logs/$name.log" 2>&1; then r=PASS; else r="FAIL(rc=$?)"; fi
  printf '%-44s %-10s %5ss  logs/%s.log\n' "$name" "$r" "$((SECONDS - t0))" "$name" | tee -a "$SUM"
}
has(){ [[ " $STEPS " == *" $1 "* ]]; }

if has data; then
  cd "$DT"
  run data_10_build_manifest   python3 scripts/10_build_manifest.py --out "$S/manifests"
  run data_11_extract_frames   python scripts/11_extract_frames.py --manifest manifests/train_videos.txt \
                                 --split train --limit 1 --short 1536 --fps 5 --out "$S/frames"
  run data_12_motion_energy    bash -c "python -u scripts/12_motion_energy.py --manifest manifests/train_videos.txt --split train \
                                 --limit 1 --workers 1 --frames-root '$S/frames' | tee /dev/stderr | grep -q 'err=0'"
  run data_13_frame_weights    bash -c "python -u scripts/13_frame_weights.py --frames-root '$S/frames' --split train \
                                 && test -f '$S/frames/train/frame_weights.npy'"
fi

if has dapt; then
  cd "$DT"
  for arm in uniform content motion motion_weighted motion_ibotlocal_high motion_ibotlocal_low; do
    run "dapt_run_${arm}_e1"   env SMOKE=1 ARM=$arm BUDGET=e1 bash scripts/dapt_run.sh
  done
  run dapt_run_motion_weighted_long  env SMOKE=1 ARM=motion_weighted BUDGET=long bash scripts/dapt_run.sh
  run dapt_weighted_sampling_active  grep -q "temporal importance-weighted sampling ENABLED" "$S/logs/dapt_run_motion_weighted_e1.log"
  run dapt_arms_rejects_bad_arm      bash -c '! (source scripts/dapt_arms.sh && dapt_overrides nope e1 x y)'
  run dapt_train_requires_arm        bash -c '! env -u ARM bash scripts/dapt_train.sh start'
  run gcp_scripts_syntax             bash -c 'for f in scripts/*.sh; do bash -n "$f" || exit 1; done'
  run patch_reproduces_worktree      bash experiments/check_patch.sh
fi

if has ship; then
  cp "$REGISTRY" "$S/registry_copy.md"
  run ship_weights                   env WEIGHTS_REGISTRY="$S/registry_copy.md" bash "$DT/scripts/ship_weights.sh" \
                                       "$SMOKE_BB" "$S/shipped/dapt_smoke.pth" "smoke" "smoke"
  run ship_weights_row_in_table      bash -c "tail -n 3 '$S/registry_copy.md' | grep -q '$(basename "$SMOKE_BB")'"
fi

if has splits; then
  cd "$BENCH"
  for prof in recording acquisition; do
    run "make_splits_check_$prof"    bash -c "python stage1_out/benchmark/dataset/make_splits.py --check --profile $prof | grep -q near_dup_new"
  done
fi

if has bench; then
  cd "$BENCH"
  for bb in resnet50_fpn dinov3_b_fpn dinov3_dapt_b_fpn; do
    run "bench_E0_$bb"          python run_benchmark_server.py --all-seeds --smoke --split-profile acquisition --dataset $E0_DS --backbone $bb
  done
  for bb in resnet50_fpn dinov2_s_fpn dinov2_b_fpn biomedclip_fpn dinov3_b_fpn dinov3_dapt_b_fpn; do
    run "bench_E2_frozen_$bb"   python run_benchmark_server.py --all-seeds --smoke --protocol frozen --backbone $bb
    run "bench_E2_finetune_$bb" python run_benchmark_server.py --all-seeds --smoke --protocol finetune --dataset cellasp icsi_seg holding_pip --backbone $bb
  done
  run bench_E3_weights_override env DINOV3_DAPT_B_WEIGHTS="$SMOKE_BB" python run_benchmark_server.py --all-seeds --smoke --protocol frozen --dataset holding_pip --backbone dinov3_dapt_b_fpn
  for bb in resnet50_fpn dinov2_b_fpn dinov3_b_fpn dinov3_dapt_b_fpn; do
    run "bench_E4_$bb"          python run_benchmark_hires_ablation.py --all-seeds --smoke --imgsz 320 --protocol frozen --dataset $DET4 --backbone $bb
  done
  run bench_verify_v2_perclass_ap python verify_v2_perclass_ap.py --run-dir "$SMOKE_RES/resnet50_fpn__detect__holding_pip__frozen__seed42"
fi

if has reports; then
  cd "$BENCH"
  run report_summarize_recording     bash -c "python summarize_bench.py '$SMOKE_RES' | grep -q FROZEN"
  run report_summarize_acquisition   bash -c "python summarize_bench.py '$SMOKE_RES' --split-profile acquisition | grep -q FROZEN"
  run report_summarize_ablation_dir  bash -c "python summarize_bench.py '$SMOKE_ABL' | grep -q FROZEN"
  run report_where_ours_refuses_ambiguous env R="$SMOKE_RES" bash -c 'out=$(python where_ours.py "$R" 2>&1); echo "$out"; echo "$out" | grep -q "several DAPT variants"'
  run report_where_ours              python where_ours.py "$SMOKE_RES" "${OURS[@]}"
  run report_plot_bench              python plot_bench.py "$SMOKE_RES" --plots "$S/plots"
  run report_make_ppt_figs           python make_ppt_figs.py "$SMOKE_RES" --plots "$S/plots" "${OURS[@]}"
  run report_make_report             python make_report.py "$SMOKE_RES" --plots "$S/plots" --out "$S/report_smoke.html" "${OURS[@]}"
fi

if has paper; then
  cd "$DT"
  export PAPER_RESULTS_DIR="$SMOKE_RES" PAPER_RES_ABL_DIR="$SMOKE_ABL" BENCH_ROOT="$BENCH"
  run paper_status                   bash -c "python experiments/paper.py status | grep -q E4"
  run paper_commands                 bash -c "python experiments/paper.py commands | grep -q '### E0'"
  run paper_tokens                   bash -c "python experiments/paper.py tokens | grep -q needle_tip"
  run paper_tables                   bash -c "python experiments/paper.py tables | grep -q 'T3'"
fi
echo "smoke finished $(date '+%F %T')" >> "$SUM"
