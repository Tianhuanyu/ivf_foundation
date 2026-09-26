#!/usr/bin/env bash
# =============================================================================
# dapt_arms.sh — the ONLY place that defines DAPT experiment arms and training budgets.
# Sourced by scripts/dapt_run.sh (local) and scripts/_gcp_dinov3_dapt.sh (GCP).
#
# Every arm = repos/dinov3/dinov3/configs/train/dapt_vitb16.yaml + the overrides below, so
# arms differ ONLY in the lines listed here (the E1 controlled-ablation requirement).
#
#   dapt_overrides <arm> <budget> <frames_train_dir> <repo_weights>
#       -> prints one dotlist override per line (feed to train.py after --output-dir)
#   dapt_arm_needs_motion <arm>   -> exit 0 if the arm reads .me.png sidecars
#   dapt_arm_needs_weights <arm>  -> exit 0 if the arm reads frame_weights.npy
# =============================================================================

DAPT_ARMS="uniform content motion motion_weighted motion_ibotlocal_high motion_ibotlocal_low"
DAPT_BUDGETS="e1 long"

_dapt_arm_lines(){
  case "$1" in
    uniform)               ;;
    content)               echo "crops.crop_sampler=content" ;;
    motion|motion_weighted) echo "crops.crop_sampler=motion" ;;
    motion_ibotlocal_high) printf '%s\n' "crops.crop_sampler=motion" "ibot_local.enabled=true" \
                                          "ibot_local.direction=mask_high_saliency" ;;
    motion_ibotlocal_low)  printf '%s\n' "crops.crop_sampler=motion" "ibot_local.enabled=true" \
                                          "ibot_local.direction=mask_low_saliency" ;;
    *) echo "!! unknown ARM '$1' (valid: $DAPT_ARMS)" >&2; return 1 ;;
  esac
}

# Budget = schedule only. Batch size is separate (DINO_BATCH), since it's a hardware choice.
_dapt_budget_lines(){
  case "$1" in
    e1)   ;;                                                  # 6 x 500 = 3000 iters (yaml default)
    long) printf '%s\n' "train.OFFICIAL_EPOCH_LENGTH=2500" "optim.epochs=8" \
                        "checkpointing.period=2500" "train.num_workers=8" ;;   # 8 x 2500 = 20000 iters
    *) echo "!! unknown BUDGET '$1' (valid: $DAPT_BUDGETS)" >&2; return 1 ;;
  esac
}

dapt_arm_needs_motion(){ case "$1" in motion*) return 0 ;; *) return 1 ;; esac; }
dapt_arm_needs_weights(){ [ "$1" = "motion_weighted" ]; }

dapt_overrides(){
  local arm="$1" budget="$2" frames="$3" weights="$4" extra=""
  _dapt_arm_lines "$arm" >/dev/null || return 1
  _dapt_budget_lines "$budget" >/dev/null || return 1
  dapt_arm_needs_weights "$arm" && extra=":extra=weighted"
  echo "train.dataset_path=Frames:root=${frames}${extra}"
  echo "student.pretrained_weights=${weights}"
  _dapt_arm_lines "$arm"
  _dapt_budget_lines "$budget"
}
