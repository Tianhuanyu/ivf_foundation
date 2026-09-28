#!/usr/bin/env bash
# =============================================================================
# upload_frame_cache.sh — 把本地 frames_hires/train(帧 + .me.png 运动图 + frame_weights.npy)按阶段打包上传到桶,
# 作为 GCP 训练用的帧缓存。WSL 里运行:
#   ./scripts/upload_frame_cache.sh            # 全部阶段(约 2.5 小时:打包约 10 分钟,其余是上传)
#   ONLY="Egg_toInseminationDisp" ./scripts/upload_frame_cache.sh     # 只处理指定阶段(测试用)
#
# 为什么这样做:148 万个小文件,在 WSL 里逐个读 /mnt/d 每秒只有约 110 个(光清点就要 3-4 小时),逐个上传更慢。
# 这里用 Windows 自带的 tar.exe 在 NTFS 上原生打包(实测每秒约 2900 个文件),每个阶段一个 tar,上传只受带宽限制。
#
# 断点续传:每个阶段上传成功后在桶里写 <阶段>.done(内容:阶段名 帧数 运动图数);重跑会跳过已有 .done 的阶段。
# 本地同一时间只有一个 tar(传完即删),最大的阶段 ICSI_injection 约 21GB。
# 全部阶段完成后写 DONE.txt(总帧数/总运动图数);训练 VM(_gcp_dinov3_dapt.sh)解包后会逐项核对。
# =============================================================================
set -euo pipefail
LOG_TAG="frame_cache"
source "$(dirname "$0")/_gcp_common.sh"

DEST="$BUCKET/dapt/cache/frames_hires_parts"
SRC="$DT_LOCAL/frames_hires/train"
PARTS="$DT_LOCAL/cache_parts"                        # 本地临时 tar(gitignored)
WIN_SRC='D:\Video\domain_transfer\frames_hires\train'
WIN_PARTS='D:\Video\domain_transfer\cache_parts'
WIN_TAR=/mnt/c/Windows/System32/tar.exe
mkdir -p "$PARTS"
preflight
[ -x "$WIN_TAR" ] || { echo "!! 找不到 Windows 的 tar.exe($WIN_TAR)"; exit 1; }
[ -f "$SRC/frame_weights.npy" ] || { echo "!! $SRC/frame_weights.npy 不存在,先跑 scripts/13_frame_weights.py"; exit 1; }

mapfile -t STAGES < <(find "$SRC" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
[ -n "${ONLY:-}" ] && read -r -a STAGES <<< "$ONLY"
log "阶段: ${STAGES[*]}"

for st in "${STAGES[@]}"; do
  if gcloud storage ls "$DEST/$st.done" >/dev/null 2>&1; then
    log "$st: 已完成,跳过($(gcloud storage cat "$DEST/$st.done"))"; continue
  fi
  log "$st: 打包 ..."
  t0=$SECONDS
  "$WIN_TAR" -cvf "$WIN_PARTS\\$st.tar" -C "$WIN_SRC" "$st" 2> "$PARTS/$st.list"
  tr -d '\r' < "$PARTS/$st.list" > "$PARTS/$st.list.lf"        # Windows tar 的清单是 CRLF 换行
  jpg=$(grep -c '\.jpg$' "$PARTS/$st.list.lf" || true)
  me=$(grep -c '\.me\.png$' "$PARTS/$st.list.lf" || true)
  [ "$jpg" -gt 0 ] || { echo "!! $st: 清单里没有数到 .jpg,停止(检查 $PARTS/$st.list)"; exit 1; }
  log "$st: 打包完成 $((SECONDS - t0))s,帧 $jpg,运动图 $me,$(du -h "$PARTS/$st.tar" | cut -f1)。上传 ..."
  gcloud storage cp "$PARTS/$st.tar" "$DEST/$st.tar"
  echo "$st $jpg $me" | gcloud storage cp - "$DEST/$st.done"
  rm -f "$PARTS/$st.tar" "$PARTS/$st.list" "$PARTS/$st.list.lf"
  log "$st: 完成"
done

[ -n "${ONLY:-}" ] && { log "ONLY 模式:不写 DONE.txt"; exit 0; }
gcloud storage cp "$SRC/frame_weights.npy" "$SRC/frame_weights_meta.json" "$DEST/"
n_st=$(find "$SRC" -mindepth 1 -maxdepth 1 -type d | wc -l)
gcloud storage cat "$DEST/*.done" > "$PARTS/manifest.txt"
[ "$(wc -l < "$PARTS/manifest.txt")" -eq "$n_st" ] || { echo "!! 只有 $(wc -l < "$PARTS/manifest.txt")/$n_st 个阶段完成,重跑本脚本"; exit 1; }
awk '{j += $2; m += $3} END {print "TOTAL", j, m}' "$PARTS/manifest.txt" | tee -a "$PARTS/manifest.txt"
gcloud storage cp "$PARTS/manifest.txt" "$DEST/DONE.txt"
log "全部完成:$(tail -1 "$PARTS/manifest.txt")(本地应为 868310 帧 / 617346 运动图)"
