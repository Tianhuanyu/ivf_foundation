#!/usr/bin/env bash
# 在便宜的纯 CPU 实例上:拉视频 -> 抽高分帧 -> 打包上传到桶缓存。
# 之后 A100 训练只需下载解包该缓存,不再自己抽帧(避免 GPU 空转)。
set -euo pipefail
BUCKET="${BUCKET:?need BUCKET env}"
DT_ROOT="$HOME/domain_transfer"
DATA_ROOT="$HOME/data"

echo "===== [prep 1/4] 拉取代码 + 视频 ====="
mkdir -p "$DT_ROOT" "$DATA_ROOT/Video"
gcloud storage rsync -r "$BUCKET/dapt/domain_transfer" "$DT_ROOT"
gcloud storage rsync -r "$BUCKET/dapt/data/Video" "$DATA_ROOT/Video"

echo "===== [prep 2/4] 安装 ffmpeg ====="
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg python3

echo "===== [prep 3/4] 改写清单 + 抽帧(多核)====="
cd "$DT_ROOT"
sed "s#/mnt/d#$DATA_ROOT#g" manifests/train_videos.txt > manifests/train_videos.vm.txt
python3 scripts/11_extract_frames.py --manifest manifests/train_videos.vm.txt --split train \
  --short 1536 --fps 5 --max-frames 0 --workers "$(( $(nproc) - 1 ))" --out frames_hires
NFR=$(find frames_hires/train -name '*.jpg' | wc -l)
echo "  抽得 $NFR 帧"

echo "===== [prep 4/4] 释放视频磁盘 -> 打包 -> 上传缓存 ====="
rm -rf "$DATA_ROOT/Video"                                   # 视频已在桶里,删掉腾磁盘给 tar
tar -C "$DT_ROOT" -cf "$HOME/frames_hires.tar" frames_hires  # 写到根盘(非 /tmp,避免 tmpfs)
gcloud storage cp "$HOME/frames_hires.tar" "$BUCKET/dapt/cache/frames_hires.tar"
rm -f "$HOME/frames_hires.tar"
echo "PREP DONE: $NFR frames" | gcloud storage cp - "$BUCKET/dapt/cache/PREP_DONE.txt"
echo "===== PREP FINISHED — 帧已缓存到 $BUCKET/dapt/cache/frames_hires.tar ====="
