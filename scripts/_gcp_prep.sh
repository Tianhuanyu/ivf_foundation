#!/usr/bin/env bash
# VM 端(纯 CPU 实例,由 dapt_prep.sh 启动):拉视频 -> 抽高分帧 -> motion sidecar -> 帧权重 -> 打包上传。
# 之后 A100 训练只需下载解包该缓存,不再自己处理视频(避免 GPU 空转)。
set -euo pipefail
BUCKET="${BUCKET:?need BUCKET env}"
DT_ROOT="$HOME/domain_transfer"
DATA_ROOT="$HOME/data"
NPROC="$(( $(nproc) - 1 ))"

echo "===== [prep 1/6] 拉取代码/清单 + 视频 ====="
mkdir -p "$DT_ROOT" "$DATA_ROOT/Video"
for d in scripts manifests; do gcloud storage rsync -r "$BUCKET/dapt/domain_transfer/$d" "$DT_ROOT/$d"; done
gcloud storage cp "$BUCKET/dapt/domain_transfer/requirements.txt" "$DT_ROOT/requirements.txt"
gcloud storage rsync -r "$BUCKET/dapt/data/Video" "$DATA_ROOT/Video"

echo "===== [prep 2/6] 安装 ffmpeg + python 依赖 ====="
sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg python3 python3-pip
python3 -m pip install --quiet opencv-python-headless numpy scipy

echo "===== [prep 3/6] 改写清单 + 抽帧(多核)====="
cd "$DT_ROOT"
sed "s#/mnt/d#$DATA_ROOT#g" manifests/train_videos.txt > manifests/train_videos.vm.txt
python3 scripts/11_extract_frames.py --manifest manifests/train_videos.vm.txt --split train \
  --short 1536 --fps 5 --max-frames 0 --workers "$NPROC" --out frames_hires
NFR=$(find frames_hires/train -name '*.jpg' | wc -l)
echo "  抽得 $NFR 帧"

echo "===== [prep 4/6] motion-energy sidecar(需要原始视频,必须在删视频之前)====="
python3 -u scripts/12_motion_energy.py --manifest manifests/train_videos.vm.txt --split train \
  --frames-root frames_hires --workers "$NPROC"
NME=$(find frames_hires/train -name '*.me.png' | wc -l)
echo "  sidecar $NME / $NFR 帧"

echo "===== [prep 5/6] 时间维度帧权重 ====="
python3 -u scripts/13_frame_weights.py --frames-root frames_hires --split train

echo "===== [prep 6/6] 释放视频磁盘 -> 打包 -> 上传缓存 ====="
rm -rf "$DATA_ROOT/Video"                                   # 视频已在桶里,删掉腾磁盘给 tar
tar -C "$DT_ROOT" -cf "$HOME/frames_hires.tar" frames_hires  # 写到根盘(非 /tmp,避免 tmpfs)
gcloud storage cp "$HOME/frames_hires.tar" "$BUCKET/dapt/cache/frames_hires.tar"
rm -f "$HOME/frames_hires.tar"
echo "PREP DONE: $NFR frames, $NME sidecars" | gcloud storage cp - "$BUCKET/dapt/cache/PREP_DONE.txt"
echo "===== PREP FINISHED — 帧缓存已上传到 $BUCKET/dapt/cache/frames_hires.tar ====="
