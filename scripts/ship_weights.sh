#!/usr/bin/env bash
# =============================================================================
# ship_weights.sh — 把一份 DAPT 权重交给 benchmark 项目,同时登记血缘。
# 替代"手动 cp 权重"——每次交接都在 D:\Conceivable-ML\WEIGHTS_REGISTRY.md 留痕
# (sha256 + 来源 commit/配置 + 时间 + 目标路径),不再"复制了就忘"。
#
# 用法(WSL, /mnt/d/Video/domain_transfer 下):
#   ./scripts/ship_weights.sh <权重文件> <benchmark侧目标路径> "<产出方说明>" "<训练配置说明>"
#
# 示例:
#   ./scripts/ship_weights.sh \
#     gcp_outputs/dinov3_vitb16_dapt_backbone.pth \
#     /mnt/d/Conceivable-SharedData01-23Jun2026/stage1_out/dinov3_ckpt/dinov3_vitb16_dapt.pth \
#     "domain_transfer@$(git rev-parse --short HEAD), 官方 dinov3/train/train.py + dapt_vitb16.yaml + patches/dinov3_dapt.patch" \
#     "ViT-B/16, batch 48, 20000 iters, A100-40GB"
# =============================================================================
set -euo pipefail

SRC="${1:?用法: ship_weights.sh <权重文件> <目标路径> <产出方说明> <训练配置说明>}"
DEST="${2:?缺少目标路径}"
PRODUCED_BY="${3:-未填写}"
TRAIN_CONFIG="${4:-未填写}"

REGISTRY="/mnt/d/Conceivable-ML/WEIGHTS_REGISTRY.md"
CONSUMED_BY="${SHIP_CONSUMED_BY:-待定(尚未跑 benchmark)}"

[ -f "$SRC" ] || { echo "!! 源文件不存在: $SRC"; exit 1; }
[ -f "$REGISTRY" ] || { echo "!! 找不到登记表: $REGISTRY(先按重构方案建好 D:\\Conceivable-ML)"; exit 1; }

echo "[1/3] 计算 sha256 ..."
SHA=$(sha256sum "$SRC" | awk '{print $1}')
echo "  $SHA"

echo "[2/3] 复制到 benchmark 侧: $DEST"
mkdir -p "$(dirname "$DEST")"
cp -v "$SRC" "$DEST"

echo "[3/3] 登记到 $REGISTRY"
NOW="$(date '+%Y-%m-%d %H:%M')"
FNAME="$(basename "$SRC")"
DEST_REL="$(echo "$DEST" | sed 's#.*/stage1_out/#stage1_out/#')"
printf '| `%s` | `%s` | %s | %s | %s | `%s` | %s | 未核对 |\n' \
  "$FNAME" "$SHA" "$PRODUCED_BY" "$TRAIN_CONFIG" "$NOW" "$DEST_REL" "$CONSUMED_BY" >> "$REGISTRY"

echo ""
echo "已交接并登记一行到 WEIGHTS_REGISTRY.md:"
echo "  文件: $FNAME"
echo "  sha256: $SHA"
echo "  目标: $DEST"
echo "记得手动把'被哪次 benchmark 使用'那一列(当前写的是: $CONSUMED_BY)在跑完 benchmark 后回填。"
