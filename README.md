# IVF 显微视频域迁移 — 训练操作指南

用 `D:\Video` 下 ~2925 段 IVF 显微操作视频，对 **DINOv3**(图像) 和 **V-JEPA2**(视频) 两个自监督
backbone 做**域自适应继续预训练 (DAPT)**，产出适配本域的通用特征提取器。评估用文件名里的
12 类操作阶段做 kNN / 线性探针，对比迁移前后特征质量。

> 环境/数据准备已完成（见文末「附录」）。日常只需看 **§0 → §1/§2 训练 → §3 评估**。

---

## §0. 每次开始前：激活环境

在 WSL (Ubuntu-24.04) 里：

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dapt
cd /mnt/d/Video/domain_transfer
```

（提示符出现 `(dapt)` 即可。下面所有命令都假设已在此目录。）

可选：另开一个终端实时看显存/利用率
```bash
watch -n 1 nvidia-smi
```

---

## §1. 训练 DINOv3（图像线）

脚本 `scripts/40_dinov3_dapt.py`。DINO 自蒸馏（student + EMA teacher + multi-crop）。

**① 先冒烟，确认管线跑通（几十秒）**
```bash
python scripts/40_dinov3_dapt.py --batch 16 --local-crops 6 --max-steps 8
```
看到 8 行 `step … loss=… vram_used=…` 且无报错即可。

**② 正式训练（本地，ViT-S）——这是主命令**
```bash
python scripts/40_dinov3_dapt.py \
    --weights weights/dinov3_vits16 \
    --batch 32 --local-crops 6 --out-dim 16384 \
    --lr 5e-4 --ema 0.996 \
    --max-steps 5000 \
    --save weights/dinov3_vits16_dapt
```
- 训练结束会把适配后的 backbone 存到 `weights/dinov3_vits16_dapt`（HF 格式，可直接 `AutoModel.from_pretrained` 加载）。
- ViT-S 很省显存（batch 16 仅 ~2.7GB），**本地可放心把 `--batch` 加到 64**。

**③ 换 ViT-B（本地显存仍够）**
```bash
python scripts/40_dinov3_dapt.py --weights weights/dinov3_vitb16 --batch 24 \
    --max-steps 5000 --save weights/dinov3_vitb16_dapt
```

**关键参数**
| 参数 | 含义 | 建议 |
|---|---|---|
| `--weights` | 初始/输入权重目录 | `weights/dinov3_vits16` 或 `_vitb16` |
| `--batch` | 每步图像数（每图再扩成 2+local_crops 个 crop） | 本地 32–64 |
| `--local-crops` | 局部小图数量 | 6（显存紧就调 4） |
| `--out-dim` | DINO 原型数 | 本地 16384；远程可 65536 |
| `--lr` | 学习率（DAPT 用小值） | 5e-4，效果不稳就降到 1e-4 |
| `--ema` | teacher 动量 | 0.996 |
| `--max-steps` | 训练步数 | 先 5000 看趋势，再决定加大 |
| `--save` | 输出 backbone 目录 | 起个带 `_dapt` 的名字 |

---

## §2. 训练 V-JEPA2（视频线）

脚本 `scripts/30_vjepa2_dapt.py`。掩码隐空间预测，从官方完整 ckpt 续训（含 predictor/target_encoder）。

**① 先冒烟（约 1 分钟）**
```bash
python scripts/30_vjepa2_dapt.py --frames 16 --batch 1 --accum 4 --max-steps 8
```

**② 正式训练（本地，ViT-L）——这是主命令**
```bash
python scripts/30_vjepa2_dapt.py \
    --frames 16 --batch 1 --accum 8 \
    --lr 1e-4 --ema 0.999 \
    --max-steps 3000 \
    --save weights/vjepa2_vitl_dapt.pt
```
- **本地安全档 `--frames 16 --batch 1`，显存峰值 ~10.5GB**（12GB 卡的上限附近）。若 OOM：先降 `--frames 8`。
- `--accum` 是梯度累积：有效 batch = `batch × accum`（上例 = 8），不额外吃显存。
- 存的是适配后的 `target_encoder`（`weights/vjepa2_vitl_dapt.pt`）。

**关键参数**
| 参数 | 含义 | 建议 |
|---|---|---|
| `--frames` | 每 clip 帧数 | 本地 16（OOM 降 8）；远程可 32/64 |
| `--batch` | clip 数/步 | 本地 1 |
| `--accum` | 梯度累积步数 | 8（有效 batch=8） |
| `--stride` | 抽帧间隔 | 4 |
| `--lr` / `--ema` | 学习率 / target 动量 | 1e-4 / 0.999 |
| `--max-steps` | 训练步数 | 先 3000 |
| `--save` | 输出 ckpt 路径 | `weights/vjepa2_vitl_dapt.pt` |

---

## §3. 评估（迁移前 vs 迁移后）

脚本 `scripts/50_eval_features.py`（目前支持 DINOv3 图像特征）。冻结 backbone 抽特征 →
kNN + 线性探针，**按视频分组分层划分**（无帧泄漏、12 类齐全）。

```bash
# 迁移前 baseline（原始权重）
python scripts/50_eval_features.py --weights weights/dinov3_vits16
# 迁移后（你 §1 训出来的）
python scripts/50_eval_features.py --weights weights/dinov3_vits16_dapt
```
比较两次输出的 `linear_acc` / `knn_acc`。**已知 baseline（原始 ViT-S）：kNN 70.8% / 线性探针 75.4% @ 12 类**——DAPT 后应上升。

---

## §4. 长时间训练的小技巧

后台跑并把日志写文件（关掉终端也不断）：
```bash
nohup python scripts/40_dinov3_dapt.py --batch 32 --max-steps 20000 \
      --save weights/dinov3_vits16_dapt > logs/dino_run1.log 2>&1 &
tail -f logs/dino_run1.log      # 实时看日志
```
（先 `mkdir -p logs`。）

- **想继续练 DINOv3**：把上一次 `--save` 出来的目录当作下一次的 `--weights` 即可。
- 本地按你的习惯先小步数验证第一个 epoch 跑通，再放大 `--max-steps` / 搬远程。

---

## §5. 搬到远程放大（脚本全参数化，改数值即可）

| | 本地(12GB) | 远程(大显存) |
|---|---|---|
| DINOv3 | ViT-S/B, batch 32, out-dim 16384 | ViT-L/H, batch 128+, out-dim 65536 |
| V-JEPA2 | ViT-L, frames 16, batch 1 | frames 32/64, batch 4+ |

远程唯一要重装的是环境（同 §附录，注意 GPU 架构对应的 CUDA 版本）；数据/脚本原样拷过去即可。

---

## 附录：一次性环境 & 数据准备（已完成，供复现/远程重建）

```bash
bash scripts/01_bootstrap_env.sh      # Miniconda + conda env 'dapt' + torch cu128
bash scripts/02_install_deps.sh       # timm/transformers/decord... + clone repos + V-JEPA2 权重
python3 scripts/10_build_manifest.py --val-dates 2          # 清单+标签+切分
python  scripts/11_extract_frames.py --manifest manifests/train_videos.txt --split train
python  scripts/11_extract_frames.py --manifest manifests/val_videos.txt   --split val
```
DINOv3 权重是 gated，需自行 `hf auth login` 后 `hf download facebook/dinov3-vits16-pretrain-lvd1689m --local-dir weights/dinov3_vits16`（vitb16 同理）。

**关键环境事实**：WSL Ubuntu-24.04；用户空间 Miniconda（sudo 需密码，故不用 apt）；
RTX 5070 Ti / Blackwell **sm_120 必须用 torch cu128**；GPU 12GB。

## 开发 / Git

**VS Code**：用 **Remote - WSL** 打开 `/mnt/d/Video/domain_transfer`（不要开 `D:\Video`，里面有 116GB 原始视频）。解释器选 WSL 的 `dapt` conda 环境。`.vscode/settings.json` 已把 `frames/weights/repos/logs/manifests` 排除出索引。

**依赖**：只有 **vjepa2** 是源码依赖（被 `30_*.py` import），作 git submodule 钉版本；DINOv3 走 pip 的 `transformers`（`repos/dinov3` 未使用，已 gitignore，可删）。

**只追踪代码**，数据/权重/帧/日志/manifests 全部 gitignore（体积大或含机器本地绝对路径，靠脚本重建或 GCS 同步）。

一次性初始化 git（在项目根，WSL 里执行）：
```bash
cd /mnt/d/Video/domain_transfer
rm -rf repos/dinov3                      # 未使用
rm -rf repos/vjepa2                      # 换成 submodule
git init
git submodule add https://github.com/facebookresearch/vjepa2 repos/vjepa2
git add .gitignore requirements.txt README.md .vscode scripts .gitmodules repos/vjepa2
git commit -m "Domain-transfer pipeline: DINOv3 + V-JEPA2 DAPT scripts"
```
别人复现：`git clone --recursive <你的仓库>`，再按 §附录装环境/备数据。

## 目录结构
```
domain_transfer/
  scripts/  00_detect 01_bootstrap 02_install 10_manifest 11_frames
            20_clip_dataset 30_vjepa2_dapt 40_dinov3_dapt 50_eval_features
  manifests/  all_videos.csv train_videos.txt val_videos.txt label_stats.txt
  frames/<split>/<stage>/*.jpg      抽出的帧(兼作 ImageFolder)
  repos/{dinov3,vjepa2}/            官方代码
  weights/  vjepa2_vitl.pt  dinov3_vits16/  dinov3_vitb16/  (+ *_dapt 输出)
```

## 状态
- [x] 环境 / 清单 / 抽帧 / clip 数据集
- [x] V-JEPA2 续训 (30) — 12GB 跑通, 峰值 10.5GB
- [x] DINOv3 续训 (40) — 12GB 跑通, ViT-S 2.7GB
- [x] 特征评估 (50) — baseline kNN 70.8% / 线性探针 75.4%
- [ ] 跑完整 DAPT 并对比迁移前后
