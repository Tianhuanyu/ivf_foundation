# IVF 显微域 DINOv3 DAPT — 操作手册

用 `D:\Video` 下约 2925 段 IVF 显微操作视频的抽帧，对官方 **DINOv3 ViT-B/16** 做**域自适应继续预训练（DAPT）**。
训练时用视频的运动信号引导采样：
- **空间**：local crop 裁在哪（`crop_sampler`）；
- **时间**：哪一帧多看（帧权重）。

产出的 backbone 交给下游 benchmark（`D:\Conceivable-SharedData01-23Jun2026`）评估。

- 项目背景、结论可信度、下一步：[`ONBOARDING.md`](ONBOARDING.md)
- 实验记录与结果表：[`SESSION_NOTES_motion_energy_e1.md`](SESSION_NOTES_motion_energy_e1.md)
- 流水线地图与权重血缘：`D:\Conceivable-ML\README.md`、`WEIGHTS_REGISTRY.md`
- 旧路线（自研 DINO 脚本、V-JEPA2、I-JEPA、fold）的代码和旧版 README 都在 `legacy/` 与 git 历史 `271895c` 里，不再维护。

---

## 0. 环境

WSL Ubuntu-24.04，用户空间 Miniconda，conda 环境 `dapt`。RTX 5070 Ti 12GB 是 Blackwell sm_120，**必须用 torch cu128**。

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dapt
cd /mnt/d/Video/domain_transfer
```

首次搭建环境：
```bash
bash scripts/01_bootstrap_env.sh
bash scripts/02_install_deps.sh
```
DINOv3 权重是 gated 的，需要先 `hf auth login`，再 `hf download facebook/dinov3-vitb16-pretrain-lvd1689m --local-dir weights/dinov3_vitb16`。之后用 `repos/dinov3/_convert_hf_to_repo.py` 转成 repo 格式的 `weights/dinov3_vitb16_repo.pth`。

## 1. 数据准备（一次性，产物都不进 git）

```bash
python3 scripts/10_build_manifest.py --val-dates 2            # 清单、标签、按日期切分
python  scripts/11_extract_frames.py --manifest manifests/train_videos.txt --split train   # → frames_hires/
```

motion-energy sidecar 和帧权重要在**原生 Windows** 上生成。WSL 下有问题，见 SESSION_NOTES §2.3。
```
C:\ProgramData\miniconda3\python.exe -u scripts\12_motion_energy.py --manifest manifests\train_videos.txt --split train --workers 4
C:\ProgramData\miniconda3\python.exe -u scripts\13_frame_weights.py --frames-root frames_hires --split train
```

- `12_motion_energy.py` 产出 `frames_hires/train/<stage>/<frame>.jpg.me.png`。
- `13_frame_weights.py` 产出 `frames_hires/train/frame_weights.npy`。
- 目前 sidecar 覆盖约 71% 的帧。缺 sidecar 的帧会自动回退到 uniform 裁剪。

## 2. 训练（`repos/dinov3` 官方 trainer）

`repos/dinov3` 是官方 clone 加我们的改动。改动的完整备份是 [`patches/dinov3_dapt.patch`](patches/dinov3_dapt.patch)。

**改了 `repos/dinov3` 之后要重新生成 patch**，否则 GCP 上打的是旧版本：
```bash
cd repos/dinov3
(git diff; for f in $(git ls-files --others --exclude-standard | grep -v __pycache__); do git diff --no-index /dev/null "$f"; done) > ../../patches/dinov3_dapt.patch
```

每个实验臂对应一个 config（`repos/dinov3/dinov3/configs/train/`），各臂之间只差一行：

| config | 臂 |
|---|---|
| `dapt_vitb16_official_compare_base.yaml` | uniform（对照） |
| `dapt_vitb16_official_compare_adaptive.yaml` | content（形态学显著性） |
| `dapt_vitb16_official_compare_motion.yaml` | motion（空间） |
| `dapt_vitb16_official_compare_motion_weighted.yaml` | motion + 时间加权（**主方法**） |
| `dapt_vitb16_official_compare_motion_ibotlocal_{high,low}.yaml` | 附录：ibot_local |
| `dapt_vitb16.yaml` | 长程配置（8 × 2500 = 20000 iter） |

本地训练：
```bash
cd repos/dinov3
PYTHONPATH=. torchrun --nproc_per_node=1 dinov3/train/train.py \
    --config-file dinov3/configs/train/<config>.yaml \
    --output-dir <与 yaml 里 output_dir 相同>
```
- ⚠️ **`--output-dir` 必须显式传**。否则 yaml 里的 `output_dir` 会被 CLI 默认值 `./local_dino` 覆盖。

从 checkpoint 抽出 backbone：
```bash
python repos/dinov3/_extract_dapt_backbone_param.py --ckpt <output_dir>/ckpt/<iter> --out weights/<name>_backbone.pth
```

## 3. GCP 训练（A100-40GB）

在 WSL 的项目根目录下运行。GCP 项目是 `hidden-outrider-390502`，实例名 `dapt-a100`：
```bash
REMOTE_SCRIPT=_gcp_dinov3_dapt.sh CONFIG=<config>.yaml ./scripts/dapt_train.sh start    # upload+up+pushcode+train
REMOTE_SCRIPT=_gcp_dinov3_dapt.sh ./scripts/dapt_train.sh status
./scripts/dapt_train.sh finish                                                         # fetch + 删实例
```

- **超参**：用 `DINO_BATCH`、`DINO_EPOCH_LEN`、`DINO_EPOCHS`、`REPO_WEIGHTS` 环境变量覆盖。默认值是 batch 48、8 × 2500 iter。
- **`REMOTE_SCRIPT` 没有默认值**。不传会直接拒绝执行，这是故意的，之前误跑旧脚本白烧过账单。
- **帧缓存**：VM 只从桶里拉帧缓存 `dapt/cache/frames_hires.tar`（用 `dapt_prep.sh run` 制作）。跑 motion 类 config 前，**确认缓存里包含最新的 `.me.png` 和 `frame_weights.npy`**。
  - 2026-09-22 那次 GCP 运行中，motion sidecar 缺失的比例比本地高。
  - 远端脚本启动时会打印 sidecar 覆盖率。config 要求时间加权但缓存里没有 `frame_weights.npy` 时，脚本会直接报错退出。
- **zone 会变**：A100 缺货时会自动换 zone，实际 zone 以 `.dapt_a100_zone` 为准。
- **OS Login 用户会漂移**：可能是 `thy`，也可能是 `htian_conceivable_life`，两者 `$HOME` 独立。换了用户时，远端会重装环境，多花几分钟，不是错误。

## 4. 交接权重 → benchmark

```bash
./scripts/ship_weights.sh <权重文件> <benchmark侧目标路径> "<产出方说明>" "<训练配置说明>"
```
脚本会自动算 sha256、登记到 `D:\Conceivable-ML\WEIGHTS_REGISTRY.md` 并复制过去。**不要手动 `cp`。**

benchmark 的跑法见 benchmark 仓库的 `BENCHMARK_DINOV3.md`。**那个仓库的 git 由人工管理，Claude 不做任何 git 操作。**

## 目录

```
scripts/       01-02 环境 · 10-13 数据 · dapt_prep.sh+_gcp_prep.sh 帧缓存 · dapt_train.sh+_gcp_dinov3_dapt.sh GCP · ship_weights.sh
patches/       dinov3_dapt.patch（repos/dinov3 改动的完整备份）
repos/dinov3/  官方 trainer + 改动（自带 .git，本仓库 gitignore）
repos/vjepa2/  submodule，只被 legacy 里的 V-JEPA2 脚本使用
legacy/        已归档代码与旧日志（gitignored）
frames_hires/ weights/ gcp_outputs/ manifests/ *_out/   数据和产物（gitignored）
```
