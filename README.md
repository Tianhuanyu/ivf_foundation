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

首次搭建环境（02 会装 `requirements.txt`、clone 官方 DINOv3 并打上我们的 patch）：
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

**只有一个 config**：`repos/dinov3/dinov3/configs/train/dapt_vitb16.yaml`。
实验臂（ARM）和训练量（BUDGET）是命令行覆盖项，只在 [`scripts/dapt_arms.sh`](scripts/dapt_arms.sh) 里定义，本地和 GCP 共用这一份。所以各臂之间只差表里列出的那几行：

| ARM | 相对 base 的改动 |
|---|---|
| `uniform` | 无（对照） |
| `content` | `crops.crop_sampler=content`（形态学显著性） |
| `motion` | `crops.crop_sampler=motion` |
| `motion_weighted` | motion + 数据路径加 `:extra=weighted`（**主方法**） |
| `motion_ibotlocal_{high,low}` | motion + `ibot_local.enabled=true`、`direction=mask_{high,low}_saliency`（附录） |

| BUDGET | 训练量 |
|---|---|
| `e1` | 6 × 500 = 3000 iter（E1 消融用） |
| `long` | 8 × 2500 = 20000 iter（正式长程） |

> 2026-09-26 已验证：base + 覆盖项与原来 7 个独立 yaml 合并后的配置**逐项完全一致**。那 6 个 `dapt_vitb16_official_compare_*.yaml` 目前还留在目录里，已经没有用，可以手动删掉。

本地训练（跑完自动抽出 backbone 到 `weights/dinov3_vitb16_dapt_<ARM>_<BUDGET>_backbone.pth`）：
```bash
ARM=motion_weighted BUDGET=e1 ./scripts/dapt_run.sh             # 可选 DINO_BATCH / RUN_TAG / EXTRA_OPTS
```

单独抽 backbone：
```bash
python repos/dinov3/_extract_dapt_backbone_param.py --ckpt-root <output_dir>/ckpt --out weights/<name>_backbone.pth
```

## 3. GCP 训练（A100-40GB）

在 WSL 的项目根目录下运行。GCP 项目是 `hidden-outrider-390502`，实例名 `dapt-a100`。账号相关的配置都在 `scripts/_gcp_common.sh`：
```bash
./scripts/dapt_prep.sh run                                        # 一次性：制作帧缓存（抽帧 + sidecar + 帧权重）
ARM=motion_weighted BUDGET=long ./scripts/dapt_train.sh start     # upload + up + pushcode + train
./scripts/dapt_train.sh status
./scripts/dapt_train.sh finish                                    # fetch + 删实例
```

- **`ARM` 没有默认值**。不传会直接拒绝执行，这是故意的：以前有默认值时误跑过错误的实验，白烧过账单。`BUDGET` 默认是 `long`。
- **超参**：`DINO_BATCH`（默认 48）、`DINO_EPOCH_LEN`、`DINO_EPOCHS`、`REPO_WEIGHTS` 可再覆盖。
- **产物**：`gcp_outputs/dinov3_vitb16_dapt_<ARM>_<BUDGET>_backbone.pth`，每个 run 有自己的完成标记 `DINOV3_DONE_<ARM>_<BUDGET>.txt`。
- **帧缓存**：由 `dapt_prep.sh` 在 CPU 机上制作，包含 `.me.png` 和 `frame_weights.npy`。motion 类的 ARM 在 sidecar 覆盖率低于 `MIN_SIDECAR_COVERAGE`（默认 0.65，本地实测约 0.71）时会直接报错退出；`motion_weighted` 缺少 `frame_weights.npy` 时也会直接报错。
  - ⚠️ 桶里现有的缓存是 2026-09-26 之前做的，sidecar 不全。跑 motion 类 ARM 之前，先删掉 `gs://mlflow-artifacts-ai-a100/dapt/cache/frames_hires.tar`，再用 `dapt_prep.sh run` 重做。
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
scripts/       01-02 环境 · 10-13 数据(+_common.py) · dapt_arms.sh 实验臂定义 · dapt_run.sh 本地训练
               dapt_prep.sh+_gcp_prep.sh 帧缓存 · dapt_train.sh+_gcp_dinov3_dapt.sh GCP(+_gcp_common.sh) · ship_weights.sh
patches/       dinov3_dapt.patch（repos/dinov3 改动的完整备份）
repos/dinov3/  官方 trainer + 改动（自带 .git，本仓库 gitignore）
legacy/        已归档代码与旧日志（gitignored）
frames_hires/ weights/ gcp_outputs/ manifests/ *_out/   数据和产物（gitignored）
```
