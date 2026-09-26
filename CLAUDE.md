# CLAUDE.md — IVF 显微域 Foundation Model（训练侧）

本仓库正在交接。使用者多半是**刚接手、不熟悉代码的人**。你的首要任务是帮他们准确理解代码和结论，其次才是改代码。

## 项目一句话
在官方 DINOv3 ViT-B/16 上，用 IVF 显微视频抽帧做域自适应继续预训练（DAPT），产出本域的 foundation backbone。权重交给下游 benchmark（`D:\Conceivable-SharedData01-23Jun2026`）评估。

## 先读这些（按顺序）
1. `ONBOARDING.md`：交接总览、结论可信度、下一步。
2. `SESSION_NOTES_motion_energy_e1.md`：最新实验记录、结果表、复现命令（§7）。
3. `README.md`：操作手册。§1/§1.1 是**已归档**路径，只作历史参考。
4. `D:\Conceivable-ML\README.md` 和 `WEIGHTS_REGISTRY.md`：跨仓库流水线和权重血缘。

## 代码真相在哪
- **活跃训练代码在 `repos/dinov3/`**：独立的官方 clone，改动未提交。本仓库 gitignore 了它，`patches/dinov3_dapt.patch` 是改动的完整备份。重点文件：
  - `dinov3/data/crop_sampler.py`
  - `dinov3/data/datasets/frames.py`
  - `dinov3/data/masking.py`
  - `dinov3/train/ssl_meta_arch.py`
  - `dinov3/configs/train/dapt_vitb16.yaml`（唯一的 base config；实验臂和训练量在 `scripts/dapt_arms.sh`）
- `scripts/`：数据准备（10–13）、GCP 编排（`dapt_train.sh`、`_gcp_*.sh`）、权重交接（`ship_weights.sh`）。
- `legacy/`：已废弃的代码（自研 DINO、V-JEPA2、I-JEPA、fold、HF 格式评估脚本、旧日志）。不要在上面开发，也不要推荐。
- 论文主线：DINOv3 DAPT + motion 引导采样（空间 crop 加时间帧权重）。消融臂是 uniform / content / motion / motion_weighted。CoarseFineFPN 和 ibot_local 不属于主线。

## 回答问题时的规则
- 引用具体文件和行号（`path:line`），并区分"文档/代码里写明的"和"你推断的"。
- **实验结论要附带可信度**：
  - E1 全部是单 seed、只在 holding_pip 上。
  - 2 epoch 和 8 epoch 的 benchmark 结果**不可比**。
  - SESSION_NOTES §5.3 的"CoarseFineFPN 在 needle_tip 上翻盘"是跨协议比较；同协议的 plain FPN + motion_weighted 结果（§8.4，0.1047）反而更高。不要把它当成已确认的结论复述。
  - ibot_local 的 A/B 也被混淆了：处理组没有时间加权、跑在 GCP 上，对照组跑在本地。
  - benchmark 在 2026-09-26 换成了锁死的分组划分（benchmark 仓库 `stage1_out/benchmark/dataset/split_lists/`，见其 `doc/CLAUDE.md` 契约修订 A1）。**此前的所有 benchmark 数字，包括 report.html 和 E1，都作废。**
- 数据和权重都不在 git 里（`frames_hires/`、`weights/`、`gcp_outputs/`、`*_out/`、`manifests/`）。回答前先确认本机上是否存在。

## 硬约束
- **不要读取、打印或移动 `keydump.txt`**：可能含凭据。
- **在 `D:\Conceivable-SharedData01-23Jun2026` 里禁止任何 git 操作**，并遵守其 `doc/CLAUDE.md`。
- `D:\Video\*.mp4` 原始视频只读。
- 启动 GCP 实例或训练前先征得用户同意，这会产生费用。`ARM` 必须显式指定。
- 训练一律通过 `scripts/dapt_run.sh`（本地）或 `scripts/dapt_train.sh`（GCP），不要直接调 torchrun。新增实验臂只改 `scripts/dapt_arms.sh`，不要复制 yaml。
- 改了 `repos/dinov3` 之后，按 README §2 重新生成 `patches/dinov3_dapt.patch`。

## 环境
- WSL Ubuntu-24.04，conda 环境 `dapt`，项目路径 `/mnt/d/Video/domain_transfer`。
- GPU 是 RTX 5070 Ti 12GB（sm_120，必须用 torch cu128）。
- motion-energy 生成在原生 Windows 上跑。
- gcloud 从 Windows 侧走 `wsl.exe` 调用更可靠。GCP 实例的实际 zone 见 `.dapt_a100_zone`。
