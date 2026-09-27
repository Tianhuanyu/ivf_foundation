# 交接指南：IVF 显微域 Foundation Model

> 写给接手这个项目的人。读完这一份应该能知道：项目在做什么、已经做了哪些决定、代码在哪、下一步按什么顺序跑哪些脚本。
> - **论文主线、主张、预先登记的判定规则**：[`PAPER.md`](PAPER.md)
> - **实验进度、命令、出表**：`python experiments/paper.py status | commands | tokens | tables`
> - **GCP 上一键跑完全部实验（防崩溃、可续跑）**：`experiments/run_all.sh`，用法见 [`GCP_WORKFLOW.md`](GCP_WORKFLOW.md) 第一部分
> - **操作手册**（环境、训练、GCP）：[`README.md`](README.md)
> - 用 Claude Code 打开本仓库时，[`CLAUDE.md`](CLAUDE.md) 会自动加载，可以直接问 Claude（见 §9）。
>
> 最后更新：2026-09-27。

---

## 0. 30 秒版

- **目标**：为 IVF 显微操作场景（ICSI 注射、卵母细胞操作等）做一个领域基础模型（通用视觉 backbone），在本域的检测、分割、分类任务上与 ImageNet CNN、DINOv2 和原版 DINOv3 比较。
- **方法**：在官方 DINOv3 ViT-B/16 上，用约 2925 段无标注显微视频的抽帧做**域自适应继续预训练（DAPT）**，并用**视频里的运动**决定 local crop 裁在哪、哪些帧多看。
- **论文**（投 MICCAI / MIDL）：
  - **C1**：DAPT 提升 DINOv3；
  - **C2**：运动引导采样优于均匀采样；
  - **C3**：小目标的瓶颈在 ViT 的 token 尺寸；
  - **E0**（辅助）：在没见过的采集批次上，ViT 相对 CNN 有没有优势。
- **当前状态**：**代码、数据划分、评估规则都已就绪；有效的实验结果为零。** 2026-09-26 之前的所有 benchmark 数字都基于有泄漏的旧 split，**已全部作废**。
- **下一步**：先跑 **E0**。它只需要 30 个 run，不需要训练新权重；它的结果决定后面的 E1–E4 值不值得跑。

---

## 1. 已经做的决定（2026-09-26）

| # | 决定 | 在哪里记录 | 为什么 |
|---|---|---|---|
| 1 | 论文主干定为 DINOv3 DAPT 加运动引导采样。V-JEPA2、I-JEPA、fold、自研 DINO 脚本全部归档 | PAPER.md | 只有 motion 这条线有正向信号，其余是阴性结果或被混淆 |
| 2 | **Split 修复**：切分单位是"录制 + 近重复帧"，锁定在 benchmark 仓库的 `split_lists/` | benchmark `doc/CLAUDE.md` **A1** | 旧代码把 val 按文件名一分为二当 test，test 与 train 大量近重复（例如 holding_pip 72/131 张） |
| 3 | **检测主表用 1024 px**，320 px 作为消融（E4） | **A2** | 320 px 下关键目标不到 1 个 ViT token（needle_tip 0.42），在结构上对所有 ViT 不利 |
| 4 | **单 seed 加 test 集 bootstrap 95% 置信区间**，比较用配对 bootstrap；只在关键对比结论不确定时才补 seed 43/44 | **A3** | 医学影像领域普遍只跑单次训练，但必须报告不确定度（Christodoulou et al., MICCAI 2024） |
| 5 | **E0 使用 `acquisition` split**：test 来自训练中没见过的采集日、病例或会话 | A1 补充、PAPER.md | 拉大训练集和测试集的差距，避免任务层面的过拟合 |
| 6 | 实验规模暂定为单 seed（E0 30、E2 108、E3 36、E4 16 个 run）；**E1–E4 暂缓，等 E0 结果** | PAPER.md | E1 每臂约 16–17 A100 小时，先确认 ViT 路线值得投入 |

---

## 2. 代码在哪

| 位置 | 作用 | git |
|---|---|---|
| `D:\Video\domain_transfer\`（本仓库） | 数据准备、DAPT 训练、GCP 编排、权重交接、**论文实验编排**（`experiments/`） | `Tianhuanyu/ivf_foundation`，当前在分支 `handoff-cleanup`，**未 push** |
| `repos/dinov3/`（本仓库内） | 官方 DINOv3 trainer 加我们的改动 | 独立的官方 clone，改动的完整备份是 `patches/dinov3_dapt.patch` |
| `D:\Conceivable-SharedData01-23Jun2026\` | benchmark：数据集、锁定 split、backbone、runner、报告 | **人工管理 git，Claude 不做任何 git 操作** |
| `D:\Conceivable-ML\` | 流水线地图、`WEIGHTS_REGISTRY.md`（权重血缘） | ⚠️ 实际**不是 git 仓库**（README 写的 submodule 与实际不符） |
| `D:\Video\*.mp4` | 原始视频，约 125GB，**只读** | — |

```
D:\Video\*.mp4
  │ 10 → 11 → 12 → 13（清单、抽帧、运动能量图、帧权重）
  ▼
frames_hires/train/<stage>/*.jpg (+ .me.png) + frame_weights.npy
  │ ARM=<臂> BUDGET=<e1|long>  dapt_run.sh（本地） / dapt_train.sh（GCP）
  ▼
weights/ 或 gcp_outputs/  dinov3_vitb16_dapt_<ARM>_<BUDGET>_backbone.pth
  │ DINOV3_DAPT_B_WEIGHTS=<pth>（run_id 会自动带上权重 sha）
  ▼
benchmark：run_benchmark_server.py / run_benchmark_hires_ablation.py → stage1_out/benchmark_results*/
  │ result.json + test_samples.npz（逐样本记录，用于 bootstrap）
  ▼
experiments/paper.py tables → experiments/out/tables.md（T0–T3）
```

---

## 3. 论文实验一览（详见 PAPER.md）

| 实验 | 回答什么 | 规模 | 前置条件 |
|---|---|---|---|
| **E0**（辅助，**先跑**） | 在没见过的采集批次上，ViT（原版 DINOv3、已交付的 DAPT 权重 `d7282330`）是否优于 CNN | 3 个 backbone × 5 个数据集 × 2 个协议 = 30 run | 无 |
| E1 | 训练 4 个 DAPT 臂（uniform / content / motion / motion_weighted），训练量相同 | 4 次训练，每次约 16–17 A100 小时（BUDGET=long） | 重做 GCP 帧缓存 |
| E2（C1） | 主 benchmark：6 个 backbone × 9 个数据集 × 2 个协议 | 108 run | E1 的 motion_weighted |
| E3（C2） | 4 个 DAPT 臂的 frozen 消融，与 uniform 做配对比较 | 36 run | E1 |
| E4（C3） | 检测改用 320 px 重跑，和主表的 1024 px 对比 | 16 run | E1 的 motion_weighted（其余 backbone 可以先跑） |
| E5 | 附录：ibot_local | 可选 | — |

**E0 的判定**：
- ViT 在多数检测集上不输 CNN：推进 E1–E4。
- CNN 明确领先：论文转向"基础模型在 IVF 显微域的局限，以及 token 尺寸规律"，或者改用 CNN 基础模型（例如 DINOv3 的 ConvNeXt 蒸馏版，权重是否可下载还需核实）。

**已有的事实**（不依赖待跑实验）：
- `paper.py tokens` 的结果：320 px 下，needle_tip、oocyte_4x、cell 的短边中位数只有 0.4–0.9 个 token；到 1024 px 是 1.3–2.8 个。
- 单 seed 的 bootstrap 置信区间宽度：检测约 ±0.04，分类约 ±0.07。

---

## 4. 脚本清单：按推进顺序

**阶段 0：看进度、拿命令**（训练仓库，任何时候都可以跑）

| 脚本 | 作用 |
|---|---|
| `python experiments/paper.py status` | 每个实验完成了多少，根据磁盘上的权重和结果自动统计 |
| `python experiments/paper.py commands [E0 E1 …]` | 列出还缺的部分该跑什么命令，按顺序；**只打印，不会执行** |
| `python experiments/paper.py tokens` | 目标尺寸换算成 token 数的分析，本地几分钟 → `experiments/out/table_tokens.md` |
| `python experiments/paper.py tables` | 用已有结果生成 T0–T3 表 → `experiments/out/tables.md` |
| `bash experiments/run_all.sh status` | 一键脚本的进度：哪些步骤完成、哪一步在 VM 上跑、第几次尝试 |
| `bash experiments/test_run_all.sh` | 用假的 GCP 测一键脚本的续跑、重试逻辑，不花钱，几秒钟 |

**开跑前：冒烟测试**（每次改代码后、正式训练或上传前都跑一遍；最近一次结果见 [`experiments/SMOKE_RESULTS.md`](experiments/SMOKE_RESULTS.md)）

| 脚本 | 作用 |
|---|---|
| `nohup setsid bash experiments/smoke.sh > smoke_out/smoke.log 2>&1 &`（WSL） | **唯一的冒烟脚本**：数据准备、6 个 DAPT 臂加 long、GCP 脚本静态检查、patch 检查、权重交接、split 检查、E0/E2/E3/E4 的 benchmark、报告脚本、`paper.py`，约 3.5 小时。**必须后台脱离运行**，进度看 `smoke_out/SUMMARY.txt`；只跑其中几组可以用 `STEPS="…"` |
| `bash experiments/check_patch.sh`（WSL） | 单独检查 patch 能否在全新的上游 clone 上复现工作区 |
| 各入口的冒烟开关 | benchmark runner 加 `--smoke`；训练用 `SMOKE=1 ARM=… scripts/dapt_run.sh` |

**阶段 1：E0**（benchmark 仓库，GPU 机器；命令由 `paper.py commands E0` 给出）

| 脚本 | 作用 |
|---|---|
| `run_benchmark_server.py --all-seeds --split-profile acquisition --dataset cellasp holding_pip routine2_coc cvit_incubator cvit_workstation --backbone <bb> --skip-done` | 分别对 `resnet50_fpn`、`dinov3_b_fpn`、`dinov3_dapt_b_fpn` 各跑一次。**建议先用 `--dry-run` 看配置，再单独跑一个 1024 px 检测 run 测速** |
| `python experiments/paper.py tables` | 看 T0（ViT − CNN 的配对 Δ）并按上面的规则判定 |

**阶段 2：训练 DAPT 臂（E1）**（训练仓库，WSL）

| 脚本 | 作用 |
|---|---|
| `scripts/10_build_manifest.py` → `11_extract_frames.py` → `12_motion_energy.py` → `13_frame_weights.py` | 本地数据准备，已经做过；12 和 13 在原生 Windows 上跑 |
| `scripts/dapt_prep.sh run` | 在 GCP CPU 机上制作完整帧缓存（抽帧、运动能量图、帧权重）。**桶里的旧缓存不全，要先删掉旧的** |
| `ARM=<臂> BUDGET=<e1\|long> scripts/dapt_run.sh` | 本地训练一个臂，跑完自动抽出 backbone |
| `ARM=<臂> BUDGET=long scripts/dapt_train.sh start / status / finish` | GCP A100 训练；`finish` 下载结果并删除实例 |
| `scripts/dapt_arms.sh` | **唯一**定义实验臂和训练量的地方（不直接运行，被上面两个脚本读取） |
| `scripts/ship_weights.sh <pth> <目标> "<来源>" "<配置>"` | 把权重交给 benchmark，并登记到 `WEIGHTS_REGISTRY.md` |

**阶段 3：benchmark（E2–E4）**（benchmark 仓库）

| 脚本 | 作用 |
|---|---|
| `[DINOV3_DAPT_B_WEIGHTS=<pth>] run_benchmark_server.py --all-seeds --backbone <bb> [--protocol frozen] --skip-done` | E2 主矩阵、E3 消融（不同 DAPT 权重会写进各自的目录，不会互相覆盖） |
| `run_benchmark_hires_ablation.py --all-seeds --imgsz 320 --protocol frozen --dataset <4 个检测集> --backbone <bb> --skip-done` | E4 分辨率消融 |
| `verify_v2_perclass_ap.py --run-dir <run>` | 单个 run 的逐类别 AP，用于 C3 看 needle_tip 等小目标 |

**阶段 4：报告**（benchmark 仓库；默认读主 split，加 `--split-profile acquisition` 读 E0 的结果）

| 脚本 | 作用 |
|---|---|
| `summarize_bench.py` | 文本总表：单 seed 显示置信区间，另有 DAPT − raw 的配对 Δ |
| `where_ours.py [--ours <变体>]` | ours 在每个数据集上排第几 |
| `plot_bench.py` / `make_ppt_figs.py` / `make_report.py` | 图和 HTML 报告，输出到 `<results_dir>_plots/`。`make_report.py` 的叙述文字是 8 月写的，有新结果后要重写 |

**不要重新运行的**（已锁定）：`stage1_out/benchmark/dataset/make_splits.py`。重跑它会覆盖锁定的 split。只读检查可以用 `--check`。

**公共模块**（不直接运行，改规则时改这里）：

| 模块 | 负责什么 |
|---|---|
| `runner.py` | 受控变量：数据集路径、imgsz、epoch、batch、lr、seed |
| `report_common.py` | 报告规则：主指标、排除的数据集、置信区间、DAPT 变体区分、旧结果作废 |
| `bootstrap.py` | 置信区间计算 |
| `weights_id.py` | run_id 里的权重标识 |
| `dataset/splits.py` | 两套 split |

以上都在 benchmark 仓库的 `stage1_out/benchmark/` 下。

---

## 5. 结论可信度（旧结果的教训）

- **所有旧数字都作废**：`report.html`、SESSION_NOTES 里的 E1 表，都基于旧 split，而且检测用的是 320 px。它们只能用来看方向。
- **旧结果里的几个"结论"本身就被混淆了**，不要复述：
  - "CoarseFineFPN 让 needle_tip 翻盘"：拿 8 epoch 的结果比 2 epoch 的，同协议对照反而更差。
  - "ibot_local 没用"：处理组和对照组差了不止一个变量（有无时间加权、在不同机器上跑）。
- **规则**：只做同协议、同 imgsz、同 split 的比较；两个模型比较用配对 bootstrap；判定标准以 PAPER.md 为准，**不要看到结果后再改**。

---

## 6. 环境速查

- **本地**：WSL Ubuntu-24.04，conda 环境 `dapt`，RTX 5070 Ti 12GB（**必须用 torch cu128**）。
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate dapt && cd /mnt/d/Video/domain_transfer
  ```
- **原生 Windows Python**（`C:\ProgramData\miniconda3\python.exe`）：用来跑 12、13 号脚本，以及 `paper.py`、报告脚本。它**不能**加载 DINOv3 hub 模型（缺少 `termcolor`），涉及 DINOv3 的 benchmark 要在 WSL `dapt` 环境里跑。
- **GCP**：项目 `hidden-outrider-390502`，实例 `dapt-a100`，桶 `gs://mlflow-artifacts-ai-a100`，配置都在 `scripts/_gcp_common.sh`。

---

## 7. 踩过的坑

- **不要绕过 launcher 直接调 torchrun**：base config 里的路径故意留成必填项，`--output-dir` 也必须显式传。
- **`ARM` 没有默认值**：这是故意的，之前误跑错误的实验白烧过 GCP 账单。
- **桶里的旧帧缓存 sidecar 不全**：跑 motion 类的臂之前，先删掉它，再用 `dapt_prep.sh run` 重做。
- **GCP zone 会变**（以 `.dapt_a100_zone` 为准）；**OS Login 用户会漂移**（`thy` 和 `htian_conceivable_life` 两个账户，`$HOME` 独立，换了会重装环境）。
- **本地长跑务必定期存 checkpoint**：WSL 意外重启丢过进度。
- **epoch 太少时学习率调度会失效**：只能用来检查"会不会崩"，不能看效果。
- **1024 px 检测比 320 px 慢得多**：批量开跑前先单独测一个 run。

---

## 8. 仍待处理 ⚠️

- [ ] **训练仓库 push**：分支 `handoff-cleanup` 还没合回 main，也没 push。
- [ ] **benchmark 仓库提交**（由人工操作）：本轮新增和修改了 runner、report_common、bootstrap、weights_id、splits、两套 split 清单、训练循环、报告脚本、`archive/`，以及契约修订 A1–A3。
- [ ] **自动模式删不掉、需要手动删的文件**：`repos/dinov3/dinov3/configs/train/dapt_vitb16_official_compare_*.yaml`（6 个，已经没有用，删完要重新生成 patch，命令见 README §2），以及本仓库根目录的 `.gitmodules`。
- [ ] **`keydump.txt`**：文件名看起来像凭据，交接前自行确认，必要时轮换密钥并删除，**不要转交**。
- [ ] **访问权限**：GCP 项目的 IAM 和桶、GitHub 协作者、HuggingFace 的 DINOv3 gated 权重。
- [ ] **大文件**：`frames_hires/`（66GB）、`weights/`、`gcp_outputs/` 都不在 git 里，要告诉接手人是拷盘、从 GCS 拉，还是重新生成。

---

## 9. 用 Claude 辅助理解

在本仓库根目录打开 Claude Code，`CLAUDE.md` 会自动提供项目上下文。可以这样问：

- "E0 该怎么跑？先给我 dry-run 的命令，再告诉我怎么根据 T0 判定。"
- "`acquisition` split 是怎么定义的？cellasp 为什么特殊处理？"
- "配对 bootstrap 在 `bootstrap.py` 里是怎么实现的？检测的 mAP50 为什么能按图重采样？"
- "按数据流带我走一遍 motion 臂：从 `12_motion_energy.py` 到 `crop_sampler.py`。"
- "新增一个 DAPT 实验臂需要改哪些文件？"
- "PAPER.md 里 C2 的判定规则是什么？什么情况下要补 seed？"
