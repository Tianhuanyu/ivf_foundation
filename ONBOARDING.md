# 交接指南：IVF 显微域 Foundation Model

> 写给接手这个项目的人。读完这一份应该能知道：项目在做什么、代码在哪、哪些是活的、结论可信到什么程度、下一步做什么。
> 细节以 [`README.md`](README.md)（操作手册）和 [`SESSION_NOTES_motion_energy_e1.md`](SESSION_NOTES_motion_energy_e1.md)（最新实验记录）为准。
> **论文主线和验证实验**：[`PAPER.md`](PAPER.md) 加 [`experiments/paper.py`](experiments/paper.py)。先跑 `python experiments/paper.py status` 看进度。
> 用 Claude Code 打开本仓库时，[`CLAUDE.md`](CLAUDE.md) 会自动加载，Claude 可以直接回答关于代码和结论的问题（见 §9）。

---

## 0. 30 秒版

- **目标**：为 IVF 显微操作场景（ICSI 注射、卵母细胞操作等）做一个领域 foundation model。它是一个通用视觉 backbone，要在本域检测、分割、分类任务上超过 ImageNet CNN、DINOv2 和原版 DINOv3。
- **方法**：在官方 DINOv3 ViT-B/16 上，用约 2925 段无标注显微视频的抽帧做域自适应继续预训练（DAPT）。视频线（V-JEPA2）跑通过，但还没接入评估。
- **两部分**：
  - Part 1 **Benchmark 评估**：决定"好不好"。
  - Part 2 **Foundation model 设计**：产出权重。
  - 两部分通过 `ship_weights.sh` 交接权重。
- **当前状态**：
  - motion 引导的裁剪加时间加权，对 oocyte 和 holding pipette 类别有明显提升。
  - needle_tip（细长小目标）是主要难点，诊断为 ViT patch 分辨率瓶颈。
  - **所有 E1 结论都是单 seed、单数据集（holding_pip），还没坐实。**

---

## 1. 三个位置

| 位置 | 作用 | git |
|---|---|---|
| `D:\Video\domain_transfer\`（本仓库） | Part 2：数据准备、DAPT 训练、GCP 编排、权重交接 | `Tianhuanyu/ivf_foundation`，⚠️ 见 §8 |
| `D:\Video\domain_transfer\repos\dinov3\` | **当前活跃的训练代码**：官方 DINOv3 trainer 加我们的改动 | 独立的官方 clone，我们的改动**未提交**，备份在 [`patches/dinov3_dapt.patch`](patches/dinov3_dapt.patch) |
| `D:\Conceivable-SharedData01-23Jun2026\` | Part 1：benchmark（backbone 注册表、检测/分割头、报告） | **人工管理 git，Claude 禁止做 git 操作**（见其 `doc/CLAUDE.md`） |
| `D:\Conceivable-ML\` | 伞形仓库：流水线地图加 [`WEIGHTS_REGISTRY.md`](../../Conceivable-ML/WEIGHTS_REGISTRY.md)（权重血缘） | submodule |
| `D:\Video\*.mp4` | 原始视频，约 125GB，**只读**，不要在这里开 IDE 索引 | — |

```
D:\Video\*.mp4
  │ 10_build_manifest → 11_extract_frames → 12_motion_energy → 13_frame_weights
  ▼
frames_hires/<split>/<stage>/*.jpg (+ *.me.png 运动能量 sidecar)
  │ ARM=<臂> BUDGET=<e1|long> scripts/dapt_run.sh（本地）或 scripts/dapt_train.sh（GCP）
  │   = repos/dinov3 官方 train.py + dapt_vitb16.yaml + scripts/dapt_arms.sh 里的覆盖项
  ▼
<output_dir>/ckpt/<iter>  →  _extract_dapt_backbone_param.py（launcher 自动调用）  →  *_backbone.pth
  │ scripts/ship_weights.sh（sha256 + 登记 WEIGHTS_REGISTRY）
  ▼
SharedData01/stage1_out/dinov3_ckpt/  →  run_benchmark_*.py  →  report.html / per-class AP
```

---

## 2. Part 1：Benchmark 评估

**要回答的问题**：本域上哪个 backbone 的表征最好？DAPT 有没有用、用在哪儿？

- **四家 backbone**：`resnet50_fpn` / `dinov2_b_fpn` / `dinov3_b_fpn`（原版）/ `dinov3_dapt_b_fpn`（我们的）。每家都有粗-细变体 `*_cf_fpn`。
- **受控变量**：只变 backbone 和协议（frozen/finetune），检测头、imgsz、lr、epoch、seed 全部固定。**frozen 档最关键**，它衡量纯表征质量。
- **完整设计**：9 个数据集 × 3 个任务族 × frozen/finetune × 3 个 seed（`run_benchmark_server.py`）。
- **高分辨率档**：`run_benchmark_hires_ablation.py --imgsz 1024`，粗-细用 `--coarse-fine`。要同时报精度和延迟。
- **实际常用的快速回路**：只跑 `holding_pip` 检测（frozen、imgsz 1024、batch 8），看 per-class AP，重点是 needle_tip 和 oocyte_4x。
- **训练侧原来的 12 类操作阶段 kNN / 线性探针**（`legacy/scripts/50_eval_features.py`）已经归档：它只能读 HF 格式的权重。以后如果想要一个便宜的表征质量指标，需要改成能读 repo 格式的 `.pth`。

关键文件（benchmark 仓库）：
- `BENCHMARK_DINOV3.md`：跑法和设计依据。
- `stage1_out/benchmark/backbone/{dinov3.py,coarsefine.py}`：backbone 适配器，以及 CoarseFineFPN。
- `stage1_out/benchmark/registry.py`：backbone 注册表。

---

## 3. Part 2：Foundation Model 设计

### 3.1 活跃代码 vs 历史代码

| 状态 | 内容 |
|---|---|
| ✅ 活跃 | `repos/dinov3` 官方 trainer 加改动：`dinov3/data/crop_sampler.py`（local crop 裁在哪）、`dinov3/data/datasets/frames.py`（帧数据集，`extra=weighted` 打开时间加权）、`dinov3/data/masking.py` 与 `train/ssl_meta_arch.py`（ibot_local）、`configs/train/dapt_vitb16*.yaml`、`_extract_dapt_backbone_param.py`（从 ckpt 抽出 backbone） |
| ✅ 活跃 | 本仓库 `scripts/`：`10`–`13` 数据准备、`dapt_train.sh` 加 `_gcp_dinov3_dapt.sh`（GCP A100）、`ship_weights.sh` |
| ⛔ 已归档（2026-09-26 清理） | `legacy/`（gitignored），包括：<br>• 自研 `40_dinov3_dapt.py` 和 `structure_score*.py`：有 koleo 坍缩和 GCP 卡死问题，是促成转向官方 trainer 的原因<br>• V-JEPA2 视频线 `30_vjepa2_dapt.py` + `20_clip_dataset.py`：没有下游任务，不在论文主线里<br>• `41_fine_ijepa.py`、`fold_utils.py`、`51_eval_dapt.py`<br>• `50_eval_features.py`：只能读 HF 格式权重，读不了官方 trainer 产出的 `.pth`<br>• `00_detect.sh`、`verify_env.py`<br>• `legacy/dinov3_extras/`：旧版 `_extract_dapt_backbone.py`、`_convert_hf_grayscale_to_repo.py`、`dapt_vits16.yaml`<br>• `legacy/logs/`：8 月的旧日志<br>**不要在这些代码上继续开发。** |

### 3.2 设计维度（每个对应一个 yaml 臂，除标注的一行外其余超参完全相同）

| 维度 | ARM（定义在 `scripts/dapt_arms.sh`） | 改了什么 |
|---|---|---|
| 空间：local crop 裁在哪 | `uniform` / `content`（形态学加结构张量打分）/ `motion`（运动能量） | `crops.crop_sampler` |
| 时间：哪一帧多看 | `motion_weighted` | `dataset_path: ...:extra=weighted`，帧权重为 p90 运动分数的 γ=0.4 次方 |
| 预训练目标 | `motion_ibotlocal_high` / `_low` | 在 local crop 上做运动显著性引导的掩码预测（`ibot_local:` 段） |
| 下游架构 | benchmark 侧 `--coarse-fine --cf-fuse xattn` | 高分辨率切块加交叉注意力，针对 patch 分辨率瓶颈 |

---

## 4. 当前结果与可信度 ⚠️

完整结果表见 SESSION_NOTES §4、§5.3、§8.4。结论按可信度排：

1. **motion 加时间加权有效**（plain FPN，2 epoch）：聚合 map50_95 从 0.284 升到 0.373，oocyte_4x 从 0.267 升到 0.536。方向一致，但只有单 seed。
2. **needle_tip 的尺寸瓶颈诊断**：目标短边约 21px，只有约 1.3 个 patch token。这是实测框尺寸，数据本身可靠。
3. **"CoarseFineFPN 让 needle_tip 翻盘"是被训练量混淆的结论，目前不能成立**：
   - SESSION_NOTES §5.3 用的是 8 epoch 的 cf-xattn（0.0964），对比的却是 2 epoch 的 uniform（0.0861）。
   - §8.4 补的同协议（8 epoch plain FPN + motion_weighted）needle_tip 是 **0.1047，反而更高**。
   - README 状态区和 SESSION_NOTES 核心结论里"诊断 → 换架构 → 正向验证闭环"的说法，需要按这一点修正。
4. **ibot_local（high 方向）的阴性结果本身也被混淆了**，low 方向还没跑 benchmark。
   - 处理组的 config 继承自 `motion`，没有时间加权，而且是在 GCP 上跑的：GCP 帧缓存的 sidecar 缺失比例比本地高（2026-09-22 的日志里有 15,749 次回退到 uniform）。
   - 对照组 `motion_weighted` 是在本地跑的，带时间加权。
   - 两组差了不止一个变量，所以"ibot_local 没用"这个结论也不成立。
5. **benchmark 数据划分：2026-09-26 已修复，旧结果全部作废。**
   - 旧代码把 val 按文件名排序后一分为二当 val/test，导致部分 val 集没有标注，并且 test 与 train 之间有大量近重复帧。例如 holding_pip test 有 72/131 张、routine2 test 有 92/127 张在 train 里有近重复图。
   - 现在的切分单位是"录制 + 近重复帧连通分量"，按 70/15/15 分配，结果锁死在 benchmark 仓库的 `stage1_out/benchmark/dataset/split_lists/`。9 个数据集的跨 split 近重复都是 0。
   - 规则写在 benchmark 仓库 `doc/CLAUDE.md` 的"契约修订 A1"里，审计过程见 `stage1_out/split_audit/REPORT.md`。
   - **因此 `report.html` 和上面所有 E1 数字都基于旧划分，不能再引用，全部需要重跑。**

**规则**：
- **2 epoch 和 8 epoch 的 benchmark 结果不能直接比较。**
- 任何对比都必须同协议、同 imgsz、同 batch，并且至少 3 个 seed（42/43/44）。

---

## 5. 建议的下一步（按优先级）

0. ~~先修 benchmark 的数据划分~~：已完成（2026-09-26）。下一步是在新划分上重跑全部 benchmark：所有 backbone × 数据集 × 协议 × 3 seed。
1. **统一协议重跑 E1**：8 epoch 协议，uniform / content / motion / motion_weighted / cf-xattn 各跑 3 个 seed，只跑 holding_pip。先确定真实的排序。
   - 所有臂都在同一台机器上、用同一份帧缓存跑。`_gcp_dinov3_dapt.sh` 已修复，会保留 config 里的 `:extra=weighted`；修复之前，GCP 上跑 motion_weighted 会被静默降级成均匀抽帧。
2. 把胜出的配置放到完整 benchmark 上验证（9 个数据集，frozen 加 finetune），不能只看 holding_pip。
3. 如果 cf-xattn 在多 seed 下仍有优势，再试 **cf 架构 + motion_weighted 组合**（两者理论上正交，可以叠加）。
4. needle_tip 的治本方向：从 bbox 检测改成针尖加针柄的关键点检测（文献见 SESSION_NOTES §5.2）。
5. 视频线：先定一个视频类下游任务（比如操作阶段识别），再决定 V-JEPA2 是否继续投入。
6. 补 motion-energy 帧级覆盖缺口：injection 缺 29%、perfZP 缺 44% 的帧。

---

## 6. 环境与跑法速查

- **本地**：WSL Ubuntu-24.04，conda 环境 `dapt`，RTX 5070 Ti 12GB（Blackwell，**必须用 torch cu128**）。
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate dapt && cd /mnt/d/Video/domain_transfer
  ```
- **例外**：motion-energy 生成在**原生 Windows** 上跑（`C:\ProgramData\miniconda3\python.exe`），WSL 下有问题，原因见 SESSION_NOTES §2.3。
- **完整复现命令**：SESSION_NOTES §7（抽帧 → sidecar → 训练 → 抽 backbone → benchmark）。
- **本地训练**：`ARM=<臂> BUDGET=e1 ./scripts/dapt_run.sh`（README §2）。
- **GCP**（项目 `hidden-outrider-390502`，实例 `dapt-a100`）：README §3。
  ```bash
  ./scripts/dapt_prep.sh run                                     # 一次性：帧缓存
  ARM=<臂> BUDGET=long ./scripts/dapt_train.sh start|status|finish
  ```
- **权重交接**：`./scripts/ship_weights.sh <权重> <benchmark侧路径> "<来源>" "<配置>"`，不要手动 `cp`。

---

## 7. 踩过的坑（接手时最容易重踩的）

- **不要绕过 launcher 直接调 torchrun**：`dapt_vitb16.yaml` 里的数据路径和权重路径故意留成必填项（`???`），`--output-dir` 也必须显式传，否则会落到默认的 `./local_dino`。`dapt_run.sh` / `_gcp_dinov3_dapt.sh` 已经处理了这些。
- **`ARM` 没有默认值**，不传会直接报错。这是故意的，之前误跑错误的实验白烧过 GCP 账单。
- **桶里现有的帧缓存 sidecar 不全**（2026-09-26 之前做的）。跑 motion 类 ARM 之前先删掉它，用 `dapt_prep.sh run` 重做。
- **GCP zone 会变**：A100 缺货时会自动换 zone，实际 zone 以 `.dapt_a100_zone` 为准。
- **GCP OS Login 账户会漂移**：可能是 `thy`，也可能是 `htian_conceivable_life`，两者 `$HOME` 独立，换账户会触发重装环境。
- **本地长跑务必定期存 checkpoint**：WSL 意外重启丢过训练进度。
- **本地调 gcloud**：从 Windows 侧走 `wsl.exe` 调用更可靠。
- **epoch 太少时学习率调度会失效**：smoke test 用 `optim.epochs=1` 时，warmup 加 cosine 被压进 300 步，lr 几乎一直是 0，只能用来检查"会不会崩"。
- **CoarseFineFPN 需要足够训练量**：新增的交叉注意力是零初始化 LayerScale，2 epoch 下会整体崩溃，至少给 8 epoch。

---

## 8. 交接前需要原负责人处理 ⚠️

- [ ] **push**：交接相关的改动已经提交在分支 `handoff-cleanup` 上，还没有合回 main、也没有 push 到 `Tianhuanyu/ivf_foundation`。
- [ ] **`repos/dinov3` 的改动只存在于工作区和 `patches/dinov3_dapt.patch`**：已核对 patch 与当前改动一致（2026-09-22）。建议在一个 fork 上提交成分支，至少把 patch 随本仓库一起提交。
- [ ] **仓库根目录的 `keydump.txt`**：已 gitignore，但文件名暗示含凭据。交接前确认内容，必要时轮换密钥并删除，**不要把它转交给别人**。
- [ ] **访问权限**：
  - GCP 项目 `hidden-outrider-390502` 的 IAM 和桶权限。
  - GitHub 仓库的协作者权限。
  - HuggingFace gated 权重 `facebook/dinov3-*`，需要接手人自己申请。
- [ ] **大文件的位置**：`frames_hires/`（66GB）、`weights/`、`gcp_outputs/`、各 `*_out/` checkpoint 都不在 git 里。需要告诉接手人是拷盘、从 GCS 桶拉，还是重新生成。
- [ ] **benchmark 仓库的 git 由人工管理**：确认它的最新状态已提交。

---

## 9. 用 Claude 辅助理解

在本仓库根目录打开 Claude Code，`CLAUDE.md` 会自动提供项目上下文。可以这样问：

- "按数据流顺序带我走一遍 motion 臂：从 `12_motion_energy.py` 生成 sidecar，到 `crop_sampler.py` 选出 crop 位置。"
- "`patches/dinov3_dapt.patch` 对官方 DINOv3 改了哪些地方？每处改动的目的是什么？"
- "ibot_local 在 `ssl_meta_arch.py` 里是怎么接入 loss 的？和原版 iBOT 的区别是什么？"
- "我想用 3 个 seed 重跑 E1 的 8 epoch 对比，给出完整命令并估算 GCP 时长。"
- "SESSION_NOTES 里哪些结论是同协议对比，哪些不是？"
- "新增一个 crop_sampler 臂，需要改哪些文件？"
