# 论文主线：IVF 显微操作视频的运动引导域自适应基础模型

> 目标投稿：医学影像会议（MICCAI / MIDL）。
> 本文件是论文的**主线与验收标准**。实验的执行、进度统计和出表全部由 [`experiments/paper.py`](experiments/paper.py) 完成（`status` / `commands` / `tokens` / `tables`），两者要保持同步。
> 2026-09-26 定稿。此前所有 benchmark 数字都基于有泄漏的旧 split，**全部作废**（见 benchmark 仓库 `doc/CLAUDE.md` 契约修订 A1），不在本文件中引用。

## 一句话

显微操作视频里的**运动**是免费的监督信号。在 DINOv3 的域自适应继续预训练（DAPT）中，用运动决定 local crop 裁在哪、哪些帧多看，可以得到更好的 IVF 显微基础模型。它什么时候有效，由**目标在 ViT 里能覆盖几个 token** 决定。

## 贡献与主张

| # | 主张 | 验证实验 | 产出 |
|---|---|---|---|
| **C1** | 通用基础模型在 IVF 显微域不会自动胜出；在域内视频上做 DAPT 可以稳定提升 DINOv3，并缩小或反转它与 ImageNet CNN 的差距 | **E2** 主 benchmark：6 个 backbone × 9 个数据集 × frozen/finetune，单 seed 加 test 集 bootstrap 置信区间（契约修订 A3），锁定 split，检测用 1024 px（A2） | 表 T1 |
| **C2**（方法） | 运动引导采样（空间 crop 加时间帧权重）优于均匀采样，也优于不用运动的显著性采样（content） | **E1** 在同一训练预算下训练 4 个 DAPT 臂，然后 **E3** 在所有数据集上跑 frozen（单 seed），用配对 bootstrap 和 uniform 比较 | 表 T2 |
| **C3**（分析） | 小目标检测的瓶颈在 ViT 的 tokenization：目标短边不到 1–2 个 token 时，任何预训练改进都救不回来；把检测输入从 320 提到 1024 px，ViT 的收益明显大于 CNN | **E4** 把主表的检测降到 320 px 重跑（4 个 backbone × 4 个检测集 × frozen，单 seed），和主表的 1024 px 对比；加上 `tokens` 分析和逐类别 AP | 表 T3、表 tokens |
| 附录 | 改训练目标（ibot_local：运动引导的 local crop 掩码预测）不如改采样；CoarseFineFPN 属于探索性结果 | E5（可选） | — |

**数据与基准**本身也是贡献：约 2925 段 IVF 显微操作视频用于 DAPT；9 个显微检测、分割、分类数据集，并提供按"录制 + 近重复帧"分组的无泄漏 split。

## 方法要点（C2 的实现）

- **底座**：官方 DINOv3 ViT-B/16，用官方 trainer 继续预训练（DINO + iBOT + KoLeo）。global crop 是整帧缩放，local crop 是原生分辨率裁剪。
- **空间**：`crop_sampler=motion`。local crop 的位置按运动能量图采样；运动能量图来自 `12_motion_energy.py`，在原生帧率上做逐像素基线归一化的帧差，不需要任何标注。
- **时间**：`extra=weighted`。帧按运动分数的 p90 的 γ=0.4 次方加权抽取（`13_frame_weights.py`，参考 MGSampler）。
- **对照臂**：`uniform` 是标准的域内继续预训练；`content` 是形态学加结构张量的显著性采样，用来回答"起作用的是运动，还是任何显著性都行"。
- **一致性**：各臂只相差 `scripts/dapt_arms.sh` 里的覆盖项。已验证合并后的配置与原来的独立 yaml 逐项一致。

## 预先登记的判定规则

结果出来之前先写死，防止事后挑选。

**统计方式**（契约修订 A3）：每个配置只训练 1 个 seed。单个结果报告 test 集 bootstrap 的 95% 置信区间（1000 次重采样）。两个模型比较时用**配对 bootstrap**：两者在同一批重采样图片上算差值 Δ，报告 Δ 的 95% 置信区间。

- **C2 成立**的条件：在 6 个主数据集的 frozen 结果上，`motion_weighted − uniform` 的配对 Δ 在 **≥ 4/6** 个数据集上为正，平均 Δ > 0，并且至少 **3/6** 个数据集的 95% 置信区间完全在 0 以上。同时 `motion_weighted` 或 `motion` 相对 `uniform` 的平均 Δ 要大于 `content` 的。
  - **补 seed 的条件**：如果正向数据集达到 4/6，但置信区间在 0 以上的不足 3 个（方向对但不确定），只给 `uniform` 和 `motion_weighted` 这一对补 seed 43、44，再按均值判定。
  - **不成立时的降级方案**：论文改为"benchmark + C3 分析"，C2 作为阴性消融如实报告。
- **C1 成立**的条件：ours（`motion_weighted`）相对 DINOv3-raw 的配对 Δ，在两个协议下都在多数数据集上为正，且其中多数的置信区间在 0 以上。它和 CNN 的相对位置如实报告，不作为 C1 的成立条件。
- **C3 成立**的条件：从 320 px 到 1024 px，ViT 类 backbone 在"短边不到 2 个 token"的检测集上，mAP 提升大于 CNN 的提升。并且逐类别来看，越小的类别（needle_tip、oocyte_4x、cell）提升越大。

## 已经可以确定的事实（不依赖待跑实验）

- **小目标 token 覆盖**（`python experiments/paper.py tokens`，数据取自锁定 split 的全部标注框）：在主 benchmark 的 320 px 下，关键检测目标的短边中位数都**不到 1 个 token**：needle_tip 0.42、oocyte_4x 0.71、cvit 两个数据集的 cell 0.71 和 0.89，89–100% 的框不足 1 token。到 1024 px 时是 1.3–2.8 个 token。这正是 C3 的前提，也解释了旧结果里 CNN 在检测上占优的现象。
- **Split 泄漏已修复**：9 个数据集的 val/test 与 train 之间，近重复图（256-bit dHash，Hamming ≤ 12）数量都是 0。旧规则下，例如 holding_pip test 有 72/131 张、routine2 test 有 92/127 张有近重复。

## 已做的决定

0. **单 seed 加 test 集 bootstrap 置信区间**（2026-09-26，契约修订 A3）。DAPT 预训练和下游 benchmark 都只跑 seed 42；依据是医学影像领域普遍只跑单次训练，但必须报告不确定度（Christodoulou et al., MICCAI 2024）。只有关键对比结论不确定时才补 seed（见判定规则）。论文里要写明：不确定度只反映 test 集的抽样波动，不包括训练随机性。

1. **检测主表使用 1024 px**（2026-09-26，方案 B，写入契约修订 A2）。
   - 理由：320 px 下关键目标不足 1 个 token，这在结构上对所有 ViT 不利，比较的就不再是表征质量。
   - 320 px 保留为分辨率消融（E4），用来支撑 C3。
   - 分类仍用 224，分割仍用 320。BiomedCLIP 固定 224 px 的局限在检测上会更明显，结果表里要标注。

## 仍待决定

2. **实验规模与算力预算**：E1 到 E3 的规模还没定，暂缓执行。主要可缩减项：DAPT 训练量（e1 或 long）、是否包含 3 个不进主表的数据集、是否保留 DINOv2-S。

## 执行顺序与算力

```bash
python experiments/paper.py status        # 进度
python experiments/paper.py commands      # 缺什么就给出对应的命令（不会自动执行）
```

| 步骤 | 内容 | 估算 |
|---|---|---|
| 0 | 重做帧缓存（旧缓存的 sidecar 不全）：`dapt_prep.sh run` | CPU 机，数小时 |
| E1 | 4 个 DAPT 臂 × BUDGET=long（20000 iter） | 每臂约 16–17 A100 小时（按 e1 实测 1.0 s/it @ batch16 推算，第一个臂跑完后校准） |
| E2 | 108 个 run，检测用 1024 px | 至少 9 A100 小时（320 px 实测约 5 分钟/run；1024 px 更慢，先测速） |
| E3 | 36 个 run，其中 main 臂的 9 个与 E2 共用 | 约 2–3 A100 小时 |
| E4 | 16 个 run，320 px | 约 1.5 A100 小时 |
| 出表 | `paper.py tokens`、`paper.py tables` → `experiments/out/` | 本地，几分钟 |

## 威胁效度与对应措施

| 风险 | 措施 |
|---|---|
| split 泄漏或近重复 | 按录制加近重复帧分组的锁定 split，并有 dHash 验证（契约修订 A1） |
| 不同权重的结果互相覆盖 | benchmark 的 run_id 带权重 sha，报告按权重区分变体（`weights_id.py`、`report_common.py`） |
| 只挑有利的 seed 或指标 | 固定 seed 42；报告 bootstrap 置信区间，比较用配对 bootstrap；每个任务的主指标固定（Acc / mIoU / mAP50），判定规则和补 seed 的条件预先登记 |
| 跨协议比较（旧 E1 的教训） | 所有臂同一训练预算、同一 benchmark 协议；`paper.py` 按权重 sha 选取结果 |
| 评测集太小 | egg_in_well、sperm_needle、routine3_coc 不进主表（`report_common.DROP`，理由写在代码里） |
| BiomedCLIP 固定 224 px | 已在契约中预先登记为局限；E4 不纳入 BiomedCLIP |
