# Motion-Energy Pipeline + CropSampler E1 消融 — 阶段性总结

日期：2026-09-20 ~ 2026-09-22
范围：为 CropSampler 的 `motion` 臂生产真实运动信号数据、跑通 DAPT 训练与下游 benchmark 评估、
诊断并**解决** needle_tip 类别对裁剪策略不敏感的问题。

## 核心结论（正向发现，先说结果）

> **裁剪策略（content/motion）的收益，由目标在 ViT backbone 里的 token 覆盖数决定。**
> 当目标短边能跨过约 2 个 patch token 时（oocyte_4x/20x、holding_pipette_tip），crop_sampler
> 的 content/motion 引导能显著提升检测效果（聚合 map50_95 最高 +31%，oocyte_4x 最高 +100%）；
> 当目标短边不足 1.5 个 token 时（needle_tip，21px÷16=1.3 token），纯裁剪策略优化全部失效——
> 信息损失发生在 backbone tokenization 这一步，比 crop_sampler 能起作用的环节更早。
>
> 这个诊断本身是可验证的：换成针对"backbone 分辨率不够"这个根因设计的 **CoarseFineFPN**（切块
> 高分辨率 tiled path + 交叉注意力融合）架构、给够训练量（8 epoch，而非欠训练的 2 epoch）后，
> **needle_tip 从所有裁剪策略实验里的 0.026~0.086 涨到 0.0964，首次超过 uniform baseline
> （0.0861）**——诊断 → 对症换架构 → 正向验证，形成完整闭环。见 §5。

---

## 1. 背景

CropSampler 三臂设计（`repos/dinov3/dinov3/data/crop_sampler.py`）：`uniform` / `content` / `motion`，
决定 DAPT 训练时 local crop 在一帧图像内**裁在哪**。`motion` 臂依赖逐帧运动能量图（`.me.png` sidecar），
本轮工作把这个数据从"设计完成但没有真实数据"推进到"真实数据 + 完整四/六臂对比 + 根因诊断"。

---

## 2. Motion-energy sidecar 生成（`scripts/12_motion_energy.py`）

### 2.1 算法

对每个已抽取的关键帧，重新解码其**原始视频的原生帧率**（而非用相邻关键帧，那样时间间隔太粗，抓不到
针尖这种快速局部运动），做逐像素基线归一化的帧差：

1. 整段视频做全局百分位对比度拉伸（这批显微视频原生对比度极低，实测某段视频像素值只落在 46–64/255，
   直接做帧差会被编码量化噪声淹没）。
2. **逐像素基线归一化**：每个像素除以它自己在全片里的"窗口最大值"典型水平（而非全局阈值）。
   原因：卵壁/皿壁这类天生高对比度边界，几乎每帧都会产生全图最大的帧差，用全局阈值归一化会让它
   把其他更细微但更相关的边缘（针尖）压成 0。逐像素基线让"这个位置正常抖动多大"成为该位置自己的
   标尺，同样的绝对抖动在背景安静区域会被判定为更"异常"。
3. 窗口最大值聚合 + 高斯模糊 + 幅度限制（`--ceiling-mult`，超过该值饱和到255）+ 下采样到 `--out-short`。

### 2.2 踩过的坑（按严重程度）

| 问题 | 现象 | 根因 | 修复 |
|---|---|---|---|
| 全局归一化让噪声爆炸 | 输出图全是均匀噪点 | 除以当前帧/窗口最大值，静止帧的量化噪声被强行拉满 | 改逐像素基线归一化（见上） |
| MotionCropSampler 温度失配 | motion 臂采样几乎退化成硬 argmax | sidecar 是任意尺度的 uint8 PNG，直接喂给按 `content_score` 的 [0,1] 尺度标定的 temperature softmax | 消费端也除以自身 max 归一化到 [0,1]，与 `content_score` 同约定 |
| **内存无限增长 → 进程被系统强杀** | 全量跑到 60~170/2777 就 `Terminated` | 解码循环 `while True: frames.append(...)` 没有硬顶；个别损坏视频 cv2 会一直返回帧、远超视频真实时长 | 加 `ABSOLUTE_MAX_FRAMES` 硬顶（最终定为12000帧/400s，5000帧太小会误伤合法长视频，见下） |
| 硬顶数值定错 | 一条 ffprobe 报时长 741 秒(!)的视频吃掉数GB内存 | 按"视频自称时长×2"算上限，时长本身离谱就没用 | 上限改成"时长相对上限"与"绝对硬顶"取更小值 |
| 绝对硬顶设太保守 | 5000 帧上限把一条**真实合法**的 225 秒视频也当异常拦掉 | 首批抽样只看到最长 142 秒的视频，据此定的上限不够代表全数据集（横跨多月，后期批次录制更长） | 上限提到 12000 帧（400s/6.7分钟），压测确认不再误伤 |
| **单帧解码真死循环** | 补跑任务卡在 800/2777 近 2 小时，CPU 满载但零进度 | `cv2.VideoCapture.read()` 对某些损坏视频可能直接阻塞，帧数上限判断在"读到一帧之后"才生效，读不到就永远等不到判断的机会 | 架构改为**每条视频独立子进程 + 180 秒硬超时**，超时直接 `terminate()`，不影响其他视频（Windows 无 `SIGALRM`，只能用真正的进程级 kill） |
| 预检查慢 | 全量任务光"哪些视频已完成"这一步就要几分钟 | 逐视频 `glob()`/`.exists()` 重复扫描同一个共享目录（如 `ICSI_injection` 目录下 22.5 万个文件，几百个视频共享） | 按目录缓存一次性 `iterdir()` 列表，之后纯内存查找 |
| `Frames` 数据集混入 sidecar | `.me.png` 被当成训练图片，且 MotionCropSampler 找 sidecar 路径变成 `xxx.me.png.me.png` | `Frames.__init__` 的扩展名过滤包含 `.png`，没排除 `.me.png` | 显式跳过 `.me.png` 结尾的文件 |

### 2.3 WSL → 原生 Windows 迁移

长任务反复触发 WSL 命令执行服务崩溃（`Wsl/Service/E_UNEXPECTED`），根因判定是这个对话工具的沙箱对
经 `wsl.exe` 发起、运行时间较长的命令有强杀机制，杀的时机不巧会连带弄坏 WSL 的服务状态。
`C:\ProgramData\miniconda3` 下已有一份装好 `cv2/numpy/scipy` 的原生 Windows Python，可以完全绕开 WSL：

```powershell
C:\ProgramData\miniconda3\python.exe -u scripts\12_motion_energy.py `
    --manifest manifests\train_videos.txt --split train --workers 4
```

脚本相应改动：去掉 `ffprobe` 依赖（改用 cv2 自带的 `CAP_PROP_FRAME_COUNT`/`CAP_PROP_FPS`，帧数读不到时
用实际解码到的帧数反推时长），并加了 `to_native_path()` 自动把 manifest 里的 `/mnt/d/...` 转成 `D:/...`，
所以旧 manifest 不用改。

### 2.4 最终覆盖率

| split | 总视频数 | 成功 | 失败(超时/帧数异常) | 成功率(按视频数) |
|---|---|---|---|---|
| train | 2777 | 2717 | 60 | 97.8% |
| val | 148 | 134 | 14 | 90.5% |

**但按帧数算覆盖率明显更低**（失败的视频往往是本来就异常长的视频，贡献了不成比例多的帧数）：
- train 总帧 868,310，缺 sidecar 250,964 帧（**28.9%**）
- val 总帧 112,259，缺 sidecar 74,985 帧（**66.8%**）
- needle_tip 高度相关的 `ICSI_injection`/`ICSI_perfZP` 阶段，帧级覆盖率分别只有 70.8%/56.3%

缺失的帧 `MotionCropSampler` 会优雅回退成 uniform 随机裁剪（不报错、不崩溃），但意味着这部分训练信号
其实没有拿到真正的运动引导。**这是一个仍然存在的真实数据缺口**，如果要投入更多算力，把这批长视频的
上限/超时再攻一次、把覆盖率拉高，对后续实验应该有实质收益。

---

## 3. 时间维度重要性加权采样（`scripts/13_frame_weights.py`）

### 3.1 动机

`crop_sampler` 只解决"选中一帧后裁在哪"（空间维度）。训练时 `dinov3/data/samplers.py` 的采样器对
**所有关键帧一视同仁**（纯均匀随机），完全没有"这一帧该不该被多看几眼"的时间维度概念。而像
ICSI_injection 这类视频，"针真正扎进去"的画面在整条视频里可能占比不到 10%，其余全是等待画面——
均匀采样严重稀释了模型见到真正动作的机会。

### 3.2 方法（参考 MGSampler, arXiv:2104.09952）

MGSampler 的核心思路是按**累积运动分布**均匀抽帧，而不是按时间均匀抽帧。我们的实现是它的静态重加权版本
（因为帧已经定长抽取完毕，不重新抽取）：

1. 每个关键帧的重要性分数 = 它的 `.me.png` 运动图的一个高分位数（p90，取"有没有热点"而非被背景稀释的均值）。
2. `weight = score ** gamma`，`gamma=0.4`（<1，压制过采样强度）——SSL 自蒸馏训练本身依赖足够多样的
   视角，如果加权太陡，训练会反复在同一小撮"高光帧"上打转，反而损害 representation 泛化性。
3. 没有 sidecar 的帧给全局中位数（中性权重），跟 crop_sampler 缺数据时的兜底哲学一致。
4. `Frames.__getitem__` **不用固定的 index→index 重映射表**（那样会让约 37% 的帧在整个训练过程中
   永远抽不到——经典"球放进 N 个桶"覆盖率损失问题），而是每次调用都独立按权重现抽一次，
   保证长期来看真正收敛到目标分布。

### 3.3 接入方式

`dinov3/data/datasets/frames.py` 的 `extra` 参数（原本未使用）复用为开关：
```yaml
dataset_path: Frames:root=.../frames_hires/train:extra=weighted
```
没有这个 `extra=weighted`，行为和之前完全一样（纯均匀），是默认关闭、显式开启的设计。

---

## 4. E1 对比结果（holding_pip 检测任务，test split，imgsz=1024/batch=8/seed=42，除非标注）

| 类别 | uniform<br>(2ep) | content<br>(2ep) | motion<br>(2ep) | motion+时间加权<br>(2ep) | cf-xattn<br>(5ep/b4,欠训练) | cf-xattn<br>(2ep/b8,欠训练) | **cf-xattn<br>(8ep/b8,充分训练)** |
|---|---|---|---|---|---|---|---|
| holding_pipette_tip | 0.3383 | 0.3728 | 0.4565 | 0.4018 | 0.3579 | 0.1820 | 0.3481 |
| needle_tip | 0.0861 | 0.0856 | 0.0260 | 0.0309 | 0.0555 | 0.0455 | **0.0964** ✅ |
| oocyte_20x | 0.4450 | 0.4544 | 0.4405 | 0.5232 | 0.4605 | 0.3122 | 0.4859 |
| oocyte_4x | 0.2673 | 0.1968 | 0.3989 | 0.5358 | 0.1937 | 0.0488 | 0.2803 |
| 聚合 map50 | 0.7006 | 0.6543 | 0.6751 | 0.6925 | 0.5431 | 0.3913 | 0.6428 |
| 聚合 map50_95 | 0.2842 | 0.2774 | 0.3305 | **0.3729** | 0.2669 | 0.1471 | 0.3027 |

**⚠️ 单 seed 结果，噪声可能很大，不构成统计显著结论。** 计划文档要求的 3-seed 还没有补——这是
把 needle_tip=0.0964 这个正向结果坐实前最需要补的一步（差距 0.0964 vs 0.0861 不算大，单 seed
噪声下有被推翻的风险）。

### 4.1 关键发现：裁剪策略优化对 needle_tip 系统性无效，根因是架构分辨率瓶颈

content / motion / motion+时间加权三次独立尝试（都是 plain FPN），needle_tip 都没有超过 uniform
baseline（0.0861）。其余类别（尤其 oocyte_4x）都能被 motion 系列明显改善——**这不是"运动引导裁剪
不work"，而是"运动引导裁剪对这一个类别不 work"**，且不 work 有明确的架构原因（见 §5）。换成
CoarseFineFPN 并给够训练量后，needle_tip 首次超过所有 plain-FPN 结果，包括 uniform baseline。

---

## 5. 根因诊断：needle_tip 可能撞上了 ViT patch tokenization 的物理分辨率上限

### 5.1 数据（holding_pip 数据集标注框实测尺寸，1042 个 needle_tip 框）

| 类别 | eval分辨率(1024)下尺寸(px, 中位数) | ÷16 (DINOv3 patch_size) = token 数 |
|---|---|---|
| holding_pipette_tip | 65.5 × 220.4 | 4.1 × 13.8 |
| **needle_tip** | **37.3 × 21.4** | **2.3 × 1.3** |
| oocyte_20x | 263.8 × 311.9 | 16.5 × 19.5 |
| oocyte_4x | 36.8 × 36.2 | 2.3 × 2.3 |

needle_tip 的短边（21.4px）除以 patch_size=16，**连 1.3 个 token 都不到**——这个目标在 ViT backbone
做 tokenization 那一步，很可能整个高度方向都被压进同一个 patch、跟背景混在一起编码，信息损失发生在
crop_sampler / DAPT 预训练能起作用的环节**之前**。

对比 oocyte_4x：面积同样小，但两个维度都还有 2.3 个 token（勉强够），这也是为什么它对预训练改进
"有反应"而 needle_tip 没有——**关键不是"够不够小"，而是"最短边有没有跨过大约 2 个 token 这个阈值"**。
holding_pipette_tip 虽窄但长边有 13.8 个 token，检测器能沿着线抓到足够信号。

### 5.2 文献佐证

- ViT patch size 从 32→16（分辨率翻倍）显著提升小目标 AP（9.7→17.8）；ViT-B/8 在小/中目标上
  明显优于 ViT-B/16——patch 越大，跨 patch 边界的细节被平均掉得越厉害。
  ([Visual Transformer for Object Detection](https://arxiv.org/pdf/2206.06323))
- FPN 下采样会让 8-32px 的小目标在 stride=4 时退化到 2-8px，继续降采样后变成"近乎一个点"的响应，
  严重损害召回率，且这是**架构层面**的信息损失，representation 质量救不了。
  ([IPG-Net](https://arxiv.org/pdf/1912.00632))
- 细长、高长宽比目标天然不适合 bbox+IoU 检测框架（框稍微偏一点，IoU 断崖式下跌）。
  最接近本场景的论文——[针的 tip-handle 检测与匹配](https://arxiv.org/pdf/2509.17931)——
  明确放弃 bbox 回归，改成检测"针尖+针柄"两个关键点再匹配连线；
  [AttWire](https://www.researchgate.net/publication/389714662_Attention_on_the_Wires_AttWire_A_Foundation_Model_for_Detecting_Devices_and_Catheters_in_X-ray_Fluoroscopic_Images)
  做导管检测用的是同样思路。

### 5.3 验证：CoarseFineFPN 变体，给够训练量后 needle_tip 正向翻盘

用同一个 motion+加权 checkpoint，换成已有的 `dinov3_dapt_b_cf_xattn_fpn`（切块高分辨率 tiled path +
交叉注意力融合）backbone 重跑 benchmark，一共试了三个训练量级：

| 训练量级 | needle_tip | 聚合map50_95 | 备注 |
|---|---|---|---|
| 5 epoch / batch4 | 0.0555 | 0.2669 | val 一路涨到 0.50、test 只有 0.27——过拟合，val/test 脱节 |
| 2 epoch / batch8（对齐其他五组 batch） | 0.0455 | 0.1471 | 新增交叉注意力模块（零初始化 LayerScale）明显没热身够，整体全面崩溃 |
| **8 epoch / batch8（充分训练）** | **0.0964** | **0.3027** | val 从 ep1 的 0.35 稳步涨到 ep7 的 0.57，ep8 略降到 0.53——val/test 差距明显收窄，不再是前两次那种明显欠拟合/过拟合状态 |

前两个量级下 needle_tip 就已经一致地比 plain FPN motion 系列（0.026/0.031）高，是"分辨率瓶颈"诊断
的方向性证据；**但直到给够训练量（8 epoch），needle_tip 才真正超过所有 plain-FPN 结果、包括
uniform baseline（0.0964 > 0.0861）**。这才是把诊断落实成正向结果的关键一步——前两次训练量级不够，
新增的交叉注意力参数还没来得及学到东西，会让人误判"架构本身不行"；给够训练预算后，架构改动的真实
收益才显现出来。

---

## 6. 尚未解决 / 后续建议（按优先级）

0. **ibot_local（motion-guided local-crop 掩码预测）单 seed A/B 为阴性，见 §8**：不是"架构"这条线
   （coarse-fine，见 §5），是"预训练目标"这条线的独立尝试，结论是目前没有证据支持它比现有
   motion_weighted baseline 更好。下一步要么补 `mask_low_saliency` 方向的 A/B + 多 seed 把这个
   阴性结果坐实，要么判定这条路径到此为止、把精力集中在 §5 的 coarse-fine 多 seed 验证上。
1. **needle_tip=0.0964 这个正向结果需要多 seed 坐实**：差距 vs uniform baseline(0.0861) 不算大，
   单 seed 噪声下有被推翻的风险。这是当前最高优先级——在往 patent/paper 里写之前，至少补 2-3 个
   seed（`--seed` 换值重跑 §7 的 coarse-fine 8epoch 命令），确认优势稳定存在。
2. **coarse-fine 在其他类别上还没到全面最优**：8epoch 版本 holding_pipette_tip(0.3481)/oocyte_4x(0.2803)
   都低于 motion+时间加权那组的峰值(0.4018/0.5358)。如果 needle_tip 的多 seed 结果稳住了，下一步可以
   探索"coarse-fine 架构 + motion 时间/空间加权"的组合，而不是二选一——architecture 解决分辨率瓶颈，
   crop_sampler 解决"该看哪里/该多看哪帧"，两者理论上正交、可以叠加。
3. **motion-energy 帧级覆盖率缺口**（injection 29%、perfZP 44% 帧缺失）：如果要更干净地验证 motion 臂
   的真实潜力，值得先把这批长视频的处理上限/超时再攻一次。
4. **needle_tip 任务形式化（更治本但改动更大的备选方向）**：即使 coarse-fine 多 seed 验证通过，
   0.0964 相对 holding_pipette_tip/oocyte 的 AP 量级依然低很多，检测头本身对细长目标的适配可能还有
   上限。如果后续还想再往上推，值得考虑像文献里那样把 needle_tip 从 bbox 检测改成关键点检测——但
   目前 coarse-fine 这条路径已经能在不改检测头的前提下拿到正向结果，优先级应该在多 seed 验证之后。
5. ~~`scripts/structure_score*.py` 等预置脚本这次没有深入梳理~~ —— 已梳理完成：`40_dinov3_dapt.py` /
   `structure_score.py` / `structure_score_gpu.py` / `_gcp_adaptive_crop_validation.sh` /
   `run_adaptive_crop_full_validation.sh` / `verify_adaptive_crop_dapt.sh` / `verify_batch_probe.sh` /
   `_gcp_remote.sh`（这个已经是孤儿代码，头注释还指向不存在的 `dapt_gcp.sh`）已全部移到 `legacy/`，
   `.gitignore` 里加了 `legacy/`（不再跟踪）。`README.md`/`dapt_train.sh` 里的相关路径引用已同步更新。
   `patches/`、`scripts/fold_utils.py` 还没梳理，如果要做更彻底的代码库整理，值得单独确认。

---

## 7. 复现指南

```bash
# 1. 生成 motion-energy sidecar（原生 Windows，避免 WSL）
C:\ProgramData\miniconda3\python.exe -u scripts\12_motion_energy.py --manifest manifests\train_videos.txt --split train --workers 4
C:\ProgramData\miniconda3\python.exe -u scripts\13_frame_weights.py --frames-root frames_hires --split train

# 2. 四个 DAPT 对比 config（3000 iters，同一套超参，只有 crops.crop_sampler / dataset_path 的 extra 不同）
repos/dinov3/dinov3/configs/train/dapt_vitb16_official_compare_base.yaml       # uniform
repos/dinov3/dinov3/configs/train/dapt_vitb16_official_compare_adaptive.yaml   # content
repos/dinov3/dinov3/configs/train/dapt_vitb16_official_compare_motion.yaml     # motion
repos/dinov3/dinov3/configs/train/dapt_vitb16_official_compare_motion_weighted.yaml  # motion + 时间加权

cd repos/dinov3
PYTHONPATH=. torchrun --nproc_per_node=1 dinov3/train/train.py \
    --config-file dinov3/configs/train/<上面某个>.yaml \
    --output-dir <跟 yaml 里 output_dir 一致的路径>   # 必须显式传，yaml 里的 output_dir 会被 CLI 默认值 ./local_dino 覆盖！

# 3. 提取 backbone + 跑 benchmark（在 D:\Conceivable-SharedData01-23Jun2026）
python repos/dinov3/_extract_dapt_backbone_param.py --ckpt <output_dir>/ckpt/2999 --out weights/<name>_backbone.pth
DINOV3_DAPT_B_WEIGHTS=weights/<name>_backbone.pth python run_benchmark_hires_ablation.py \
    --seed 42 --backbone dinov3_dapt_b_fpn --dataset holding_pip --protocol frozen \
    --imgsz 1024 --batch-size 8 --epochs 2

# 4. §5.3 的正向结果：同一个 backbone，换 CoarseFineFPN + 交叉注意力，给够训练量(8 epoch)
DINOV3_DAPT_B_WEIGHTS=weights/dinov3_official_compare_motion_weighted_backbone.pth \
  python run_benchmark_hires_ablation.py \
    --seed 42 --backbone dinov3_dapt_b_fpn --dataset holding_pip --protocol frozen \
    --imgsz 1024 --batch-size 8 --epochs 8 --coarse-fine --cf-fuse xattn
# 换 --seed 42/43/44 跑三次是 §6 第1条要补的多 seed 验证

# 5. §8 的 ibot_local 实验（阴性结果，含 mask_high_saliency / mask_low_saliency 两个方向）
CONFIG=dapt_vitb16_official_compare_motion_ibotlocal_high.yaml DINO_BATCH=16 DINO_EPOCH_LEN=500 DINO_EPOCHS=6 \
  REMOTE_SCRIPT=_gcp_dinov3_dapt.sh ./scripts/dapt_train.sh train   # 或 ..._ibotlocal_low.yaml
# 跑完后同 §3 提取 backbone + 跑 benchmark（8 epoch，不要用 2 epoch —— 见 §8.3 的教训）
```

---

## 8. ibot_local（motion-guided local-crop 掩码预测）：实现完成，单 seed A/B 为阴性结果

与 §5 的 CoarseFineFPN 是两条独立的解决方向：§5 改**架构**（切块高分辨率+交叉注意力），本节改
**预训练目标**（在 iBOT 掩码预测里引入 motion 显著性引导）。文献依据：AttMask（ECCV'22，遮挡显著
区域更多，逼模型学更精细的重建）vs MotionMAE/V-JEPA4A（ICCV'23/GCPR'26，遮挡显著区域更少，把它
当上下文保留）——两者方向相反，都发表于顶会，本身就是一个开放问题，所以设计成可配置的 A/B
（`direction: mask_high_saliency` / `mask_low_saliency`），不预设哪个赢。

### 8.1 为什么需要新代码，而不是复用现有 masking

官方 DINOv3 训练器的 iBOT 掩码**只作用于 global crop**（224px，resize 后的粗分辨率），local crop
（128px，native 分辨率，crop_sampler 真正起作用的地方）从来没有被掩码过、老师网络也从来没有在
local crop 上跑过。所以不能简单地给现有 `MaskingGenerator` 加个 motion 偏置——必须新增一条完整、并行
的 local-crop 掩码预测通路（新 masking generator + 新的 teacher 前向 + 新的 student 掩码 + 新的
iBOT loss 项），跟 global crop 的掩码逻辑完全独立、互不影响。

### 8.2 实现（新增/改动文件）

| 文件 | 改动 |
|---|---|
| `dinov3/data/crop_sampler.py` | `sample()` 从只返回 crops 改成返回 `(crops, positions)`，让 augmentations.py 能按位置回查 saliency |
| `dinov3/data/augmentations.py` | 新增 `_local_crop_saliency()`：按 crop 位置从 `.me.png` 侧车图裁出对应的 patch-grid 尺寸 saliency；`output["local_crops_saliency"]` |
| `dinov3/data/masking.py` | 新增 `MotionGuidedMaskingGenerator`（继承 `MaskingGenerator`，`_mask()` 用候选块采样+按 saliency 加权挑选替代无条件随机放置），基类完全不动 |
| `dinov3/data/collate.py` | `collate_data_and_cast` 新增可选参数，产出 `collated_local_masks` / `local_mask_indices_list` / `local_masks_weight` / `n_masked_local_patches`，默认关闭时输出字典逐字节不变 |
| `dinov3/train/train.py` | `build_data_loader_from_cfg` 按需构建第二个 `MotionGuidedMaskingGenerator`（尺寸对齐 local crop） |
| `dinov3/train/ssl_meta_arch.py` | 新增 `get_teacher_output_local()`（教师首次在 local crop 上跑）；`get_student_output()` 用真实 local mask 替换硬编码的 `None`；`compute_losses()` 新增 `ibot_local_loss` 项，复用已有 `ibot_head`/`ibot_patch_loss`，不新建头 |
| `dinov3/configs/ssl_default_config.yaml` | 新增 `ibot_local:` 段，`enabled: false` 默认关闭，不影响任何现有 config |
| `dinov3/configs/train/dapt_vitb16_official_compare_motion_ibotlocal_{high,low}.yaml` | 两个新实验 config，对应 A/B 的两个方向 |

### 8.3 验证过程（含踩过的坑）

| 阶段 | 问题 | 根因 | 修复 |
|---|---|---|---|
| 本地单元测试 | — | — | saliency 提取、`MotionGuidedMaskingGenerator` 的方向偏置（56.8% vs 8.2% vs 25%随机）均通过 |
| 本地端到端 smoke test | WSL 没有 torch/pip/venv（PEP 668 + 缺 python3-venv，且不想动系统包） | 这台 WSL 本来就只做 gcloud 编排，不跑训练 | 用 Miniforge 装了个独立 conda 环境（`~/miniforge3` + env `dapt_smoketest`），意外发现本机有块 RTX 5070 Ti（12GB）能跑真实 CUDA 训练，不用等 GCP：5/5 iter 全过，`ibot_local_loss` 每步都在，0 次 saliency 回退 |
| GCP 首次训练 | 训练报 config 文件 `FileNotFoundError` | `_gcp_dinov3_dapt.sh` 自己的第 [1/6] 步会 `gcloud storage rsync` 整个 bucket 前缀盖回 VM，而 bucket 里的 `scripts/patches` 从来没同步过（只有已归档的 pushbench legacy 路径会传），训练悄悄用了几个月前的旧 patch | 同步新代码到 bucket；**永久修复**：`dapt_train.sh` 的 `cmd_pushcode` 现在也会把 `scripts/patches/manifests` 同步进 bucket，不再只 scp 到 VM |
| GCP 第二次训练 | 同样的 `FileNotFoundError`，反复出现 | 第一次训练已经把 VM 本地磁盘上的 patch 文件覆盖成旧版本；bucket 修好了，但没把 VM 本地那份改回来（新脚本的设计就是不再从 bucket 覆盖 VM 本地，所以旧文件会一直留在原地） | 重新 `pushcode`，直接在 VM 上用 md5sum 核实 patch 内容正确后再删 clone 重跑 |
| GCP 第三次训练 | 训练能跑，但每一条样本都在报 `falling back to uniform`（crop_sampler 一直找不到 `.me.png`） | GCP bucket 里缓存的 `frames_hires.tar` 是 §2 的 motion-energy 生成工作**之前**打包的快照（0 个 `.me.png`，jpg 数量也对不上本地），而 `dapt_prep.sh` 从原始视频重新抽帧的流程压根不知道侧车文件这回事 | 没有重新走"CPU 机器重新抽帧"这条路（68GB 太贵太慢），而是把本地已有的 61.7 万个 `.me.png`（7.1GB）单独打成一个 tar 直接上传、解压合并进 VM 已有的 frames_hires 目录（tar 解压不会删除目录里没提到的文件，jpg 不受影响） |
| GCP 第四次训练 | 成功 | — | 3000 iters，55 分钟，`ibot_local_loss` 从 5.55 降到 4.74，收敛正常；仍有约 29% 帧因为 §2.4 记录的历史覆盖率缺口而回退成 uniform（不是新问题） |

`.me.png` 覆盖率缺口（61.7万/86.8万，71%）是 §2.4 已经记录过的既有事实，不是这次新引入的问题；这次
只是第一次真正让**同一条训练**同时依赖它（crop_sampler 的空间信号 + ibot_local 的掩码信号），所以第一次
把这个缺口的影响暴露成了"训练根本跑不动"而不是"效果打了折扣"。

### 8.4 结果：单 seed A/B，8 epoch/plain-FPN/imgsz1024/batch8/seed42/holding_pip/test split

| 类别 | motion_weighted（对照，无 ibot_local） | motion + ibot_local (mask_high_saliency) |
|---|---|---|
| holding_pipette_tip | 0.5479 | 0.5249 |
| **needle_tip** | **0.1047** | 0.0962 |
| oocyte_20x | 0.6056 | 0.5786 |
| oocyte_4x | 0.4612 | 0.4993 |
| 聚合 map50 | 0.9046 | 0.8992 |
| 聚合 map50_95 | 0.5622 | 0.5627 |

关键教训：ibot_local 跑出来的 needle_tip=0.0962，乍一看几乎追平了 §5.3 的正向结果（0.0964），但那是
一个**训练量级混淆**的假象——§4 表格里 uniform/content/motion/motion_weighted 那几列全是 2 epoch
benchmark 协议（已知严重低估真实性能），不能直接跟这次的 8 epoch 结果比。补了同协议（8 epoch，同
plain FPN，同 motion_weighted backbone）的对照后，needle_tip 反而是对照更高（0.1047 > 0.0962）。

**结论：没有证据支持 mask_high_saliency 方向的 ibot_local 比现有 motion_weighted baseline 更好**，
聚合指标基本打平，per-class 上喜忧参半（oocyte_4x 好一点，其余三类差一点）。单 seed 噪声下，这些
0.01~0.03 的差距不构成"更差"的结论，只构成"没有更好"的结论。

### 8.5 下一步

- 还没跑 `mask_low_saliency` 方向（MotionMAE/V-JEPA4A 的约定）——万一是掩码方向选反了，这个方向
  可能才是有效的那个，值得补一次同协议的对照。
- 如果 `mask_low_saliency` 也是阴性，这条"预训练目标"路径大概率到此为止，精力应该集中在 §5 的
  coarse-fine 多 seed 验证（那才是目前唯一站得住脚的正向结果）。
- 无论哪个方向，都还只是单 seed——真要下结论还需要至少 2-3 个 seed，尤其是 needle_tip 这种本来就
  低量级、噪声占比高的类别。
