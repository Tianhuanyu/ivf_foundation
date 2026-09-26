# 冒烟测试结果（2026-09-26）

- **目的**：正式训练或上传之前，确认论文流程里的每个脚本都能跑通。
- **口径**：只验证代码路径能走通，不看指标。训练 5 个 iter，benchmark 每个配置 1 个 epoch × 3 步、batch 2，所以指标接近 0 是正常的。
- **不碰真实产物**：
  - 冒烟输出在训练仓库的 `smoke_out/`，以及 benchmark 仓库的 `stage1_out/benchmark_results_smoke/`、`benchmark_results_imgsz320_smoke/`；
  - 权重登记写的是登记表的副本。
- **怎么重跑**：
  - WSL：`nohup setsid bash experiments/smoke_wsl.sh > smoke_out/smoke_wsl.log 2>&1 &`，一定要后台脱离运行；
  - Windows：`C:\ProgramData\miniconda3\python.exe experiments\smoke_windows.py`。
- **每一步的日志**：`smoke_out/logs/`。

**结果：WSL 36 项 + Windows 17 项，全部 PASS。** 其中 3 处是冒烟脚本本身的问题（下面标 ⓘ），已修复并重跑通过。

## 训练仓库：数据准备

| 步骤 | 环境 | 结果 | 耗时 |
|---|---|---|---|
| `10_build_manifest.py`（输出到 smoke_out） | WSL | ✅ | 32s |
| `11_extract_frames.py --limit 1` | WSL | ✅ 60 帧 | 2s |
| `12_motion_energy.py --limit 1` | Windows | ✅ 60 个运动图 | 12s |
| `13_frame_weights.py` | Windows | ✅ 生成 frame_weights.npy | <1s |

## 训练仓库：DAPT 训练（`dapt_run.sh SMOKE=1`，5 iter，训练完自动抽 backbone）

| ARM / BUDGET | 结果 | 耗时 | 备注 |
|---|---|---|---|
| uniform / e1 | ✅ | 95s | |
| content / e1 | ✅ | 93s | |
| motion / e1 | ✅ | 87s | |
| motion_weighted / e1 | ✅ | 84s | 日志确认时间加权已生效（868,310 帧），crop 按运动采样 |
| motion_ibotlocal_high / e1 | ✅ | 80s | |
| motion_ibotlocal_low / e1 | ✅ | 85s | |
| motion_weighted / long | ✅ | 77s | long 的覆盖项在前、冒烟覆盖项在后，后者生效 |

## 训练仓库：GCP、交接、patch（静态检查，GCP 本身不在冒烟范围内）

| 步骤 | 结果 | 备注 |
|---|---|---|
| `dapt_arms.sh` 拒绝未知的 ARM | ✅ | |
| `dapt_train.sh start` 不传 ARM 时拒绝执行 | ✅ | |
| GCP 相关 6 个 shell 脚本的语法检查 | ✅ | 这些脚本需要真实的 GCP 实例才能完整测试；第一次正式运行时要盯着 `status` 输出 |
| `check_patch.sh`：在全新的上游 clone 上打 patch，结果与工作区一致 | ✅ ⓘ | 原来的检查直接对 Windows 工作区（CRLF 换行）做 `git apply -R`，是误报；已替换为模拟 GCP 的检查：打上后 16 个文件、0 个不同 |
| `ship_weights.sh`（写登记表副本） | ✅ | 同时修了一个 bug：新登记的行以前会落在"待办"一节下面，现在追加到专用的交接记录表里 |

## benchmark 仓库（WSL，`--smoke`）

| 实验 | 覆盖范围 | 结果 |
|---|---|---|
| E0 | ResNet50、DINOv3-raw、DINOv3-DAPT × 5 个数据集 × 2 个协议，acquisition split | ✅ 30/30 |
| E2 frozen | 6 个 backbone × 9 个数据集 | ✅ 54/54 |
| E2 finetune | 6 个 backbone × cellasp、icsi_seg、holding_pip | ✅ 18/18 |
| E3 | 用 `DINOV3_DAPT_B_WEIGHTS` 换一份权重 | ✅ 生成了单独的 run 目录（`__w86f921cc`），不覆盖默认权重的结果（`__wd7282330`） |
| E4 | 4 个 backbone × 4 个检测集，320 px | ✅ 16/16 |
| `verify_v2_perclass_ap.py` | 对一个检测 run 算逐类别 AP | ✅ |

每个 run 都写出了 `test_samples.npz`，用于 bootstrap。

## 报告与论文脚本（Windows，读冒烟结果）

| 步骤 | 结果 |
|---|---|
| `make_splits.py --check`（两套 split，只读；锁定清单的时间戳没有变） | ✅ ✅ |
| `summarize_bench.py`：recording 和 acquisition 两套 split，外加 320 px 消融目录 | ✅ ✅ ✅ |
| `where_ours.py`：有多个 DAPT 变体、又没指定 `--ours` 时**拒绝执行** | ✅ ⓘ |
| `where_ours.py`、`make_ppt_figs.py`、`make_report.py`（带 `--ours dinov3_dapt_b_fpn@dapt`） | ✅ ✅ ✅ ⓘ |
| `plot_bench.py` | ✅ |
| `paper.py status`、`commands`、`tokens`、`tables`（T0–T3 全部生成） | ✅ ✅ ✅ ✅ |

ⓘ 这三项第一次没指定 `--ours` 时报错退出。那是预期中的保护：冒烟结果里有两份 DAPT 权重，脚本拒绝替你随便选一个。冒烟脚本已改为显式传入 `--ours`，并单独加了一项检查这个保护。

## 冒烟中得到的实用信息

- **1024 px 检测很慢**：即使训练只有 3 步，每个配置也要约 1.5–2.5 分钟，主要花在对完整 val 和 test 集做推理上。正式 E0 每个 run 训练 25 个 epoch，开跑前务必先测一个 run 的速度。
- **显存**：冒烟用的是 batch 2。正式配置（检测 batch 8、1024 px）的显存是否够，要在目标机器上确认。本地 12GB 显卡很可能不够 finetune。
- **WSL 的稳定性**：凡是长任务都用 `nohup setsid ... &` 脱离启动，只用短命令查看日志。这样可以避开工具强杀 `wsl.exe` 的问题，本次 3.5 小时的测试全程没有中断。
