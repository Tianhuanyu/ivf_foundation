# GCP 指令清单（直接复制粘贴）

> 位置：`D:\Video\domain_transfer\GCP_WORKFLOW.md`，WSL 里的路径是 `/mnt/d/Video/domain_transfer/GCP_WORKFLOW.md`。最后更新：2026-09-27。
> - 全部在 **WSL** 里执行，从上往下一条一条粘贴。
> - 🔁 表示这条要重复执行，直到出现注释里写的结果。
> - ⛔ 表示先检查，符合条件再往下走。
> - 实验设计和判定规则见 [PAPER.md](PAPER.md)。

## 当前进度

| 步骤 | 状态 |
|---|---|
| 本地冒烟 | ✅ 已完成（全部通过，见 `experiments/SMOKE_RESULTS.md`） |
| 上传帧缓存 | ✅ 已完成：`gs://mlflow-artifacts-ai-a100/dapt/cache/frames_hires_parts/`，12 个 tar，868,310 帧 / 617,346 个运动图 |
| **1. E0** | ⏭ **从这里开始** |
| 2. E1 训练 4 个 DAPT 臂 | E0 判定通过后再做 |
| 3. E2 + E3 + E4 | E1 完成后再做 |
| 4. 出表 | 最后 |

---

## 0. 进入环境（每次开新终端都要做）
```bash
wsl -d Ubuntu-24.04
```
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate dapt
```
```bash
gcloud auth login
```
```bash
gcloud config set project hidden-outrider-390502
```

## 1. E0：ViT 对 CNN
```bash
cd /mnt/d/Conceivable-SharedData01-23Jun2026
```
先在 VM 上冒烟（会问是否创建实例，输入 `y`）：
```bash
RETRY_MIN=10 BENCH_CMD="bash paper_jobs.sh E0 smoke" ./bench_gcp.sh start
```
🔁 直到出现 `BENCH DONE`：
```bash
./bench_gcp.sh status
```
取回结果并删除实例（输入 `y`）。下一步 `start` 会在任意有货的 zone 新建实例：
```bash
./bench_gcp.sh finish
```
⛔ 应该是 `[PASS] …runs OK`，并且没有 OOM：
```bash
tail -n 5 stage1_out/gcp_bench_logs/bench_run.log
```
正式运行：
```bash
RETRY_MIN=10 BENCH_CMD="bash paper_jobs.sh E0 run" ./bench_gcp.sh start
```
🔁 直到出现 `BENCH DONE`：
```bash
./bench_gcp.sh status
```
取回结果并删除实例（会问确认，输入 `y`）：
```bash
./bench_gcp.sh finish
```
```bash
cd /mnt/d/Video/domain_transfer && python experiments/paper.py tables
```
⛔ 看 `experiments/out/tables.md` 里的 T0：ViT 在多数检测集上不输 CNN，才继续第 2 步。

## 2. E1：训练 4 个 DAPT 臂
```bash
cd /mnt/d/Video/domain_transfer
```
先在 GCP 上冒烟（5 个 iter，会问是否创建实例，输入 `y`）：
```bash
RETRY_MIN=10 ARM=motion_weighted BUDGET=e1 DINO_EPOCH_LEN=5 DINO_EPOCHS=1 DINO_BATCH=8 ./scripts/dapt_train.sh start
```
🔁 直到出现 `DINOV3_DONE_motion_weighted_e1.txt`：
```bash
./scripts/dapt_train.sh status
```
取回结果并删除实例（输入 `y`）：
```bash
./scripts/dapt_train.sh finish
```
⛔ 应该能看到 `dinov3_vitb16_dapt_motion_weighted_e1_backbone.pth`：
```bash
ls -la gcp_outputs/
```

### 2a. uniform（约 16–17 小时）
```bash
RETRY_MIN=10 ARM=uniform BUDGET=long ./scripts/dapt_train.sh start
```
🔁 直到出现 `DINOV3_DONE_uniform_long.txt`：
```bash
./scripts/dapt_train.sh status
```
取回结果并删除实例（输入 `y`）：
```bash
./scripts/dapt_train.sh finish
```
```bash
./scripts/ship_weights.sh gcp_outputs/dinov3_vitb16_dapt_uniform_long_backbone.pth /mnt/d/Conceivable-SharedData01-23Jun2026/stage1_out/dinov3_ckpt/dinov3_vitb16_dapt_uniform_long.pth "domain_transfer@$(git rev-parse --short HEAD), ARM=uniform BUDGET=long" "ViT-B/16, 20000 it, batch 48, A100"
```

### 2b. content
```bash
RETRY_MIN=10 ARM=content BUDGET=long ./scripts/dapt_train.sh start
```
🔁 直到出现 `DINOV3_DONE_content_long.txt`：
```bash
./scripts/dapt_train.sh status
```
```bash
./scripts/dapt_train.sh finish
```
```bash
./scripts/ship_weights.sh gcp_outputs/dinov3_vitb16_dapt_content_long_backbone.pth /mnt/d/Conceivable-SharedData01-23Jun2026/stage1_out/dinov3_ckpt/dinov3_vitb16_dapt_content_long.pth "domain_transfer@$(git rev-parse --short HEAD), ARM=content BUDGET=long" "ViT-B/16, 20000 it, batch 48, A100"
```

### 2c. motion
```bash
RETRY_MIN=10 ARM=motion BUDGET=long ./scripts/dapt_train.sh start
```
🔁 直到出现 `DINOV3_DONE_motion_long.txt`：
```bash
./scripts/dapt_train.sh status
```
```bash
./scripts/dapt_train.sh finish
```
```bash
./scripts/ship_weights.sh gcp_outputs/dinov3_vitb16_dapt_motion_long_backbone.pth /mnt/d/Conceivable-SharedData01-23Jun2026/stage1_out/dinov3_ckpt/dinov3_vitb16_dapt_motion_long.pth "domain_transfer@$(git rev-parse --short HEAD), ARM=motion BUDGET=long" "ViT-B/16, 20000 it, batch 48, A100"
```

### 2d. motion_weighted
```bash
RETRY_MIN=10 ARM=motion_weighted BUDGET=long ./scripts/dapt_train.sh start
```
🔁 直到出现 `DINOV3_DONE_motion_weighted_long.txt`：
```bash
./scripts/dapt_train.sh status
```
```bash
./scripts/dapt_train.sh finish
```
```bash
./scripts/ship_weights.sh gcp_outputs/dinov3_vitb16_dapt_motion_weighted_long_backbone.pth /mnt/d/Conceivable-SharedData01-23Jun2026/stage1_out/dinov3_ckpt/dinov3_vitb16_dapt_motion_weighted_long.pth "domain_transfer@$(git rev-parse --short HEAD), ARM=motion_weighted BUDGET=long" "ViT-B/16, 20000 it, batch 48, A100"
```

## 3. E2 + E3 + E4：benchmark
```bash
cd /mnt/d/Conceivable-SharedData01-23Jun2026
```
先在 VM 上冒烟：
```bash
RETRY_MIN=10 BENCH_CMD="bash paper_jobs.sh E2 smoke && bash paper_jobs.sh E3 smoke && bash paper_jobs.sh E4 smoke" ./bench_gcp.sh start
```
🔁 直到出现 `BENCH DONE`：
```bash
./bench_gcp.sh status
```
取回结果并删除实例（输入 `y`）。下一步 `start` 会在任意有货的 zone 新建实例：
```bash
./bench_gcp.sh finish
```
⛔ 应该是 `[PASS]`：
```bash
tail -n 5 stage1_out/gcp_bench_logs/bench_run.log
```
正式运行（中断后重新执行同一条命令就会续跑）：
```bash
RETRY_MIN=10 BENCH_CMD="bash paper_jobs.sh E2 run && bash paper_jobs.sh E3 run && bash paper_jobs.sh E4 run" ./bench_gcp.sh start
```
🔁 直到出现 `BENCH DONE`：
```bash
./bench_gcp.sh status
```
```bash
./bench_gcp.sh finish
```

## 4. 出表
```bash
cd /mnt/d/Video/domain_transfer && python experiments/paper.py status
```
```bash
python experiments/paper.py tables
```
结果在 `experiments/out/tables.md`，按 PAPER.md 的判定规则看 C1–C3。

---

## 出问题时

**一直抢不到 A100：改用 flex-start 排队（推荐）**
加上 `FLEX=1`，GCP 会把创建请求排进队列，等该 zone 有空闲 A100 时自动创建实例，不用我们反复轮询。
- 最多排队 `FLEX_WAIT`（默认 6h）。
- 实例最长运行 `FLEX_RUN`，到时间会被**自动删除**，所以一定要设得比任务时长大。结果在任务结束时已经回传到桶，自动删除不会丢数据。
- 排队只在一个 zone 里进行，所以用 `ZONES` 指定和桶同地区的 us-east1-b。

E0 正式运行（估计 15–30 小时，所以设 72h）：
```bash
FLEX=1 FLEX_RUN=72h ZONES=us-east1-b BENCH_CMD="bash paper_jobs.sh E0 run" ./bench_gcp.sh start
```
E1 每个臂（估计 16–17 小时，设 36h），例如：
```bash
FLEX=1 FLEX_RUN=36h ZONES=us-east1-b ARM=uniform BUDGET=long ./scripts/dapt_train.sh start
```
> flex-start 是否支持 a2-highgpu-1g，要第一次真正提交时才能确认。如果报"不支持该机型"，就去掉 `FLEX=1`，回到 `RETRY_MIN=10` 的轮询方式。

**已关机的实例重新开机失败（`instances start` 报 STOCKOUT）**：已关机的实例只能在原来的 zone 重新开机。先取回结果，再删掉旧实例，重新 `start` 就会在其他 zone 新建：
```bash
./bench_gcp.sh fetch
```
```bash
./bench_gcp.sh down
```
训练这边同理：先 `./scripts/dapt_train.sh fetch`，再 `./scripts/dapt_train.sh down`。所以每个任务结束时都用 `finish`，不要把关机的实例留着。

**A100 没货（报 `ZONE_RESOURCE_POOL_EXHAUSTED` / `STOCKOUT`）**：这是 GCP 暂时没有 GPU，不是配额问题（每个地区都有 16 张 A100 的配额）。
- 脚本默认依次尝试这些 zone：us-east1-b（和桶同一地区）、us-central1-a/b/c/f、us-west1-b、us-west3-b、us-west4-b。
- 在任意 `start` 命令前加 `RETRY_MIN=10`，全部没货时每 10 分钟自动再试一轮，直到抢到为止。例如：
```bash
RETRY_MIN=10 BENCH_CMD="bash paper_jobs.sh E0 smoke" ./bench_gcp.sh start
```
- 还可以扩到欧洲和亚洲。注意：从 us-east1 的桶往这些地区拉数据，会产生少量跨区流量费。
```bash
ZONES="europe-west4-a europe-west4-b asia-northeast1-a asia-northeast1-c" RETRY_MIN=10 BENCH_CMD="bash paper_jobs.sh E0 smoke" ./bench_gcp.sh start
```


查看当前有哪些实例：
```bash
gcloud compute instances list
```
登录训练 VM 看日志（日志在 `~/dapt.log`）：
```bash
cd /mnt/d/Video/domain_transfer && ./scripts/dapt_train.sh ssh
```
登录 benchmark VM 看日志（日志在 `~/bench.log`）：
```bash
cd /mnt/d/Conceivable-SharedData01-23Jun2026 && ./bench_gcp.sh ssh
```
- 每个任务跑完都会自动关机，关机后不计 GPU 费；`finish` 会先取回结果，再删除实例。
- 改过代码以后，先重跑本地冒烟：`cd /mnt/d/Video/domain_transfer && nohup setsid bash experiments/smoke.sh > smoke_out/smoke.log 2>&1 &`，然后用 `cat smoke_out/SUMMARY.txt` 查看。
- 本地的帧有变化时才需要重传帧缓存：`./scripts/upload_frame_cache.sh`，已完成的阶段会自动跳过。
