# GCP 指令清单（直接复制粘贴）

> 位置：`D:\Video\domain_transfer\GCP_WORKFLOW.md`，WSL 里的路径是 `/mnt/d/Video/domain_transfer/GCP_WORKFLOW.md`。最后更新：2026-09-27。
> - 全部在 **WSL** 里执行，从上往下一条一条粘贴。
> - **推荐用第一部分的一键脚本**。第二部分是手动逐步执行的备用方式，两者做的事情完全一样。
> - 实验设计和判定规则见 [PAPER.md](PAPER.md)。

## 当前进度

| 步骤 | 状态 |
|---|---|
| 本地冒烟 | ✅ 已完成（全部通过，见 `experiments/SMOKE_RESULTS.md`） |
| 上传帧缓存 | ✅ 已完成：`gs://mlflow-artifacts-ai-a100/dapt/cache/frames_hires_parts/`，12 个 tar，868,310 帧 / 617,346 个运动图 |
| **1. E0** | ⏳ 2026-09-28 已用 `run_all.sh`（FLEX=1）启动，VM 在 us-east1-b。用 `bash experiments/run_all.sh status` 看进度 |
| 2. E1 训练 4 个 DAPT 臂 | E0 判定通过后再做。**4 个臂并行**，各用一台 VM（`dapt-a100-uniform/-content/-motion/-motion-weighted`），约 1 天 |
| 3. E2 + E3 + E4 | E1 完成后再做 |
| 4. 出表 | 最后 |

---

# 第一部分：一键跑完（推荐）

`experiments/run_all.sh` 按顺序跑完全部任务：E0 冒烟 → E0 → **暂停等你批准** → E1 冒烟 → **4 个臂同时训练**（每个臂一台 VM，启动间隔 90 秒；4 个都结束后，把训练成功的臂逐个交接给 benchmark）→ E2/E3/E4 冒烟 → E2/E3/E4 → 出表。
每一步都是：建 VM → 等结果 → 取回 → 删 VM → 校验。

**防崩溃机制：**
- **本脚本挂了**（WSL 重启、关机、终端关了）：重新执行启动命令即可。已完成的步骤会跳过；正在 VM 上跑的任务不会重复提交，只是接着等它。
- **VM 挂了**（flex-start 到时间、被抢占、任务失败）：自动删掉旧 VM，新建一台续跑，每步最多重试 6 次（`MAX_ATTEMPTS`）。
  - 训练：每 1000 个 iter 存一个断点，每 5 分钟同步到桶。新 VM 从最新断点继续，最多损失约 1000 个 iter。
  - 4 个臂互不影响：某个臂的 VM 挂了只重跑那一个臂；某个臂最终失败，其他 3 个照样完成并交接，脚本停下来报告是哪个臂失败。重新启动只会续跑没完成的臂。
  - benchmark：每 10 分钟把结果同步到桶。新 VM 先拉回已完成的 run，跳过它们，最多损失正在跑的那一个 run。
- **A100 没货**：普通方式每 10 分钟自动再试所有 zone；flex-start 方式在 GCP 队列里排队等卡。
- 不管成功还是失败，VM 最后都会自动关机，不会空烧钱。

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
⚠️ 下面所有命令都要在训练仓库目录里执行。这一条**单独**粘贴，不要和后面的启动命令拼成一行（行尾的 `&` 会把整行都放到后台执行，`cd` 也在后台，你的终端就不会切换目录）：
```bash
cd /mnt/d/Video/domain_transfer
```
```bash
mkdir -p run_state
```

## 1. 启动（后台运行，关掉终端也不会停）
⚠️ 这一步会开始产生 GCP 费用，而且中间不会再问确认。

**推荐：flex-start 排队。** 请求会排进 GCP 的队列，us-east1-b（和桶同地区）一有空闲 A100 就自动建 VM，比轮询更容易抢到，价格也更低：
```bash
FLEX=1 ZONES=us-east1-b nohup setsid bash experiments/run_all.sh >> run_state/run_all.log 2>&1 &
```
普通方式：依次试 8 个美国 zone，全部没货就每 10 分钟重试一轮：
```bash
nohup setsid bash experiments/run_all.sh >> run_state/run_all.log 2>&1 &
```
> ✅ 2026-09-28 已验证：flex-start 支持 a2-highgpu-1g。8 个 zone 都没货时，flex 在 us-east1-b 马上就建好了 VM。每次最多排队 2h（GCP 上限），没排到会自动重新排队。

## 2. 查看进度（随时可以看）
```bash
bash experiments/run_all.sh status
```
```bash
tail -n 30 run_state/run_all.log
```
确认脚本在运行（应该只有一行 `bash experiments/run_all.sh`）：
```bash
pgrep -af run_all.sh
```
日志里怎么看：
- `第 N 轮所有 zone 都没有容量` → 暂时没有 A100，在自动重试，不用管。一直这样可以切换到 flex-start。
- `SSH 就绪` → VM 已经建好，任务开始跑。
- `<步骤>: 结果 = done` 然后 `✅ 完成` → 这一步成功。
- `结果 = failed` 或 `结果 = lost` → VM 出了问题，脚本会自动新建 VM 续跑，不用管。
- `!!` 开头 → 脚本停下来了，需要人工处理（见下面的"如果脚本报错停止了"）。

## 停止脚本 / 切换启动方式（普通 ↔ flex-start）
停止要用下面这条。**不要用 `pkill -f run_all.sh`**：它只会杀掉主脚本，正在重试 zone 的子进程还会继续跑，而且会一直占着锁，导致下次启动报"已经在运行"。
```bash
bash experiments/run_all.sh stop
```
- **还没建好 VM 时**（日志最后是 `没有容量` 或 `排队`，还没出现 `SSH 就绪`）：停掉再用第 1 步里另一种命令启动，不会丢任何东西。
- **VM 已经在跑任务时**：停掉脚本不会影响 VM 上的任务。重新启动后，脚本会接着等这个任务，不会重复提交。
- 用 flex-start 排队时停掉脚本，GCP 那边可能还留着排队中的实例。停掉之后查一下，有 `bench-a100` 或 `dapt-a100` 却不需要的话，就用第二部分里的 `down` 命令删掉：
```bash
gcloud compute instances list --filter="name~a100"
```

## 3. E0 完成后：脚本会停下，等你判定
⛔ 日志最后一行会出现 `GATE`。看 `experiments/out/tables.md` 里的 T0，按 PAPER.md 判定：ViT 在多数检测集上不输 CNN，才继续。决定继续就批准，然后再执行一次第 1 步的启动命令（用哪种方式都可以）：
```bash
touch run_state/APPROVE_E1
```
```bash
FLEX=1 ZONES=us-east1-b nohup setsid bash experiments/run_all.sh >> run_state/run_all.log 2>&1 &
```

## 4. 全部完成
日志最后一行是 `全部完成`，结果在 `experiments/out/tables.md`，按 PAPER.md 的判定规则看 C1–C3。

## gcloud 登录过期（日志里出现 `!! gcloud 登录已过期`）
公司账号每隔一段时间就要重新登录（2026-09-28 实际遇到，大约 1 天一次）。登录过期时脚本会**暂停等你**：不消耗重试次数，也不会误以为 VM 丢了。VM 上的任务照常在跑。重新登录后，脚本在 5 分钟内自动继续：
```bash
gcloud auth login
```

## 如果脚本报错停止了
- `bash experiments/run_all.sh status` 可以看到停在哪一步，`run_state/run_all.log` 里有原因。
- 修好之后重新执行启动命令，会从停下的那一步继续。
- 如果某一步已经重试 6 次都失败，先删掉 `run_state/<步骤名>.attempts` 来重置计数，再启动。
- 脚本自带锁，重复启动会直接报"已经在运行"并退出。如果确定没有在跑却还是报这个，先执行 `bash experiments/run_all.sh stop`，再查一下是否还有残留进程：
```bash
pgrep -af run_all.sh
```

---

# 第二部分：手动逐步执行（备用）

> 🔁 表示这条要重复执行，直到出现注释里写的结果。⛔ 表示先检查，符合条件再往下走。
> 不要和 `run_all.sh` 同时用：两者用的是同一台 VM 和同一个桶路径。

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
- 每次最多排队 `FLEX_WAIT`（默认 2h，这是 GCP 允许的上限）。到时间还没排到，会过 `RETRY_MIN` 分钟后重新排队。
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
> ✅ 2026-09-28 已验证 flex-start 支持 a2-highgpu-1g。

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
