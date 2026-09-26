# GCP 命令顺序（直接复制粘贴）

- 全部在 WSL 里执行，从上往下一条一条粘贴。
- 标 🔁 的命令要重复执行，直到出现注释里写的结果。
- 标 ⛔ 的地方先检查，符合条件再往下走。
- 实验设计见 PAPER.md。

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

## 1. 本地冒烟（约 3.5 小时）

```bash
cd /mnt/d/Video/domain_transfer
```
```bash
nohup setsid bash experiments/smoke.sh > smoke_out/smoke.log 2>&1 &
```
🔁 直到最后一行是 `smoke finished`：
```bash
cat smoke_out/SUMMARY.txt
```
⛔ 全部 PASS 才继续。

## 2. 上传帧缓存（直接用本地已处理好的帧，不用开 CPU 机）

本地 `frames_hires/train` 里已经有抽好的帧（86.8 万张）、运动能量图（61.7 万个 `.me.png`）和 `frame_weights.npy`，和本地冒烟、训练用的是同一份数据。直接同步到桶，中断后重新执行同一条命令会接着传：
```bash
cd /mnt/d/Video/domain_transfer
```
```bash
gcloud storage rsync -r frames_hires/train gs://mlflow-artifacts-ai-a100/dapt/cache/frames_hires/train
```
⛔ 检查帧权重已上传：
```bash
gcloud storage ls -l gs://mlflow-artifacts-ai-a100/dapt/cache/frames_hires/train/frame_weights.npy
```
⛔ 检查帧数应该是 868310：
```bash
gcloud storage ls "gs://mlflow-artifacts-ai-a100/dapt/cache/frames_hires/train/**" | grep -c "\.jpg$"
```
删掉旧的打包缓存（它缺运动图；训练 VM 会优先用上面的目录，删掉是为了避免混淆）：
```bash
gcloud storage rm gs://mlflow-artifacts-ai-a100/dapt/cache/frames_hires.tar
```
> 备用方案：`./scripts/dapt_prep.sh run` 会在 CPU 机上从原始视频重新生成打包缓存（数小时）。只有本地帧丢失时才需要。

## 3. E0：ViT 对 CNN

```bash
cd /mnt/d/Conceivable-SharedData01-23Jun2026
```
先在 VM 上冒烟，会问是否创建实例，输入 `y`：
```bash
BENCH_CMD="bash paper_jobs.sh E0 smoke" ./bench_gcp.sh start
```
🔁 直到出现 `BENCH DONE`：
```bash
./bench_gcp.sh status
```
```bash
./bench_gcp.sh fetch
```
⛔ 应该是 `[PASS] …runs OK`，并且没有 OOM：
```bash
tail -n 5 stage1_out/gcp_bench_logs/bench_run.log
```
正式运行：
```bash
BENCH_CMD="bash paper_jobs.sh E0 run" ./bench_gcp.sh start
```
🔁 直到出现 `BENCH DONE`：
```bash
./bench_gcp.sh status
```
取回结果并删除实例，会问确认，输入 `y`：
```bash
./bench_gcp.sh finish
```
```bash
cd /mnt/d/Video/domain_transfer && python experiments/paper.py tables
```
⛔ 看 `experiments/out/tables.md` 里的 T0：ViT 在多数检测集上不输 CNN，才继续第 4 步。

## 4. E1：训练 4 个 DAPT 臂

```bash
cd /mnt/d/Video/domain_transfer
```
先在 GCP 上冒烟（5 个 iter），会问是否创建实例，输入 `y`：
```bash
ARM=motion_weighted BUDGET=e1 DINO_EPOCH_LEN=5 DINO_EPOCHS=1 DINO_BATCH=8 ./scripts/dapt_train.sh start
```
🔁 直到出现 `DINOV3_DONE_motion_weighted_e1.txt`：
```bash
./scripts/dapt_train.sh status
```
```bash
./scripts/dapt_train.sh fetch
```
⛔ 应该能看到 `dinov3_vitb16_dapt_motion_weighted_e1_backbone.pth`：
```bash
ls -la gcp_outputs/
```

**uniform**（约 16–17 小时）：
```bash
ARM=uniform BUDGET=long ./scripts/dapt_train.sh start
```
🔁 直到出现 `DINOV3_DONE_uniform_long.txt`：
```bash
./scripts/dapt_train.sh status
```
取回结果并删除实例，会问确认，输入 `y`：
```bash
./scripts/dapt_train.sh finish
```
```bash
./scripts/ship_weights.sh gcp_outputs/dinov3_vitb16_dapt_uniform_long_backbone.pth /mnt/d/Conceivable-SharedData01-23Jun2026/stage1_out/dinov3_ckpt/dinov3_vitb16_dapt_uniform_long.pth "domain_transfer@$(git rev-parse --short HEAD), ARM=uniform BUDGET=long" "ViT-B/16, 20000 it, batch 48, A100"
```

**content**：
```bash
ARM=content BUDGET=long ./scripts/dapt_train.sh start
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

**motion**：
```bash
ARM=motion BUDGET=long ./scripts/dapt_train.sh start
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

**motion_weighted**：
```bash
ARM=motion_weighted BUDGET=long ./scripts/dapt_train.sh start
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

## 5. E2 + E3 + E4：benchmark

```bash
cd /mnt/d/Conceivable-SharedData01-23Jun2026
```
先在 VM 上冒烟：
```bash
BENCH_CMD="bash paper_jobs.sh E2 smoke && bash paper_jobs.sh E3 smoke && bash paper_jobs.sh E4 smoke" ./bench_gcp.sh start
```
🔁 直到出现 `BENCH DONE`：
```bash
./bench_gcp.sh status
```
```bash
./bench_gcp.sh fetch
```
⛔ 应该是 `[PASS]`：
```bash
tail -n 5 stage1_out/gcp_bench_logs/bench_run.log
```
正式运行（中断后重新执行同一条命令即可续跑）：
```bash
BENCH_CMD="bash paper_jobs.sh E2 run && bash paper_jobs.sh E3 run && bash paper_jobs.sh E4 run" ./bench_gcp.sh start
```
🔁 直到出现 `BENCH DONE`：
```bash
./bench_gcp.sh status
```
```bash
./bench_gcp.sh finish
```

## 6. 出表

```bash
cd /mnt/d/Video/domain_transfer && python experiments/paper.py status
```
```bash
python experiments/paper.py tables
```
结果在 `experiments/out/tables.md`，按 PAPER.md 的判定规则看 C1–C3。

---

## 出问题时

- **查看有哪些实例：**
```bash
gcloud compute instances list
```
- **登录训练 VM 看日志**（日志在 `~/dapt.log`）：
```bash
./scripts/dapt_train.sh ssh
```
- **登录 benchmark VM 看日志**（在 benchmark 仓库目录下执行；日志在 `~/bench.log`）：
```bash
./bench_gcp.sh ssh
```
- **CPU 机中途退出，删掉残留实例：**
```bash
./scripts/dapt_prep.sh down
```
- 每个任务跑完都会自动关机，关机后不计 GPU 费。`finish` 会先取回结果，再删除实例。
