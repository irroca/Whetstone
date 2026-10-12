# records：重要运行的原始记录

每次重要运行留下的文本记录都在这里，共 1.6 MB，`git clone` 就有，复盘不需要下载任何大文件。
权重和语料在私有仓库里，每个文件的说明和取回方法见 [`docs/artifacts.md`](../docs/artifacts.md)。

目录结构和 `results/` 一致，每个运行都有 `runs/<运行 ID>/{meta.json, metrics.jsonl, summary.json}`，可以直接用 `analyze_runs.py` 读：

```bash
python analyze_runs.py list records/pretrain_v2 records/ablation_v1/*
python analyze_runs.py show records/pretrain_v2
python analyze_runs.py compare records/ablation_v1/* --metric loss --split val
python analyze_runs.py plot records/pretrain_v2 --out pretrain_v2.html
```

**脱敏**：消融（`ablation_v1/`）跑在一台不便公开的电脑上，记录里的绝对路径改成了相对仓库根目录的路径，主机名换成了 `redacted`，
其余内容原样保留；未改动的原件在私有仓库的 `ablation_v1-records.tar.gz` 里。其他目录的记录是在 AutoDL 实例上写的，没有改动。

## pretrain_v2/：正式预训练（2026-10-11，H800 PCIe）

| 文件 | 是什么 |
|---|---|
| `runs/pretrain_20261011-141555_c21027/meta.json` | 启动命令、全部参数、git commit（`e223385`）、主机、输入数据文件的大小和指纹 |
| `runs/pretrain_20261011-141555_c21027/metrics.jsonl` | 每 10 次更新一行训练指标（loss、学习率、grad norm、吞吐），每 500 次更新一行验证 loss，共 1931 行 |
| `runs/pretrain_20261011-141555_c21027/summary.json` | 最终结果：总 token、耗时、平均吞吐、最终验证 loss |
| `train.log` | 训练进程的完整日志 |
| `probes.json` | 三个探针：留出集上各来源的 bits per byte、中英混杂率、1 位和 2 位加法的正确率 |
| `samples.json` | 固定随机种子的生成样例（中英文续写、代码、解方程、加法），`docs/experiments.md` 引用了它 |
| `pretrain_final.config.json` | 模型结构，与私有仓库里权重旁边的那份相同 |
| `SHA256SUMS` | 在服务器上算的 `pretrain_final.pth` 和 `epoch_1_checkpoint.pth` 的 SHA-256，用来核对取回的权重 |

## autodl_h800/：H800 实例上的吞吐优化、试跑和环境

| 文件 | 是什么 |
|---|---|
| `bench.json`、`bench.log` | 优化前的基线：`bench_train.py` 在正式模型上扫 batch size 和 compile（tok/s、峰值显存、MFU） |
| `bench_ablation/<commit>.json`、`bench_ablation.log` | 每项吞吐优化单独一个 commit，逐个测出来的吞吐，对应 `docs/status.md` §3.3 里每项优化的收益；`final.json` 是最终代码上 batch 16/32/64 的结果 |
| `bench_ablation/coord_descent.json`、`max_autotune.json`、`inductor_bench.log` | 两个 Inductor 调优选项的对比：收益在测量误差以内，都没有采用 |
| `bench_ablation/trial_after_metrics.jsonl` | 优化后的代码上试跑的逐步指标，用来和优化前对比 loss |
| `trial.log` | 试跑一：跑到第一个断点后杀掉进程再续训，检查断点续训 |
| `trial2.log` | 试跑二：优化后的代码与优化前的 loss 是否一致 |
| `gpu_monitor.csv` | AutoDL 面板每 5 秒一次的采样：GPU 利用率、显存、CPU、内存，覆盖正式训练及前后各 10 分钟；原始 SQLite 数据库在私有仓库 |
| `environment/` | 实例环境：`setup.log`（安装）、`preflight.log`（无卡模式下的 CPU 预检）、`tests.log`（单测）、`finish.log`（语料上传后在实例上的校验）、`pip_freeze.txt`（Python 3.12.3、Ubuntu 22.04 上的包版本） |
| `scripts/` | 当时在实例上临时写的脚本，`docs/status.md` §3.3 的数字由它们得出，见下表 |

| 脚本 | 做什么 |
|---|---|
| `ablation_bench.sh` | 依次切到每个优化 commit 跑 `bench_train.py`，生成 `bench_ablation/<commit>.json` 和 `final.json` |
| `inductor_bench.sh` | 打开 Inductor 的 coordinate descent tuning 和 max autotune 各跑一次 |
| `profile_step.py` | profile 一次编译后的完整更新，看 GPU 时间花在哪些内核上 |
| `attn_bench.py` | 在正式运行的注意力形状上比较各个 SDPA 后端（bf16 前向加反向），并核对 cuDNN 与 flash 的数值 |
| `sdpa_select.py` | 哪些“优先用 cuDNN 注意力”的写法在 `torch.compile` 之后仍然生效 |
| `explain.py` | 用 cuDNN 后端顺序时，编译后的前向加 loss 是否还是一整张图 |
| `trial_kill_resume.sh` | 试跑一的杀进程和续训 |
| `run_rows.py`、`run_health.py` | 正式训练中看进度：区间吞吐、loss、ETA；以及 loss 尖峰、非有限值、吞吐波动的检查 |
| `train_val_gap.py` | 每次验证前 500 次更新的平均训练 loss 与该次验证 loss 的差距 |
| `bench_rows.py`、`trial_rows.py` | 把基准 JSON 和试跑指标打印成表 |

## ablation_v1/：数据配比消融（2026-10-03～04，Apple M5 Pro）

| 文件 | 是什么 |
|---|---|
| `report.md`、`report.json` | 6 个臂的汇总表：val loss、各来源 bpb、中英混杂率、加法正确率 |
| `<臂>/runs/<运行 ID>/` | 每个臂的 meta、逐步指标和 summary（已脱敏） |
| `<臂>/probes.json`、`<臂>/pretrain_final.config.json` | 每个臂的探针结果和模型结构 |
| `run.log`、`run.sh` | `run_ablation.py` 全流程的日志和启动脚本（已脱敏） |

臂的命名规则、各自的结构和参数量见 `docs/artifacts.md` 的 ablation-v1 一节。

## mixture_v2/：正式预训练语料的构建报告（不含数据本身）

| 文件 | 是什么 |
|---|---|
| `manifest.json` | `datatools.prepare` 的总报告：每个来源的文档数、token 数、过滤和去重、去污染结果（含删掉文档最多的评测题） |
| `sources/*.jsonl.done.json` | 每个来源的拉取报告：参数、预算、实际拉到的量、分词器指纹 |
| `train.meta.json`、`val.meta.json`、`holdout.meta.json` | 三个划分分词后的 token 数、文档数和各来源的量 |
| `SHA256SUMS` | 分词后文件（`train.bin`、`val.bin`、`holdout.jsonl` 等）的 SHA-256，取回语料后用来核对 |
