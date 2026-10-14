# records：重要运行的原始记录

每次重要运行留下的文本记录都在这里，共 11 MB，`git clone` 就有，复盘不需要下载任何大文件。
权重和语料在私有仓库里，每个文件的说明和取回方法见 [`docs/artifacts.md`](../docs/artifacts.md)。

目录结构和 `results/` 一致，每个运行都有 `runs/<运行 ID>/{meta.json, metrics.jsonl, summary.json}`，可以直接用 `analyze_runs.py` 读：

```bash
python analyze_runs.py list records/pretrain_v2 records/ablation_v1/*
python analyze_runs.py show records/pretrain_v2
python analyze_runs.py compare records/ablation_v1/* --metric loss --split val
python analyze_runs.py plot records/pretrain_v2 --out pretrain_v2.html
python analyze_runs.py compare records/posttrain_v1/*_s1 --metric accuracy --split val      # 四种 GRPO 变体在 dev 上的曲线
python analyze_grpo.py records/posttrain_v1/grpo_s1/grpo_metrics.jsonl records/posttrain_v1/dapo_lite_s1/grpo_metrics.jsonl --window 50
```

**脱敏**：消融（`ablation_v1/`）、Mac 上的后训练冒烟（`posttrain_mac_smoke/`）和 `posttrain_v2/` 跑在一台不便公开的电脑上，记录里的绝对路径改成了相对仓库根目录的路径，
主机名换成了 `redacted`，其余内容原样保留（以后的记录用 `python3 scripts/redact_records.py <目录>` 做这一步，它还会检查有没有漏网的）；未改动的原件在私有仓库的 `ablation_v1-records.tar.gz` 和 `posttrain_v1-records.tar.gz` 里。
其他目录的记录是在 AutoDL 实例上写的，没有改动。

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

## posttrain_v1/：第一轮后训练（2026-10-12，RTX 4090）

SFT 一次，再从同一个 SFT 权重跑 4 种 GRPO 变体 × 3 个种子，结果和分析在 `docs/experiments.md` 的 Ablation grid 一节，
权重在私有仓库的 `posttrain-v1` 组。

| 文件 | 是什么 |
|---|---|
| `run.log` | `scripts/autodl/posttrain.sh` 的完整日志：造数据（含 5 个题目文件的 sha256）、SFT、12 个 GRPO 跑次的训练日志和每次评测，最后是汇总表 |
| `gpu.csv` | 每 10 秒一次 `nvidia-smi`：GPU 利用率、显存、功耗，覆盖整个运行 |
| `SHA256SUMS` | 在服务器上算的 13 份权重的 SHA-256，用来核对取回的权重 |
| `sft/` | SFT 的 `runs/<运行 ID>/`（命令、参数、git commit、逐步 loss）、`dev.json` / `test.json`（`eval_env.py` 的打分）、`sft_final.config.json` |
| `<变体>_s<种子>/grpo_metrics.jsonl` | 每步一行训练指标（奖励、准确率、零方差组比例、熵、KL、grad norm、每步秒数），每 25 步一行 dev 评测；`analyze_grpo.py` 读它 |
| `<变体>_s<种子>/runs/<运行 ID>/` | 同一份指标的 `analyze_runs.py` 格式，加上 meta 和 summary |
| `<变体>_s<种子>/dev.json`、`test.json` | 最终策略在 dev 和 test 上的贪心准确率、按题型拆开的准确率、前 5 个样例 |
| `<变体>_s<种子>/grpo_final.config.json` | 模型结构，与私有仓库里权重旁边的那份相同 |
| `summarize.py` | 由上面这些文件重算 `docs/experiments.md` 里后训练的每个数字：每个跑次和每种变体的准确率（按题型拆开）、变体之间的 Welch t、训练动态的分段均值、dev 曲线、每次跑的耗时和显存。只用标准库：`python3 records/posttrain_v1/summarize.py` |

## posttrain_trial/：正式后训练之前的试跑（同一台 4090）

| 文件 | 是什么 |
|---|---|
| `run.log`、`gpu.csv` | `OUT=… STEPS=20 VARIANTS=grpo bash scripts/autodl/posttrain.sh` 的日志和 GPU 监控 |
| `sft/` | 试跑的 SFT，dev / test 和正式跑逐位相同（CUDA 上 SFT 可复现） |
| `grpo_s1/` | 20 步的 GRPO：dev 从 0.636 掉到 0.588（20 步的余弦调度第 2 步就到满学习率） |
| `probe_mac_config/`、`probe_mac_config.log` | 照 Mac 冒烟的设置跑的 100 步诊断，dev 升到 0.764，确认 CUDA 上没有问题 |

## autodl_4090/：4090 实例的环境

`environment/` 下：`tests.log`（实例上的全量单测，522 passed、1 skipped）、`pip_freeze.txt`、`versions.txt`（Python 3.12.3、torch 2.8.0+cu128、
transformers 5.19.0、Ubuntu 22.04、RTX 4090 24GB）。AutoDL 面板的监控数据库只在私有仓库的 `posttrain_v1-records.tar.gz` 里。

## posttrain_mac_smoke/：Mac 上的后训练冒烟（2026-10-12，Apple M5 Pro，已脱敏）

`docs/experiments.md` 的 Mac smoke 一节由这些记录得出。

| 目录 | 是什么 |
|---|---|
| `sft_e1/` | SFT 1 个 epoch（65 秒），dev 0.618 |
| `grpo_probe_lr1e5/` | 100 步 GRPO 探测（lr 1e-5），dev 0.618 → 0.778；`train.log` 是完整日志 |
| `grpo_smoke/` | 流程冒烟：30 步、每步 8 题 × 8 条，rollout 分桶之前，每步平均 5.6 秒 |
| `grpo_timing_b8/` | 同样 8 × 8，分桶之后：8 步，每步平均 3.2 秒 |
| `grpo_timing_b32/` | 每步 32 × 8 的两次：训练侧不分块的那次（`runs/*_0dd6ae`）把内存换页换满，7 步里每步 22–153 秒，中途停掉，所以没有 `summary.json`；`--micro_batch_size 32` 的那次（`runs/*_26bf7a`）4 步，每步 8.5–9.2 秒 |

## posttrain_v2/：DPO-only 基线（2026-10-12，Mac，已脱敏）

从 `posttrain-v1` 的 SFT 权重出发，只做 DPO，结果和分析在 `docs/experiments.md` 的 “DPO alone does not move accuracy” 一段。

| 文件 | 是什么 |
|---|---|
| `dpo/<跑次>/` | 6 个 DPO 跑次，每个有 `train.log`、`runs/<运行 ID>/`、`dev.json`（`eval_env.py` 的打分）；其中三个（`dpo_e1_lr1e5`、`dpo_e1_lr1e6`、`dpo_onpolicy_lr1e-6`）还有 `test.json`。`dpo_e1_lr1e5` 是 lr 1e-5 跑满一个 epoch，格式被打坏；`dpo_s20_lr1e5` / `dpo_s40_lr1e5` 是同样设置只更新 20 / 40 步；`dpo_e1_lr1e6` 用手工损坏的对；`dpo_onpolicy_lr1e-6` / `dpo_onpolicy_lr5e-6` 用 on-policy 的对 |
| `onpolicy_pairs.jsonl`、`.stats.json` | 训练用的 2022 对：`mine_pairs.py` 对 SFT 策略在 `grpo_prompts.jsonl` 的前 3000 道题上各采 8 条（温度 1.0），每个有对有错的题组取一对（对的作 chosen，错的作 rejected）。MPS 上采样不逐位可复现，重新挖出来的对会有出入，所以存了这份 |
| `heldout_pairs.jsonl` | 500 对手工损坏的留出对：`generate_data --split preference --n 1500 --seed 77`，去掉 dev / test 的题和训练对里出现过的 prompt、再去重后取前 500 |
| `onpolicy_heldout_pairs.jsonl`、`.stats.json` | 548 对 on-policy 的留出对：同样的挖法，题取 `grpo_prompts.jsonl` 的第 3001–3800 道，`--seed 5` |
| `pref_accuracy.py` | 给定检查点和一份偏好对，算 DPO 自己的准确率（对 SFT 参考的隐式奖励差）和不需要参考的原始排序准确率（chosen 的序列对数概率是否更高）：`PYTHONPATH=. python3 records/posttrain_v2/pref_accuracy.py --pairs <对> <检查点>...` |
| `pref_accuracy.json`、`pref_accuracy_onpolicy.json` | 上面的输出：各个 DPO 检查点在两份留出对上的结果（SFT 作参考行） |

## curriculum_v1/、curriculum_v1_600/、pyfunc_v3/：难度课程与带修复的 5 种子网格、pyfunc 混合 RL 池（2026-10-14，一张 RTX 5090）

`docs/experiments.md` 末两节由这些记录得出。每个跑次目录有 `dev.json`、`test.json`（`eval_env.py` 逐题）、`grpo_metrics.jsonl`、`runs/`；
`sft_dev.json` / `sft_test.json` 是起点（三位数加减法，SFT 权重是 `posttrain-v1-sft`）。权重没有存档（每个 0.4GB，实例已释放），`scripts/autodl/parallel_grid.sh` 可以重跑。

| 目录 | 是什么 |
|---|---|
| `curriculum_v1/` | 三位数加减法、300 步、全部带 `--truncated_reward 0`：`grpo`、`dr_grpo`、`dapo_lite`、`grpo_cur`、`dapo_lite_cur` 各 5 个种子。`python3 records/curriculum_v1/summarize.py` 算均值、种子标准差、零方差组比例、三位数层的采样占比、配对 McNemar |
| `curriculum_v1_600/` | 同样设置 600 步，只有 `grpo_cur`、`dapo_lite_cur` 的 3 个种子；基线取 `posttrain_v5`（有修复重跑的格子）和 `posttrain_v4`（没失控的格子）：`python3 records/curriculum_v1/summarize.py records/curriculum_v1_600` |
| `pyfunc_v3/` | pyfunc 第二轮，`sft/` 是起点（A 族 SFT + B 族各 3 条），`grpo_B_s*` 只用 B 族 prompt，`grpo_AB_s*` 把 40% 换成 A 族 prompt，200 步、3 个种子：`python3 records/pyfunc_v3/summarize.py` |
