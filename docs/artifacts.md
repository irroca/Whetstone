# 大文件存放说明：权重、语料和原始记录

git 里只放代码、文档和小体积的训练记录（`records/`，每个文件的说明见 [`records/README.md`](../records/README.md)）。
模型权重、分词后的语料和未脱敏的原始记录太大或不宜公开，统一放在两个**私有**仓库里。两边内容完全相同，每个文件都有 SHA-256：

| 存储 | 位置 | 什么时候用 |
|---|---|---|
| 魔搭 ModelScope 私有模型仓库 | `irroca/whetstone-artifacts`，每组一个文件夹 | 国内（包括 AutoDL 实例）默认用它，下载快 |
| GitHub 私有仓库的 Release | `irroca/whetstone-artifacts`，每组一个 Release，tag 就是组名 | 主存档；海外网络或挂代理时用。国内直连实测只有 70–150 KB/s |

每组文件旁边都有一份 `README.md`（内容就是本文对应的一节）和 `SHA256SUMS`，单独下载下来也分得清。

## 怎么取回

```bash
pip install modelscope && modelscope login                     # 默认来源是魔搭；令牌在 modelscope.cn 的“访问令牌”页
scripts/fetch_artifacts.sh pretrain-v2 mixture-v2-eval         # 从头做后训练需要的全部：基座模型 + 验证集、留出集、评测集（约 1 GB）
scripts/fetch_artifacts.sh posttrain-v1-sft                    # 接着做后训练（DPO、蒸馏、重跑 GRPO）的起点：SFT 权重 + 记录和题目（0.4 GB）
scripts/fetch_artifacts.sh posttrain-v1                        # 再加上 12 个 GRPO 策略（共 5.2 GB）
scripts/fetch_artifacts.sh pretrain-v2 mixture-v2 ablation-v1 posttrain-v1  # 全部（约 19 GB）
scripts/fetch_artifacts.sh --source github pretrain-v2         # 改从 GitHub 拉，需要先 gh auth login
scripts/fetch_artifacts.sh --with-optimizer pretrain-v2        # 连优化器断点一起拉（1.2 GB），只有接着预训练才需要
```

脚本先下载到 `results/artifacts/<组名>/`，逐个核对 SHA-256，再把文件放到训练脚本和文档预期的位置（下面各表的“放回到”一列）。
`mixture-v2` 会自动把分段拼起来、解压出 `train.bin` 并再校验一次。中断了就重跑同一条命令：已经下完的文件不会重下，
GitHub 那边下了一半的文件也会接着下。缺文件或校验不符时脚本会列出来并退出，这一组不会放回去。

## 总览

| 组 | 大小 | 内容 | 后训练需要吗 |
|---|---|---|---|
| `pretrain-v2` | 0.4 GB，加优化器断点 1.6 GB | 正式预训练出的 100M 基座模型，以及这次运行的原始记录 | **需要**，它是 SFT 的起点 |
| `mixture-v2` | 12.9 GB | 分词后的正式预训练语料，以及验证集、留出集、评测集 | 只用到验证集、留出集和评测集（0.6 GB），取回时写 `mixture-v2-eval` 就只拉这部分 |
| `ablation-v1` | 1.0 GB | 6 个数据配比消融小模型，以及它们的分词器和记录 | 不需要；可以拿来快速调试流程 |
| `posttrain-v1` | 5.2 GB | 第一轮后训练：SFT 冷启动的权重、12 个 GRPO 策略（4 种变体 × 3 个种子），以及记录和题目 | 接着做 DPO、蒸馏或重跑 GRPO 都从这里的 SFT 权重出发，取回时写 `posttrain-v1-sft` 就只拉它和记录（0.4 GB） |

来龙去脉：

```
configs/ablation_v1.json --run_ablation.py--> ablation-v1 --定下中文占比和词表--> configs/mixture_v2.json, tokenizer/v1_32k
configs/mixture_v2.json --datatools.prepare + tokenize_corpus--> mixture-v2
mixture-v2/train.bin --pretrain.py（H800，8.3 小时）--> pretrain-v2/pretrain_final.pth
pretrain-v2/pretrain_final.pth --sft.py--> posttrain-v1/sft --grpo.py × 12（RTX 4090，3.7 小时）--> posttrain-v1/<变体>_s<种子>
```

## 命名约定

- 组名是 `<阶段>-v<N>`，和本地目录 `results/<阶段>_v<N>/`、git 里的 `records/<阶段>_v<N>/` 一一对应。
- 一组里有多个模型时，文件名是 `<跑次>.<原文件名>`（如 `grpo_s1.grpo_final.pth`），取回时放回 `results/<阶段>_v<N>/<跑次>/<原文件名>`。
- 以后的产物照此办理（下一轮后训练是 `posttrain-v2`）：在本文加一节同样格式的说明，写明起点权重（哪一组的哪个文件）、训练数据、commit 和结果，
  再给 `scripts/fetch_artifacts.sh` 加上这一组。
- 上传：文件、`README.md`（由本文那一节生成）和 `SHA256SUMS` 先放进 `results/artifacts/<组名>/`。魔搭用
  `modelscope upload irroca/whetstone-artifacts results/artifacts/<组名> <组名> --repo-type model`；GitHub 用
  `scripts/upload_github_release.sh <组名> "<标题>"`，已传完的文件会跳过，重跑到输出 `GITHUB OK` 为止。两边都传完后，
  用 `fetch_artifacts.sh` 从魔搭取回核对一遍。

---

## pretrain-v2：正式预训练的 100M 基座模型

2026-10-11 在一张 H800 PCIe 上完成的正式预训练：运行 ID `pretrain_20261011-141555_c21027`，代码 commit `e223385`。

- **模型**：99.5M 参数（嵌入 25.2M，输入输出共享；其余 74.3M），`dim 768 / 12 层 / 12 个注意力头 / 3 个 KV 头 / FFN 2048 / RoPE θ=1e6 / 上下文 2048`，
  分词器 `tokenizer/v1_32k`（词表 32768，在 git 里）。
- **训练**：`mixture-v2` 的 `train.bin`，9.91B token，1 个 epoch，8.3 小时；batch 32 × 累积 8 × 2048 token，峰值学习率 6e-4。
- **结果**：最终验证集 loss 2.5539（ppl 12.86）。曲线、探针和生成样例的分析见 Whetstone 仓库的 `docs/experiments.md` 和 `records/pretrain_v2/`。

| 文件 | 大小 | 是什么 | 作用 / 谁会读它 | 什么时候需要 | 放回到 |
|---|---|---|---|---|---|
| `pretrain_final.pth` | 398 MB | 训练结束时的模型权重（fp32 `state_dict`，只有权重） | 后训练的起点：`sft.py --pretrained_path`；`chat.py --model_mode 0`、`probes.py`、`eval_ppl.py` 也读它 | **后训练和评测都要** | `results/pretrain_v2/pretrain_final.pth` |
| `pretrain_final.config.json` | 225 B | 模型结构（dim、层数、头数、词表大小等） | `resolve_model_config` 靠它恢复结构，尤其是无法从权重形状推出的 `n_heads` | 永远和 `.pth` 放在一起 | `results/pretrain_v2/` |
| `epoch_1_checkpoint.pth` | 1.19 GB | 训练结束时的完整断点：同一份权重，加上 AdamW 每个参数的一阶、二阶矩，随机数状态，进度（epoch、step、global_step）和结构配置 | `pretrain.py --resume_from` 用它接着训 | **只有**在这次预训练的基础上继续预训练时才要，后训练用不到 | `results/pretrain_v2/` |
| `pretrain_v2-records.tar.gz` | 0.5 MB | 这次运行的全部原始记录，未脱敏（内容见下） | 复盘、画图、写博客 | 复盘时 | 在仓库根目录解压，得到 `results/pretrain_v2/` 和 `results/autodl_h800/` |
| `SHA256SUMS` | — | 以上每个文件的 SHA-256 | `fetch_artifacts.sh` 下载后逐个核对 | 自动使用 | — |
| `README.md` | — | 本说明 | — | — | — |

`pretrain_v2-records.tar.gz` 里有：

- `results/pretrain_v2/`：`train.log`（完整训练日志）；`runs/pretrain_20261011-141555_c21027/` 下的 `meta.json`（命令、全部参数、git commit、
  数据文件指纹）、`metrics.jsonl`（每 10 次更新一行训练指标，每 500 次更新一行验证指标，共 1931 行）、`summary.json`（最终结果）；
  `probes.json`（三个探针：各来源 bits per byte、中英混杂率、加法正确率）；`samples.json`（固定种子的生成样例）；`loss_curve.png`；
  `report.html`（`analyze_runs.py plot` 的输出）；`pretrain_final.config.json`。
- `results/autodl_h800/`：吞吐基准（`bench.json` / `bench.log`、逐 commit 的 `bench_ablation/`、`inductor_bench.log`）；断点续训和 loss 等价性
  试跑（`trial.log`、`trial2.log`）；当时写的分析脚本（`server_scripts/`，如 `profile_step.py`）；实例环境（`server_logs/`：安装、preflight、
  测试日志和 `pip_freeze.txt`）；AutoDL 面板每 5 秒一次的 GPU 利用率和显存监控（`autopanel/autopanel.monitor.db`，SQLite）；
  服务器上算的权重校验和 `pth.sha256`。

没有保留：`latest_checkpoint.pth`，第 18500 次更新时的中间断点，已被 `epoch_1_checkpoint.pth` 取代。

## mixture-v2：正式预训练语料（已分词）

按 `configs/mixture_v2.json` 用 `datatools.prepare` 构建（2026-10-09～10），再用 `datatools.tokenize_corpus` 以 `tokenizer/v1_32k` 编码。
正式预训练（`pretrain-v2`）读的就是这里的 `train.bin`。

格式：每篇文档存成 `bos + 正文 + eos`，token 是 uint16；`.idx` 是每篇文档的起始偏移（uint64）；`.meta.json` 记录 token 数、文档数、
各来源的量和分词器指纹 `00853aafd17a84f5716c704484c56273`。分词器对不上时 `MemmapPretrainDataset` 会拒绝读取。

| 划分 | 文档数 | token 数 | 用途 |
|---|---|---|---|
| train | 8,393,795 | 9.914B | 预训练 |
| val | 42,451 | 50.3M | 训练中算验证 loss（`pretrain.py --val_data_path`）；以后也能用来看后训练有没有让语言能力退化 |
| holdout | 42,680 | 48.9M | 从未参与训练或调参；`probes.py` 读 `holdout.jsonl` 算各来源 bits per byte 和中英混杂率 |

train 中各来源的 token 数：en_web 3.17B、code 2.27B、zh_web 1.98B、math 1.58B、books 0.56B、synthetic_textbook 0.34B。

| 文件 | 大小 | 是什么 | 作用 / 什么时候需要 | 放回到 |
|---|---|---|---|---|
| `train.bin.zst.part-aa` … `part-af` | 6 个，共 12.3 GB | `train.bin`（19.8 GB）经 zstd 压缩后切成的 6 段（GitHub 单文件上限 2 GiB） | 只有重新预训练或继续预训练才要。还原：`cat train.bin.zst.part-* \| zstd -d -o train.bin`，取回脚本会自动做 | 解压成 `datasets/mixture_v2/train.bin` |
| `train.bin.sha256` | 76 B | 解压后 `train.bin` 的 SHA-256 | 校验还原结果 | — |
| `train.idx`、`train.meta.json` | 67 MB、1 KB | train 的文档偏移和元信息 | 和 `train.bin` 一起用 | `datasets/mixture_v2/` |
| `val.bin`、`val.idx`、`val.meta.json` | 101 MB | 验证集（已分词） | 算验证 loss | `datasets/mixture_v2/` |
| `val.jsonl` | 195 MB | 同一个验证集的原文 | 需要原文时，例如换分词器重新编码 | `datasets/mixture_v2/` |
| `holdout.bin`、`holdout.idx`、`holdout.meta.json` | 98 MB | 留出集（已分词） | 留出集上的 loss | `datasets/mixture_v2/` |
| `holdout.jsonl` | 190 MB | 留出集原文，每条带 `source` 字段 | **`probes.py` 必需**，评测后训练模型时也要 | `datasets/mixture_v2/` |
| `manifest.json` | 6 KB | `prepare` 的总报告：每个来源拉了多少、过滤掉多少，去重和去污染（decontamination）的结果 | 复盘数据处理 | `datasets/mixture_v2/` |
| `mixture_v2-support.tar.gz` | 3.5 MB | 三样东西，见下 | 做 GRPO 评测时要用其中的评测集 | 在仓库根目录解压 |
| `SHA256SUMS`、`README.md` | — | 校验和、本说明 | — | — |

`mixture_v2-support.tar.gz` 里有：

1. `datasets/eval/`：7 个评测集，由 `datatools.fetch_evals` 统一成 `{"question","answer"}` 格式：GSM8K 1319 题、MATH-500 500、HumanEval 164、
   MBPP 500、MMLU 14042、TAL-SCQ5K 中文 2000 / 英文 1978。它们既是构建语料时去污染的比对对象，也是 `grpo.py --eval_path` 的评测集。
2. `datasets/mixture_v2/sources/*.jsonl.done.json`：每个来源的拉取报告（参数、预算、实际拉到的量），以及原始的 `SHA256SUMS`。
3. `results/data_v2/`：构建和上传语料过程的原始日志与脚本，未脱敏。

没有保留：未分词的 `train.jsonl`（38 GB）和各来源的原始 JSONL（36 GB）。需要时用 `python3 -m datatools.prepare configs/mixture_v2.json` 重新拉取，
约一天，其中 starcoderdata 需要 Hugging Face 授权。

## ablation-v1：6 个数据配比消融小模型

2026-10-03～04 在 Apple M5 Pro 上跑的数据消融（`configs/ablation_v1.json`，`run_ablation.py`），用来决定正式预训练的中文占比和词表大小。
结论（中文 20%、词表 32k）和完整结果见 Whetstone 仓库的 `docs/status.md` §4 步骤 4 和 `records/ablation_v1/report.md`。

六个模型结构相同：`dim 512 / 8 层 / 8 个头（没有 GQA）/ FFN 1408 / 上下文 512`，各训练 3 亿 token、1 个 epoch，只有数据配比和词表不同。
文档里叫它们“29M 代理模型”：29M 是这个结构配 6400 词表时的参数量，配 32k 词表时是 42.5M（非嵌入部分都是 25.7M）。

命名 `zhXX_vNNk`：`zhXX` 是训练数据里中文网页的占比，`vNNk` 是分词器的词表大小。

| 臂 | 中文占比 | 词表 | 参数量 | val loss | 综合 bpb | 备注 |
|---|---|---|---|---|---|---|
| `zh00_v32k` | 0% | 32k | 42.5M | 4.656 | 1.658 | 不含中文的对照组 |
| `zh15_v32k` | 15% | 32k | 42.5M | 3.500 | 1.296 | |
| `zh30_v32k` | 30% | 32k | 42.5M | 3.455 | 1.287 | 词表对比的基准 |
| `zh45_v32k` | 45% | 32k | 42.5M | 3.452 | 1.290 | |
| `zh30_v16k` | 30% | 16k | 34.1M | 3.237 | 1.312 | 小词表 |
| `zh30_v48k` | 30% | 48k | 50.9M | 3.508 | 1.277 | 大词表 |

val loss 只能在同一词表内比较；跨词表看 bpb（留出集上每个 UTF-8 字节的 bit 数，越低越好）。

| 文件 | 大小 | 是什么 | 作用 / 什么时候需要 | 放回到 |
|---|---|---|---|---|
| `<臂>.pretrain_final.pth` | 6 个，136–203 MB | 每个臂训练结束时的权重（fp32） | 复现消融的探针；也可以当作小模型，几分钟就能跑完一轮，用来快速调试 SFT / GRPO 流程 | `results/ablation_v1/<臂>/pretrain_final.pth` |
| `<臂>.pretrain_final.config.json` | 6 个 | 对应的模型结构 | 和权重放在一起 | `results/ablation_v1/<臂>/pretrain_final.config.json` |
| `ablation_v1-support.tar.gz` | 22 MB | 使用和评测这些权重所需的东西，见下 | 加载 v16k / v48k 两个臂，或重算任何一个臂的 val loss、探针时 | 在仓库根目录解压，得到 `datasets/ablation_v1/` |
| `ablation_v1-records.tar.gz` | 0.2 MB | 原始记录，未脱敏：`report.md` / `report.json`（汇总表），每个臂的 `runs/`（命令、参数、逐步指标）和 `probes.json`，整个消融的 `run.log` / `run.sh` | 复盘 | 在仓库根目录解压，得到 `results/ablation_v1/` |
| `SHA256SUMS`、`README.md` | — | 校验和、本说明 | — | — |

`ablation_v1-support.tar.gz` 里有：三个消融分词器 `tokenizers/v16k`、`v32k`、`v48k`（v32k 和 git 里的 `tokenizer/v1_32k` 完全相同；v16k 和 v48k
只存在于这里，加载那两个臂时用 `--tokenizer_path datasets/ablation_v1/tokenizers/v16k` 之类的参数）；按三种词表分别编码的验证集
`val_v16k` / `val_v32k` / `val_v48k`（`.bin` / `.idx` / `.meta.json`）；数据池的 `pool_spec.json`、`pool/manifest.json`、`pool/val.jsonl`，以及
探针用的 `pool/holdout.jsonl`。

没有保留：每个臂的 `epoch_1_checkpoint.pth` / `latest_checkpoint.pth`（带优化器状态，各 0.4–0.6 GB，消融已经结束，不会再续训）；
各臂的训练数据 `arms/*/train.bin` 和数据池的 `pool/train.jsonl`，`run_ablation.py` 能从数据源重新生成。

## posttrain-v1：SFT 冷启动和 12 个 GRPO 策略

2026-10-12 在 AutoDL 的一张 RTX 4090 24GB 上跑的第一轮后训练：`scripts/autodl/posttrain.sh`，代码 commit `58c7bb0`，15:55–19:39，共 3.7 小时。

- **起点**：`pretrain-v2` 的 `pretrain_final.pth`。13 个模型的结构都和它相同（99.5M 参数，分词器 `tokenizer/v1_32k`）。
- **数据**：一两位数加减法，`scripts/arith_data.sh` 用固定种子生成（就是下面的 `posttrain_v1-data.tar.gz`）。test 500 题只用来报最终成绩，
  dev 500 题在训练中看、用来选模型；两者都不进 SFT、DPO 和 GRPO 的数据。
- **SFT**：4000 条、1 个 epoch（125 步）、batch 32、lr 1e-4、`--max_seq_len 128`，训练 13 秒。
- **GRPO**：12 个跑次都从这一个 SFT 权重出发，4 种变体 × 3 个种子，各 500 步：每步 32 题 × 8 条（temperature 1.0），lr 1e-5（10% 预热，
  余弦退到 1e-6），micro-batch 64，每 25 步在 dev 上评一次。每个跑次 16–20 分钟。
- **结果**（贪心准确率）：SFT 的 test 0.628、dev 0.636；12 个 GRPO 策略的 test 在 0.938–0.962 之间（平均 0.948），dev 0.962–0.982。
  四种变体之间的差别不比种子之间大。分析见 Whetstone 仓库的 `docs/experiments.md` 和 `records/posttrain_v1/`。

跑次的命名是 `<变体>_s<种子>`：

| 变体 | `grpo.py` 的参数 | test（种子 1 / 2 / 3） | dev（种子 1 / 2 / 3） |
|---|---|---|---|
| `grpo` | 默认：`seq_mean`、优势除以组内标准差、不加 KL | 0.954 / 0.944 / 0.944 | 0.972 / 0.968 / 0.978 |
| `grpo_kl` | `--kl_coeff 0.02` | 0.946 / 0.938 / 0.942 | 0.976 / 0.966 / 0.974 |
| `dr_grpo` | `--normalize_advantage_std False --aggregation dr_grpo` | 0.948 / 0.954 / 0.962 | 0.962 / 0.968 / 0.978 |
| `dapo_lite` | `--filter_zero_variance True --clip_eps_high 0.28 --aggregation token_mean` | 0.950 / 0.954 / 0.944 | 0.982 / 0.972 / 0.978 |

dev 上最高的是 `dapo_lite_s1`（0.982，test 0.950）。按“用 dev 选模型”的约定，要拿一个 GRPO 策略往下做（比如当蒸馏的教师）就用它。

| 文件 | 大小 | 是什么 | 作用 / 什么时候需要 | 放回到 |
|---|---|---|---|---|
| `sft.sft_final.pth` | 398 MB | SFT 冷启动后的权重（fp32 `state_dict`） | 12 个 GRPO 跑次的共同起点；以后做 DPO-only 基线（`dpo.py --policy_path`）、重跑 GRPO 也从它开始；`eval_env.py`、`chat.py --model_mode 1` 也读它 | `results/posttrain_v1/sft/sft_final.pth` |
| `<跑次>.grpo_final.pth` | 12 个，各 398 MB | 每个 GRPO 跑次第 500 步的权重 | 评测、比较变体、往下做的起点；`chat.py --model_mode 4` | `results/posttrain_v1/<跑次>/grpo_final.pth` |
| `sft.sft_final.config.json`、`<跑次>.grpo_final.config.json` | 13 个 | 对应的模型结构 | 永远和权重放在一起 | 和权重同一个目录 |
| `posttrain_v1-records.tar.gz` | 1.3 MB | 原始记录，未脱敏（内容见下） | 复盘 | 在仓库根目录解压 |
| `posttrain_v1-data.tar.gz` | 0.3 MB | 这一轮用到的 5 个题目文件 | 复现或重新评测。`scripts/arith_data.sh` 能逐字节重新生成，这份是兜底 | 在仓库根目录解压，得到 `datasets/arith_v1/` |
| `SHA256SUMS`、`README.md` | — | 校验和、本说明 | — | — |

`posttrain_v1-data.tar.gz` 里的 5 个文件：`test.jsonl` 500 题、`dev.jsonl` 500 题、`sft.jsonl` 4000 条、`preference.jsonl` 2000 对（给 DPO 的，
这一轮没用）、`grpo_prompts.jsonl` 2 万道题（GRPO 每步从中抽 32 道）。sha256 的前 16 位：test `174cc7c11019914c`、dev `e855599f44e9d965`、
sft `daebf99e1448c8e3`、preference `234dedcdc995007a`、grpo_prompts `59134139878d71a2`。

`posttrain_v1-records.tar.gz` 里有：

- `results/posttrain_v1/`：`run.log`（`posttrain.sh` 的完整日志：造数据、SFT、12 个 GRPO 跑次、每次评测和最后的汇总表）；`gpu.csv`
  （每 10 秒一次 `nvidia-smi`：GPU 利用率、显存、功耗）；每个跑次的 `dev.json` / `test.json`（`eval_env.py` 的输出：准确率、按题型拆开的
  准确率、前 5 个样例）、`grpo_metrics.jsonl`（每步一行训练指标，每 25 步一行 dev 评测）和 `runs/<运行 ID>/`（命令、全部参数、git commit、
  逐步指标）；SFT 的同样几样。
- `results/posttrain_trial/`：正式跑之前在同一台 4090 上的试跑：SFT 加 20 步 GRPO（`grpo_s1/`），以及照 Mac 冒烟的配置跑的 100 步诊断
  （`probe_mac_config/`），连同日志和 GPU 监控。
- `results/autodl_4090/`：实例环境（`environment/`：单测日志 `tests.log`、`pip_freeze.txt`，以及 Python、torch、CUDA 的版本）和 AutoDL
  面板的监控数据库（`autopanel/autopanel.monitor.db`，SQLite）。
- `results_smoke/`：Mac 上的冒烟记录（SFT、100 步 GRPO 探测、测吞吐的几次短跑，还有 10-03 的一次预训练冒烟），未脱敏；git 里的
  `records/posttrain_mac_smoke/` 是其中后训练部分脱敏后的副本。

没有保留：SFT 的 `epoch_1_checkpoint.pth`（1.19 GB，带优化器状态；SFT 只要十几秒，重训比续训简单）；试跑和 Mac 冒烟的权重（都被这一组取代）。
