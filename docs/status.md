# 项目状态与路线（交接文档）

**最后更新**：2026-10-09，本地 Mac session。
本文档是新 session 的入口——先读这里，再读 `AGENTS.md`。

---

## 0. 一句话现状

算法链路（五阶段后训练）和数据管线都已实现，CPU 单测覆盖；语料方案和模型规模已定。
**消融 #1（中文占比）和 #5（词表大小）已跑完**（§4 步骤 4）：定为中文 20%、32k 词表、10B token，
词表就用消融里评测过的那个（`tokenizer/v1_32k`），配比是 `configs/mixture_v2.json`。

**正式数据集正在本机生成**（§4 步骤 5，10-09 20:49 重新开始，预计 10-10 03:00 前后出完）。训练代码已经为租卡
准备好：四个阶段共用一个训练循环，续训与不中断逐位一致，checkpoint 原子写入，有 `bench_train.py`
在开卡第一个小时量吞吐和显存、`probes.py` 评测任意 checkpoint（§3.4）。下一步是租卡正式预训练，
完整计划在 §4 步骤 6。

仓库里现有的一切数字都是 29M 玩具规模的**实现验证**，或 3 亿 token 消融代理模型的**相对比较**，
不是能力声明。

---

## 1. 已完成，可以依赖的部分

### 模型与训练
- ~29M Llama 风格解码器（RoPE / RMSNorm / SwiGLU / 可选 GQA / 权重共享），架构**全部走 CLI**
  （`--dim` / `--n_layers` / `--n_heads` / `--n_kv_heads` / ...），不需要改源码做尺寸消融
- 五个阶段都能跑通：`pretrain.py` → `sft.py` → `distill.py` → `dpo.py` → `grpo.py`
- 架构解析优先级 **显式 CLI > checkpoint 记录 > 库默认值**；`*_final.pth` 旁边写
  `*.config.json` sidecar（因为 `n_heads` 无法从张量形状反推）
- 真正的跨尺寸蒸馏（teacher / student 各自从自己的 checkpoint 解析架构）
- 注意力走 `F.scaled_dot_product_attention`（MPS 除外，见 §3.2），显式 score 矩阵的实现留作对照

### RLVR / GRPO（[PR #4](https://github.com/irroca/Whetstone/pull/4)）
- `envs/`：可验证奖励环境，accuracy 与 format 双分量分开上报
- `losses.py`：组相对优势、clipped policy loss、k3 KL；Dr. GRPO / DAPO 的改动都是**开关**
- `rollout.py`：分组 rollout、逐行 completion mask、log-prob 重算
- `analyze_grpo.py`：窗口平均 + 多组并排比较

### 数据管线（[PR #5](https://github.com/irroca/Whetstone/pull/5)）
- `datatools.prepare`：按配比 spec 跑 拉取 → 过滤 → 精确去重 → 去污染 → 划分 → manifest，全程流式
- `datatools.stats` / `filters` / `dedup`（MinHash）/ `decontaminate`（13-gram + LCS）/ `split` /
  `tokenizer_stats` / `fetch_evals`
- `train_tokenizer.py` 已改成 CLI 驱动
- `datatools.tokenize_corpus` + `dataset.MemmapPretrainDataset`：预 tokenize 成 `uint16` 的 `.bin`，
  `pretrain.py` 按扩展名自动选读取器（§3.1）
- `prepare --probe`、`where`（按上游元数据过滤）、`hf.columns`（只下载需要的 parquet 列）、
  `cleaners`（过滤前改写文本，目前用来删 StarCoder 元数据行）、输出记录带 `source` 字段、
  manifest 里报告命中最多的评测项（§4 步骤 3 的冒烟里加的）
- `fetch_evals` 有七个去污染集：GSM8K / MATH-500 / TAL-SCQ5K 中英 / MMLU / HumanEval / MBPP，
  代码集带着测试（`test` / `test_list`），将来的代码环境可以直接用

### 训练记录（[PR #8](https://github.com/irroca/Whetstone/pull/8) / [#9](https://github.com/irroca/Whetstone/pull/9)）
- 每次训练写 `{save_dir}/runs/{run_id}/`：`meta.json`（参数 + git commit + 数据指纹）、
  逐点 `metrics.jsonl`、`summary.json`（失败也写）；`analyze_runs.py` 做 list / show / compare / plot
- 五个阶段都有 held-out 验证（`--val_data_path`），DPO 看偏好准确率；设备自动选 cuda > mps > cpu

### 数据消融
- `run_ablation.py`：一个 spec 跑完 数据池 → 分词器 → 各组语料 → 训练 → 评测 → 汇总表，
  每个阶段可续跑（§4 步骤 4）
- `probes.py`：每个源的 bits per byte（跨词表可比）、语言混淆率、few-shot 加法；
  也是 CLI，可以评测任意 checkpoint（§3.4）

### 训练循环（§3.4）
- `trainer.py`：pretrain / SFT / KD / DPO 共用的 epoch 循环，各阶段只提供「一个 batch 怎么算损失」
- 日志、验证、存档只在优化器更新之后发生；续训从同一批数据、同一个随机数状态接着跑
- `build_optimizer`：只对矩阵做 weight decay，β₂ = 0.95，CUDA 上用 fused AdamW；`--max_steps`；
  `pretrain.py --compile`
- `bench_train.py`：吞吐、峰值显存、MFU、flash attention 检查

### 调研与决策（`docs/corpus-plan.md`）
- 语料候选清单（中英 web / 代码 / 数学 / 书籍 / 可验证任务），含规模、许可证、获取上的坑
- 三份可引用的公开配比（SmolLM3、SmolLM2、CCI3.0-HQ）
- **已定**：模型 ~100M（`--dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3`，vocab 32k）、
  中文语料 `epfml/FineWeb2-HQ` 的 `cmn_Hani`、仓库公开、正式训练租卡

### 已经跑出来的实验结论（`docs/experiments.md`）
- **零奖励是 GRPO 的吸收态**：无 KL 锚定时策略在第 50 步崩溃，此后奖励恒 0 → 组内零方差 →
  优势恒 0 → `grad_norm` 精确为 0，剩下 100 步完全没有梯度，再也起不来
- **它只学会了格式**：accuracy 全程 0.000，而 `hack_rate` 与 `format_rate` 完全重合
- **eval 曲线平不等于优化器坏了**：另一组 100 步 greedy eval 完全不动，但权重确实在变

---

## 2. PR 历史：`main` 已完整，但中间出过一次合并顺序事故

**结论先说：`main` 现在是完整的**（PR #4 + #5 + #6 全部落地），直接从 `main` 开工即可。

```bash
git clone https://github.com/irroca/Whetstone.git && cd Whetstone
HF_HUB_OFFLINE=1 python3 -m pytest tests/ -q      # 应为 509 passed（含 #13）
```

下面这段记录下来，是因为它是一个很容易再犯一次的坑：

- **PR #4**（Mini-RLVR）以 **squash** 方式合进 `main`
- **PR #5**（数据管线 + 改名 Whetstone + 架构 CLI）的 base 是 #4 的分支，
  它在 #4 合入 `main` **之后 21 秒**才合进 #4 的分支
- 结果 `main` 拿到的是 **#5 之前**的树：#5 显示为「已合并」，但它的工作全部滞留在 #4 的分支上，
  `main` 上还是 `Config.py` / `SFT.py` / `spongebob_tokenizer/`，完全没有 `datatools/`
- **PR #6** 的分支持有两者的完整线性历史、是 `main` 的严格超集，直接 target `main` 一次补齐

| PR | 内容 | 状态 |
|----|------|------|
| [#4](https://github.com/irroca/Whetstone/pull/4) | Mini-RLVR：可验证奖励环境 + 从零 GRPO | 已合并（squash）|
| [#5](https://github.com/irroca/Whetstone/pull/5) | 数据管线、架构 CLI、改名 Whetstone | 已合并，但当时**没进 main** |
| [#6](https://github.com/irroca/Whetstone/pull/6) | 本交接文档 + 补齐 `main` | 已合并 |
| [#7](https://github.com/irroca/Whetstone/pull/7) | 交接文档跟上 `main` | 已合并 |
| [#8](https://github.com/irroca/Whetstone/pull/8) | MPS 设备自动选择 + 训练记录体系 | 已合并 |
| [#9](https://github.com/irroca/Whetstone/pull/9) | 训练记录与复盘体系（`runlog` + 验证集 + `analyze_runs.py`）| 已合并 |
| [#10](https://github.com/irroca/Whetstone/pull/10) | memmap 预训练语料 + 数据管线首次真实数据冒烟 | 已合并（squash），但只含前 4 个提交 |
| [#11](https://github.com/irroca/Whetstone/pull/11) | 消融编排 + 补上 #10 合并后才推的 5 个提交（SDPA、代码源修复）| 已合并（squash）|
| [#12](https://github.com/irroca/Whetstone/pull/12) | 消融结论、正式配比与词表、去污染修复、共享训练循环、GPU 准备 | 已合并（squash）|
| [#13](https://github.com/irroca/Whetstone/pull/13) | `prepare` 锁输出目录、按源续跑、`--only`（正式数据集就是用它重跑的）| 待合并 |

**教训**：stacked PR 要么严格按自下而上的顺序合，要么在合之前把上层 PR 的 base 直接改成 `main`。
`squash` 合并会切断祖先关系，所以一旦顺序错了，后续那个 PR 的内容不会自动跟过来，
而且再合时会因为历史分叉产生一堆「两边都改了同一文件」的假冲突。

**第二次是同一类事故**：#10 合并之后，又往它的分支推了 5 个提交，GitHub 照样显示「已合并」，
但这 5 个提交不在 `main` 上。**往已有 PR 推提交之前，先 `gh pr view <n> --json state` 确认它还开着。**
补救是把这些提交 rebase 到 `main` 上开新 PR：`main` 的 squash 提交和原分支末端的 tree 相同时，
`git rebase --onto origin/main <原末端>` 不会改变任何一个提交的 tree。

---

## 3. 开始正式训练之前的工程事

都是在 CPU 上跑不出来、但一上真实规模就会立刻爆的问题。3.1、3.2、3.4 已完成，3.3 留给开卡后的
第一个小时：

### 3.1 ~~【阻塞】`dataset.py` 把整个 JSONL 读进内存~~（已完成）

原来 `PretrainDataset` 把所有行 `json.loads` 进一个 Python list（10B token 约 30GB 文本，直接 OOM），
每个样本每个 epoch 都在训练循环里重新 tokenize。现在：

- `datatools/tokenize_corpus.py` 把 JSONL 预 tokenize 成 `<prefix>.bin`（文档首尾相接，每篇
  `bos + text + eos`，词表 ≤ 65536 用 `uint16`）+ `.idx`（文档偏移）+ `.meta.json`
  （计数、每个 `source` 的 token 数、分词器指纹）。三个文件写完才改名，被 kill 不会留下半份语料
- `dataset.MemmapPretrainDataset`：`np.memmap` 读，窗口 `max_seq_len`、步长 `max_seq_len - 1`，
  每个 token 每 epoch 恰好当一次预测目标，`X`/`Y` 形状与 JSONL 读取器相同。分词器指纹对不上、
  文件大小与 meta 不符（拷贝不完整）都直接报错
- `pretrain.py` 通过 `build_pretrain_dataset` 按扩展名选读取器，训练集和验证集都是；
  老的 `PretrainDataset` 保留给 fixtures 和单测
- 单测断言了每篇文档存下来的 token 与 `PretrainDataset` 喂给模型的逐个相同

`SFTDataset` / `PreferenceDataset` 仍是整文件读入，SFT 规模下没问题。

实测（冒烟语料 793 万 token，`--max_seq_len 512`）：

- **JSONL 读取器其实一直在丢数据**：每篇只取前 512 个 token，只用到 28% 的 token，书籍只用到 1%，
  实际配比里书籍从 6.4% 变成 0.2%。这是换 `.bin` 最重要的理由，比内存和速度更要紧
- 建完数据集后常驻内存：JSONL +38MB（文件 23.7MB），memmap +1MB；单进程每秒产出 loss token：
  JSONL 67 万，memmap 3856 万（快 58 倍）
- 预 tokenize：793 万 token 用 1.0 秒
- 端到端：29M 模型在 MPS + bf16 上读 `.bin` 跑完 1 epoch（970 步，384 秒，约 2.07 万 token/s），
  验证 loss 6.82 → 5.04。训练 token 数 7,930,209 = 15,519 个窗口 × 511，与"每个 token 恰好当一次
  目标"吻合；运行记录里有两个 `.bin` 的指纹和 manifest 里的配比

### 3.2 ~~【阻塞 8GB 本地卡】注意力显式构造完整 score 矩阵~~（已完成）

原来每层都物化 `(B, n_heads, q_len, kv_len)` 的 score 张量，softmax 还升到 fp32，8GB 卡上
seq 2048 会先在这里爆。现在 `model.Attention` 走 `F.scaled_dot_product_attention`：

- 不带缓存、没有 padding 的整段前向（训练的热路径）用 `is_causal=True`、不传 mask，CUDA 才能选
  flash attention；单 token 解码也不传 mask
- 带缓存的多 token 续写和带 padding 的 batch 构造显式布尔 mask。**续写不能用 `is_causal`**：
  SDPA 的因果 mask 是左上角对齐的，`q_len < kv_len` 时会让新 token 看不到自己前面的缓存。
  单测故意把这个条件改坏过，续写的两个用例会失败
- 显式 score 矩阵的旧实现保留为 `Attention.use_sdpa = False`，单测在 prefill / 解码 / 续写 /
  padding / 续写 + chunk mask 五种情况 × MHA / GQA 上逐位置对照，另外对照训练梯度

实测（同一初始化、同一批真实 token，旧实现 vs SDPA）：

| 设备 | 配置 | 旧实现 | SDPA |
|------|------|--------|------|
| CPU fp32 | 100M，seq 2048，bs 1，一步前向 + 反向 | 激活 +4.80 GB，2.0 秒 | **+2.33 GB，1.4 秒** |
| MPS bf16 | 29M 代理，seq 512，bs 8 | 25.6k–26.1k token/s | 23.6k–23.8k token/s |
| MPS bf16 | 100M，seq 2048，bs 4 | 4,471 token/s，27.6 GB | 4,170 token/s，23.4 GB |

loss 在两条路径上一致到小数点后 3–4 位。**MPS 上的 SDPA 训练时仍然物化 score 矩阵**
（seq 2048 只省 15% 显存），而且慢 7–8%，所以 MPS 一律走显式路径（`model.py` 里按设备判断），
本机消融的速度不受影响。CUDA 上的收益还没量，租到卡之后和 §3.3 一起量。

`--top_p`、repetition penalty 这些生成逻辑不受影响。

### 3.3 显存与吞吐的实测校准

`docs/corpus-plan.md` 里的「~92 小时」是按 `FLOPs/token ≈ 6N` 加一个假设的 30k token/s 估的。
**第一组真实测量已经有了**（Apple M5 Pro / 20 核 GPU / 48GB 统一内存，fp32 除非注明）：

| 配置 | cpu | mps | mps + bf16 |
|------|-----|-----|-----------|
| 29M 消融代理（`dim512 L8`, v6400, seq512, bs8） | 4,574 | 19,631 | — |
| 100M 目标（`dim768 L12`, v32k, seq512, bs4） | 1,445 | 5,717 | **9,055** |
| 100M 目标（seq1024, bs2） | 1,202 | 4,974 | — |

按这组数推算：

- **消融在本机可行**：0.4B token 的一组，MPS 上约 5.7 小时，一晚上跑一组；CPU 要 24 小时
- **正式训练在本机不可行**：10B token 即便 mps+bf16 也要约 **13 天**。租卡的计划不变
- 假设的 30k token/s 对这台 Mac 明显偏乐观。租到卡之后要**重新量一遍**再排时间

还没量的：`--batch_size` / `--max_seq_len` 的 OOM 边界（本机 48GB 统一内存不紧张，
8GB 的 5060Ti 或租来的卡才是真约束），以及 SDPA 在 CUDA 上的提升（§3.2，CPU 和 MPS 已量）。
**这些留给开卡后的第一个小时**，用 `bench_train.py` 一条命令量完（§4 步骤 6）。它跑的就是
`pretrain.py` 的训练步（同一个模型、优化器、autocast、损失），只是数据换成随机 token：

```bash
python3 bench_train.py --tokenizer_path tokenizer/v1_32k --dim 768 --n_layers 12 --n_heads 12 \
  --n_kv_heads 3 --max_seq_len 2048 --batch_sizes 8 16 32 --compile False True \
  --peak_tflops 312 --out results/bench.json
```

每个 micro-batch × 编译开关输出 token/s、每次更新耗时、峰值显存、TFLOPS 和 MFU，OOM 记一行
继续下一个；CUDA 上先检查训练前向能不能走 flash attention（只允许 flash 后端跑一次前向，
不能就报原因）。FLOPs 按每个矩阵权重 6 次 + 因果注意力 `6 × 层数 × seq × dim` 算：
100M 目标在 seq 2048 是 0.71 GFLOP/token。

本机 MPS + bf16 的参考（100M，seq 512，后台在跑数据任务）：batch 4 6.1k token/s，batch 8 7.4k。

已做：`torch.compile`（`pretrain.py --compile`，只包训练前向）、fused AdamW（CUDA 上自动）。
没做：gradient checkpointing（100M 在 80GB 卡上用不着），以及 RoPE 改成实数实现：现在的复数乘法
Inductor 不能融合，退回 eager 执行，开编译时大约损失 1–2%，等实测编译的收益再决定。

### 3.4 ~~训练循环在梯度累积和续训上的问题~~（已完成）

原来四个 DataLoader 阶段各有一份几乎相同的 `train_epoch`，问题也是四份：

- **验证和存档在每个 micro-batch 上判断**：累积 16 步时，`global_step` 停在 1000 的那 16 个
  micro-batch 每个都会触发一次第 1000 步的验证和存档
- **续训不保证数据相同**：`shuffle=True` 每次重新打乱，跳过的只是 batch 的**个数**；RNG 状态不进
  checkpoint，dropout 的随机数也接不上
- **checkpoint 原地写**：写到一半被回收（租的卡会发生），新旧两份一起坏
- **每个 micro-batch 都 `loss.item()`**：强制 GPU 同步，CUDA 上白白损失吞吐
- 优化器是 `AdamW` 默认值：对 RMSNorm 的增益也做 weight decay，β₂ = 0.999

现在 `trainer.py` 一处实现，四个阶段只提供 step 函数（pretrain / SFT 是 `lm_step`，KD 另记 ce / kd，
DPO 记 `dpo_loss`）：

- 日志、验证、存档只在优化器更新之后判断；`--log_step` / `--val_every` / `--save_step` 都按
  **优化器更新**计数（以前 `--log_step` 按 micro-batch）。epoch 末尾不满一个累积窗口的部分也是一次更新
- 每个 epoch 的顺序是 `seed + epoch` 决定的排列；checkpoint 的 `step` 改为「本 epoch 已消耗的
  batch 数」，续训的 sampler 直接从后面开始，不加载跳过的数据；RNG 状态跟权重一起存取。
  DataLoader 自己的 generator 也要固定：否则它每创建一次迭代器就从全局 RNG 取一个种子，续训的
  dropout 会因此错开一位
- 单测：训到第 7 次更新存档，换一个不同初始化的模型从存档续训，结束时**每个参数逐位相等**；
  模型里有 dropout。去掉排列的种子、去掉 RNG 恢复、让 DataLoader 用全局 RNG，三种改法各自都会让它失败
- `atomic_torch_save`：写临时文件、`fsync`、`os.replace`；`*_final.pth` 和 sidecar 也一样
- 指标在设备上累加、记日志时才读；token 数从 CPU 上的 mask 算；CUDA 上 `pin_memory` + `non_blocking`
- `build_optimizer`：只对二维以上的权重 decay（`--weight_decay 0.1`），`--adam_beta1/2` 默认
  0.9 / 0.95，CUDA 上 `fused=True`；GRPO 也用它
- `--max_steps`：训够这么多次更新就停，学习率调度按它展开；`pretrain.py --compile`
- `tests/test_stages.py` 在 fixtures 上把 pretrain → SFT → KD / DPO 依次跑通，外加一次从 epoch
  checkpoint 续训

---

## 4. 下一步的实验路线

按依赖顺序，每步都有明确的验收标准。

### 步骤 1：合并 #4 和 #5
见 §2。

### 步骤 2：拉评测集，闭上去污染的环 ✅
```bash
python3 -m datatools.fetch_evals --decontamination_only --update_spec configs/mixture_v1.json
```
2026-10-03 完成：GSM8K 1319 / MATH-500 500 / TAL-SCQ5K 中英各 2000 / MMLU 14042 /
HumanEval 164 / MBPP 500，共 20,525 条；`decontaminate.against` 已写回 spec（按字段拆开后
索引 45,778 项）。代码两个集是冒烟加进代码源之后补的：之前去污染只查数学和 MMLU，代码部分
等于没查。评测集在 `datasets/eval/`，git 忽略，换机器要重拉。

### 步骤 3：小规模跑通数据管线 ✅
```bash
python3 -m datatools.prepare configs/mixture_v1.json --probe 3      # 先查所有源的 spec 问题
python3 -m datatools.prepare configs/mixture_v1.json --scale 0.001 --out_dir datasets/smoke_full
```
完整配比、1000 万 token，最后一次运行（所有修复之后，产物在 `datasets/smoke_full/`）：

| 源 | token | 文档 | 读取 | 保留率 | token/篇 | 主要拒绝原因 |
|----|------:|-----:|-----:|------:|--------:|------|
| zh_web | 3.00M | 1895 | 2337 | 81% | 1584 | `too_short` 277，`low_cjk_ratio` 160 |
| en_web | 2.81M | 1867 | 1877 | 99% | 1504 | — |
| code | 2.00M | 859 | 911 | 94% | 2329 | `too_short` 26，`too_long` 17，`duplicate_lines` 9 |
| math | 1.42M | 767 | 845 | 91% | 1855 | `repetitive` 59，`duplicate_lines` 18 |
| books | 0.51M | 7 | 11 | 64% | 72948 | `too_long` 2（钦定版圣经、莎士比亚全集），意大利语《神曲》，《大宪章》判为 `repetitive` |
| synthetic | 0.30M | 276 | 278 | 99% | 1090 | — |

划分 train 5605 / val 37 / holdout 29；对 45,778 个评测片段做 13-gram 匹配，命中 2 篇（都不是代码），
LCS 复核后删除 0 篇。网速好的时候全程 3 分 21 秒，差的时候（几十 KB/s）13 分钟，大头是网络。
memmap 端到端（§3.1）用的是更早的无代码 800 万 token 版本 `datasets/smoke/`。

**跑出来并已修掉的十一个问题**：
1. `datasets`（HF）不在 `requirements.txt` 里，而且本地 `datasets/` 目录会被当成同名的命名空间包，
   报 `cannot import name 'load_dataset' from 'datasets' (unknown location)`
2. **FineWeb2-HQ 每行带 768 维 embedding**：上游列被原样写进输出，中文那份比正文大 9.3 倍；下载
   也一样，1000 行的行组不裁列 25.6MB、只读正文 3.8MB。现在 `to_record` 只保留 schema 字段，
   parquet 源用 `hf.columns` 裁列，还要把 fsspec 的预读从 5MiB 调到 64KB，否则读完正文会接着
   预读进 embedding 列（8.7MB）
3. **古腾堡**：`max_chars: 400000` 筛掉 32% 的书且专挑长篇（中位数 25 万字符、98% 在 100 万内），
   改为 200 万；约 5% 不是英文，用新加的 `where: {"metadata.language": "en"}` 过滤
4. **长文档的报告失真**：书籍报 18% 保留率（实为约 89%），为凑 7 本书拉了 289 本。原因是
   tokenize 按 256 条一批，预算在批中间填满后剩下的记录全被计入"读取"。现在每条记录带计数快照，
   填满时回退；缓冲区字符数超过剩余预算时提前 flush
5. **去污染误杀**：旧规则删掉 186 篇（3.9%），逐条核对几乎全是误杀：标点算作单元，一串 `-` 命中所有
   markdown 表格；短答案按 n-gram 索引，MMLU 的 `1,2,3` 一项就命中 76 篇。改为标点不算单元、
   短答案只精确匹配、题干至少 8 个单元
6. **LCS 复核只看文档前 2000 个单元、且对整篇算**：书里更靠后的泄漏不会被比对，短题目的词又会从
   整页各处被零散凑齐。改为在共享 n-gram 处对齐后再算。之前以为抓到的"MATH-500 真泄漏"其实是
   一份和题目共享三角形面积行列式记号的公式表
7. **`--probe` 打印完结果后卡死不退出**：pyarrow 25 在还有 parquet 读请求没完成时退出，会死锁在
   线程池的析构里（工作线程在解释器收尾时被 CPython 结束，析构一直在等它）。`del` + `gc.collect()`
   和关掉 `pre_buffer` 都不管用。`prepare` 的入口改成跑完 atexit 后直接 `os._exit`
8. **去污染不查代码**：评测集只有数学和 MMLU。补了 HumanEval（164）和 MBPP（500），参考实现放
   `answer`，这样 `return [x for x in strings if substring in x]` 这类短惯用写法只做精确匹配，
   不会误杀用到它的代码文件（放 `solution` 就会，测试里断言了）
9. **starcoderdata 的内容里带仓库元数据**：49% 的文件第一行是
   `<reponame>…<filename>…<gh_stars>…`。新加 `cleaners`，代码源用 `starcoder_metadata` 在过滤前删掉
   这一行；只删整行都是元数据段的，代码里的 `'-f <filename>'` 不动。修完后 0 篇带标记
10. **散文用的重复度规则在删正常代码**：字符 10-gram 重复度拒掉 15% 的代码文件，2k 字符以内 2%、
   超过 2 万字符 57%；232 篇里只有 10 篇是生成代码。代码源关掉这条后保留率 80% → 94%，
   同样 200 万 token 从 1238 篇变成 869 篇（中位数 1840 → 2481 字符），长文件不再被系统性删掉
11. **`--progress_every` 从不打印或刷屏**：文档按 tokenize 批次（256 篇）一起计数，取模判断对 500
   永远不成立、对 128 每一行都成立（1200 篇刷了 945 行）。改为跨过下一个整数倍时打印
12. **重复行比例把单独一行的 `"""`、`)`、`},` 也算进去**，docstring 多的代码文件因此显得一半是重复行。
   改为只统计含字母或数字的行，所有源通用（导航菜单的行都有字，照样抓得到）。在 1557 个代码文件上
   这条规则的拒绝 27 → 17 篇，数学 20 → 18，其余源已保留的文档没有一篇因此被拒；上表是改完后
   重跑的结果

**值得在消融里核实的观察（没改）**：中文 `min_chars: 200` 删掉 12% 的文档，`min_cjk_ratio: 0.5`
删掉 7%。后者可能专删中英混排的技术文章，而这正是代码/数学能力需要的，见 §8。代码里 2.4% 的文件
头部带生成标记（以 Django migration 为主），**决定不过滤**：按标记过滤会误删 Colab / nbdev 导出的
人写代码，大段重复的生成文件已经被重复行规则拦下。

验收（已满足）：每个源 `fill` ≈ 100%，没有 `ran out of data`；没有哪条规则删掉大半；
`split.top_matches` 里没有一项删掉大量文档。

### 步骤 4：消融 #1（中文占比）和 #5（词表大小）✅
这两个决定其余所有配置，所以先做。

**结果**（`results/ablation_v1/report.md`；holdout 上的 bits per byte，越低越好；混淆率是中文开头
续写成别的语言的比例）：

| 组 | 词表 | 中文 | zh | en | code | math | 加权平均 | zh→en |
|----|------|----:|----:|----:|----:|----:|----:|----:|
| zh00_v32k | 32k | 0% | 2.914 | **1.235** | **0.864** | **1.167** | 1.658 | 39% |
| zh15_v32k | 32k | 15% | 1.665 | 1.251 | 0.881 | 1.188 | 1.296 | 1% |
| zh30_v32k | 32k | 30% | 1.584 | 1.271 | 0.901 | 1.212 | 1.287 | 3% |
| zh45_v32k | 32k | 45% | **1.534** | 1.295 | 0.925 | 1.241 | 1.290 | 2% |
| zh30_v16k | 16k | 30% | 1.616 | 1.293 | 0.917 | 1.238 | 1.312 | 1% |
| zh30_v48k | 48k | 30% | 1.579 | 1.256 | 0.895 | 1.203 | **1.277** | 2% |

- **中文的收益在 15% 之前就拿到了九成**：0 → 15% 中文 bpb 降 1.249，15 → 45% 只再降 0.131。
  中文从 15% 加到 45%，en / code / math 各变差 3.5% / 5.0% / 4.5%
- **中文 0% 的组 39% 的中文提示会续写成英文**，有中文的组都在 3% 以内
- **定为 20%**：按 15% 与 30% 两组线性插值，20% 比 30% 中文差 3.4%，code / math / en 好 1.5% /
  1.3% / 1.0%。项目的目标能力是算术和代码，而中文的大头已经在 15% 拿到
- **词表定 32k**：16k → 32k 加权平均降 1.9%，32k → 48k 只再降 0.8%，而 48k 在 dim 768、seq 2048
  时每个 token 多 11% 的 FLOPs，logits 显存多一半
- **加法探针在所有组都是 1–9%**，3 亿 token 的代理模型还没有算术能力，这一列区分不出组

**决定落地**：`tokenizer/v1_32k/` 直接用消融里评测过的 v32k，不在正式语料上重训（重训出来的就是
另一个没评测过的分词器）。它训练的样本取自数据池的 train 划分，而划分按内容哈希、盐是同一个
`seed`，同一篇文档在正式数据集里也只会落进 train，不会把 holdout 漏进分词器。
`configs/mixture_v2.json`：zh 20%，其余五个源按 v1 的比例缩放（en 32% / code 22.9% / math 16% /
books 5.7% / synthetic 3.4%），10B token，**按 v1_32k 计数**（按 zh_6400 计只会得到约 7.5B 个
v1_32k token）。

```bash
python3 run_ablation.py configs/ablation_v1.json plan   # 各组配比、数据池大小、产物清单；不动任何文件
python3 run_ablation.py configs/ablation_v1.json run    # 全部阶段；被打断后重跑同一条命令续上
```

六组，每组 3 亿 token（按各组自己的分词器计）。代理模型 `--dim 512 --n_layers 8 --n_heads 8`，
非 embedding 参数 25.7M，三种词表一样；seq 512、batch 32、lr 1e-3、bf16：

| 组 | 词表 | zh_web | en_web | code | math | books | synthetic | 本机预计 |
|----|------|------:|------:|-----:|-----:|------:|---------:|--------:|
| zh00_v32k | 32k | 0% | 40.0% | 28.6% | 20.0% | 7.1% | 4.3% | 4.0 h |
| zh15_v32k | 32k | 15% | 34.0% | 24.3% | 17.0% | 6.1% | 3.6% | 4.0 h |
| zh30_v32k | 32k | 30% | 28.0% | 20.0% | 14.0% | 5.0% | 3.0% | 4.0 h |
| zh45_v32k | 32k | 45% | 22.0% | 15.7% | 11.0% | 3.9% | 2.4% | 4.0 h |
| zh30_v16k | 16k | 30% | 28.0% | 20.0% | 14.0% | 5.0% | 3.0% | 3.4 h |
| zh30_v48k | 48k | 30% | 28.0% | 20.0% | 14.0% | 5.0% | 3.0% | 4.9 h |

预计耗时来自同一代理在 MPS + bf16 上实测的吞吐：32k 2.08 万 token/s（19.2GB），16k 2.43 万，
48k 1.70 万（24.2GB）。

设计（规则写在 `datatools/ablation.py` 的模块文档里，README「数据消融」一节有用法）：

- **一个数据池，所有组从里面切**：`prepare` 只拉一次，约 7.7 亿个 zh_6400 token（每个源取任何一组
  需要的最大量，乘 1.75 的余量，再除以 train 划分的占比）。每组取每个源的**前**若干篇，15% 组的中文
  是 30% 组的前缀
- **配额按各组自己的分词器计数**，每组训练的 token 数相同。余量 1.75 覆盖的是新词表比 zh_6400
  压缩得更好（冒烟语料上最多 1.57 倍，代码 / 48k）。不够时切数据那一步报错，不会静默少给
- **三个分词器在同一份样本上训**：数据池 train 划分里按基础配比取的 1 亿个 zh_6400 token。
  `train_tokenizer.py` 改成了数字逐位切分（原来 `1987` 是一个 token，`2024` 切成 `20|24`，`12345`
  切成 `12|345`，算术没有一致的单位）。冒烟语料上 32k 词表的代价：数学 3.47 → 3.19 字符/token，
  其余源 0–3%
- **评测在数据池的 holdout 上**：每个源的 bits per byte（跨词表可比）、按基础配比加权的平均、
  语言混淆率（中文开头续写成英文的比例，以及反过来）、4-shot 一位数 / 两位数加法。
  代码执行通过率没做：这个规模的基座模型大概率全是 0，区分不出组；代码先看 bpb
- 已知偏差：窗口固定 512 token，词表大的组每个窗口覆盖的文本更多，bpb 对大词表略有利

搭的时候发现并修掉的问题：

1. **验证集只看了第一个源**：`build_val_loader` 按文件顺序读，`--val_batches 20` 只评估前 20 个
   batch，而 `prepare` 写出的划分按源分组，那 33 万个 token 全是中文。中文 0% 那组的 val 曲线会完全
   测在它没训过的语言上。现在验证集按一个固定排列读，每次评估还是同一批样本，但样本取自整个划分。
   pretrain / SFT / KD / DPO 共用这个函数，一起修好了
2. **数据池按全部文档定量，但组和分词器样本只读 train 划分**：正式配置里 val + holdout 只占 1%，被
   余量掩盖了；端到端测试用 20% 时直接切不够。现在目标除以 train 的占比
3. 端到端测试另外说明了余量为什么必须有：合成的重复代码上，300 词表的小分词器反而比 zh_6400
   压缩得好 1.34 倍，1.3 的余量不够，报错信息准确指向了「加大 `pool_margin`」

产物：`results/ablation_v1/report.md`（汇总表）、每组的 `probes.json`、`runs/` 里的完整曲线
（`analyze_runs.py compare results/ablation_v1/zh*_v32k --metric loss --split val`）。

**运行记录**：10-03 19:51 开始训练，每秒约 2.0–2.2 万 token，六组跑完后自动评测、汇总。

- 跑在 `screen` 会话里，不依赖 Cursor。一开始挂在 agent 的 shell 下，退出 Cursor 会连带杀掉训练；
  训到第 1900 步时停掉，在 `screen` 里从头重跑。两次第 51 步的 loss 都是 9.2713，固定种子下可复现

- 数据池 18:36–19:14（38 分钟，网络是瓶颈）：6 个源 fill 都是 100%；train 489,537 / val 2,544 /
  holdout 2,466 篇。中文保留率 89%（`too_short` 14,233、`low_cjk_ratio` 7,894），代码 95%，数学 90%，
  书籍 300 本。去污染删 252 篇，过半是误杀，见 §8 第 7 条
- 分词器 29–39 秒一个；每组切 3 亿 token 约 1 分钟，六组都在配额上方 0.05% 以内
- 第一组（zh00_v32k）第 1000 步 val loss 6.12。比同期训练 loss 高很多是预期的：验证集里 31% 是这组
  没见过的中文
- **消融从一个固定在 `eea8acb` 的 worktree 里跑**，这样一天里主仓库切分支、改代码都不会让后面几组
  用上另一份代码。跑完后已删除（数据和结果在主仓库的 `datasets/`、`results/` 里，worktree 里只是软链接）

### 步骤 5：正式数据集 ⏳ 正在跑
分词器已定（步骤 4），这一步只剩出数据：

```bash
python3 -m datatools.fetch_evals --decontamination_only --update_spec configs/mixture_v2.json
python3 -m datatools.prepare configs/mixture_v2.json --probe 3
python3 -m datatools.prepare configs/mixture_v2.json --out_dir datasets/mixture_v2 --progress_every 50000
for s in val holdout train; do
  python3 -m datatools.tokenize_corpus datasets/mixture_v2/$s.jsonl \
    --tokenizer tokenizer/v1_32k --out datasets/mixture_v2/$s
done
```

**进度**：前两条已完成（七个评测集重拉、六个源 probe 全部 `ok`）。后两条由
`zsh results/data_v2/run.sh` 在 `screen` 会话 `whetstone-data` 里跑，日志 `results/data_v2/run.log`，
代码固定在 worktree `../whetstone-data-v2` 的 `cebd574`（PR #13 的提交；`datasets` 是指回主仓库的
软链接）。10-09 20:49 开始，分两路并行拉：

- **五个公开源经 `hf-mirror.com`**，进程看不到 HF token（`HF_TOKEN_PATH=/nonexistent`）。断网之后，
  到官方 CDN（`us.gcp.cdn.hf.co`）的单条连接只有 120–350 KB/s，镜像有 3 MB/s，抽查的 10MB 逐字节
  相同。zh_web 实测约 3000 万 token/分钟，是第一次构建的 3 倍，五个源约 5 小时
- **gated 的 starcoderdata 单独一路**，拉进 `datasets/mixture_v2_code/`，完成后连同完成标记挪进
  `datasets/mixture_v2/sources/`。20:49 起带主 token 走官方源，受慢线路所限只有 270–390 万
  token/分钟，2.29B 要 10–14 小时。**21:33 切到镜像，从头重拉**，约 4500 万 token/分钟，
  预计 22:25 拉完。镜像这一路由 `code_mirror.sh`（screen `whetstone-code`）跑
- **主 token 不发给镜像**。镜像用的是专门新建的 fine-grained token，存为
  `~/.cache/huggingface/token.mirror`，只经 `HF_TOKEN_PATH` 传给进程。`switch_code.sh` 先向官方
  核对权限，要求是 fine-grained、没有针对具体账号的权限、没有写权限。实际这个 token 还带了个人名下
  仓库的只读权限，检查没通过；用户确认接受后手动切换（见 `run.log` 的 `[switch]` 行）。
  **构建完成后在 HF 上删掉这个 token，并 `rm ~/.cache/huggingface/token.mirror`**
- 切换之后，原 `run.sh` 会在公开源拉完时以 `EXIT=1 (open lane 0, code lane 143)` 退场，这是预期的；
  `code_mirror.sh` 等它放开锁，再自动重跑 `run.sh` 收尾

两路都完成后，`prepare` 离线再跑一遍，复用全部六个源，只做去污染和划分，然后 tokenize。现在的瓶颈
是公开源那一路：zh_web 之后还有 en_web 3.2B、math 1.6B、books 0.57B、synthetic_textbook 0.34B，
按约 3200 万 token/分钟，预计 10-10 01:00 前后拉完，整个构建约 03:00 出完。20:31–20:46 那一次走的
是慢线路，日志在 `run.attempt2.log`。

**第一次构建丢了 3.6 小时**（日志 `results/data_v2/run.attempt1.log`）：16:30 起跑，19:56 断网约
13 分钟，HF 客户端自己的重试扛了过去，20:09 已拉到 zh_web 的 1954M/2000M；但断网期间又手动起了
两份 `run.sh`，它们以 `"w"` 重开 `sources/zh_web.jsonl`，把它截成一个表面 7.9GB、实际只有 339MB
数据的稀疏文件。现在两层都防住了：`run.sh` 整体加锁（`zsystem flock`），`prepare` 锁 `--out_dir`，
第二份都会直接退出。`prepare` 按源续跑，`run.sh` 在它失败时每 10 分钟自动重试、最多 8 次，
每次只重拉没完成的那个源。**日志里刷 `Retrying` 时不要手动重启**，那是客户端在正常重试；
任务真的退出了，`run.sh` 会自己接上。

**出数据之前修掉的去污染误杀**（原 §8 第 7 条）：选择题的通用题干（`下列说法正确的是．`、
`Which of the following statements is true?`）按自身长度索引，命中每一篇带这句话的文档。
`fetch_evals` 现在把选项拼进 `question`（`题干\nA. …\nB. …`），模型看到的题目本来也带选项；
题干里没有任何字母数字的题（上游是图片，22 道 TAL-SCQ5K 英文题）直接跳过，否则选项字母本身
凑够 8 个单元，又会去命中 `a a b b c c` 这类序列。在消融数据池上重跑去污染：**删除 252 → 57 篇**
（zh 4 / en 4 / code 17 / math 31 / synthetic 1），剩下的基本是真泄漏：MBPP / HumanEval 的参考解
出现在代码里，MATH / GSM8K 原题出现在 FineMath 里，TAL 原题出现在中文教育网页上。

验收：manifest 里每个源 `fill` ≈ 100%、没有 `ran out of data`；`split.top_matches` 里没有一项删掉
大量文档；`train.bin` 约 20GB（10B 个 `uint16`）。`.bin` 绑定分词器指纹，换词表必须重跑
`tokenize_corpus`（不重跑会直接报错）。上传到服务器的就是 `datasets/mixture_v2/` 下的
`{train,val}.{bin,idx,meta.json}`、`holdout.jsonl`（给 `probes.py`）和 `manifest.json`。
验收之后删掉镜像用的 `token.mirror`（HF 上删 token，本地 `rm`）。

**v1_32k 让所有旧 checkpoint 失效**——`resolve_model_config` 会在 `vocab_size` 不匹配时直接报错，
这是设计如此。

### 步骤 6：正式预训练（~100M / 10B token，租卡）
本机 MPS 跑 10B token 要约 13 天（§3.3），所以租卡。按 0.71 GFLOP/token、MFU 35–45% 估
（**开卡后用 `bench_train.py` 的实测替换**）：

| 卡 | 估计吞吐 | 10B token | 费用（按 AutoDL 时价） |
|----|--------:|--------:|------:|
| RTX 4090 24GB | 7–9 万 token/s | 31–40 小时 | ¥60–110 |
| A800 80GB | 15–18 万 | 15–19 小时 | ¥75–120 |
| H800 80GB | 25–40 万 | 7–11 小时 | ¥60–160 |

**无卡模式下先做完的事**（AutoDL 无卡模式 0.5 核 / 2GB / ¥0.1 每小时，不占 GPU）：

1. 选 PyTorch 2.x + CUDA 12 + Python 3.12 的镜像，`git clone`，`pip install -r requirements.txt`
2. 上传 `datasets/mixture_v2/` 下步骤 5 列出的文件（约 20GB）到 `/root/autodl-tmp`
3. `HF_HUB_OFFLINE=1 python -m pytest tests/ -q`（0.5 核会慢，但能跑）
4. 写好启动脚本（见下），确认路径都指向 `/root/autodl-tmp`

无卡模式做不了的：任何 CUDA 相关的（吞吐、显存、编译、flash 检查），以及重 CPU 的活（别在上面
跑 `prepare` 或 `tokenize_corpus`，在本机做完再传）。

**开卡后的第一个小时**：

1. `nvidia-smi`；`python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name())"`
2. `bench_train.py`（§3.3 的命令，`--peak_tflops` 按卡填）：选最大不 OOM、吞吐最高的 micro-batch，
   定编译开不开；**flash attention 必须是 `ok`**
3. 用正式命令加 `--max_steps 200 --save_step 100` 试跑：loss 在降、验证在出数；中途 kill 一次，
   用 `--resume_from <save_dir>/latest_checkpoint.pth` 续上。确认后删掉这个目录
4. 正式启动，挂在 `tmux` / `screen` 里，命令后接 `; /usr/bin/shutdown`，跑完自动关机停止计费

**正式命令**（A800 的例子；`--batch_size` 以 bench 为准，`batch_size × accumulation_steps × 2047`
保持约 50 万 token，共约 1.9 万次更新）：

```bash
python3 pretrain.py --dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3 \
  --tokenizer_path tokenizer/v1_32k --max_seq_len 2048 \
  --data_path /root/autodl-tmp/mixture_v2/train.bin --val_data_path /root/autodl-tmp/mixture_v2/val.bin \
  --epochs 1 --batch_size 16 --accumulation_steps 16 --learning_rate 6e-4 \
  --log_step 10 --val_every 500 --val_batches 50 --save_step 500 \
  --dtype bfloat16 --compile True --save_dir /root/autodl-tmp/results/pretrain_v2
```

- 学习率 6e-4、50 万 token 的 batch 是 GPT-3 125M 的设定；10% 线性预热后余弦退到 10%
- `--save_step 500` 约半小时一次；实例被回收就用同一条命令加 `--resume_from` 续上，数据顺序与
  不中断时完全相同（§3.4）
- 训完：`probes.py --checkpoint .../pretrain_final.pth --holdout .../holdout.jsonl
  --tokenizer_path tokenizer/v1_32k`，把结果和 `runs/` 拷回本机，填 `docs/experiments.md`

**AutoDL 的坑**：

- 一个账号同时只能有一个无卡实例；无卡开机会释放 GPU，再开卡时那台机器的卡可能已被别人占用。
  选空闲卡多的机器
- `/root/autodl-tmp` 是本地数据盘，没有冗余，实例释放就没了；`/root/autodl-fs` 是同区域共享的
  文件存储（免费 20GB）；系统盘只有 30GB，**数据和 checkpoint 都别放系统盘**（一份带优化器状态的
  checkpoint 约 1.2GB）
- 访问 HF / GitHub：`source /etc/network_turbo`，或 `export HF_ENDPOINT=https://hf-mirror.com`
- 计费从开机到关机，跑完一定要关机

### 步骤 7：重建后训练四阶段
SFT / KD / DPO 的数据要基于新语料和可验证任务重做（`envs.generate_data` 负责算术那部分；
代码任务的环境还没写，见 §5）。然后 GRPO——这次 accuracy 应该终于有方差了，这是整个项目
最关键的验证点。

---

## 5. 长期计划

### 阶段 A：把可验证任务做实（当前重心）
- 100M 双语模型，预训练 + 后训练全链路跑通
- 算术任务上 GRPO 让 **accuracy 真正上升**（不只是 format rate）——这是上一轮没做到的
- 填满 `docs/experiments.md` 的消融表

### 阶段 B：第二个环境——代码任务
`envs/base.py` 的 `TaskEnv` 接口已经留好（`sample_task` / `render` / `score`），加环境不用动
训练代码。代码环境的奖励是**执行单元测试**，比算术更接近真实 RLVR，也是 Anthropic
Fellows 那类岗位明确在做的「creating RL environments」。

要点：沙箱执行（子进程 + 超时 + 资源限制）、测试用例来源（`PRIME-RL/Eurus-2-RL-Data` 已经带
测试用例，`fetch_evals` 里已注册）、部分通过的奖励整形（pass@k 还是通过率）。

### 阶段 C：算法侧的消融与技术博客
现在都是开关，可以直接跑：
- Dr. GRPO（去 std 归一化 + 常数分母）vs GRPO
- DAPO（token-level loss + clip-higher + dynamic sampling）
- 难度课程（用 `big_math` 的 `llama8b_solve_rate` 筛中等难度题，避开零方差组）
- 熵下界 vs KL 锚定，验证崩溃是 KL 问题还是一般的探索问题

博客建议题目仍然偏现象而不是教程，例如《在小模型上，GRPO 学会的是计算还是格式？》。
素材已经有了一半（§1 末尾那三条结论）。

### 明确不做
PPO critic、PRM（过程奖励模型）、MoE、多卡并行、推理服务化、量化。

---

## 6. 从云端切到本地：环境差异

| | 云端（之前） | 本地 Mac（现在） | 5060Ti / 租的卡 |
|---|---|---|---|
| 加速器 | 无 | **Apple M5 Pro，MPS** | CUDA |
| `--device` 默认 | `cpu` | **`mps`**（自动选）| `cuda` |
| `--dtype` | 只能 `float32` | `bfloat16` | `bfloat16` |
| 数据 | `datasets/` 是空的 | 评测集 `datasets/eval/`、冒烟语料 `datasets/smoke_full/`、消融数据 `datasets/ablation_v1/`，正式数据集 `datasets/mixture_v2/` 生成中（都被 git 忽略）| 从本机上传 `mixture_v2` |

本地环境搭建：

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
HF_HUB_OFFLINE=1 python -m pytest tests/ -q     # 502 passed，其中端到端消融测试约 30 秒
```

要注意的几点：

- **测试和训练带上 `HF_HUB_OFFLINE=1`**。分词器就在仓库里，但 `transformers` 默认每次联网检查更新，
  测试套件因此从 47 秒变 16 秒（其中只有 4 秒是 CPU 时间）。**跑 `prepare` / `fetch_evals` 时
  反过来要去掉它**，否则报 `OfflineModeIsEnabled`；shell 里 `export` 过的话用 `env -u HF_HUB_OFFLINE ...`
- **HF `datasets` 是数据管线的依赖**（已在 `requirements.txt`）。没装时仓库自己的 `datasets/` 目录会被
  当成同名命名空间包，报的是看不出原因的 `cannot import name 'load_dataset' from 'datasets' (unknown location)`
- **设备自动选择是 cuda > mps > cpu**（`train_utils.resolve_device()`）。Mac 上不用传 `--device`
- **MPS 上别用 fp16**：实测把 loss 从 fp32 的 `-0.3278` 变成 `+0.0018`，bf16 则是 `-0.3276`。
  fp16 的指数范围扛不住这个模型的注意力路径
- **`--dtype float16` 才会启用 GradScaler**，`bfloat16` 不需要也不会启用
  （`build_autocast_scaler` 里的逻辑），别以为是 bug
- 系统自带的 Python 是 3.9，**跑不了**（`transformers` 5.x 要 3.10+）。必须用 3.12 的 venv
- **`bigcode/starcoderdata`（代码语料）和 `big_math` 是 gated 的**（自动批准），要先在 HF 上接受条款，
  再 `hf auth login`（或设 `HF_TOKEN`）。其余语料和七个去污染评测集都不需要。本机已登录并接受了
  starcoderdata 的条款，`big_math` 还没有。`hf` 命令在 venv 里（`.venv/bin/hf`），系统 Python 没有
- **磁盘预算**：10B token 的 JSONL 约 30GB，预 tokenize 成 `uint16` 后约 20GB，
  加上 HF 缓存，留 100GB 比较稳妥。不要下全量语料——FineWeb2-HQ 的 `cmn_Hani` 是 784GB，
  我们用 `streaming=True` 只取需要的量
- `datasets/`、`results*/`、`*.pth` 都在 `.gitignore` 里

---

## 7. 新 session 最容易踩的坑

`AGENTS.md` 里有完整清单，这里只列最容易造成「以为是 bug」的几条：

1. **GRPO 的 `loss` 恒为 0**（on-policy + `seq_mean` 聚合时）。这是数学上的必然：ratio ≡ 1 且
   组内优势之和为 0。看 `grad_norm`，别看 loss
2. **eval 曲线完全不动 ≠ 优化器坏了**。先查权重 delta 和 `grad_norm` 再怀疑算法
3. **GRPO 需要冷启动，但不能过**。策略不会输出环境格式 → 全 0 奖励 → 无梯度；
   SFT 训到完全饱和（熵极低）→ 采不出奖励方差 → 同样无梯度
4. **一次 `generate` 只能处理一个 prompt 的 group**。模型没有 left-padding 的 RoPE 偏移
5. **`generate` 产出的 token id 必须 `clone()`** 才能参与需要 backward 的前向
   （inference_mode 张量 + embedding 反向会保存索引）
6. **微批必须共用整批的分母**（`grpo_policy_loss(..., normalizer=...)`），有单测断言
   「分块损失之和 == 单次全批损失」，别「简化」掉
7. **配比 spec 的 `decontaminate.against` 为空时静默什么都不查**
8. **`single_char_frac` 只能在同一书写系统内比较**。中文单字本身就是有意义的单位，
   63% 是正常的，不是碎片化
9. **JSONL 预训练读取器每篇只取前 `max_seq_len` 个 token**，不报错、loss 照常下降，但配比已被改写
   （冒烟语料上只用到 28% 的 token、书籍 1%）。正式训练和消融一律用 `tokenize_corpus` 产出的 `.bin`
10. **自己写的脚本流式读 HF parquet、读到一半就退出，会卡死在退出阶段**（pyarrow 25：在途的读请求
   拿不到 GIL，线程池析构永远等下去）。`prepare` 已经规避了；临时脚本结束时先 flush，
   再 `os._exit(0)`
11. **跨词表别比 loss**。词表大的每个 token 要预测的信息更多，val loss 只在同一词表内可比；
   消融的结论看 `probes.json` 里每个源的 bits per byte
12. **`prepare` 的日志刷 `Retrying` / `Got disconnected from remote data host` 不等于任务挂了**。
   HF 客户端会自己重试（实测扛过 13 分钟断网）。在它重试时再起一份，正是第一次正式构建丢掉
   3.6 小时的原因；现在第二份会被锁拒掉，而 `prepare` 真退出了就重跑同一条命令，已完成的源会复用

---

## 8. 未决的问题

1. ~~§3.1 和 §3.2 谁先做？~~ 都已完成
2. **中文过滤阈值会不会专删技术文章？** 冒烟里 `min_cjk_ratio: 0.5` 删掉 7% 的中文文档，中英混排、
   带代码片段的技术文章最容易落到这条线以下，而它们正对代码/数学能力有用。抽 50 篇被拒的看一眼
   再决定，可以作为消融 #1 的附带项
3. ~~代码要不要过滤生成文件？~~ 不过滤（已定）。2.4% 的文件带生成标记，按标记过滤会误删 Colab /
   nbdev 导出的人写代码，大段重复的生成文件已经被重复行规则拦下
4. ~~`duplicate_lines` 对代码偏严~~ 已改：只统计含字母或数字的行（§4 步骤 3 第 12 条）。考虑过再豁免
   短行，在代码上能多放回 4 篇，但多出来的都是比例卡在 0.50 上下的边界文件，为此给代码单独加一个
   参数不值
5. **代码任务的沙箱怎么做？** 阶段 B 的核心设计问题。子进程 + 超时是底线，要不要上容器取决于
   数据源可信度
7. ~~去污染会被只有通用题干的选择题误杀~~ 已修：选项拼进题目，消融数据池上删除 252 → 57 篇（§4 步骤 5）
6. **租什么卡？** 估算见 §4 步骤 6，A800 / H800 都在 ¥160 以内；以 `bench_train.py` 的实测为准。
   如果显存和预算宽裕，`docs/corpus-plan.md` 里 ~185M / 18B token 的档位也在射程内

改名已全部完成：代码、文档、远端仓库都是 **`irroca/Whetstone`**。如果你手上还有指向旧名的
clone，GitHub 会一直重定向，但建议顺手改掉：

```bash
git remote set-url origin https://github.com/irroca/Whetstone.git
```
