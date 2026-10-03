# 项目状态与路线（交接文档）

**最后更新**：2026-10-03，本地 Mac session。
本文档是新 session 的入口——先读这里，再读 `AGENTS.md`。

---

## 0. 一句话现状

算法链路（五阶段后训练）和数据管线都已实现，CPU 单测覆盖；语料方案和模型规模已定。
**数据管线已经在真实语料上跑通了千分之一规模的完整冒烟**（§4 步骤 3，六个源全部含代码），
暴露并修掉了十二个问题：去污染此前删掉的文档几乎全是误杀，代码源有 15% 的正常源码被一条散文用的
重复度规则删掉。预训练已经改读预 tokenize 的 memmap 语料（§3.1），注意力换成了 SDPA（§3.2）。

**正式训练还没开始**。消融 #1（中文占比）和 #5（词表大小）的编排脚本已经写好，正在本机上跑
（§4 步骤 4，六组共约一天）。跑完按结果定配比和词表，再出正式数据集、重训分词器（§4 步骤 5）。

仓库里现有的一切数字都是 29M 玩具规模的**实现验证**，不是能力声明。

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
- `probes.py`：每个源的 bits per byte（跨词表可比）、语言混淆率、few-shot 加法

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
HF_HUB_OFFLINE=1 python3 -m pytest tests/ -q      # 应为 483 passed
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
| [#11](https://github.com/irroca/Whetstone/pull/11) | 消融编排 + 补上 #10 合并后才推的 5 个提交（SDPA、代码源修复）| 待合并 |

**教训**：stacked PR 要么严格按自下而上的顺序合，要么在合之前把上层 PR 的 base 直接改成 `main`。
`squash` 合并会切断祖先关系，所以一旦顺序错了，后续那个 PR 的内容不会自动跟过来，
而且再合时会因为历史分叉产生一堆「两边都改了同一文件」的假冲突。

**第二次是同一类事故**：#10 合并之后，又往它的分支推了 5 个提交，GitHub 照样显示「已合并」，
但这 5 个提交不在 `main` 上。**往已有 PR 推提交之前，先 `gh pr view <n> --json state` 确认它还开着。**
补救是把这些提交 rebase 到 `main` 上开新 PR：`main` 的 squash 提交和原分支末端的 tree 相同时，
`git rebase --onto origin/main <原末端>` 不会改变任何一个提交的 tree。

---

## 3. 开始正式训练之前必须做的三件工程事

这三件都是在云端 CPU 上跑不出来、但一上真实规模就会立刻爆的问题。按优先级排：

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

顺带可以考虑（不阻塞）：gradient checkpointing、`torch.compile`、fused AdamW。

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

### 步骤 4：消融 #1（中文占比）和 #5（词表大小）⏳ 正在跑
这两个决定其余所有配置，所以先做。

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

**进度**（2026-10-03）：数据、分词器、六组语料都已就绪，19:51 开始训练，按实测吞吐（每秒约 2.0–2.2 万
token，32k 词表每组约 4.1 小时）六组 10-04 晚上训完，然后自动跑评测和汇总。

- 跑在 `screen` 会话 `whetstone-ablation` 里，不依赖 Cursor：`screen -r whetstone-ablation` 接上去看，
  `Ctrl-A D` 离开。日志 `results/ablation_v1/run.log`，中断后重跑 `zsh results/ablation_v1/run.sh` 续上
- 一开始是挂在 agent 的 shell 下跑的，那样退出 Cursor 会连带杀掉训练；训到第 1900 步时停掉，
  在 `screen` 里从头重跑，没有用续训（续训不保证数据顺序和不中断时一样）。两次第 51 步的 loss
  都是 9.2713，固定种子下可复现

- 数据池 18:36–19:14（38 分钟，网络是瓶颈）：6 个源 fill 都是 100%；train 489,537 / val 2,544 /
  holdout 2,466 篇。中文保留率 89%（`too_short` 14,233、`low_cjk_ratio` 7,894），代码 95%，数学 90%，
  书籍 300 本。去污染删 252 篇，过半是误杀，见 §8 第 7 条
- 分词器 29–39 秒一个；每组切 3 亿 token 约 1 分钟，六组都在配额上方 0.05% 以内
- 第一组（zh00_v32k）第 1000 步 val loss 6.12。比同期训练 loss 高很多是预期的：验证集里 31% 是这组
  没见过的中文
- **消融从一个固定在 `eea8acb` 的 worktree 里跑**（`../whetstone-ablation-v1`，`datasets/`、
  `results/` 是指回主仓库的软链接），这样一天里主仓库切分支、改代码都不会让后面几组用上另一份代码。
  跑完之前别删它；中断了就在那个目录里重跑同一条 `run` 命令续上

### 步骤 5：正式数据集 + 重训 tokenizer
```bash
python3 -m datatools.prepare configs/mixture_v1.json --out_dir datasets/prepared
python3 train_tokenizer.py --data datasets/prepared/train.jsonl --out tokenizer/v1_32k --vocab_size 32768
python3 -m datatools.tokenizer_stats --probe --tokenizer tokenizer/v1_32k tokenizer/zh_6400
for s in train val; do
  python3 -m datatools.tokenize_corpus datasets/prepared/$s.jsonl \
    --tokenizer tokenizer/v1_32k --out datasets/prepared/$s
done
```
验收：新词表在代码上的 `chars/token` 明显高于 2.23（旧词表的值），中文不显著变差。
`.bin` 绑定分词器指纹，换词表必须重跑 `tokenize_corpus`（不重跑会直接报错）。

**注意重训 tokenizer 会让所有旧 checkpoint 失效**——`resolve_model_config` 会在 `vocab_size`
不匹配时直接报错，这是设计如此。本地若还有想留的 29M 权重，先归档。

### 步骤 6：正式预训练（~100M / ~10B token，租卡）
```bash
python3 pretrain.py --dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3 \
  --tokenizer_path tokenizer/v1_32k --max_seq_len 2048 \
  --data_path datasets/prepared/train.bin --val_data_path datasets/prepared/val.bin \
  --val_every 500 --save_dir results --dtype bfloat16
```
前提是 §3.2 已经做完（§3.1 已完成）。填 `docs/experiments.md` 的 loss / PPL 表。

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
| 数据 | `datasets/` 是空的 | 有评测集 `datasets/eval/`、完整冒烟语料 `datasets/smoke_full/`，以及 memmap 端到端用过的无代码版 `datasets/smoke/`（git 忽略）| 需重新拉 |

本地环境搭建：

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
HF_HUB_OFFLINE=1 python -m pytest tests/ -q     # 483 passed，其中端到端消融测试约 30 秒
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
7. **去污染会被只有通用题干的选择题误杀，出正式数据集之前要修**。消融数据池（0.77B token）的去污染
   删了 252 篇，其中 136 篇来自两道 TAL-SCQ5K 题：题干只有 `下列说法正确的是．` 和
   `下列说法中正确的是（~ ~ ~ ）．`，内容全在选项里。题干按自身长度索引（8 个字，刚好够
   `MIN_GRAM`），于是命中每一篇带这句话的文档。MMLU 的 `Which one of the following statements is
   true:`（10 篇）同理。修法：`fetch_evals` 把选项拼进 `question` 再索引，模型看到的题目本来就带选项。
   这只占数据池文档的 0.05%，而且所有组共用同一个数据池，不影响消融
6. **租什么卡、租多久？** 取决于 §3.3 的实测结果。如果显存宽裕，`docs/corpus-plan.md` 里
   ~185M / 18B token 的档位也在射程内

改名已全部完成：代码、文档、远端仓库都是 **`irroca/Whetstone`**。如果你手上还有指向旧名的
clone，GitHub 会一直重定向，但建议顺手改掉：

```bash
git remote set-url origin https://github.com/irroca/Whetstone.git
```
