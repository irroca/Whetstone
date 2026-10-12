# posttrain_v3：更难的题（三位数加减法）和第一个代码环境（2026-10-13，Mac，已脱敏）

结果和分析在 `docs/experiments.md` 的 “Harder tasks” 一节；`python3 records/posttrain_v3/summarize.py` 重算那里的每个数字。
每个跑次一个种子，MPS 上训练不逐位可复现。没有权重（每个 398MB，本地 `results/` 里有，不进存档仓库）。

| 目录 | 是什么 |
|---|---|
| `arith/sft/` | 第一轮的 SFT 策略（只见过 1–2 位数）在 `datasets/arith_v2` 的 dev / test 上的打分，含逐题结果（`per_item`） |
| `arith/sft123/` | 用 1–3 位数的 `sft.jsonl` 重新做 SFT（1 个 epoch）作参照，含训练日志 |
| `arith/<变体>_s1/` | `scripts/mac_posttrain_v3.sh` 的 GRPO 跑次（300 步，lr 2e-5，32 题 × 8 条）：`dev.json`、`test.json`、`grpo_metrics.jsonl`（含新的 `entropy_all`）、`runs/` |
| `pyfunc/sft_all_families/` | 在 22 个模板族上 SFT：见过的族 1.000，没见过的 0.000 |
| `pyfunc/sft_A_only.train.log` | 只在 12 个族上 SFT 的训练日志（loss 0.0000）；它在另外 10 个族上 0.000 的打分只在对话里看过，没存文件 |
| `pyfunc/sft_ab30/` | 12 个族 + 另外 10 个族各 3 条的 SFT：`dev_B` / `test_B` / `test_A` / `test_C` 的打分（逐族、逐题）。**没有 GRPO 跑次**：那一次因为内存换页停掉了 |
| `summarize.py` | 重算脚本 |

数据用 `MAX_DIGITS=3 bash scripts/arith_data.sh datasets/arith_v2`（算术）和 `scripts/mac_pyfunc_v2.sh`（代码，依赖 `pyfunc-env`
那个 PR 里的环境）按固定种子重新生成。
