# posttrain_v4: the 3-digit seed grid on an RTX 4090 (600 steps)

GRPO / Dr. GRPO / DAPO-lite × seeds 1-3 from the first round's SFT policy on `datasets/arith_v2`
(`scripts/mac_posttrain_v3.sh` with `STEPS=600`, run on the 4090). Numbers: `docs/experiments.md`
("Seed grid on the 3-digit task"), recomputed by `summarize.py`. Weights are not here.

| Path | What |
|---|---|
| `<variant>_s<seed>/` | `dev.json`, `test.json` (per item), `grpo_metrics.jsonl`, `grpo_final.config.json`, `runs/` |
| `sft_dev.json`, `sft_test.json` | the shared starting policy (dev 0.300 / test 0.290) |
| `eos_probe.json` | `probe_eos.py`: P(EOS) after the first `</answer>` of each policy's own greedy completion |
| `run.log` | the queue log |
| `summarize.py` | recompute script |
