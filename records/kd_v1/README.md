# kd_v1: distilling the 100M RL policy into a 42.5M student, and RL on the student (Mac, MPS)

Teacher `posttrain_v1/dapo_lite_s1` (dev 0.982 / test 0.950 on `datasets/arith_v1`), student the ablation model
`zh30_v32k` (dim 512, 8 layers, same 32k vocabulary). Numbers: `docs/experiments.md` ("Distillation"),
recomputed by `summarize.py`. No weights here (`*.pth` stay in `results/`).

| Path | What |
|---|---|
| `<arm>_s<seed>/` | `scripts/mac_kd_v1.sh`: `sft` (plain CE), `kd05` / `kd10` (alpha 0.5 / 1.0, temperature 2), `kd05t1` / `kd10t1` (temperature 1); seeds 1-3; `dev.json`, `test.json` (per item), `runs/` |
| `rl_dapo_lite_s1/`, `rl_grpo_s1/` | `scripts/mac_student_rl.sh`: GRPO from `sft_s1`, 300 steps, lr 2e-5; adds `grpo_metrics.jsonl` |
| `queue_*.log` | the three queue logs |
| `summarize.py` | recompute script |
