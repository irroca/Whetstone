# posttrain_v5: the EOS-runaway fix (`--truncated_reward 0`) on the four cells that ran away

`scripts/mac_posttrain_v3.sh` with `GRPO_EXTRA='--truncated_reward 0'`, 600 steps, same seeds and starting
policy as `records/posttrain_v4/` (4090). `grpo_s1` (which never ran away) is included as a determinism check:
its dev and test results are identical item for item. Numbers: `docs/experiments.md` ("Seed grid ... the fix"),
recomputed by `summarize.py`. No weights here. `dr_grpo_s1` was started once and stopped when the plan changed;
the run recorded here is the complete rerun.

| Path | What |
|---|---|
| `<variant>_s<seed>/` | `dev.json`, `test.json` (per item), `grpo_metrics.jsonl`, `grpo_final.config.json`, `runs/` |
| `run.log` | both queue logs |
| `summarize.py` | recompute script (reads `../posttrain_v4/` for the runs without the fix) |
