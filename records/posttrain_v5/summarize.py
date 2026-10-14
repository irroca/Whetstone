"""Recompute the numbers quoted in docs/experiments.md for the ``--truncated_reward 0`` reruns.

Each run here is the same variant and seed as a run in ``records/posttrain_v4/`` (scripts/mac_posttrain_v3.sh,
600 steps, 4090) with ``GRPO_EXTRA='--truncated_reward 0'``, so a completion that never emitted EOS is scored as
wrong. The four cells that ran away in v4 were rerun, plus grpo_s1 and dr_grpo_s1's first attempt (stopped when
the plan changed, not recorded)::

    python3 records/posttrain_v5/summarize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V4 = HERE.parent / "posttrain_v4"
sys.path.insert(0, str(HERE.parents[1]))
from paired import paired_compare  # noqa: E402


def load(path: Path):
    return json.loads(path.read_text())


def rows(run: Path):
    return [json.loads(x) for x in (run / "grpo_metrics.jsonl").read_text().splitlines() if x]


def tail(rs, key, lo=501, hi=600):
    v = [r[key] for r in rs if "completion_len" in r and lo <= r["step"] <= hi and key in r]
    return sum(v) / len(v) if v else float("nan")


def first_over(rs, limit=100.0):
    t = [r for r in rs if "completion_len" in r]
    for i in range(9, len(t)):
        if sum(r["completion_len"] for r in t[i - 9 : i + 1]) / 10 > limit:
            return t[i]["step"]
    return None


def main() -> None:
    print(f"{'run':<13} {'v4 test':>8} {'fix test':>8} {'v4 dev':>7} {'fix dev':>8} {'v4 len':>7} {'fix len':>8} "
          f"{'v4 trunc':>9} {'fix trunc':>9} {'v4 onset':>9} {'fix onset':>9} {'fix s/step':>10}  paired fix vs v4 (test)")
    for run in sorted(p for p in HERE.iterdir() if p.is_dir()):
        a, b = V4 / run.name, run
        ra, rb = rows(a), rows(b)
        ta, tb, da, db = load(a / "test.json"), load(b / "test.json"), load(a / "dev.json"), load(b / "dev.json")
        c = paired_compare(ta["per_item"], tb["per_item"])
        sec = [r["sec"] for r in rb if "sec" in r]
        print(f"{run.name:<13} {ta['accuracy']:8.3f} {tb['accuracy']:8.3f} {da['accuracy']:7.3f} {db['accuracy']:8.3f} "
              f"{ta['completion_len']:7.1f} {tb['completion_len']:8.1f} {tail(ra, 'truncated_frac'):9.2f} "
              f"{tail(rb, 'truncated_frac'):9.2f} {str(first_over(ra)):>9} {str(first_over(rb)):>9} "
              f"{sum(sec) / len(sec):10.2f}  +{c['only_b']} -{c['only_a']} p {c['p_value']:.2g}")


def combined() -> None:
    """The 3x3 grid with the v5 rerun wherever there is one and the v4 run elsewhere (the five v4 runs that never
    ran away were not rerun; grpo_s1, which was, came out bit-identical)."""
    import statistics as st

    print("\ncombined grid (test): variant mean / sd / per seed; * = v5 rerun with --truncated_reward 0")
    for v in ("grpo", "dr_grpo", "dapo_lite"):
        vals, marks = [], []
        for s in (1, 2, 3):
            name = f"{v}_s{s}"
            rerun = (HERE / name / "test.json").exists()
            vals.append(load((HERE if rerun else V4) / name / "test.json")["accuracy"])
            marks.append("*" if rerun else " ")
        print(f"  {v:<10} {st.mean(vals):.3f}  sd {st.stdev(vals):.3f}  " + " / ".join(f"{x:.3f}{m}" for x, m in zip(vals, marks)))


if __name__ == "__main__":
    main()
    combined()
