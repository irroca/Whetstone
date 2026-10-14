"""Recompute the numbers quoted in docs/experiments.md for the 3-digit seed grid (RTX 4090) from this directory.

``<variant>_s<seed>/`` are scripts/mac_posttrain_v3.sh runs (despite the name it ran on a 4090) at 600 steps:
``dev.json`` / ``test.json`` (eval_env.py, per item) and ``grpo_metrics.jsonl``. ``sft_*.json`` is the shared
starting policy::

    python3 records/posttrain_v4/summarize.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from paired import paired_compare  # noqa: E402

VARIANTS = ("grpo", "dr_grpo", "dapo_lite")
SEEDS = (1, 2, 3)


def load(path: Path):
    return json.loads(path.read_text())


def metrics(run: Path):
    return [json.loads(x) for x in (run / "grpo_metrics.jsonl").read_text().splitlines() if x]


def mean(rows, key, lo, hi):
    vals = [r[key] for r in rows if "completion_len" in r and lo <= r["step"] <= hi and key in r]
    return sum(vals) / len(vals) if vals else float("nan")


def onset(rows, limit=100.0):
    """First step where the mean completion length of the 10 steps ending there exceeds ``limit`` tokens."""
    train = [r for r in rows if "completion_len" in r]
    for i in range(9, len(train)):
        if sum(r["completion_len"] for r in train[i - 9 : i + 1]) / 10 > limit:
            return train[i]["step"]
    return None


def main() -> None:
    sft = {s: load(HERE / f"sft_{s}.json") for s in ("dev", "test")}
    print(f"SFT start: dev {sft['dev']['accuracy']:.3f} test {sft['test']['accuracy']:.3f}\n")
    print(f"{'run':<13} {'dev':>6} {'test':>6} {'len ev':>6} {'len 1-100':>9} {'len 501-600':>11} {'trunc 501-600':>13} "
          f"{'onset':>6} {'silent 501-600':>14} {'s/step':>7}")
    table = {}
    for v in VARIANTS:
        for s in SEEDS:
            run = HERE / f"{v}_s{s}"
            dev, test, rows = load(run / "dev.json"), load(run / "test.json"), metrics(run)
            table[(v, s)] = (dev["accuracy"], test["accuracy"], rows)
            sec = [r["sec"] for r in rows if "sec" in r]
            print(f"{v + '_s' + str(s):<13} {dev['accuracy']:6.3f} {test['accuracy']:6.3f} "
                  f"{test.get('completion_len', float('nan')):6.1f} {mean(rows, 'completion_len', 1, 100):9.1f} "
                  f"{mean(rows, 'completion_len', 501, 600):11.1f} {mean(rows, 'truncated_frac', 501, 600):13.2f} "
                  f"{str(onset(rows)):>6} {mean(rows, 'silent_group_frac', 501, 600):14.2f} {st.mean(sec) if sec else float('nan'):7.2f}")
    print("\nper variant (test): mean, sd, per seed")
    for v in VARIANTS:
        t = [table[(v, s)][1] for s in SEEDS]
        d = [table[(v, s)][0] for s in SEEDS]
        print(f"  {v:<10} dev {st.mean(d):.3f}  test {st.mean(t):.3f}  sd {st.stdev(t):.3f}  "
              + " / ".join(f"{x:.3f}" for x in t))
    allt = [(v, s, table[(v, s)][1], onset(table[(v, s)][2])) for v in VARIANTS for s in SEEDS]
    run_t = [t for _, _, t, o in allt if o is not None]
    ok_t = [t for _, _, t, o in allt if o is None]
    print(f"\nlength runaway (onset not None): {len(run_t)} of 9 runs, test mean {st.mean(run_t):.3f}; "
          f"without: {len(ok_t)} runs, test mean {st.mean(ok_t):.3f}")
    print("\npaired on test, same seed, A vs B (only_b = B right and A wrong)")
    for a, b in (("grpo", "dr_grpo"), ("grpo", "dapo_lite"), ("dr_grpo", "dapo_lite")):
        for s in SEEDS:
            c = paired_compare(load(HERE / f"{a}_s{s}" / "test.json")["per_item"],
                               load(HERE / f"{b}_s{s}" / "test.json")["per_item"])
            print(f"  seed {s} {a} vs {b}: +{c['only_b']} -{c['only_a']} p {c['p_value']:.2g}")
    print("\nper type (test), mean over the three seeds")
    for v in VARIANTS:
        groups: dict[str, list[float]] = {}
        for s in SEEDS:
            for k, g in load(HERE / f"{v}_s{s}" / "test.json")["by_group"].items():
                groups.setdefault(k, []).append(g["accuracy"])
        print(f"  {v:<10} " + "  ".join(f"{k.replace('op=', '').replace(' n_digits=', '/')} {st.mean(x):.3f}" for k, x in sorted(groups.items())))


if __name__ == "__main__":
    main()
