"""Recompute the post-training numbers quoted in docs/experiments.md from this directory.

Reads each run's dev.json / test.json (eval_env.py), grpo_metrics.jsonl (grpo.py), run.log and
gpu.csv, all written by scripts/autodl/posttrain.sh; needs nothing outside the standard library::

    python3 records/posttrain_v1/summarize.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
from datetime import datetime
from itertools import pairwise
from pathlib import Path

VARIANTS = ("grpo", "grpo_kl", "dr_grpo", "dapo_lite")
SEEDS = (1, 2, 3)
GROUPS = {"op=+ n_digits=1": "1d+", "op=+ n_digits=2": "2d+", "op=- n_digits=1": "1d-", "op=- n_digits=2": "2d-"}
WINDOWS = ((1, 50), (51, 150), (151, 300), (301, 500))
DYNAMICS = ("silent_group_frac", "entropy", "kl", "grad_norm", "clip_frac", "completion_len", "sec")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent)
    root = parser.parse_args().dir

    def score(run: str, split: str) -> dict:
        return json.loads((root / run / f"{split}.json").read_text())

    def metrics(run: str) -> list[dict]:
        return [json.loads(line) for line in (root / run / "grpo_metrics.jsonl").read_text().splitlines() if line]

    def by_group(result: dict) -> str:
        return "  ".join(f"{GROUPS[g]} {result['by_group'][g]['accuracy']:.3f}" for g in GROUPS)

    for split in ("dev", "test"):
        result = score("sft", split)
        items = ", ".join(f"{GROUPS[g]} {result['by_group'][g]['items']}" for g in GROUPS)
        print(f"SFT {split}: {result['accuracy']:.3f}  |  {by_group(result)}  (items: {items})")

    runs = [f"{v}_s{s}" for v in VARIANTS for s in SEEDS]
    print("\nPer run: dev, test, then test by group")
    for run in runs:
        dev, test = score(run, "dev"), score(run, "test")
        print(f"{run:>13}: dev {dev['accuracy']:.3f}  test {test['accuracy']:.3f}  |  {by_group(test)}"
              f"  |  format {test['format_rate']:.3f}  hack {test['hack_rate']:.3f}")

    print("\nPer variant, over seeds: test mean / sd / [min, max], dev mean, test by group")
    tests = {}
    for v in VARIANTS:
        vr = [f"{v}_s{s}" for s in SEEDS]
        tests[v] = [score(r, "test")["accuracy"] for r in vr]
        groups = "  ".join(
            f"{name} {st.mean(score(r, 'test')['by_group'][g]['accuracy'] for r in vr):.3f}" for g, name in GROUPS.items()
        )
        print(f"{v:>10}: {st.mean(tests[v]):.3f} / {st.stdev(tests[v]):.4f} / [{min(tests[v]):.3f}, {max(tests[v]):.3f}]"
              f"  dev {st.mean(score(r, 'dev')['accuracy'] for r in vr):.3f}  |  {groups}")
    pooled = [t for v in VARIANTS for t in tests[v]]
    groups = "  ".join(
        f"{name} {st.mean(score(r, 'test')['by_group'][g]['accuracy'] for r in runs):.3f}" for g, name in GROUPS.items()
    )
    print(f"{'all 12':>10}: {st.mean(pooled):.3f} / {st.stdev(pooled):.4f} / [{min(pooled):.3f}, {max(pooled):.3f}]"
          f"  |  {groups}")
    best = max(runs, key=lambda r: score(r, "dev")["accuracy"])
    print(f"best on dev: {best} (dev {score(best, 'dev')['accuracy']:.3f}) -> test {score(best, 'test')['accuracy']:.3f}")
    print(f"binomial standard error of one 500-item score at 0.95: {math.sqrt(0.95 * 0.05 / 500):.4f}")

    print("\nWelch's t between variants' test scores (3 seeds each)")
    for i, a in enumerate(VARIANTS):
        for b in VARIANTS[i + 1:]:
            va, vb = st.variance(tests[a]) / len(SEEDS), st.variance(tests[b]) / len(SEEDS)
            t = (st.mean(tests[a]) - st.mean(tests[b])) / math.sqrt(va + vb)
            df = (va + vb) ** 2 / (va**2 / (len(SEEDS) - 1) + vb**2 / (len(SEEDS) - 1))
            print(f"  {a} vs {b}: t {t:+.2f}, df {df:.1f}")

    print("\nTraining dynamics, mean over seeds, per window of steps " + ", ".join(f"{lo}-{hi}" for lo, hi in WINDOWS))
    for v in VARIANTS:
        rows = {s: metrics(f"{v}_s{s}") for s in SEEDS}
        print(f"{v}:")
        for key in DYNAMICS:
            means = []
            for lo, hi in WINDOWS:
                per_seed = [st.mean(r[key] for r in rows[s] if lo <= r["step"] <= hi and key in r) for s in SEEDS]
                means.append(f"{st.mean(per_seed):.4f}")
            print(f"  {key:>17}: {'  '.join(means)}")
        curves = [{r["step"]: r["eval_accuracy"] for r in rows[s] if "eval_accuracy" in r} for s in SEEDS]
        steps = [k for k in sorted(curves[0]) if k % 50 == 0]
        print("  dev curve: " + "  ".join(f"{k}: {st.mean(c[k] for c in curves):.3f}" for k in steps))

    print("\nWall clock from run.log")
    stamps = []
    for line in (root / "run.log").read_text().splitlines():
        m = re.match(r"^=== (\S+) seed (\d) (\d\d-\d\d \d\d:\d\d:\d\d)", line) or re.match(
            r"^=== (start|done)() (\d\d-\d\d \d\d:\d\d:\d\d)", line
        )
        if m:
            name = f"{m[1]}_s{m[2]}" if m[2] else m[1]
            stamps.append((name, datetime.strptime(m[3], "%m-%d %H:%M:%S")))
    for (name, start), (_, end) in pairwise(stamps):
        print(f"  {name:>13}: {(end - start).total_seconds() / 60:.1f} min")
    print(f"  total: {(stamps[-1][1] - stamps[0][1]).total_seconds() / 3600:.2f} h")

    samples = []
    for line in (root / "gpu.csv").read_text().splitlines():
        fields = [f.strip() for f in line.split(",")]
        when = datetime.strptime(fields[0], "%Y/%m/%d %H:%M:%S.%f").replace(year=1900)
        samples.append((when, int(fields[1].split()[0]), int(fields[2].split()[0])))
    print(f"\nGPU (gpu.csv, every 10 s): median utilization {st.median(u for _, _, u in samples):.0f}%")
    for (name, start), (_, end) in pairwise(stamps[1:]):
        run = [(when, mem) for when, mem, _ in samples if start <= when < end]
        peak = max(mem for _, mem in run)
        settled = next(when for when, mem in run if mem >= 0.95 * peak)
        print(f"  {name:>13}: memory reserved {peak / 1024:.1f} GiB, 95% of it after "
              f"{(settled - start).total_seconds() / 60:.1f} min")


if __name__ == "__main__":
    main()
