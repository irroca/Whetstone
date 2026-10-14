"""Recompute the curriculum / fix-grid numbers in docs/experiments.md from the committed records.

    python3 records/curriculum_v1/summarize.py [records/curriculum_v1]

Layout: ``<run>/{dev,test}.json`` (``eval_env.py`` output), ``<run>/grpo_metrics.jsonl``; runs are named
``<variant>_s<seed>`` and ``sft_{dev,test}.json`` is the starting policy.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
sys.path.insert(0, str(ROOT.resolve().parents[1]))
from paired import paired_compare  # noqa: E402


def acc(path: Path) -> float:
    return json.load(open(path))["accuracy"]


def train_rows(run: Path) -> list[dict]:
    return [r for r in map(json.loads, open(run / "grpo_metrics.jsonl")) if "lr" in r]


def mean(xs):
    return sum(xs) / len(xs)


def main() -> None:
    runs: dict[str, dict[int, Path]] = defaultdict(dict)
    for d in sorted(ROOT.iterdir()):
        m = re.fullmatch(r"(.+)_s(\d+)", d.name)
        if m and (d / "test.json").exists():
            runs[m.group(1)][int(m.group(2))] = d
    if ROOT.name.endswith("_600"):  # the 600-step baselines are the 4090 grid: the fix rerun (v5) where there is one, else v4
        for name in ("grpo", "dr_grpo", "dapo_lite"):
            for seed in (1, 2, 3):
                for v in ("posttrain_v5", "posttrain_v4"):
                    d = ROOT.parent / v / f"{name}_s{seed}"
                    if (d / "test.json").exists():
                        runs[name][seed] = d
                        break
    print(f"sft: dev {acc(ROOT / 'sft_dev.json'):.3f}  test {acc(ROOT / 'sft_test.json'):.3f}\n")
    print(f"{'variant':<15}{'n':>2} {'dev mean':>9} {'test mean':>10} {'test sd':>8} {'test per seed':<34}{'silent(last100)':>16}{'3-digit share':>14}")
    for name, seeds in runs.items():
        dev = [acc(d / "dev.json") for d in seeds.values()]
        test = [acc(d / "test.json") for d in seeds.values()]
        silent = mean([mean([r["silent_group_frac"] for r in train_rows(d)[-100:]]) for d in seeds.values()])
        share = []
        for d in seeds.values():
            rows = train_rows(d)
            w = [sum(v for k, v in r.items() if k.startswith("cur_w") and "n_digits=3" in k) for r in rows[-100:]]
            share.append(mean(w) if rows and any("cur_w" in k for k in rows[0]) else float("nan"))
        sd = statistics.stdev(test) if len(test) > 1 else float("nan")
        per_seed = " ".join(f"{t:.3f}" for t in test)
        print(f"{name:<15}{len(test):>2} {mean(dev):>9.3f} {mean(test):>10.3f} {sd:>8.3f} {per_seed:<34}{silent:>16.3f}{mean(share):>14.3f}")
    print("\npaired McNemar on test (pooled over matching seeds; items, not seeds, are the unit):")
    for a, b in (("grpo", "grpo_cur"), ("dapo_lite", "dapo_lite_cur"), ("grpo", "dr_grpo"), ("grpo", "dapo_lite")):
        if a in runs and b in runs:
            only_a = only_b = 0
            for s in sorted(set(runs[a]) & set(runs[b])):
                r = paired_compare(json.load(open(runs[a][s] / "test.json"))["per_item"],
                                   json.load(open(runs[b][s] / "test.json"))["per_item"])
                only_a += r["only_a"]; only_b += r["only_b"]
            print(f"  {a} vs {b}: only {a} right {only_a}, only {b} right {only_b}, p = {mean_p(only_a, only_b):.3g}")


def mean_p(a: int, b: int) -> float:
    from paired import mcnemar_exact
    return mcnemar_exact(a, b)


if __name__ == "__main__":
    main()
