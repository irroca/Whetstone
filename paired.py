"""Paired comparison of two ``eval_env.py`` result files on the same task file.

Two checkpoints scored on the same 500 items differ by a few points far more often than they
differ in skill: ``dev`` moved by 1.2 points and ``test`` by -5.2 for the same DPO run. The items
are shared, so the right test is paired. McNemar's exact test looks only at the items on which
exactly one checkpoint was right and asks whether they split evenly:

    python3 paired.py results/sft_test.json results/dpo_test.json

It needs the ``per_item`` rows ``eval_env.py`` writes, and refuses two files whose questions differ.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Sequence


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value: ``min(only_a, only_b)`` or fewer of ``n`` fair coin flips, doubled.

    Computed in log space, since 2**-n underflows long before a few thousand discordant items.
    """
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    log_terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
                 for i in range(k + 1)]
    peak = max(log_terms)
    tail = math.exp(peak) * sum(math.exp(x - peak) for x in log_terms)
    return min(1.0, 2 * tail)


def paired_compare(a: Sequence[dict], b: Sequence[dict]) -> dict:
    """Outcome counts and McNemar p for two ``per_item`` lists over the same tasks, in order."""
    if len(a) != len(b):
        raise ValueError(f"different numbers of items: {len(a)} vs {len(b)}")
    for i, (x, y) in enumerate(zip(a, b)):
        if x["question"] != y["question"]:
            raise ValueError(f"item {i} is a different question in the two files")
    both = sum(x["correct"] and y["correct"] for x, y in zip(a, b))
    only_a = sum(x["correct"] and not y["correct"] for x, y in zip(a, b))
    only_b = sum(y["correct"] and not x["correct"] for x, y in zip(a, b))
    neither = len(a) - both - only_a - only_b
    n = max(len(a), 1)
    return {
        "items": len(a), "both": both, "only_a": only_a, "only_b": only_b, "neither": neither,
        "accuracy_a": (both + only_a) / n, "accuracy_b": (both + only_b) / n,
        "difference": (only_b - only_a) / n, "p_value": mcnemar_exact(only_a, only_b),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="McNemar test between two eval_env.py result files")
    parser.add_argument("a", help="result JSON of the baseline")
    parser.add_argument("b", help="result JSON of the candidate")
    args = parser.parse_args(argv)
    rows = []
    for path in (args.a, args.b):
        with open(path, encoding="utf-8") as fh:
            result = json.load(fh)
        if "per_item" not in result:
            raise SystemExit(f"{path} has no per_item rows: rerun eval_env.py with this version")
        rows.append(result["per_item"])
    r = paired_compare(*rows)
    print(
        f"{r['items']} items: A {r['accuracy_a']:.3f}, B {r['accuracy_b']:.3f} (B - A {r['difference']:+.3f})\n"
        f"  both right {r['both']}, only A {r['only_a']}, only B {r['only_b']}, neither {r['neither']}\n"
        f"  McNemar exact p = {r['p_value']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
