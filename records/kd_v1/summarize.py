"""Recompute the numbers quoted in docs/experiments.md for the distillation round from this directory.

``<arm>_s<seed>/`` are scripts/mac_kd_v1.sh runs (sft, kd05, kd10, kd05t1, kd10t1: the 42.5M student, 3 seeds),
``rl_<variant>_s1/`` scripts/mac_student_rl.sh (GRPO from sft_s1). Reads each run's dev/test JSON
(eval_env.py, with per-item rows) and grpo_metrics.jsonl; paired tests come from ``paired.py``::

    python3 records/kd_v1/summarize.py
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from paired import paired_compare  # noqa: E402


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    print("arm      dev mean  test mean  test sd   test per seed")
    for arm in ("sft", "kd05", "kd10", "kd05t1", "kd10t1"):
        runs = [HERE / f"{arm}_s{s}" for s in (1, 2, 3)]
        dev = [load(r / "dev.json")["accuracy"] for r in runs]
        test = [load(r / "test.json")["accuracy"] for r in runs]
        print(f"{arm:<8} {st.mean(dev):8.3f} {st.mean(test):9.3f} {st.stdev(test):8.3f}   "
              + " / ".join(f"{t:.3f}" for t in test))
    print("\nstudent GRPO from sft_s1 (one seed), paired against sft_s1")
    for split in ("dev", "test"):
        base = load(HERE / "sft_s1" / f"{split}.json")
        print(f"  {split}: sft_s1 {base['accuracy']:.3f}")
        for variant in ("rl_dapo_lite_s1", "rl_grpo_s1"):
            run = load(HERE / variant / f"{split}.json")
            c = paired_compare(base["per_item"], run["per_item"])
            print(f"    {variant}: {run['accuracy']:.3f}  +{c['only_b']} -{c['only_a']}  p {c['p_value']:.2g}")
    for variant in ("rl_dapo_lite_s1", "rl_grpo_s1"):
        rows = [json.loads(line) for line in (HERE / variant / "grpo_metrics.jsonl").read_text().splitlines() if line]
        ev = [(r["step"], round(r["eval_accuracy"], 3)) for r in rows if "eval_accuracy" in r]
        print(f"\n{variant} dev accuracy by step: {ev}")


if __name__ == "__main__":
    main()
