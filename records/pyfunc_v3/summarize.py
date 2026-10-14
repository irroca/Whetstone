"""pyfunc round two: does mixing the SFT-only families (A) into the RL pool prevent forgetting?

    python3 records/pyfunc_v3/summarize.py

Reads ``<run>/{dev_B,test_B,test_A,test_C}.json`` (``eval_env.py --group_by family``) for the SFT policy
(``sft/``) and for ``grpo_{B,AB}_s{seed}`` (scripts/pyfunc_v3.sh, 200 steps, ``--truncated_reward 0``).
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT.parents[1]))
from paired import paired_compare  # noqa: E402

SPLITS = ("dev_B", "test_B", "test_A", "test_C")


def load(run: Path, split: str) -> dict:
    return json.load(open(run / f"{split}.json"))


def seeds(runs, split) -> str:
    return " ".join("%.2f" % load(r, split)["accuracy"] for r in runs)


def main() -> None:
    sft = ROOT / "sft"
    arms = {arm: sorted(ROOT.glob(f"grpo_{arm}_s*")) for arm in ("B", "AB")}
    print(f"{'run':<9}{'n':>2}" + "".join(f"{s:>17}" for s in SPLITS))
    print(f"{'sft':<9}{'':>2}" + "".join(f"{load(sft, s)['accuracy']:>17.3f}" for s in SPLITS))
    for arm, runs in arms.items():
        cells = []
        for s in SPLITS:
            v = [load(r, s)["accuracy"] for r in runs]
            cells.append(f"{statistics.mean(v):.3f} ± {statistics.stdev(v) if len(v) > 1 else 0:.3f}")
        print(f"{'grpo ' + arm:<9}{len(runs):>2}" + "".join(f"{c:>17}" for c in cells))
        print(f"{'  seeds':<11}" + "".join(f"{seeds(runs, s):>17}" for s in SPLITS))
    print("\ntest_A per family, mean over seeds: sft / B / AB")
    fam: dict = {}
    for name, runs in (("sft", [sft]), ("B", arms["B"]), ("AB", arms["AB"])):
        per = [load(r, "test_A")["by_group"] for r in runs]
        for k in per[0]:
            fam.setdefault(k, {})[name] = statistics.mean(p[k]["accuracy"] for p in per)
    for k, v in sorted(fam.items()):
        print(f"  {k.removeprefix('family='):<20}" + "  ".join(f"{v[n]:.2f}" for n in ("sft", "B", "AB")))
    print("\npaired McNemar, AB vs B (pooled over matching seeds):")
    for s in ("test_B", "test_A"):
        a = b = 0
        for x, y in zip(arms["B"], arms["AB"]):
            r = paired_compare(load(x, s)["per_item"], load(y, s)["per_item"])
            a += r["only_a"]; b += r["only_b"]
        from paired import mcnemar_exact
        print(f"  {s}: only B right {a}, only AB right {b}, p = {mcnemar_exact(a, b):.3g}")


if __name__ == "__main__":
    main()
