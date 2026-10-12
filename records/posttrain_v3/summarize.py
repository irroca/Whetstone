"""Recompute the numbers quoted in docs/experiments.md for the harder-task round from this directory.

``arith/`` is the 1-3 digit GRPO grid (scripts/mac_posttrain_v3.sh), ``pyfunc/`` the code environment
(scripts/mac_pyfunc_v2.sh). Reads each run's dev/test JSON (eval_env.py, with per-item rows) and
grpo_metrics.jsonl; paired tests come from ``paired.py`` at the repo root::

    python3 records/posttrain_v3/summarize.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paired import paired_compare  # noqa: E402

WINDOWS = ((1, 50), (51, 100), (101, 200), (201, 300))
DYNAMICS = ("silent_group_frac", "entropy", "entropy_all", "clip_frac", "grad_norm", "completion_len", "groups_used", "sec")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metrics(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "grpo_metrics.jsonl").read_text().splitlines() if line]


def digits(result: dict) -> str:
    out = []
    for n in (1, 2, 3):
        rows = [v for k, v in result["by_group"].items() if k.endswith(f"n_digits={n}")]
        items = sum(r["items"] for r in rows)
        out.append(f"{n}d {sum(r['accuracy'] * r['items'] for r in rows) / items:.3f}" if items else f"{n}d -")
    return "  ".join(out)


def window_means(rows: list[dict], key: str, lo: int, hi: int) -> float | None:
    values = [r[key] for r in rows if lo <= r.get("step", 0) <= hi and key in r]
    return st.mean(values) if values else None


def fmt(x: float | None) -> str:
    return "   -  " if x is None else f"{x:6.3f}"


def arith(root: Path) -> None:
    runs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "test.json").exists())
    if not runs:
        return
    print("== arith (1-3 digit +/-), SFT trained on 1-2 digit only")
    for name in ("sft", "sft123"):
        if (root / name / "test.json").exists():
            print(f"{name:>14}: dev {load(root / name / 'dev.json')['accuracy']:.3f}  "
                  f"test {load(root / name / 'test.json')['accuracy']:.3f}  |  {digits(load(root / name / 'test.json'))}")
    sft = load(root / "sft" / "test.json")
    print("\nPer run: dev, test, test by digits, paired against SFT (McNemar exact)")
    tests = {}
    for run in runs:
        if run.name in ("sft", "sft123"):
            continue
        dev, test = load(run / "dev.json"), load(run / "test.json")
        tests[run.name] = test
        cmp = paired_compare(sft["per_item"], test["per_item"])
        print(f"{run.name:>16}: dev {dev['accuracy']:.3f}  test {test['accuracy']:.3f}  |  {digits(test)}  |  "
              f"format {test['format_rate']:.3f}  len {test['completion_len']:.1f}  |  "
              f"+{cmp['only_b']} -{cmp['only_a']} p {cmp['p_value']:.2g}")
    names = list(tests)
    if len(names) > 1:
        print("\nPaired between runs (test): only-row-wins / only-column-wins, p")
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                c = paired_compare(tests[a]["per_item"], tests[b]["per_item"])
                print(f"  {a} vs {b}: {c['only_a']} / {c['only_b']}, p {c['p_value']:.2f}")
    print("\nTraining dynamics (mean per window of steps)")
    print(f"{'run':>16} {'window':>9} " + " ".join(f"{k[:10]:>10}" for k in ("accuracy",) + DYNAMICS))
    for run in runs:
        if not (run / "grpo_metrics.jsonl").exists():
            continue
        rows = metrics(run)
        for lo, hi in WINDOWS:
            vals = [window_means(rows, k, lo, hi) for k in ("accuracy",) + DYNAMICS]
            print(f"{run.name:>16} {lo:>4}-{hi:<4} " + " ".join(f"{fmt(v):>10}" for v in vals))


def pyfunc(root: Path) -> None:
    runs = sorted(p for p in root.iterdir() if p.is_dir() and (p / "test_B.json").exists())
    if not runs:
        return
    print("\n== pyfunc: families A (SFT), B (3 SFT examples, then RL), C (never trained)")
    base = None
    for run in runs:
        scores = {s: load(run / f"{s}.json") for s in ("dev_B", "test_B", "test_A", "test_C") if (run / f"{s}.json").exists()}
        print(f"{run.name:>14}: " + "  ".join(f"{s} {r['accuracy']:.3f}" for s, r in scores.items()))
        if run.name.startswith("sft"):
            base = scores
    if base is None:
        return
    for run in runs:
        if run.name.startswith("sft"):
            continue
        for split in ("test_B", "test_A", "test_C"):
            if (run / f"{split}.json").exists() and split in base:
                c = paired_compare(base[split]["per_item"], load(run / f"{split}.json")["per_item"])
                print(f"  {run.name} vs SFT on {split}: +{c['only_b']} -{c['only_a']}, p {c['p_value']:.2g}")
    print("\nPer family, test_B / test_C (SFT -> each run)")
    for split in ("test_B", "test_C"):
        families = sorted(base[split]["by_group"])
        print(f"  {split}")
        for fam in families:
            cells = [f"{base[split]['by_group'][fam]['accuracy']:.2f}"]
            for run in runs:
                if run.name.startswith("sft") or not (run / f"{split}.json").exists():
                    continue
                cells.append(f"{load(run / f'{split}.json')['by_group'][fam]['accuracy']:.2f}")
            print(f"    {fam:>20}: " + " -> ".join(cells))
    for run in runs:
        if (run / "grpo_metrics.jsonl").exists():
            rows = metrics(run)
            last = max(r["step"] for r in rows)
            print(f"\n{run.name}: sampled accuracy by window / silent / entropy_all")
            for lo, hi in ((1, 50), (51, 100), (101, 150), (151, last)):
                print(f"  {lo:>3}-{hi:<3}: " + " ".join(f"{fmt(window_means(rows, k, lo, hi))}"
                                                        for k in ("accuracy", "silent_group_frac", "entropy_all")))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent)
    root = parser.parse_args().dir
    if (root / "arith").is_dir():
        arith(root / "arith")
    if (root / "pyfunc").is_dir():
        pyfunc(root / "pyfunc")


if __name__ == "__main__":
    main()
