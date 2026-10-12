"""Formal run health: spikes, non-finite values, throughput spread, and a pause-aware ETA."""
import glob
import json
import math
import time

TOTAL_UPDATES = 18919
path = sorted(glob.glob("/root/autodl-tmp/results/pretrain_v2/runs/*/metrics.jsonl"))[-1]
rows = [json.loads(line) for line in open(path) if line.strip()]
train = [r for r in rows if r["split"] == "train"]
vals = [r for r in rows if r["split"] == "val"]

bad = [r["step"] for r in train if not all(math.isfinite(r[k]) for k in ("loss", "grad_norm"))]
print(f"train rows {len(train)}, val rows {len(vals)}, non-finite at steps: {bad or 'none'}")

after = [r for r in train if r["step"] > 2000]
peak = max(after, key=lambda r: r["grad_norm"])
print(f"grad_norm after warmup: max {peak['grad_norm']:.3f} at step {peak['step']}, "
      f"median {sorted(r['grad_norm'] for r in after)[len(after) // 2]:.3f}")
jumps = sorted(((b["loss"] - a["loss"], b["step"]) for a, b in zip(train, train[1:]) if b["step"] > 1000), reverse=True)[:3]
print("largest row-to-row train-loss rises after step 1000:", ", ".join(f"+{d:.3f} at {s}" for d, s in jumps))

rates = []
for a, b in zip(train, train[1:]):
    if b["elapsed_s"] > a["elapsed_s"] and b["step"] > 20:
        rates.append(((b["tokens"] - a["tokens"]) / (b["elapsed_s"] - a["elapsed_s"]), b["step"]))
steady = sorted(r for r, s in rates)
print(f"interval tok/s: median {steady[len(steady) // 2]:,.0f}, p5 {steady[len(steady) // 20]:,.0f}, "
      f"min {steady[0]:,.0f} (intervals holding a validation/checkpoint pause are the low tail)")
first, last = train[0], train[-1]
per_update = (last["elapsed_s"] - first["elapsed_s"]) / (last["step"] - first["step"])
left = (TOTAL_UPDATES - last["step"]) * per_update
print(f"step {last['step']}/{TOTAL_UPDATES} ({last['step'] / TOTAL_UPDATES:.1%}), {last['tokens'] / 1e9:.2f}B tokens; "
      f"{per_update:.3f} s/update since step 1 incl. pauses -> {left / 3600:.2f} h left, "
      f"training ends ~{time.strftime('%H:%M', time.localtime(time.time() + left))}")
print("val:", "  ".join(f"{v['step']}:{v['loss']:.3f}" for v in vals))
