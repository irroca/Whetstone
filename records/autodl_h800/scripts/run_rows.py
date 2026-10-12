"""Formal run progress: interval throughput, loss, and ETA from the newest metrics.jsonl."""
import glob
import json
import sys
import time

TOTAL_UPDATES = 18919
TOKENS_PER_UPDATE = 32 * 8 * 2047

pattern = sys.argv[1] if len(sys.argv) > 1 else "/root/autodl-tmp/results/pretrain_v2/runs/*/metrics.jsonl"
rows = [json.loads(line) for path in sorted(glob.glob(pattern)) for line in open(path) if line.strip()]
train = [r for r in rows if r["split"] == "train"]
vals = [r for r in rows if r["split"] == "val"]
every = int(sys.argv[2]) if len(sys.argv) > 2 else 1
prev = None
for r in train:
    if prev is not None and r["elapsed_s"] > prev["elapsed_s"] and (r["step"] % (10 * every) == 0 or r is train[-1]):
        rate = (r["tokens"] - prev["tokens"]) / (r["elapsed_s"] - prev["elapsed_s"])
        print(f"step {r['step']:>6}  loss {r['loss']:.4f}  lr {r['lr']:.2e}  gnorm {r['grad_norm']:.3f}  {rate:,.0f} tok/s")
    prev = r
for v in vals:
    print(f"val step {v['step']:>6}  loss {v['loss']:.4f}  ppl {v['ppl']:.1f}")
if len(train) > 20:
    a, b = train[-21], train[-1]
    rate = (b["step"] - a["step"]) / (b["elapsed_s"] - a["elapsed_s"])
    left = (TOTAL_UPDATES - b["step"]) / rate
    print(f"last 200 updates: {rate * TOKENS_PER_UPDATE:,.0f} tok/s; step {b['step']}/{TOTAL_UPDATES}, "
          f"{left / 3600:.2f} h left, ETA {time.strftime('%m-%d %H:%M', time.localtime(time.time() + left))}")
