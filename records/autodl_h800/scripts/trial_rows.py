import glob
import json

path = sorted(glob.glob("/root/autodl-tmp/results/pretrain_v2_trial/runs/*/metrics.jsonl"))[-1]
prev = None
for line in open(path):
    r = json.loads(line)
    rate = ""
    if r["split"] == "train" and prev is not None:
        rate = f"{(r['tokens'] - prev['tokens']) / (r['elapsed_s'] - prev['elapsed_s']):,.0f} tok/s over the interval"
    print(r["step"], r["split"], round(r["loss"], 4), rate)
    if r["split"] == "train":
        prev = r
