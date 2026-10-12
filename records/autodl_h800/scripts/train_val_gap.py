"""Train loss averaged over the 500 updates before each validation, against that validation."""
import glob
import json

path = sorted(glob.glob("/root/autodl-tmp/results/pretrain_v2/runs/*/metrics.jsonl"))[-1]
rows = [json.loads(line) for line in open(path) if line.strip()]
train = [r for r in rows if r["split"] == "train"]
for v in (r for r in rows if r["split"] == "val"):
    window = [r["loss"] for r in train if v["step"] - 500 < r["step"] <= v["step"]]
    mean = sum(window) / len(window)
    print(f"step {v['step']:>6}  train(500-step mean) {mean:.4f}  val {v['loss']:.4f}  val - train {v['loss'] - mean:+.4f}")
