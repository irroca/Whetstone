#!/bin/bash
source /root/Whetstone/scripts/autodl/env.sh
cd /root/Whetstone
out=/root/autodl-tmp/results/bench_ablation
mkdir -p $out
common="--tokenizer_path tokenizer/v1_32k --dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3 --max_seq_len 2048 --accumulation_steps 8 --peak_tflops 756"
for c in ecaf51d 46c459d ffbb829 c122a93; do
  git checkout -q $c
  echo "=== $c $(git log -1 --format=%s) $(date +%T)"
  python bench_train.py $common --batch_sizes 32 --compile True --out $out/$c.json 2>&1 | grep --line-buffered -v -E "UserWarning|warnings.warn"
done
git checkout -q throughput
echo "=== $(git rev-parse --short HEAD) final grid $(date +%T)"
python bench_train.py $common --batch_sizes 16 32 64 --compile True --attention_backends cudnn --out $out/final.json 2>&1 | grep --line-buffered -v -E "UserWarning|warnings.warn"
echo "ABLATION DONE $(date +%T) at $(git rev-parse --short HEAD) dirty=$(git status --short | wc -l)"
