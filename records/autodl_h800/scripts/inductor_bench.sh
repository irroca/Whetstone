#!/bin/bash
source /root/Whetstone/scripts/autodl/env.sh
cd /root/Whetstone
out=/root/autodl-tmp/results/bench_ablation
common="--tokenizer_path tokenizer/v1_32k --dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3 --max_seq_len 2048 --accumulation_steps 8 --peak_tflops 756 --batch_sizes 32 --compile True --attention_backends cudnn"
echo "=== coordinate_descent_tuning $(date +%T)"
TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 python bench_train.py $common --out $out/coord_descent.json 2>&1 | grep --line-buffered -v -E "UserWarning|warnings.warn" | tail -2
echo "=== max_autotune + coordinate_descent_tuning $(date +%T)"
TORCHINDUCTOR_MAX_AUTOTUNE=1 TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1 python bench_train.py $common --out $out/max_autotune.json 2>&1 | grep --line-buffered -v -E "UserWarning|warnings.warn|AUTOTUNE|SingleProcess|^  " | tail -2
echo "INDUCTOR DONE $(date +%T)"
