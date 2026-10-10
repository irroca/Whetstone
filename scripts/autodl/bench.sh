#!/bin/bash
# Throughput, peak memory and MFU of the formal model on this card, to pick the micro-batch and
# whether to compile. PEAK_TFLOPS is the card's dense bf16 peak (H800 SXM 989, A800 312).
export PATH=/root/miniconda3/bin:$PATH HF_HUB_OFFLINE=1
cd "$(dirname "$0")/../.."
mkdir -p /root/autodl-tmp/results
python bench_train.py --tokenizer_path tokenizer/v1_32k --dim 768 --n_layers 12 --n_heads 12 \
  --n_kv_heads 3 --max_seq_len 2048 --batch_sizes ${BATCH_SIZES:-16 32 64} --compile False True \
  --accumulation_steps 8 --peak_tflops ${PEAK_TFLOPS:?set PEAK_TFLOPS, e.g. 989 for H800 SXM} \
  --out /root/autodl-tmp/results/bench.json
