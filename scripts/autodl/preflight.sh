#!/bin/bash
# CPU check of the formal data path before paying for a GPU: a tiny model reads the real train.bin
# and val.bin through MemmapPretrainDataset (tokenizer fingerprint and size checks included), saves,
# and the probes run on the result against the real holdout. Shapes match the formal run except the
# model width. Needs ~1GB, so not alongside the test suite on a 2GB no-GPU instance.
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH HF_HUB_OFFLINE=1
cd "$(dirname "$0")/../.."
out=/root/autodl-tmp/results/preflight
rm -rf $out
python pretrain.py --dim 64 --n_layers 2 --n_heads 4 --n_kv_heads 1 \
  --tokenizer_path tokenizer/v1_32k --max_seq_len 2048 \
  --data_path /root/autodl-tmp/mixture_v2/train.bin --val_data_path /root/autodl-tmp/mixture_v2/val.bin \
  --epochs 1 --batch_size 1 --accumulation_steps 2 --max_steps 4 --learning_rate 6e-4 \
  --log_step 1 --val_every 2 --val_batches 2 --save_step 2 --num_workers 2 \
  --dtype float32 --device cpu --save_dir $out
python probes.py --checkpoint $out/pretrain_final.pth --holdout /root/autodl-tmp/mixture_v2/holdout.jsonl \
  --tokenizer_path tokenizer/v1_32k --device cpu --max_tokens_per_source 4096 \
  --confusion_prompts 2 --arithmetic_items 4 --out $out/probes.json
ls -la $out
echo "PREFLIGHT EXIT=0"
