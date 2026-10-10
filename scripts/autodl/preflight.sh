#!/bin/bash
# CPU check of the formal data path before paying for a GPU. train.bin and val.bin are opened
# through MemmapPretrainDataset at the formal 2048 tokens (tokenizer fingerprint and size checks
# included) and both ends of each are decoded; then a tiny model trains 4 updates with validation
# and saving, and the probes run on it against the real holdout. That training reads val.bin at 512
# tokens because the no-GPU instance has 2GB, ~0.3GB of it AutoDL's own services: at 2048 tokens
# fp32 logits over the 32k vocabulary take ~1GB in the backward pass, and on train.bin the sampler's
# permutation of every window is a Python list of millions of ints. Not alongside the test suite.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/../.."
data=/root/autodl-tmp/mixture_v2
out=/root/autodl-tmp/results/preflight
rm -rf $out
python - $data <<'EOF'
import sys
from transformers import AutoTokenizer
from dataset import MemmapPretrainDataset

tokenizer = AutoTokenizer.from_pretrained("tokenizer/v1_32k")
for split in ("train", "val"):
    ds = MemmapPretrainDataset(f"{sys.argv[1]}/{split}.bin", tokenizer, max_length=2048)
    print(f"{split}: {ds.num_tokens:,} tokens, {len(ds):,} windows of 2048")
    print("  first:", repr(tokenizer.decode(ds[0][0][:48].tolist())))
    print("  last: ", repr(tokenizer.decode(ds[len(ds) - 1][1][-48:].tolist())))
EOF
python pretrain.py --dim 64 --n_layers 2 --n_heads 4 --n_kv_heads 1 \
  --tokenizer_path tokenizer/v1_32k --max_seq_len 512 \
  --data_path $data/val.bin --val_data_path $data/val.bin \
  --epochs 1 --batch_size 1 --accumulation_steps 2 --max_steps 4 --learning_rate 6e-4 \
  --log_step 1 --val_every 2 --val_batches 2 --save_step 2 --num_workers 2 \
  --dtype float32 --device cpu --save_dir $out
python probes.py --checkpoint $out/pretrain_final.pth --holdout $data/holdout.jsonl \
  --tokenizer_path tokenizer/v1_32k --device cpu --max_seq_len 512 --max_tokens_per_source 4096 \
  --confusion_prompts 2 --arithmetic_items 4 --out $out/probes.json
ls -la $out
echo "PREFLIGHT EXIT=0"
