#!/bin/bash
# 200-update rehearsal of the formal command in a scratch directory. Validation runs at updates 100
# and 200; a run killed after 100 continues with
#   bash scripts/autodl/trial.sh --resume_from /root/autodl-tmp/results/pretrain_v2_trial/latest_checkpoint.pth
# With --max_steps the LR schedule spans those 200 updates, so do not change it when resuming.
export PATH=/root/miniconda3/bin:$PATH HF_HUB_OFFLINE=1
cd "$(dirname "$0")/../.."
python pretrain.py --dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3 \
  --tokenizer_path tokenizer/v1_32k --max_seq_len 2048 \
  --data_path /root/autodl-tmp/mixture_v2/train.bin --val_data_path /root/autodl-tmp/mixture_v2/val.bin \
  --epochs 1 --batch_size ${BATCH:-16} --accumulation_steps ${ACCUM:-16} --learning_rate 6e-4 \
  --log_step 10 --val_every 100 --val_batches 50 --save_step 100 --num_workers 2 \
  --dtype bfloat16 --compile ${COMPILE:-True} --save_dir /root/autodl-tmp/results/pretrain_v2_trial \
  --max_steps 200 "$@"
