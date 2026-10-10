#!/bin/bash
# The formal pretraining run (docs/status.md §4 step 6); start it with start_pretrain.sh. After a
# crash it resumes from the latest checkpoint, which replays the exact data order of an
# uninterrupted run; once the final weights exist it runs the probes. BATCH x ACCUM x 2047 should
# stay near 0.5M tokens.
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/../.."
out=/root/autodl-tmp/results/pretrain_v2
mkdir -p $out
args=(--dim 768 --n_layers 12 --n_heads 12 --n_kv_heads 3
      --tokenizer_path tokenizer/v1_32k --max_seq_len 2048
      --data_path /root/autodl-tmp/mixture_v2/train.bin --val_data_path /root/autodl-tmp/mixture_v2/val.bin
      --epochs 1 --batch_size ${BATCH:-16} --accumulation_steps ${ACCUM:-16} --learning_rate 6e-4
      --log_step 10 --val_every 500 --val_batches 50 --save_step 500 --num_workers 2
      --dtype bfloat16 --compile ${COMPILE:-True} --save_dir $out)
for attempt in 1 2 3 4; do
  [ -f $out/pretrain_final.pth ] && break
  resume=()
  [ -f $out/latest_checkpoint.pth ] && resume=(--resume_from $out/latest_checkpoint.pth)
  echo "=== attempt $attempt $(date '+%m-%d %H:%M:%S') at $(git rev-parse --short HEAD) ${resume[*]}"
  python pretrain.py "${args[@]}" "${resume[@]}"
  echo "=== pretrain exited with $? $(date '+%m-%d %H:%M:%S')"
done
if [ -f $out/pretrain_final.pth ]; then
  python probes.py --checkpoint $out/pretrain_final.pth --holdout /root/autodl-tmp/mixture_v2/holdout.jsonl \
    --tokenizer_path tokenizer/v1_32k --out $out/probes.json
  echo "=== probes exited with $? $(date '+%m-%d %H:%M:%S')"
fi
