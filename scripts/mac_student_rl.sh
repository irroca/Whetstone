#!/bin/bash
# GRPO on the 42.5M student of scripts/mac_kd_v1.sh, starting from its plain-SFT checkpoint (seed 1), as the
# alternative to distillation: does RL lift a small model that KD could not?
#   PYTHON=.venv/bin/python VARIANTS="dapo_lite grpo" STEPS=300 bash scripts/mac_student_rl.sh
# Same task and settings as the 100M grid on arith_v1 (32 prompts x 8, max_new_tokens 96), lr 2e-5 as on the
# Mac harder-task grid. A finished run is skipped; an unfinished one is moved aside (grpo.py cannot resume).
set -eo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
data=${DATA:-datasets/arith_v1}
sft=${SFT:-results/kd_v1/sft_s1/sft_final.pth}
out=${OUT:-results/kd_v1}
variants=${VARIANTS:-dapo_lite grpo}
tok=(--tokenizer_path tokenizer/v1_32k)
py=${PYTHON:-python3}

variant_flags() {
  case $1 in
    grpo) flags=() ;;
    dapo_lite) flags=(--filter_zero_variance True --clip_eps_high 0.28 --aggregation token_mean) ;;
    *) echo "unknown variant: $1" >&2; return 1 ;;
  esac
}

echo "=== start $(date '+%m-%d %H:%M:%S') at $(git rev-parse --short HEAD): variants '$variants'"
for variant in $variants; do
  run=$out/rl_${variant}_s1
  if [ ! -f $run/grpo_final.pth ]; then
    if [ -d $run ]; then mv $run $run.unfinished.$(date +%s); fi
    variant_flags $variant
    echo "=== $variant $(date '+%m-%d %H:%M:%S') ${flags[*]}"
    $py grpo.py --policy_path $sft "${tok[@]}" --seed 1 \
      --data_path $data/grpo_prompts.jsonl --eval_path $data/dev.jsonl --eval_size 500 --eval_every ${EVAL_EVERY:-25} \
      --rl_steps ${STEPS:-300} --batch_size 32 --group_size 8 --micro_batch_size 32 \
      --max_new_tokens 96 --max_seq_len 192 --learning_rate ${LR:-2e-5} --dtype bfloat16 \
      --log_step 10 --save_step 0 --save_dir $run ${flags[@]+"${flags[@]}"} || echo "=== $variant failed"
  fi
  if [ -f $run/grpo_final.pth ]; then
    for split in dev test; do
      [ -f $run/$split.json ] || $py eval_env.py --checkpoint $run/grpo_final.pth --eval_path $data/$split.jsonl \
        "${tok[@]}" --max_new_tokens 96 --out $run/$split.json
    done
  fi
done
echo "=== done $(date '+%m-%d %H:%M:%S')"
