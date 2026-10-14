#!/bin/bash
# Harder-task GRPO grid on a Mac (MPS), from the first round's SFT policy, which saw only 1-2 digit
# problems and so gets ~3% of the 3-digit ones. Runs the variants one after another (a second MPS job
# only slows both) and skips a finished run, so the command is safe to repeat after the machine was
# switched off. grpo.py cannot resume: an unfinished run is moved aside and started over.
#   GRPO_EXTRA='--truncated_reward 0' adds flags to every grpo.py call (the EOS-runaway fix, docs/experiments.md).
#   GRPO_EXTRA='--truncated_reward 0' adds flags to every grpo.py call (the EOS-runaway fix, docs/experiments.md).
#   PYTHON=.venv/bin/python VARIANTS="grpo dapo_lite" SEEDS="1 2" STEPS=300 bash scripts/mac_posttrain_v3.sh
# Needs datasets/arith_v2 (MAX_DIGITS=3 bash scripts/arith_data.sh datasets/arith_v2) and the SFT
# weights at results/posttrain_v1/sft/sft_final.pth (scripts/fetch_artifacts.sh posttrain-v1-sft).
# About 10.5 s a step at 32 prompts x 8 samples, so 300 steps is ~55 minutes a run (~35 with the
# zero-variance filter, which skips the silent groups).
set -eo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
data=${DATA:-datasets/arith_v2}
sft=${SFT:-results/posttrain_v1/sft/sft_final.pth}
out=${OUT:-results/posttrain_v3}
variants=${VARIANTS:-grpo dapo_lite dapo_lite_p2 grpo_p2 dr_grpo}
tok=(--tokenizer_path tokenizer/v1_32k)
new=120
py=${PYTHON:-python3}

variant_flags() {
  case $1 in
    grpo) flags=() ;;
    grpo_p2) flags=(--ppo_epochs 2) ;;
    dr_grpo) flags=(--normalize_advantage_std False --aggregation dr_grpo) ;;
    dapo_lite) flags=(--filter_zero_variance True --clip_eps_high 0.28 --aggregation token_mean) ;;
    dapo_lite_p2) flags=(--filter_zero_variance True --clip_eps_high 0.28 --aggregation token_mean --ppo_epochs 2) ;;
    *) echo "unknown variant: $1" >&2; return 1 ;;
  esac
}

score() {  # checkpoint: greedy dev and test accuracy, written next to it
  local split
  for split in dev test; do
    [ -f "$(dirname "$1")/$split.json" ] && continue
    $py eval_env.py --checkpoint "$1" --eval_path $data/$split.jsonl "${tok[@]}" --max_new_tokens $new \
      --out "$(dirname "$1")/$split.json" || echo "=== eval of $1 on $split failed"
  done
}

mkdir -p $out
echo "=== start $(date '+%m-%d %H:%M:%S') at $(git rev-parse --short HEAD): variants '$variants', seeds '${SEEDS:-1}'"
for split in dev test; do
  [ -f $out/sft_$split.json ] || $py eval_env.py --checkpoint $sft --eval_path $data/$split.jsonl "${tok[@]}" \
    --max_new_tokens $new --out $out/sft_$split.json
done

for seed in ${SEEDS:-1}; do
  for variant in $variants; do
    run=$out/${variant}_s$seed
    if [ ! -f $run/grpo_final.pth ]; then
      if [ -d $run ]; then mv $run $run.unfinished.$(date +%s); fi
      variant_flags $variant
      echo "=== $variant seed $seed $(date '+%m-%d %H:%M:%S') ${flags[*]}"
      $py grpo.py --policy_path $sft "${tok[@]}" --seed $seed \
        --data_path $data/grpo_prompts.jsonl --eval_path $data/dev.jsonl --eval_size 500 --eval_every ${EVAL_EVERY:-25} \
        --rl_steps ${STEPS:-300} --batch_size 32 --group_size 8 --micro_batch_size 32 \
        --max_new_tokens $new --max_seq_len 200 --env_max_digits 3 --learning_rate ${LR:-2e-5} --dtype bfloat16 \
        --log_step 10 --save_step 0 --save_dir $run ${flags[@]+"${flags[@]}"} ${GRPO_EXTRA:-} ${GRPO_EXTRA:-} || echo "=== $variant seed $seed failed"
    fi
    if [ -f $run/grpo_final.pth ]; then score $run/grpo_final.pth; fi
  done
done
echo "=== done $(date '+%m-%d %H:%M:%S')"
