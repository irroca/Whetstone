#!/bin/bash
# pyfunc, second round: does mixing the SFT-only families (A) into the RL prompt pool stop GRPO from
# forgetting them? The first round (scripts/mac_pyfunc_v2.sh) trained RL on the 10 B families only and
# test_A fell. Two arms from the same SFT policy, both with `--truncated_reward 0`:
#   B   RL pool = B families only (the control, same as round one except for the EOS fix)
#   AB  RL pool = 12000 B prompts + 8000 A prompts (40% A)
# Scored greedily on dev_B / test_B (RL families), test_A (SFT families), test_C (nobody trained on them).
#   ARMS="B AB" VARIANTS=grpo SEEDS="1 2 3" PYTHON=python bash scripts/pyfunc_v3.sh
# Needs datasets/pyfunc_v2 (scripts/mac_pyfunc_v2.sh builds it) and the SFT policy at $SFT. Reruns skip finished runs.
set -eo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
py=${PYTHON:-python3}
D=${DATA:-datasets/pyfunc_v2}
O=${OUT:-results/pyfunc_v3}
sft=${SFT:-$O/sft_ab30/sft_final.pth}
tok=(--tokenizer_path tokenizer/v1_32k)
A=sum_multiples,count_greater,clamp,repeat_join,count_vowels,is_palindrome,digit_sum,range_of_list,reverse_words,count_char,even_squares,sum_multiples_upto

variant_flags() {
  case $1 in
    grpo) flags=() ;;
    dapo_lite) flags=(--filter_zero_variance True --clip_eps_high 0.28 --aggregation token_mean) ;;
    *) echo "unknown variant: $1" >&2; return 1 ;;
  esac
}

score() {
  local split
  for split in dev_B test_B test_A test_C; do
    [ -f "$(dirname "$1")/$split.json" ] && continue
    $py eval_env.py --env pyfunc --group_by family --checkpoint "$1" --eval_path $D/$split.jsonl "${tok[@]}" \
      --max_new_tokens 96 --out "$(dirname "$1")/$split.json" | head -1 || echo "=== eval of $1 on $split failed"
  done
}

mkdir -p $O
if [ ! -f $D/grpo_prompts_AB.jsonl ]; then
  $py -m envs.generate_data --env pyfunc --split eval --n 8000 --seed 3 --families $A --exclude $D/test_A.jsonl \
    --out $D/grpo_prompts_A.jsonl
  $py - $D <<'EOF'
import json
import random
import sys

d = sys.argv[1]
rows = [json.loads(l) for l in open(f"{d}/grpo_prompts_B.jsonl", encoding="utf-8")][:12000]
rows += [json.loads(l) for l in open(f"{d}/grpo_prompts_A.jsonl", encoding="utf-8")]
random.Random(0).shuffle(rows)
with open(f"{d}/grpo_prompts_AB.jsonl", "w", encoding="utf-8") as fh:
    fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
EOF
fi
score $sft

for seed in ${SEEDS:-1}; do
  for arm in ${ARMS:-B AB}; do
    for variant in ${VARIANTS:-grpo}; do
      run=$O/${variant}_${arm}_s$seed
      pool=$D/grpo_prompts_B.jsonl
      [ "$arm" = AB ] && pool=$D/grpo_prompts_AB.jsonl
      if [ ! -f $run/grpo_final.pth ]; then
        if [ -d $run ]; then mv $run $run.unfinished.$(date +%s); fi
        variant_flags $variant
        echo "=== $variant $arm seed $seed $(date '+%m-%d %H:%M:%S') ${flags[*]}"
        $py grpo.py --env pyfunc --policy_path $sft "${tok[@]}" --seed $seed \
          --data_path $pool --eval_path $D/dev_B.jsonl --eval_size 150 --eval_every ${EVAL_EVERY:-25} \
          --rl_steps ${STEPS:-200} --batch_size 32 --group_size 8 --micro_batch_size 32 \
          --max_new_tokens 96 --max_seq_len 272 --learning_rate ${LR:-1e-5} --dtype bfloat16 \
          --log_step 10 --save_step 0 --save_dir $run --truncated_reward 0 ${flags[@]+"${flags[@]}"} \
          || echo "=== $variant $arm seed $seed failed"
      fi
      if [ -f $run/grpo_final.pth ]; then score $run/grpo_final.pth; fi
    done
  done
done
echo "=== done $(date '+%m-%d %H:%M:%S')"
