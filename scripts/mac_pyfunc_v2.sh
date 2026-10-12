#!/bin/bash
# pyfunc on a Mac: can RL teach a policy code tasks that SFT only touched on, and does it carry over to
# task families that neither stage saw? Three disjoint sets of families (envs/pyfunc.py):
#   A  12 families, SFT only             -> test_A  (does RL forget what SFT taught?)
#   B  10 families, 3 SFT examples each,
#      then RL on fresh instances        -> dev_B / test_B  (what RL adds)
#   C   4 families nobody trained on     -> test_C  (does it transfer?)
# SFT on A alone leaves B at 0% (loss 0.0000: the model memorises one template per family) and GRPO then
# has no signal; three examples per B family put sampled accuracy near 30%, where there is.
#   PYTHON=.venv/bin/python VARIANTS="grpo dapo_lite" SEEDS=1 STEPS=200 bash scripts/mac_pyfunc_v2.sh
# Reruns skip finished stages; grpo.py cannot resume, so an unfinished run is moved aside.
set -eo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
py=${PYTHON:-python3}
D=${DATA:-datasets/pyfunc_v2}
O=${OUT:-results/pyfunc_v2}
variants=${VARIANTS:-grpo dapo_lite}
tok=(--tokenizer_path tokenizer/v1_32k)
A=sum_multiples,count_greater,clamp,repeat_join,count_vowels,is_palindrome,digit_sum,range_of_list,reverse_words,count_char,even_squares,sum_multiples_upto
B=dedupe,rotate_left,title_case,adjacent_diff_sum,prefix_sums,long_words,every_kth,join_numbers,has_duplicate,first_min_index
C=gcd,fib,is_prime,second_largest
gen() { $py -m envs.generate_data --env pyfunc "$@"; }

variant_flags() {
  case $1 in
    grpo) flags=() ;;
    dapo_lite) flags=(--filter_zero_variance True --clip_eps_high 0.28 --aggregation token_mean) ;;
    *) echo "unknown variant: $1" >&2; return 1 ;;
  esac
}

score() {  # checkpoint: greedy accuracy on every test set, per family, written next to it
  local split
  for split in dev_B test_B test_A test_C; do
    [ -f "$(dirname "$1")/$split.json" ] && continue
    $py eval_env.py --env pyfunc --group_by family --checkpoint "$1" --eval_path $D/$split.jsonl "${tok[@]}" \
      --max_new_tokens 96 --out "$(dirname "$1")/$split.json" | head -1 || echo "=== eval of $1 on $split failed"
  done
}

mkdir -p $D $O
if [ ! -f $D/grpo_prompts_B.jsonl ]; then
  gen --split eval --n 300 --seed 777 --distinct --families $C --out $D/test_C.jsonl
  gen --split eval --n 300 --seed 777 --distinct --families $B --out $D/test_B.jsonl
  gen --split eval --n 300 --seed 777 --distinct --families $A --out $D/test_A.jsonl
  gen --split eval --n 300 --seed 778 --distinct --families $B --exclude $D/test_B.jsonl --out $D/dev_B.jsonl
  gen --split sft --n 3000 --seed 0 --families $A --exclude $D/test_A.jsonl --out $D/sft_A.jsonl
  gen --split sft --n 30 --seed 5 --families $B --exclude $D/test_B.jsonl $D/dev_B.jsonl --out $D/sft_B30.jsonl
  gen --split eval --n 20000 --seed 2 --families $B --exclude $D/test_B.jsonl $D/dev_B.jsonl --out $D/grpo_prompts_B.jsonl
  $py - $D <<'EOF'
import json
import random
import sys

d = sys.argv[1]
rows = [json.loads(line) for name in ("sft_A", "sft_B30") for line in open(f"{d}/{name}.jsonl", encoding="utf-8")]
random.Random(0).shuffle(rows)
with open(f"{d}/sft_AB30.jsonl", "w", encoding="utf-8") as fh:
    fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
EOF
fi

sft=$O/sft_ab30
if [ ! -f $sft/sft_final.pth ]; then
  $py sft.py --pretrained_path results/pretrain_v2/pretrain_final.pth "${tok[@]}" --data_path $D/sft_AB30.jsonl \
    --max_seq_len 256 --epochs 1 --batch_size 32 --learning_rate 1e-4 --dtype bfloat16 --save_step 0 --save_dir $sft
fi
score $sft/sft_final.pth

for seed in ${SEEDS:-1}; do
  for variant in $variants; do
    run=$O/${variant}_s$seed
    if [ ! -f $run/grpo_final.pth ]; then
      if [ -d $run ]; then mv $run $run.unfinished.$(date +%s); fi
      variant_flags $variant
      echo "=== $variant seed $seed $(date '+%m-%d %H:%M:%S') ${flags[*]}"
      $py grpo.py --env pyfunc --policy_path $sft/sft_final.pth "${tok[@]}" --seed $seed \
        --data_path $D/grpo_prompts_B.jsonl --eval_path $D/dev_B.jsonl --eval_size 150 --eval_every ${EVAL_EVERY:-25} \
        --rl_steps ${STEPS:-200} --batch_size 32 --group_size 8 --micro_batch_size 32 \
        --max_new_tokens 96 --max_seq_len 272 --learning_rate ${LR:-1e-5} --dtype bfloat16 \
        --log_step 10 --save_step 0 --save_dir $run ${flags[@]+"${flags[@]}"} || echo "=== $variant seed $seed failed"
    fi
    if [ -f $run/grpo_final.pth ]; then score $run/grpo_final.pth; fi
  done
done
echo "=== done $(date '+%m-%d %H:%M:%S')"
