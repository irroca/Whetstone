#!/bin/bash
# Post-training on the arithmetic env (docs/status.md §4 step 7); start it with start_posttrain.sh.
# The data is regenerated from fixed seeds, SFT starts from the pretrained base, and every GRPO run
# starts from that SFT policy, one per variant and seed. eval_env.py scores each policy on dev and
# on the test set, which no stage trains or selects on. Finished stages are skipped, so the same
# command resumes after a crash; grpo.py cannot resume inside a run, so an unfinished run is moved
# aside and started over.
#   OUT=/root/autodl-tmp/results/posttrain_trial STEPS=20 VARIANTS=grpo bash scripts/autodl/posttrain.sh
set -eo pipefail
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/../.."
base=results/pretrain_v2/pretrain_final.pth
data=${DATA:-/root/autodl-tmp/arith_v1}
out=${OUT:-/root/autodl-tmp/results/posttrain_v1}
variants=${VARIANTS:-grpo grpo_kl dr_grpo dapo_lite}
tok=(--tokenizer_path tokenizer/v1_32k)
new=96

variant_flags() {
  case $1 in
    grpo) flags=() ;;
    grpo_kl) flags=(--kl_coeff 0.02) ;;
    dr_grpo) flags=(--normalize_advantage_std False --aggregation dr_grpo) ;;
    dapo_lite) flags=(--filter_zero_variance True --clip_eps_high 0.28 --aggregation token_mean) ;;
    *) echo "unknown variant: $1" >&2; return 1 ;;
  esac
}

score() {  # checkpoint: greedy dev and test accuracy, written next to it
  local split
  for split in dev test; do
    [ -f "$(dirname $1)/$split.json" ] && continue
    python eval_env.py --checkpoint $1 --eval_path $data/$split.jsonl "${tok[@]}" --max_new_tokens $new \
      --out "$(dirname $1)/$split.json" || echo "=== eval of $1 on $split failed"
  done
}

for variant in $variants; do variant_flags $variant; done
[ -f $base ] || { echo "$base is missing: scripts/fetch_artifacts.sh pretrain-v2" >&2; exit 1; }
mkdir -p $out
echo "=== start $(date '+%m-%d %H:%M:%S') at $(git rev-parse --short HEAD): variants '$variants', seeds '${SEEDS:-1}'"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash scripts/arith_data.sh $data
sha256sum $data/*.jsonl

sft=$out/sft
if [ ! -f $sft/sft_final.pth ]; then
  python sft.py --pretrained_path $base "${tok[@]}" --data_path $data/sft.jsonl --max_seq_len 128 \
    --epochs ${SFT_EPOCHS:-1} --batch_size 32 --learning_rate 1e-4 --dtype bfloat16 --save_step 0 --save_dir $sft
fi
score $sft/sft_final.pth

finished=()
for seed in ${SEEDS:-1}; do
  for variant in $variants; do
    run=$out/${variant}_s$seed
    if [ ! -f $run/grpo_final.pth ]; then
      if [ -d $run ]; then mv $run $run.unfinished.$(date +%s); fi
      variant_flags $variant
      echo "=== $variant seed $seed $(date '+%m-%d %H:%M:%S') ${flags[*]}"
      python grpo.py --policy_path $sft/sft_final.pth "${tok[@]}" --seed $seed \
        --data_path $data/grpo_prompts.jsonl --eval_path $data/dev.jsonl --eval_size 500 --eval_every ${EVAL_EVERY:-25} \
        --rl_steps ${STEPS:-500} --batch_size ${BATCH:-32} --group_size ${GROUP:-8} --micro_batch_size ${MICRO:-64} \
        --max_new_tokens $new --max_seq_len 192 --learning_rate ${LR:-1e-5} --dtype bfloat16 \
        --log_step 10 --save_step 0 --save_dir $run "${flags[@]}" || echo "=== $variant seed $seed failed"
    fi
    if [ -f $run/grpo_final.pth ]; then
      score $run/grpo_final.pth
      finished+=($run/grpo_metrics.jsonl)
    fi
  done
done

echo "=== done $(date '+%m-%d %H:%M:%S')"
if [ ${#finished[@]} -gt 0 ]; then python analyze_grpo.py "${finished[@]}" --window 50 || true; fi
python - $out <<'EOF'
import glob
import json
import os
import sys

for test in sorted(glob.glob(f"{sys.argv[1]}/*/test.json")):
    run = os.path.dirname(test)
    t = json.load(open(test))
    dev = json.load(open(f"{run}/dev.json"))["accuracy"] if os.path.exists(f"{run}/dev.json") else float("nan")
    print(f"{os.path.basename(run):>16}: dev {dev:.3f}  test {t['accuracy']:.3f}  "
          f"format {t['format_rate']:.3f}  length {t['completion_len']:.1f}")
EOF
