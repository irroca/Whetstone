#!/bin/bash
# Knowledge distillation on the arithmetic task: can the 100M RL policy teach a 42.5M student?
#   teacher  results/posttrain_v1/dapo_lite_s1/grpo_final.pth   (dev 0.982 / test 0.950 on arith_v1)
#   student  results/ablation_v1/zh30_v32k/pretrain_final.pth   (dim 512, 8 layers, same 32k vocab,
#            pretrained on 0.3B tokens: it has seen no arithmetic SFT)
# Arms, all on the same 4000-row data/sft.jsonl and the same budget:
#   sft      the student trained with plain cross-entropy (the baseline KD has to beat)
#   kd05     0.5 CE + 0.5 KL at temperature 2 (distill.py defaults)
#   kd10     KL only
#   kd05t1   like kd05 at temperature 1 (kd10t1: KL only at temperature 1)
# Each finished arm is scored by eval_env.py on dev and on test (greedy, like the GRPO runs). A finished
# run is skipped, so the command is safe to repeat.
#   PYTHON=.venv/bin/python SEEDS="1 2 3" bash scripts/mac_kd_v1.sh
# Needs datasets/arith_v1 (bash scripts/arith_data.sh datasets/arith_v1), the teacher weights
# (scripts/fetch_artifacts.sh posttrain-v1) and the ablation student (scripts/fetch_artifacts.sh ablation-v1).
set -eo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
data=${DATA:-datasets/arith_v1}
teacher=${TEACHER:-results/posttrain_v1/dapo_lite_s1/grpo_final.pth}
student=${STUDENT:-results/ablation_v1/zh30_v32k/pretrain_final.pth}
out=${OUT:-results/kd_v1}
epochs=${EPOCHS:-3}
arms=${ARMS:-sft kd05 kd10}
tok=(--tokenizer_path tokenizer/v1_32k)
py=${PYTHON:-python3}

score() {  # checkpoint
  local split
  for split in dev test; do
    [ -f "$(dirname "$1")/$split.json" ] && continue
    $py eval_env.py --checkpoint "$1" --eval_path $data/$split.jsonl "${tok[@]}" --max_new_tokens 96 \
      --out "$(dirname "$1")/$split.json" || echo "=== eval of $1 on $split failed"
  done
}

mkdir -p $out
echo "=== start $(date '+%m-%d %H:%M:%S') at $(git rev-parse --short HEAD): arms '$arms', seeds '${SEEDS:-1}', $epochs epochs"
for seed in ${SEEDS:-1}; do
  for arm in $arms; do
    run=$out/${arm}_s$seed
    final=$run/sft_final.pth; [ $arm = sft ] || final=$run/distill_final.pth
    if [ ! -f $final ]; then
      if [ -d $run ]; then mv $run $run.unfinished.$(date +%s); fi
      echo "=== $arm seed $seed $(date '+%m-%d %H:%M:%S')"
      case $arm in
        sft)
          $py sft.py --pretrained_path $student "${tok[@]}" --data_path $data/sft.jsonl --max_seq_len 192 \
            --epochs $epochs --batch_size 32 --learning_rate 1e-4 --dtype bfloat16 --seed $seed \
            --save_step 0 --save_dir $run ;;
        kd05|kd10|kd05t1|kd10t1)
          alpha=0.5; temp=${TEMPERATURE:-2.0}
          case $arm in kd10|kd10t1) alpha=1.0 ;; esac
          case $arm in kd05t1|kd10t1) temp=1.0 ;; esac
          $py distill.py --teacher_path $teacher --student_path $student "${tok[@]}" --data_path $data/sft.jsonl \
            --max_seq_len 192 --epochs $epochs --batch_size 32 --learning_rate 1e-4 --dtype bfloat16 --seed $seed \
            --alpha $alpha --temperature $temp --save_step 0 --save_dir $run ;;
        *) echo "unknown arm: $arm" >&2; exit 1 ;;
      esac
    fi
    [ -f $final ] && score $final
  done
done
echo "=== done $(date '+%m-%d %H:%M:%S')"
