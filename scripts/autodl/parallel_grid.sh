#!/bin/bash
# Several GRPO workers on one GPU, one screen each: a 100M policy at 32 prompts x 8 samples is bound by
# decoding overhead, not by the GPU, so a 32 GB card runs four at 4.7 s a step each where one alone runs 2.7.
#   bash scripts/autodl/parallel_grid.sh "grpo dr_grpo dapo_lite grpo_cur" "1 2 3 4 5" 300
# Memory is the limit: ~6.5 GB a process (the arithmetic task), ~9 GB for pyfunc. A fifth process at the same
# time (or an evaluation while four train) runs out of memory and `mac_posttrain_v3.sh` then logs
# "failed" for every seed in a few seconds; rerun the failed variants once a worker has finished.
# Each worker gets OMP_NUM_THREADS=5, so five workers fit a 25-core quota.
#   OUT=...  SFT=...  GRPO_EXTRA='--truncated_reward 0'  override the defaults below.
set -eo pipefail
variants=$1
seeds=${2:-1 2 3 4 5}
steps=${3:-300}
here=$(cd "$(dirname "$0")/../.." && pwd)
out=${OUT:-/root/autodl-tmp/results/grid}
sft=${SFT:-/root/autodl-tmp/results/posttrain_v1/sft/sft_final.pth}
mkdir -p "$out"
for variant in $variants; do
  screen -dmS "$variant" bash -c "
    source $here/scripts/autodl/env.sh
    export OMP_NUM_THREADS=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    cd $here
    OUT=$out SFT=$sft PYTHON=python VARIANTS=$variant SEEDS='$seeds' STEPS=$steps \
      GRPO_EXTRA='${GRPO_EXTRA:---truncated_reward 0}' bash scripts/mac_posttrain_v3.sh > $out/w_$variant.log 2>&1"
done
screen -ls
