#!/bin/bash
# Starts posttrain.sh in screen `posttrain`, logging to results/posttrain_v1/run.log. The instance
# shuts down when the run ends, so billing stops; NO_SHUTDOWN=1 keeps it up. posttrain.sh reads its
# settings (VARIANTS, SEEDS, STEPS, LR, BATCH, GROUP, MICRO, EVAL_EVERY, SFT_EPOCHS) from the
# environment, which screen passes on:
#   SEEDS="1 2 3" bash scripts/autodl/start_posttrain.sh
here=$(cd "$(dirname "$0")" && pwd)
log=${OUT:-/root/autodl-tmp/results/posttrain_v1}/run.log
mkdir -p $(dirname $log)
if screen -ls | grep -q '\.posttrain\b'; then
  echo "screen session 'posttrain' already exists; not starting a second run" >&2
  exit 1
fi
cmd="bash $here/posttrain.sh >> $log 2>&1"
[ "${NO_SHUTDOWN:-0}" = 1 ] || cmd="$cmd; /usr/bin/shutdown"
screen -dmS posttrain bash -c "$cmd"
echo "started: variants '${VARIANTS:-grpo grpo_kl dr_grpo dapo_lite}', seeds '${SEEDS:-1}', log $log"
