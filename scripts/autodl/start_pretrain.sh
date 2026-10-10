#!/bin/bash
# Starts pretrain.sh in screen `pretrain`, logging to results/pretrain_v2/train.log. The instance
# shuts down when the run ends, so billing stops; NO_SHUTDOWN=1 keeps it up.
#   BATCH=32 ACCUM=8 COMPILE=True bash scripts/autodl/start_pretrain.sh
here=$(cd "$(dirname "$0")" && pwd)
log=/root/autodl-tmp/results/pretrain_v2/train.log
mkdir -p $(dirname $log)
if screen -ls | grep -q '\.pretrain\b'; then
  echo "screen session 'pretrain' already exists; not starting a second run" >&2
  exit 1
fi
cmd="bash $here/pretrain.sh >> $log 2>&1"
[ "${NO_SHUTDOWN:-0}" = 1 ] || cmd="$cmd; /usr/bin/shutdown"
BATCH=${BATCH:-16} ACCUM=${ACCUM:-16} COMPILE=${COMPILE:-True} screen -dmS pretrain bash -c "$cmd"
echo "started: BATCH=${BATCH:-16} ACCUM=${ACCUM:-16} COMPILE=${COMPILE:-True}, log $log"
