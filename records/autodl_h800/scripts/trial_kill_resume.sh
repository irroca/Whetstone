#!/bin/bash
out=/root/autodl-tmp/results/pretrain_v2_trial
log=/root/autodl-tmp/results/trial.log
while [ ! -f $out/latest_checkpoint.pth ]; do sleep 2; done
sleep 15
echo "=== killing at $(date +%T) after $(tail -1 $(ls -d $out/runs/*/ | tail -1)/metrics.jsonl | cut -c1-40)" >> $log
pkill -f "[p]retrain.py --dim 768"
while pgrep -f "[p]retrain.py --dim 768" > /dev/null; do sleep 1; done
sleep 3
echo "=== resuming at $(date +%T)" >> $log
cd /root/Whetstone && BATCH=32 ACCUM=8 COMPILE=True bash scripts/autodl/trial.sh --resume_from $out/latest_checkpoint.pth >> $log 2>&1
echo "TRIAL RESUME EXIT=$?" >> $log
