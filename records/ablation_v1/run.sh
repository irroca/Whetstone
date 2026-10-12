#!/bin/zsh
# Runs every ablation stage from the pinned worktree. Run it again to resume.
cd ~/projects/whetstone-ablation-v1 || exit 1
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
log=results/ablation_v1/run.log
echo "=== start $(date '+%m-%d %H:%M:%S') at $(git rev-parse --short HEAD)" >> $log
caffeinate -is .venv/bin/python run_ablation.py configs/ablation_v1.json run 2>&1 \
  | perl -MPOSIX -ne '$|=1; print strftime("%m-%d %H:%M:%S ", localtime), $_' >> $log
echo "EXIT=${pipestatus[1]} $(date '+%m-%d %H:%M:%S')" >> $log
