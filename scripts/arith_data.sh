#!/bin/bash
# The arithmetic env's post-training data (1-2 digit + and -), regenerated identically from fixed
# seeds. test is reported once at the end and dev is watched during training; neither appears in
# the SFT rows, the DPO pairs or the GRPO prompt pool.
#   bash scripts/arith_data.sh [out_dir]                      # default datasets/arith_v1 (1-2 digits)
#   MAX_DIGITS=3 bash scripts/arith_data.sh datasets/arith_v2 # 1-3 digits; each operand's length is uniform,
#                                                             # so the larger one is 3 digits 5 times in 9
set -euo pipefail
cd "$(dirname "$0")/.."
out=${1:-datasets/arith_v1}
mkdir -p "$out"
held_out=("$out/test.jsonl" "$out/dev.jsonl")
max_digits=${MAX_DIGITS:-2}
gen() { python3 -m envs.generate_data --max_digits "$max_digits" "$@"; }
gen --split eval --n 500 --seed 777 --distinct --out "$out/test.jsonl"
gen --split eval --n 500 --seed 778 --distinct --exclude "$out/test.jsonl" --out "$out/dev.jsonl"
gen --split sft --n 4000 --seed 0 --exclude "${held_out[@]}" --out "$out/sft.jsonl"
gen --split preference --n 2000 --seed 1 --exclude "${held_out[@]}" --out "$out/preference.jsonl"
gen --split eval --n 20000 --seed 2 --exclude "${held_out[@]}" --out "$out/grpo_prompts.jsonl"
