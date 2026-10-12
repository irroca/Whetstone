"""Greedy accuracy of any checkpoint on a fixed task file of an RLVR environment.

It is the number GRPO logs as ``eval_accuracy``, measured the same way for SFT, DPO and GRPO
policies, which is what the held-out test set is for:

    python3 eval_env.py --checkpoint results/sft_final.pth --eval_path datasets/arith_v1/test.jsonl \\
        --tokenizer_path tokenizer/v1_32k --out results/sft_test.json

Accuracy is also broken down by task metadata (``--group_by``; for arithmetic the operator and
the larger operand's digit count), since the average hides which kind of problem lags.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Sequence


def evaluate_tasks(model, tokenizer, env, tasks, *, max_new_tokens: int, device: str,
                   group_by: Sequence[str] = ()) -> dict:
    """Greedy rollout per task: GRPO's eval statistics, per-group accuracy and a few examples."""
    from rollout import rollout_stats, run_rollouts

    rollouts = run_rollouts(
        model, tokenizer, env, tasks, group_size=1, max_new_tokens=max_new_tokens,
        temperature=0.0, device=device,
    )
    stats = {k: v for k, v in rollout_stats(rollouts).items() if k != "silent_group_frac"}
    counts = defaultdict(lambda: [0, 0])
    for roll in rollouts:
        key = " ".join(f"{k}={roll.task.meta[k]}" for k in group_by if k in roll.task.meta)
        if key:
            counts[key][0] += 1
            counts[key][1] += int(roll.rewards[0].correct)
    return {
        "items": len(rollouts),
        **stats,
        "by_group": {key: {"items": n, "accuracy": right / n} for key, (n, right) in sorted(counts.items())},
        "examples": [
            {"question": r.task.question, "answer": r.task.answer, "completion": r.texts[0],
             "correct": r.rewards[0].correct}
            for r in rollouts[:5]
        ],
    }


def main(argv=None) -> int:
    from transformers import AutoTokenizer

    from envs import available_envs, load_tasks, make_env
    from probes import load_model
    from rollout import build_prompt_ids
    from train_utils import resolve_device

    parser = argparse.ArgumentParser(description="greedy accuracy of one checkpoint on an env's task file")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_path", required=True, help="task JSONL: {question, answer, ...} rows")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--env", default="arithmetic", choices=available_envs())
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=0, help="evaluate the first N tasks; 0 = all")
    parser.add_argument("--group_by", default="op,n_digits", help="comma-separated task metadata keys")
    parser.add_argument("--device", default=resolve_device())
    parser.add_argument("--out", default="", help="also write the results as JSON here")
    args = parser.parse_args(argv)

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    env = make_env(args.env, seed=0)
    tasks = load_tasks(args.eval_path)[: args.limit or None]
    longest = max(build_prompt_ids(tokenizer, env, task).size(0) for task in tasks)
    model = load_model(args.checkpoint, tokenizer, longest + args.max_new_tokens, args.device)
    group_by = [key for key in args.group_by.split(",") if key]
    result = evaluate_tasks(
        model, tokenizer, env, tasks, max_new_tokens=args.max_new_tokens, device=args.device,
        group_by=group_by,
    )
    result = {
        "checkpoint": args.checkpoint, "eval_path": args.eval_path, "env": args.env,
        "max_new_tokens": args.max_new_tokens, **result, "seconds": round(time.time() - started, 1),
    }

    print(
        f"{result['items']} tasks: accuracy {result['accuracy']:.3f}, format {result['format_rate']:.3f}, "
        f"hack {result['hack_rate']:.3f}, length {result['completion_len']:.1f} ({result['seconds']}s)"
    )
    for key, row in result["by_group"].items():
        print(f"  {key:>16}: {row['accuracy']:.3f} of {row['items']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
