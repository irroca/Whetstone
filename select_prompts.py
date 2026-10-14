"""Difficulty curriculum by solve rate: keep the prompts a policy answers sometimes but not always.

A GRPO group whose samples are all right (or all wrong) has zero reward variance and no gradient. On the
arithmetic task that is most of the batch: 72% of the groups at the 3-digit SFT start and about 90% at
the end of the first round (docs/experiments.md). This samples ``--group_size`` completions per prompt
from a policy, estimates each prompt's solve rate, and writes only the prompts whose rate is strictly
between ``--min_rate`` and ``--max_rate`` (default: any mixed group), with the rate stored as task
metadata, in the layout ``grpo.py --data_path`` reads::

    python3 select_prompts.py --policy_path results/posttrain_v1/sft/sft_final.pth \
        --tokenizer_path tokenizer/v1_32k --data_path datasets/arith_v2/grpo_prompts.jsonl \
        --n_prompts 8000 --out datasets/arith_v2/grpo_prompts_mixed.jsonl

The rate is estimated at one policy, so the selection is static: a prompt that was too hard becomes
useful later, and one that was useful becomes trivial. ``--n_prompts`` larger than the training draw keeps
the pool from being replayed. Pool prompts must come from a file that excludes dev and test.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from collections.abc import Sequence


def solve_rates(rollouts: Sequence) -> list[float]:
    return [sum(r.correct for r in roll.rewards) / max(len(roll.rewards), 1) for roll in rollouts]


def select_by_solve_rate(rollouts: Sequence, min_rate: float = 0.0, max_rate: float = 1.0) -> list[dict]:
    """Task rows (question, answer, metadata, ``solve_rate``) for groups with min_rate < rate < max_rate."""
    rows = []
    for roll, rate in zip(rollouts, solve_rates(rollouts)):
        if min_rate < rate < max_rate:
            rows.append({"question": roll.task.question, "answer": roll.task.answer, **roll.task.meta,
                         "solve_rate": round(rate, 4)})
    return rows


def summarize(rollouts: Sequence, rows: Sequence[dict], group_key: str = "n_digits") -> dict:
    rates = solve_rates(rollouts)
    return {
        "prompts": len(rollouts), "kept": len(rows), "kept_frac": len(rows) / max(len(rollouts), 1),
        "all_right_frac": sum(r == 1.0 for r in rates) / max(len(rates), 1),
        "all_wrong_frac": sum(r == 0.0 for r in rates) / max(len(rates), 1),
        "mean_solve_rate": sum(rates) / max(len(rates), 1),
        "kept_by_" + group_key: dict(sorted(Counter(str(row.get(group_key)) for row in rows).items())),
        "pool_by_" + group_key: dict(sorted(Counter(str(roll.task.meta.get(group_key)) for roll in rollouts).items())),
    }


def main(argv=None) -> int:
    import torch
    from transformers import AutoTokenizer

    from envs import load_tasks, make_env
    from probes import load_model
    from rollout import build_prompt_ids, run_rollouts
    from train_utils import resolve_device, set_seed

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--policy_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--data_path", required=True, help="task file to select from (not dev / test)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--env", default="arithmetic")
    parser.add_argument("--n_prompts", type=int, default=8000)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--min_rate", type=float, default=0.0, help="keep rates strictly above this")
    parser.add_argument("--max_rate", type=float, default=1.0, help="keep rates strictly below this")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=resolve_device())
    args = parser.parse_args(argv)

    started = time.time()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    env = make_env(args.env, seed=0)
    tasks = load_tasks(args.data_path)[: args.n_prompts]
    longest = max(build_prompt_ids(tokenizer, env, task).size(0) for task in tasks)
    model = load_model(args.policy_path, tokenizer, longest + args.max_new_tokens, args.device)

    rollouts, rows = [], []
    for start in range(0, len(tasks), args.chunk):
        with torch.inference_mode():
            batch = run_rollouts(
                model, tokenizer, env, tasks[start:start + args.chunk], group_size=args.group_size,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature, device=args.device,
            )
        rollouts += batch
        rows += select_by_solve_rate(batch, args.min_rate, args.max_rate)
        print(f"{len(rollouts)}/{len(tasks)} prompts, {len(rows)} kept ({time.time() - started:.0f}s)", flush=True)

    stats = summarize(rollouts, rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    with open(args.out + ".stats.json", "w", encoding="utf-8") as fh:
        json.dump({**stats, "policy_path": args.policy_path, "data_path": args.data_path,
                   "group_size": args.group_size, "temperature": args.temperature, "seed": args.seed,
                   "min_rate": args.min_rate, "max_rate": args.max_rate}, fh, indent=2)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
