"""Mine on-policy preference pairs: sample a policy on env prompts, pair a correct completion
(chosen) with a wrong one (rejected) from the same prompt.

``envs.generate_data --split preference`` pairs the gold answer with a hand-corrupted one, and a
policy that has been through SFT already ranks every such pair correctly (records/posttrain_v2), so
DPO has nothing to learn from them. Here the rejected side is what the policy itself got wrong, and
a prompt contributes only when the group has both outcomes, the same prompts GRPO gets a gradient
from. The rows have the layout ``dpo.py`` reads (``prompt`` / ``chosen`` / ``rejected`` / ``system``)::

    python3 mine_pairs.py --policy_path results/posttrain_v1/sft/sft_final.pth \
        --tokenizer_path tokenizer/v1_32k --data_path datasets/arith_v1/grpo_prompts.jsonl \
        --n_prompts 3000 --out datasets/arith_v1/onpolicy_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections.abc import Iterable, Sequence


def pairs_from_rollouts(rollouts: Iterable, env, rng: random.Random, per_prompt: int = 1) -> list[dict]:
    """One row per (correct, wrong) pairing, ``per_prompt`` at most, for groups with both outcomes."""
    rows = []
    for roll in rollouts:
        right = [text for text, reward in zip(roll.texts, roll.rewards) if reward.correct]
        wrong = [text for text, reward in zip(roll.texts, roll.rewards) if not reward.correct]
        if not right or not wrong:
            continue
        right, wrong = sorted(set(right)), sorted(set(wrong))
        rng.shuffle(right)
        rng.shuffle(wrong)
        for i in range(min(per_prompt, len(right), len(wrong))):
            row = {"prompt": env.render(roll.task), "chosen": right[i], "rejected": wrong[i]}
            if env.system_prompt:
                row["system"] = env.system_prompt
            rows.append(row)
    return rows


def summarize(rollouts: Sequence, rows: Sequence[dict]) -> dict:
    groups = len(rollouts)
    correct = sum(reward.correct for roll in rollouts for reward in roll.rewards)
    total = sum(len(roll.rewards) for roll in rollouts)
    mixed = sum(
        any(r.correct for r in roll.rewards) and not all(r.correct for r in roll.rewards) for roll in rollouts
    )
    return {
        "prompts": groups, "samples": total, "sample_accuracy": correct / max(total, 1),
        "mixed_groups": mixed, "mixed_group_frac": mixed / max(groups, 1), "pairs": len(rows),
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
    parser.add_argument("--data_path", required=True, help="task file whose prompts are sampled (not dev / test)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--env", default="arithmetic")
    parser.add_argument("--n_prompts", type=int, default=3000)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--per_prompt", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--chunk", type=int, default=256, help="prompts sampled per call, for progress output")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=resolve_device())
    args = parser.parse_args(argv)

    started = time.time()
    set_seed(args.seed)
    rng = random.Random(args.seed)
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
        rows += pairs_from_rollouts(batch, env, rng, args.per_prompt)
        print(f"{len(rollouts)}/{len(tasks)} prompts, {len(rows)} pairs ({time.time() - started:.0f}s)", flush=True)

    stats = summarize(rollouts, rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    with open(args.out + ".stats.json", "w", encoding="utf-8") as fh:
        json.dump({**stats, "policy_path": args.policy_path, "data_path": args.data_path,
                   "group_size": args.group_size, "temperature": args.temperature, "seed": args.seed}, fh, indent=2)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
