"""Turn an RLVR environment into JSONL datasets for the other training stages.

One task definition feeds every stage, which is what makes SFT / DPO / GRPO
comparable on the same problems::

    python3 -m envs.generate_data --split eval       --n 500  --out datasets/arith_eval.jsonl --seed 777 --distinct
    python3 -m envs.generate_data --split sft        --n 4000 --out datasets/arith_sft.jsonl --exclude datasets/arith_eval.jsonl
    python3 -m envs.generate_data --split preference --n 2000 --out datasets/arith_pref.jsonl --exclude datasets/arith_eval.jsonl

* ``sft``        — gold chain-of-thought, the cold start GRPO needs to get any reward at all
* ``preference`` — gold vs. a deliberately corrupted answer, for the offline DPO baseline
* ``eval``       — questions plus ground truth, a fixed held-out set for accuracy curves

SFT and preference rows carry the env's system prompt, so they tokenize to exactly the
prompt GRPO samples from (``rollout.build_prompt_ids``). Without it the chat template
substitutes its default system prompt and the cold start trains on a different prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Iterable

from .base import Task, TaskEnv, available_envs, load_tasks, make_env

SPLITS = ("sft", "preference", "eval")


def sample_tasks(env: TaskEnv, n: int, exclude: Iterable[str] = (), distinct: bool = False) -> list[Task]:
    """Draw ``n`` tasks whose questions are not in ``exclude``, all different if ``distinct``.

    1–2 digit addition/subtraction has about 15k distinct problems and draws a one-digit
    operand half the time: drawn independently, a 4000-row SFT set contains 55% of a
    500-row eval set's questions, including every one with two one-digit operands, and
    that eval set holds only 432 distinct questions.
    """
    excluded = set(exclude)
    if not excluded and not distinct:
        return env.sample(n)
    tasks, seen = [], set()
    draws = 100 * n + 1000
    while len(tasks) < n:
        if draws == 0:
            raise ValueError(
                f"env {env.name!r} gave only {len(tasks)} of {n} tasks that are "
                f"{'distinct and ' if distinct else ''}not excluded; ask for fewer"
            )
        draws -= 1
        task = env.sample_task()
        if task.question in excluded or (distinct and task.question in seen):
            continue
        seen.add(task.question)
        tasks.append(task)
    return tasks


def build_rows(
    env: TaskEnv, split: str, n: int, rng: random.Random,
    exclude: Iterable[str] = (), distinct: bool = False,
) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    for attr in ("gold_completion", "corrupt_completion"):
        if split != "eval" and not hasattr(env, attr):
            raise TypeError(f"env {env.name!r} cannot generate {split} data: missing {attr}()")

    system = [{"role": "system", "content": env.system_prompt}] if env.system_prompt else []
    rows = []
    for task in sample_tasks(env, n, exclude, distinct):
        if split == "sft":
            rows.append(
                {
                    "conversations": system + [
                        {"role": "user", "content": env.render(task)},
                        {"role": "assistant", "content": env.gold_completion(task)},
                    ]
                }
            )
        elif split == "preference":
            row = {
                "prompt": env.render(task),
                "chosen": env.gold_completion(task),
                "rejected": env.corrupt_completion(task, rng),
            }
            if env.system_prompt:
                row["system"] = env.system_prompt
            rows.append(row)
        else:
            rows.append({"question": task.question, "answer": task.answer, **task.meta})
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RLVR datasets from an environment")
    parser.add_argument("--split", choices=list(SPLITS), required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env", type=str, default="arithmetic", choices=available_envs())
    parser.add_argument("--ops", type=str, default="+,-")
    parser.add_argument("--min_digits", type=int, default=1)
    parser.add_argument("--max_digits", type=int, default=2)
    parser.add_argument("--families", default="", help="pyfunc: comma-separated families to draw from; empty = all")
    parser.add_argument("--exclude_families", default="", help="pyfunc: comma-separated families never drawn")
    parser.add_argument(
        "--exclude", nargs="*", default=[],
        help="Task JSONL files ({question, answer} rows, e.g. the eval split) whose questions must not be drawn",
    )
    parser.add_argument("--distinct", action="store_true", help="No question appears twice in the output")
    args = parser.parse_args()

    kwargs = {"seed": args.seed}
    if args.env == "arithmetic":
        kwargs.update(
            ops=tuple(op.strip() for op in args.ops.split(",") if op.strip()),
            min_digits=args.min_digits,
            max_digits=args.max_digits,
        )
    if args.env == "pyfunc":
        kwargs.update(
            families=[f for f in args.families.split(",") if f] or None,
            exclude_families=[f for f in args.exclude_families.split(",") if f],
        )
    env = make_env(args.env, **kwargs)
    exclude = {task.question for path in args.exclude for task in load_tasks(path)}
    rows = build_rows(env, args.split, args.n, random.Random(args.seed), exclude, args.distinct)
    written = write_jsonl(args.out, rows)
    print(f"wrote {written} {args.split} rows -> {args.out}")


if __name__ == "__main__":
    main()
