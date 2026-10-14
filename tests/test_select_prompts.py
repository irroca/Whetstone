import json
import random
from types import SimpleNamespace

from envs import ArithmeticEnv
from envs.base import load_tasks
from select_prompts import select_by_solve_rate, solve_rates, summarize


def _rollout(env, task, outcomes):
    gold = env.gold_completion(task)
    wrong = env.corrupt_completion(task, random.Random(0))
    texts = [gold if ok else wrong + f" {i}" for i, ok in enumerate(outcomes)]
    return SimpleNamespace(task=task, texts=texts, rewards=[env.score(t, task) for t in texts])


def test_only_mixed_groups_are_kept_with_their_rate_and_metadata(tmp_path):
    env = ArithmeticEnv(seed=0)
    a, b, c, d = env.sample(4)
    rollouts = [
        _rollout(env, a, [True] * 4),            # solved every time: no gradient
        _rollout(env, b, [False] * 4),           # never solved: no gradient
        _rollout(env, c, [True, False, False, False]),
        _rollout(env, d, [True, True, True, False]),
    ]
    assert solve_rates(rollouts) == [1.0, 0.0, 0.25, 0.75]

    rows = select_by_solve_rate(rollouts)
    assert [r["question"] for r in rows] == [c.question, d.question]
    assert [r["solve_rate"] for r in rows] == [0.25, 0.75]
    assert rows[0]["op"] == c.meta["op"] and rows[0]["n_digits"] == c.meta["n_digits"]

    # a band keeps only the middle, and the file reads back through the loader grpo.py uses
    band = select_by_solve_rate(rollouts, min_rate=0.3, max_rate=0.9)
    assert [r["solve_rate"] for r in band] == [0.75]
    path = tmp_path / "pool.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    tasks = load_tasks(str(path))
    assert [t.question for t in tasks] == [c.question, d.question] and tasks[0].meta["solve_rate"] == 0.25

    stats = summarize(rollouts, rows)
    assert stats["kept"] == 2 and stats["all_right_frac"] == 0.25 and stats["all_wrong_frac"] == 0.25
