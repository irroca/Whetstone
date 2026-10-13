import json
import random
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from dataset import PreferenceDataset
from envs import ArithmeticEnv
from mine_pairs import pairs_from_rollouts, summarize
from rollout import build_prompt_ids

TOKENIZER = "./tokenizer/zh_6400"


def _rollout(env, task, outcomes):
    """A group whose i-th completion is gold when outcomes[i] is True, else an off-by-one answer."""
    gold = env.gold_completion(task)
    wrong = env.corrupt_completion(task, random.Random(0))
    texts = [gold if ok else wrong + f" {i}" for i, ok in enumerate(outcomes)]
    return SimpleNamespace(task=task, texts=texts, rewards=[env.score(t, task) for t in texts])


def test_only_groups_with_both_outcomes_give_pairs_and_the_sides_are_right():
    env = ArithmeticEnv(seed=0)
    a, b, c = env.sample(3)
    rollouts = [
        _rollout(env, a, [True, True, True]),    # nothing wrong to reject
        _rollout(env, b, [False, False, False]),  # nothing right to prefer
        _rollout(env, c, [True, False, True]),
    ]
    rows = pairs_from_rollouts(rollouts, env, random.Random(0))

    assert len(rows) == 1
    assert rows[0]["prompt"] == env.render(c)
    assert rows[0]["chosen"] == env.gold_completion(c)
    assert not env.score(rows[0]["rejected"], c).correct
    assert rows[0]["system"] == env.system_prompt

    stats = summarize(rollouts, rows)
    assert stats["prompts"] == 3 and stats["mixed_groups"] == 1 and stats["pairs"] == 1
    assert abs(stats["sample_accuracy"] - 5 / 9) < 1e-9


def test_per_prompt_caps_pairs_by_distinct_completions():
    env = ArithmeticEnv(seed=0)
    (task,) = env.sample(1)
    roll = _rollout(env, task, [True, True, True, False, False])
    # three identical correct texts are one distinct chosen, so only one pair exists
    assert len(pairs_from_rollouts([roll], env, random.Random(0), per_prompt=3)) == 1


def test_mined_rows_load_through_preference_dataset_with_the_rollout_prompt(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    env = ArithmeticEnv(seed=0)
    (task,) = env.sample(1)
    rows = pairs_from_rollouts([_rollout(env, task, [True, False])], env, random.Random(0))
    path = tmp_path / "pairs.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    ds = PreferenceDataset(str(path), tokenizer, max_length=128)
    chosen_x = ds[0][0]
    prompt_ids = build_prompt_ids(tokenizer, env, task)
    assert torch.equal(chosen_x[: prompt_ids.numel()], prompt_ids)
