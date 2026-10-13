import json
import random

import pytest

from envs import PyFuncEnv, load_tasks, make_env
from envs.generate_data import build_rows, write_jsonl
from envs.pyfunc import FAMILY_NAMES, extract_code


def gold(task):
    return f"```python\n{task.answer}\n```"


def test_every_family_has_a_reference_that_passes_its_own_hidden_tests():
    env = PyFuncEnv(seed=0)
    seen = set()
    for task in env.sample(400):
        seen.add(task.meta["family"])
        reward = env.score(gold(task), task)
        assert reward.correct and reward.format_ok, (task.meta["family"], task.question)
        assert reward.total == pytest.approx(1.2)
        assert len(task.meta["cases"]) >= 3
    assert seen == set(FAMILY_NAMES)


def test_sampling_is_deterministic_in_the_seed_and_varied_across_it():
    a = [t.question for t in PyFuncEnv(seed=3).sample(30)]
    assert a == [t.question for t in PyFuncEnv(seed=3).sample(30)]
    assert a != [t.question for t in PyFuncEnv(seed=4).sample(30)]


def test_the_prompt_shows_one_example_that_is_not_a_hidden_case():
    env = PyFuncEnv(seed=1)
    for task in env.sample(40):
        example = task.question.split("例如：", 1)[1]
        for args, _ in task.meta["cases"]:
            call = f"solve({', '.join(repr(a) for a in args)})"
            assert not example.startswith(call + " 返回"), task.question


def test_format_and_accuracy_are_scored_separately():
    env = PyFuncEnv(seed=0)
    task = env.sample_task()
    wrong = "```python\ndef solve(*a):\n    return None\n```"
    for text, fmt, correct in [
        (gold(task), True, True),
        (wrong, True, False),                                     # well formed, wrong: the hacked-format case
        (task.answer, False, False),                              # right code, no fence
        ("```python\n" + task.answer, False, False),              # fence never closed
        ("```python\ndef other():\n    return 1\n```", False, False),   # no solve
        ("", False, False),
    ]:
        reward = env.score(text, task)
        assert (reward.format_ok, reward.correct) == (fmt, correct), text
    assert env.score(wrong, task).hacked_format
    assert env.score(wrong, task).total == pytest.approx(0.2)


def test_the_first_block_that_defines_solve_wins():
    text = "说明\n```python\nx = 1\n```\n再来\n```python\ndef solve(a):\n    return a\n```"
    assert "def solve" in extract_code(text)
    assert extract_code("```\nprint(1)\n```") == "print(1)\n"
    assert extract_code("no code here") is None


def test_partial_credit_is_off_unless_asked_for():
    task = PyFuncEnv(seed=2).sample_task()
    half = "```python\ndef solve(*a):\n    return None\n```"
    assert PyFuncEnv(seed=2).score(half, task).accuracy == 0.0
    task = next(t for t in PyFuncEnv(seed=2).sample(200) if t.meta["family"] == "is_palindrome")
    always_false = "```python\ndef solve(s):\n    return False\n```"
    partial = PyFuncEnv(seed=2, partial_weight=0.5).score(always_false, task)
    passed = sum(1 for _, expected in task.meta["cases"] if expected is False)
    assert partial.accuracy == pytest.approx(0.5 * passed / len(task.meta["cases"]))
    assert not partial.correct


def test_score_many_matches_score_and_keeps_the_order():
    env = PyFuncEnv(seed=0, workers=4)
    task = env.sample_task()
    texts = [gold(task), "nope", gold(task), "```python\ndef solve(*a):\n    return 0\n```", gold(task)]
    assert env.score_many(texts, task) == [env.score(t, task) for t in texts]
    assert [r.correct for r in env.score_many(texts, task)] == [True, False, True, False, True]


def test_corrupted_completions_are_well_formed_but_wrong():
    env = PyFuncEnv(seed=0)
    rng = random.Random(0)
    for task in env.sample(120):
        reward = env.score(env.corrupt_completion(task, rng), task)
        assert reward.format_ok and not reward.correct, task.meta["family"]


def test_family_selection_and_exclusion():
    env = PyFuncEnv(seed=0, families=["gcd", "fib"], exclude_families=["fib"])
    assert {t.meta["family"] for t in env.sample(30)} == {"gcd"}
    held_in = PyFuncEnv(seed=0, exclude_families=["gcd"])
    assert "gcd" not in {t.meta["family"] for t in held_in.sample(300)}
    with pytest.raises(ValueError, match="unknown families"):
        PyFuncEnv(families=["nope"])
    with pytest.raises(ValueError, match="no families left"):
        PyFuncEnv(families=["gcd"], exclude_families=["gcd"])


def test_registered_and_datasets_round_trip_through_the_generic_tools(tmp_path):
    env = make_env("pyfunc", seed=5)
    rows = build_rows(env, "eval", 10, random.Random(5))
    path = tmp_path / "eval.jsonl"
    write_jsonl(str(path), rows)
    tasks = load_tasks(str(path))
    assert [t.question for t in tasks] == [r["question"] for r in rows]
    assert all(env.score(gold(t), t).correct for t in tasks), "cases survive the JSONL round trip"
    sft = build_rows(env, "sft", 3, random.Random(5))
    assert sft[0]["conversations"][0]["role"] == "system"
    assert sft[0]["conversations"][2]["content"].startswith("```python")
    pref = build_rows(env, "preference", 3, random.Random(5))
    assert {"prompt", "chosen", "rejected", "system"} <= set(pref[0])
    json.dumps(rows)
