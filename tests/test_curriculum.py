import random
from types import SimpleNamespace

import torch

from curriculum import StratumSampler
from envs import ArithmeticEnv
from envs.base import Task


def _pool():
    return [Task(question=f"{a}+{b}", answer=str(a + b), meta={"op": "+", "n_digits": d})
            for d, (a, b) in ((1, (1, 2)), (1, (2, 3)), (2, (12, 13)), (3, (123, 456)))]


def _roll(task, right, total=8):
    return SimpleNamespace(task=task, rewards=[SimpleNamespace(correct=i < right) for i in range(total)])


def test_weights_start_equal_and_follow_the_solve_rate():
    sampler = StratumSampler(_pool(), ["op", "n_digits"], floor=0.05, decay=0.0)
    assert len(set(round(w, 6) for w in sampler.weights().values())) == 1

    # 1 digit always solved, 2 digits half, 3 digits never: the mixed stratum dominates, the others sit at the floor
    for _ in range(2):
        sampler.update([_roll(_pool()[0], 8), _roll(_pool()[2], 4), _roll(_pool()[3], 0)])
    w = sampler.weights()
    assert w["op=+ n_digits=2"] > 0.8 and w["op=+ n_digits=1"] == w["op=+ n_digits=3"] < 0.06
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_sampling_follows_the_weights_and_every_stratum_stays_alive():
    sampler = StratumSampler(_pool(), ["op", "n_digits"], floor=0.05, decay=0.0)
    sampler.update([_roll(_pool()[0], 8), _roll(_pool()[2], 4), _roll(_pool()[3], 0)])
    tasks = sampler.sample(4000, torch.Generator().manual_seed(0))
    share = {k: sum(sampler.key(t) == k for t in tasks) / len(tasks) for k in sampler.names}
    assert 0.8 < share["op=+ n_digits=2"] < 0.95
    assert share["op=+ n_digits=3"] > 0  # the floor: an estimate that is wrong can still be corrected


def test_ema_moves_slowly_with_decay():
    sampler = StratumSampler(_pool(), ["n_digits"], decay=0.8, prior=0.5)
    sampler.update([_roll(_pool()[3], 0)])
    assert abs(sampler.rate["n_digits=3"] - 0.4) < 1e-9
