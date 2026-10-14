"""Adaptive prompt sampling for GRPO: spend the batch where the policy is neither always right nor always wrong.

A group whose samples all agree has zero reward variance and no gradient. ``select_prompts.py`` removes such
prompts once, from one policy. This keeps a running solve-rate estimate per *stratum* (task metadata such as
operator and digit count, so the estimate rests on many groups) and samples strata with weight
``floor + 4 p (1 - p)``: 1 at p = 0.5, ``floor`` at p = 0 or 1. The estimate follows the policy as it
learns, so a stratum that was hopeless becomes worth sampling and one that was solved stops being drawn.
It is a heuristic, not a bandit with guarantees; the floor keeps every stratum alive so an estimate that is
wrong at the start can still correct itself.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import torch


class StratumSampler:
    def __init__(self, pool: Sequence, keys: Sequence[str], *, floor: float = 0.05, decay: float = 0.8,
                 prior: float = 0.5):
        if not pool:
            raise ValueError("StratumSampler needs a non-empty pool")
        if not 0.0 <= decay < 1.0 or floor <= 0.0:
            raise ValueError("decay must be in [0, 1) and floor positive")
        self.pool, self.keys, self.floor, self.decay = pool, tuple(keys), floor, decay
        self.members: dict[str, list[int]] = defaultdict(list)
        for i, task in enumerate(pool):
            self.members[self.key(task)].append(i)
        self.names = sorted(self.members)
        self.rate = {name: prior for name in self.names}

    def key(self, task) -> str:
        return " ".join(f"{k}={task.meta[k]}" for k in self.keys if k in task.meta) or "all"

    def weights(self) -> dict[str, float]:
        raw = {n: self.floor + 4.0 * self.rate[n] * (1.0 - self.rate[n]) for n in self.names}
        total = sum(raw.values())
        return {n: w / total for n, w in raw.items()}

    def sample(self, n: int, generator: torch.Generator | None = None) -> list:
        w = self.weights()
        probs = torch.tensor([w[name] for name in self.names])
        picks = torch.multinomial(probs, n, replacement=True, generator=generator).tolist()
        tasks = []
        for p in picks:
            members = self.members[self.names[p]]
            tasks.append(self.pool[members[int(torch.randint(len(members), (1,), generator=generator))]])
        return tasks

    def update(self, rollouts: Sequence) -> None:
        """Move each drawn stratum's estimate toward this batch's mean solve rate."""
        seen: dict[str, list[float]] = defaultdict(list)
        for roll in rollouts:
            seen[self.key(roll.task)].append(sum(r.correct for r in roll.rewards) / max(len(roll.rewards), 1))
        for name, rates in seen.items():
            if name in self.rate:
                self.rate[name] = self.decay * self.rate[name] + (1.0 - self.decay) * (sum(rates) / len(rates))
