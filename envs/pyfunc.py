"""Verifiable Python-function environment (RLVR): the reward is a test suite, not a string match.

The policy gets a one-sentence spec of a small function ``solve`` and one worked example, and must
answer with a fenced code block::

    ```python
    def solve(xs):
        return sum(x for x in xs if x % 3 == 0)
    ```

The program is run in ``envs.sandbox`` against hidden ``[args, expected]`` cases generated from a
reference solution. Rewards:

* **accuracy**: every hidden case passes (all-or-nothing by default)
* **format**: a closed fenced block that defines ``solve``
* ``partial_weight`` (default 0): credit for the fraction of cases that pass. Off by default on
  purpose: with 6 cases a function that always returns the most common output scores ~0.3 of the
  fraction, and the policy finds that before it finds the algorithm.

Tasks come from ~26 parameterised families (a threshold, a divisor, a separator, ...), so the
problems are templated: this measures whether RL turns a model that *can* write these into one that
reliably does, and how that transfers to a family it never saw (``families`` /
``exclude_families``), not general programming ability. A 100M model is nowhere near HumanEval;
``datasets/eval/humaneval.jsonl`` stays a decontamination set, not a target.
"""

from __future__ import annotations

import copy
import random
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .base import Reward, Task, TaskEnv, register_env
from .sandbox import run_cases

ENTRY = "solve"
_FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(completion: str) -> str | None:
    """The first closed fenced block that defines ``solve``; failing that, the first closed block."""
    blocks = [m.group(1) for m in _FENCE.finditer(completion)]
    for block in blocks:
        if re.search(rf"^\s*def\s+{ENTRY}\s*\(", block, re.MULTILINE):
            return block
    return blocks[0] if blocks else None


# ----------------------------------------------------------------------------- families

def _ints(rng: random.Random, lo=-9, hi=30, n_lo=0, n_hi=8) -> list[int]:
    return [rng.randint(lo, hi) for _ in range(rng.randint(n_lo, n_hi))]


def _word(rng: random.Random, n_lo=0, n_hi=7) -> str:
    return "".join(rng.choice("abcdeiloprstu") for _ in range(rng.randint(n_lo, n_hi)))


def _sentence(rng: random.Random) -> str:
    return " ".join(_word(rng, 1, 6) for _ in range(rng.randint(0, 5)))


@dataclass(frozen=True)
class Family:
    """``spec`` and ``code`` are format strings over the sampled parameters; ``inputs`` draws one call."""

    name: str
    params: Callable[[random.Random], dict]
    spec: str
    code: str
    inputs: Callable[[random.Random, dict], list]


def _palindrome_input(rng, _p):
    word = _word(rng, 1, 4)
    return [word + word[::-1][rng.randint(0, 1):] if rng.random() < 0.5 else _word(rng, 0, 6)]


def _clamp_params(rng):
    lo = rng.randint(-10, 10)
    return {"lo": lo, "hi": lo + rng.randint(1, 15)}


def _distinct_ints(rng, _p):
    xs = _ints(rng, 0, 12, 2, 7)
    while len(set(xs)) < 2:
        xs.append(rng.randint(0, 12))
    return [xs]


FAMILIES: tuple[Family, ...] = (
    Family("sum_multiples", lambda r: {"k": r.randint(2, 9)},
           "返回整数列表 xs 中所有能被 {k} 整除的数之和（没有就返回 0）。",
           "def solve(xs):\n    return sum(x for x in xs if x % {k} == 0)",
           lambda r, p: [_ints(r)]),
    Family("count_greater", lambda r: {"t": r.randint(-3, 15)},
           "返回整数列表 xs 中大于 {t} 的数的个数。",
           "def solve(xs):\n    return sum(1 for x in xs if x > {t})",
           lambda r, p: [_ints(r)]),
    Family("clamp", _clamp_params,
           "把整数 x 限制在 [{lo}, {hi}] 范围内：小于 {lo} 返回 {lo}，大于 {hi} 返回 {hi}，否则返回 x。",
           "def solve(x):\n    return max({lo}, min({hi}, x))",
           lambda r, p: [r.randint(p["lo"] - 8, p["hi"] + 8)]),
    Family("repeat_join", lambda r: {"n": r.randint(2, 5), "sep": r.choice(["-", ",", "+", "|", "/"])},
           "把字符串 s 重复 {n} 次，中间用 '{sep}' 连接，返回结果。",
           "def solve(s):\n    return '{sep}'.join([s] * {n})",
           lambda r, p: [_word(r, 0, 4)]),
    Family("count_vowels", lambda r: {},
           "返回字符串 s 中元音字母（a、e、i、o、u，不分大小写）的个数。",
           "def solve(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
           lambda r, p: [_word(r).upper() if r.random() < 0.3 else _word(r)]),
    Family("is_palindrome", lambda r: {},
           "判断字符串 s 是不是回文（正着读和倒着读一样），返回 True 或 False。",
           "def solve(s):\n    return s == s[::-1]",
           _palindrome_input),
    Family("gcd", lambda r: {},
           "返回两个正整数 a 和 b 的最大公约数。",
           "def solve(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
           lambda r, p: [r.randint(1, 60), r.randint(1, 60)]),
    Family("digit_sum", lambda r: {},
           "返回非负整数 n 的各位数字之和。",
           "def solve(n):\n    return sum(int(c) for c in str(n))",
           lambda r, p: [r.choice([0, r.randint(1, 99), r.randint(100, 99999)])]),
    Family("prefix_sums", lambda r: {},
           "返回整数列表 xs 的前缀和列表：第 i 个元素是 xs[0] 到 xs[i] 之和。",
           "def solve(xs):\n    out = []\n    total = 0\n    for x in xs:\n        total += x\n        out.append(total)\n    return out",
           lambda r, p: [_ints(r)]),
    Family("dedupe", lambda r: {},
           "去掉整数列表 xs 中的重复元素，保留每个数第一次出现的顺序，返回新列表。",
           "def solve(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out",
           lambda r, p: [_ints(r, 0, 6, 0, 9)]),
    Family("second_largest", lambda r: {},
           "返回整数列表 xs 中第二大的不同数值（xs 里至少有两个不同的数）。",
           "def solve(xs):\n    return sorted(set(xs))[-2]",
           _distinct_ints),
    Family("range_of_list", lambda r: {},
           "返回非空整数列表 xs 的最大值减最小值。",
           "def solve(xs):\n    return max(xs) - min(xs)",
           lambda r, p: [_ints(r, -9, 30, 1, 8)]),
    Family("reverse_words", lambda r: {},
           "把字符串 s 中的单词（以空格分隔）顺序倒过来，用一个空格连接后返回。",
           "def solve(s):\n    return ' '.join(reversed(s.split()))",
           lambda r, p: [_sentence(r)]),
    Family("count_char", lambda r: {"c": r.choice("aeiorst")},
           "返回字符 '{c}' 在字符串 s 中出现的次数。",
           "def solve(s):\n    return s.count('{c}')",
           lambda r, p: [_word(r, 0, 9)]),
    Family("fib", lambda r: {},
           "返回第 n 个斐波那契数，其中 fib(0)=0，fib(1)=1，fib(n)=fib(n-1)+fib(n-2)。",
           "def solve(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
           lambda r, p: [r.randint(0, 20)]),
    Family("even_squares", lambda r: {},
           "返回整数列表 xs 中所有偶数的平方组成的列表，保持原来的顺序。",
           "def solve(xs):\n    return [x * x for x in xs if x % 2 == 0]",
           lambda r, p: [_ints(r)]),
    Family("sum_multiples_upto", lambda r: {"k": r.randint(2, 9)},
           "返回 1 到 n（含）之间所有 {k} 的倍数之和。",
           "def solve(n):\n    return sum(i for i in range(1, n + 1) if i % {k} == 0)",
           lambda r, p: [r.randint(0, 60)]),
    Family("every_kth", lambda r: {"k": r.randint(2, 4)},
           "返回列表 xs 中从第 0 个开始、每隔 {k} 个取一个元素组成的新列表。",
           "def solve(xs):\n    return xs[::{k}]",
           lambda r, p: [_ints(r, 0, 20, 0, 10)]),
    Family("rotate_left", lambda r: {"k": r.randint(1, 4)},
           "把非空列表 xs 循环左移 {k} 位（移出左边的元素接到右边），返回新列表。",
           "def solve(xs):\n    k = {k} % len(xs)\n    return xs[k:] + xs[:k]",
           lambda r, p: [_ints(r, 0, 20, 1, 8)]),
    Family("long_words", lambda r: {"t": r.randint(2, 5)},
           "返回字符串 s 中长度大于 {t} 的单词个数（单词以空格分隔）。",
           "def solve(s):\n    return sum(1 for w in s.split() if len(w) > {t})",
           lambda r, p: [_sentence(r)]),
    Family("adjacent_diff_sum", lambda r: {},
           "返回整数列表 xs 中所有相邻两个数之差的绝对值之和（少于两个元素时返回 0）。",
           "def solve(xs):\n    return sum(abs(a - b) for a, b in zip(xs, xs[1:]))",
           lambda r, p: [_ints(r)]),
    Family("is_prime", lambda r: {},
           "判断非负整数 n 是不是质数，返回 True 或 False。",
           "def solve(n):\n    if n < 2:\n        return False\n    return all(n % i for i in range(2, int(n ** 0.5) + 1))",
           lambda r, p: [r.randint(0, 60)]),
    Family("title_case", lambda r: {},
           "把字符串 s 中每个单词（以空格分隔）的首字母大写、其余字母小写，用一个空格连接后返回。",
           "def solve(s):\n    return ' '.join(w.capitalize() for w in s.split())",
           lambda r, p: [" ".join(_word(r, 1, 5).swapcase() for _ in range(r.randint(0, 4)))]),
    Family("has_duplicate", lambda r: {},
           "判断整数列表 xs 里有没有重复的数，有返回 True，没有返回 False。",
           "def solve(xs):\n    return len(set(xs)) != len(xs)",
           lambda r, p: [_ints(r, 0, 9, 0, 7)]),
    Family("first_min_index", lambda r: {},
           "返回非空整数列表 xs 中最小值第一次出现的下标。",
           "def solve(xs):\n    return xs.index(min(xs))",
           lambda r, p: [_ints(r, 0, 8, 1, 8)]),
    Family("join_numbers", lambda r: {"sep": r.choice([",", "-", "+", "|", ";"])},
           "把整数列表 xs 里的数依次转成字符串，用 '{sep}' 连接后返回（空列表返回空字符串）。",
           "def solve(xs):\n    return '{sep}'.join(str(x) for x in xs)",
           lambda r, p: [_ints(r)]),
)
FAMILY_NAMES = tuple(f.name for f in FAMILIES)
_BY_NAME = {f.name: f for f in FAMILIES}
_N_TESTS = 6


def _reference(source: str):
    namespace: dict = {}
    exec(source, namespace)  # noqa: S102 - trusted: our own templates
    return namespace[ENTRY]


def _draw_inputs(rng: random.Random, family: Family, params: dict) -> list[list]:
    """``_N_TESTS`` calls from the family's sampler; those include empty and tiny inputs."""
    return [family.inputs(rng, params) for _ in range(_N_TESTS)]


def _build_cases(family: Family, params: dict, rng: random.Random) -> tuple[list, list]:
    """Hidden cases (``_N_TESTS`` distinct inputs where possible) and one visible example."""
    function = _reference(family.code.format(**params))
    seen, cases = set(), []
    for args in _draw_inputs(rng, family, params) + _draw_inputs(rng, family, params):
        key = repr(args)
        if key in seen:
            continue
        seen.add(key)
        cases.append([args, function(*copy.deepcopy(args))])
        if len(cases) == _N_TESTS + 1:
            break
    example = cases.pop(rng.randrange(len(cases)))
    return cases, example


def _call_text(args: list) -> str:
    return ", ".join(repr(a) for a in args)


@register_env
class PyFuncEnv(TaskEnv):
    """Small Python functions, rewarded by running hidden tests in a sandbox."""

    name = "pyfunc"

    def __init__(
        self,
        seed: int = 0,
        families: Sequence[str] | None = None,
        exclude_families: Sequence[str] = (),
        accuracy_weight: float = 1.0,
        format_weight: float = 0.2,
        partial_weight: float = 0.0,
        workers: int = 8,
        case_seconds: float = 0.5,
        wall_seconds: float = 4.0,
    ):
        super().__init__(seed)
        chosen = list(families) if families else list(FAMILY_NAMES)
        unknown = [name for name in [*chosen, *exclude_families] if name not in _BY_NAME]
        if unknown:
            raise ValueError(f"unknown families {unknown}; available: {list(FAMILY_NAMES)}")
        self.families = tuple(name for name in chosen if name not in set(exclude_families))
        if not self.families:
            raise ValueError("no families left after excluding")
        self.accuracy_weight = accuracy_weight
        self.format_weight = format_weight
        self.partial_weight = partial_weight
        self.workers = workers
        self.case_seconds = case_seconds
        self.wall_seconds = wall_seconds

    @property
    def system_prompt(self) -> str:
        return "你是一个 Python 助手，必须只用一个 ```python 代码块回答，代码块里定义要求的函数。"

    def sample_task(self) -> Task:
        family = _BY_NAME[self._rng.choice(self.families)]
        params = family.params(self._rng)
        cases, (example_args, example_out) = _build_cases(family, params, self._rng)
        spec = family.spec.format(**params)
        question = f"{spec}\n例如：{ENTRY}({_call_text(example_args)}) 返回 {example_out!r}。"
        return Task(
            question=question,
            answer=family.code.format(**params),
            meta={"family": family.name, "entry": ENTRY, "cases": cases},
        )

    def render(self, task: Task) -> str:
        return (
            f"写一个 Python 函数 {ENTRY}：{task.question}\n"
            "把完整代码放在一个 ```python 代码块里，不要写代码块以外的内容。"
        )

    def _grade(self, completion: str, task: Task) -> Reward:
        code = extract_code(completion)
        format_ok = code is not None and re.search(rf"^\s*def\s+{ENTRY}\s*\(", code, re.MULTILINE) is not None
        passed, total = 0, len(task.meta["cases"])
        if format_ok:
            result = run_cases(
                code, task.meta.get("entry", ENTRY), task.meta["cases"],
                case_seconds=self.case_seconds, wall_seconds=self.wall_seconds,
            )
            passed = sum(result.passed)
        correct = format_ok and passed == total
        accuracy = self.accuracy_weight if correct else self.partial_weight * passed / max(total, 1)
        fmt = self.format_weight if format_ok else 0.0
        return Reward(
            total=accuracy + fmt, accuracy=accuracy, format=fmt, format_ok=format_ok, correct=correct,
            parsed_answer=f"{passed}/{total}" if format_ok else None,
        )

    def score(self, completion: str, task: Task) -> Reward:
        return self._grade(completion, task)

    def score_many(self, completions: Sequence[str], task: Task) -> list[Reward]:
        """One sandbox process per completion, a few at a time: a group of 8 is ~8 x 30 ms serially."""
        if len(completions) <= 1 or self.workers <= 1:
            return [self._grade(text, task) for text in completions]
        with ThreadPoolExecutor(max_workers=min(self.workers, len(completions))) as pool:
            return list(pool.map(lambda text: self._grade(text, task), completions))

    def gold_completion(self, task: Task) -> str:
        return f"```python\n{task.answer}\n```"

    def corrupt_completion(self, task: Task, rng: random.Random) -> str:
        """A completion that still looks like an answer but fails: mutate the reference until it does."""
        swaps = [(" + ", " - "), (" - ", " + "), (" == ", " != "), (" > ", " >= "), (" < ", " <= "),
                 ("% 2", "% 3"), ("[::-1]", ""), ("min(", "max("), ("max(", "min("), (" > ", " < "),
                 ("sum(", "len("), ("return ", "return 1 + "), ("+ 1", "+ 2"), ("0, 1", "1, 1")]
        rng.shuffle(swaps)
        for old, new in swaps:
            if old in task.answer:
                mutated = task.answer.replace(old, new, 1)
                if not self._grade(f"```python\n{mutated}\n```", task).correct:
                    return f"```python\n{mutated}\n```"
        # No mutation broke it: drop the final return so the function yields None.
        lines = task.answer.rstrip().splitlines()
        body = [line for line in lines[:-1]] + ["    pass"]
        return "```python\n" + "\n".join(body) + "\n```"
