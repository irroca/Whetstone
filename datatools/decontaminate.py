"""Remove training documents that overlap an evaluation set.

An eval number is meaningless if the answers were in the training data, and
exact-match dedup does not catch it: a benchmark question embedded in a longer
web page is byte-different from the benchmark file.

The method follows SmolLM2, which decontaminates against GSM8K / MATH / MMLU
using 13-gram matching plus a minimum longest-common-subsequence overlap ratio
of 0.6. Two stages:

1. **13-gram containment** (cheap, always on). Build the set of 13-grams of
   every eval item; flag any training document sharing one.
2. **LCS overlap ratio** (optional, stricter). For flagged documents only,
   align the matched eval item with the document at their shared n-grams and
   require the longest common subsequence there to cover at least
   ``lcs_threshold`` of the shorter text. This demotes coincidental 13-gram
   collisions back to clean: on the smoke corpus, a coordinate-geometry formula
   sheet shares the triangle-area determinant with a MATH-500 problem.

**CJK needs its own tokenization.** A 13-gram over whitespace-separated words
does not exist in Chinese. ``text_units`` emits each CJK character as its own
unit and each other run of letters/digits as a word, so a 13-gram is 13 words in
English and 13 characters in Chinese — both roughly a sentence fragment.
Punctuation and symbols are not units (as in GPT-3's contamination analysis):
as units, a run of dashes in one MMLU question matches every markdown table,
and LaTeX braces turn any formula skeleton into a "phrase".

**Answers are protected by exact matching only, unless they are long.** A
benchmark answer leaking on its own is not contamination; the question is what
makes an item recognizable. Indexing short answers as n-grams was the largest
source of false positives on real data: MMLU's ``1,2,3`` matched 76 of 4.8k
smoke-test documents, MATH's ``\\frac{3}{2}`` 35, TAL's ``等腰三角形`` most of
the rest. So an answer shorter than ``n`` units only matches a training part
that is itself that string; questions and solutions keep the shorter-gram rule.
This is asserted in the tests so it stays a known property, not a surprise.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .records import (
    TASK,
    ReadStats,
    detect_schema,
    normalize_text,
    read_jsonl,
    record_parts,
    record_text,
    write_jsonl,
)

DEFAULT_N = 13
# Below this an n-gram matches generic phrasing ("which of the following is
# true", "what is the value of b"), so shorter items fall back to exact matching.
MIN_GRAM = 8
# Bounds the quadratic LCS gate to about 600 x 1200 units per alignment.
LCS_CONTEXT = 300
_CJK = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
_UNIT_RE = re.compile(rf"[^\W_{_CJK}]+|[{_CJK}]")


def text_units(text: str) -> list[str]:
    """Split into comparison units: each CJK character, or a run of other letters/digits.

    Punctuation, symbols and underscores separate units and are dropped.
    ``"solve 12 + 7 计算结果"`` -> ``['solve', '12', '7', '计', '算', '结', '果']``
    """
    return _UNIT_RE.findall(text)


def ngrams(text: str, n: int = DEFAULT_N) -> set[tuple[str, ...]]:
    """All ``n``-grams of a text's units. Texts shorter than ``n`` yield one short gram."""
    if n < 1:
        raise ValueError("n must be >= 1")
    units = text_units(normalize_text(text, casefold=True))
    if not units:
        return set()
    if len(units) <= n:
        return {tuple(units)}
    return {tuple(units[i : i + n]) for i in range(len(units) - n + 1)}


def lcs_length(a: Sequence[str], b: Sequence[str], max_units: int = 2000) -> int:
    """Length of the longest common subsequence, with rolling rows.

    Quadratic, so inputs are truncated to ``max_units``. It only runs on
    documents that already matched an n-gram, which is rare.
    """
    a, b = a[:max_units], b[:max_units]
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, 1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def lcs_ratio(a: str, b: str, n: int = DEFAULT_N) -> float:
    """How much of the shorter text the longer one reproduces, in order, where they overlap.

    At each n-gram the texts share, the shorter text (up to ``LCS_CONTEXT``
    units either side of it) is aligned against the longer one and their LCS
    is taken over the shorter side, so a long page cannot dilute it. The best
    alignment wins; texts sharing no n-gram score 0. Measured against a whole
    page instead, a short item collects its words by chance from anywhere on
    it, and any cap on the page's length hides a leak further down a book.
    """
    units_a, units_b = text_units(normalize_text(a, True)), text_units(normalize_text(b, True))
    short, long_ = sorted((units_a, units_b), key=len)
    if not short:
        return 0.0
    size = min(n, len(short))
    offsets: dict[tuple[str, ...], int] = {}
    for q in range(len(short) - size + 1):
        offsets.setdefault(tuple(short[q : q + size]), q)
    best, covered = 0.0, 0
    for p in range(len(long_) - size + 1):
        q = offsets.get(tuple(long_[p : p + size])) if p >= covered else None
        if q is None:
            continue
        lo, hi = max(0, q - LCS_CONTEXT), min(len(short), q + size + LCS_CONTEXT)
        segment = short[lo:hi]
        start, end = p - (q - lo), p + (hi - q)
        slack = len(segment) // 2
        window = long_[max(0, start - slack) : end + slack]
        best = max(best, lcs_length(window, segment) / len(segment))
        covered = end
    return best


@dataclass
class EvalIndex:
    """n-gram index over an evaluation set.

    Eval items shorter than ``n`` are indexed at *their own* length rather than
    as one whole-item gram. Otherwise a 10-word benchmark question could never
    match a longer page, because that page only ever produces 13-unit windows.
    Items shorter than ``min_gram`` are held aside for exact matching, since a
    2-unit gram would match almost every document.
    """

    n: int = DEFAULT_N
    min_gram: int = MIN_GRAM
    gram_to_item: dict[tuple[str, ...], int] = field(default_factory=dict)
    gram_sizes: set[int] = field(default_factory=set)
    short_items: dict[str, int] = field(default_factory=dict)
    items: list[str] = field(default_factory=list)

    def add(self, text: str, min_gram: Optional[int] = None) -> None:
        """Index one eval item. Items shorter than ``min_gram`` units only match exactly."""
        units = text_units(normalize_text(text, casefold=True))
        if not units:
            return
        item_id = len(self.items)
        self.items.append(text)
        if len(units) < (self.min_gram if min_gram is None else min_gram):
            self.short_items.setdefault(" ".join(units), item_id)
            return
        size = min(len(units), self.n)
        self.gram_sizes.add(size)
        for start in range(len(units) - size + 1):
            self.gram_to_item.setdefault(tuple(units[start : start + size]), item_id)

    def match(self, text: str) -> Optional[int]:
        """Id of an eval item overlapping ``text``, else None."""
        units = text_units(normalize_text(text, casefold=True))
        if not units:
            return None
        if self.short_items:
            hit = self.short_items.get(" ".join(units))
            if hit is not None:
                return hit
        for size in sorted(self.gram_sizes):
            if len(units) < size:
                continue
            for start in range(len(units) - size + 1):
                hit = self.gram_to_item.get(tuple(units[start : start + size]))
                if hit is not None:
                    return hit
        return None

    def __len__(self) -> int:
        return len(self.items)


def build_eval_index(
    texts: Iterable[str],
    n: int = DEFAULT_N,
    min_gram: int = MIN_GRAM,
) -> EvalIndex:
    index = EvalIndex(n=n, min_gram=min_gram)
    for text in texts:
        index.add(text)
    return index


def load_eval_index(
    paths: Sequence[str],
    n: int = DEFAULT_N,
    field_name: str = "auto",
) -> EvalIndex:
    """Index an eval set from JSONL.

    ``field_name="auto"`` indexes each meaningful field *separately* (question,
    answer, each conversation turn, ...). Indexing the joined record instead
    would insert separators that never occur in natural text, so a page
    quoting only the benchmark question would slip through.
    """
    index = EvalIndex(n=n)
    for path in paths:
        for record in read_jsonl(path):
            if field_name != "auto":
                index.add(str(record.data.get(field_name, "")))
            elif detect_schema(record.data) == TASK:
                for key in ("question", "answer", "solution"):
                    if record.data.get(key):
                        # Short answers match exactly only; see the module docstring.
                        index.add(str(record.data[key]), min_gram=n if key == "answer" else None)
            else:
                for part in record_parts(record.data):
                    index.add(part)
    return index


def match_record(record: dict, index: EvalIndex) -> Optional[tuple[str, int]]:
    """First (text, eval item id) pair where a part of ``record`` hits the index.

    Checked per part for the same reason the index is built per part: an
    assistant turn that reproduces a benchmark answer is contamination even
    though the whole conversation is a different string.
    """
    for part in record_parts(record):
        if not part.strip():
            continue
        hit = index.match(part)
        if hit is not None:
            return part, hit
    return None


@dataclass
class DecontaminationReport:
    input_records: int = 0
    removed: int = 0
    ngram_hits: int = 0
    lcs_rescued: int = 0
    eval_items: int = 0
    n: int = DEFAULT_N
    lcs_threshold: Optional[float] = None
    examples: list[dict] = field(default_factory=list)

    @property
    def kept(self) -> int:
        return self.input_records - self.removed

    def to_dict(self) -> dict:
        return {**self.__dict__, "kept": self.kept}


def decontaminate(
    records: Sequence[dict],
    index: EvalIndex,
    lcs_threshold: Optional[float] = None,
) -> tuple[list[dict], DecontaminationReport]:
    """Drop records overlapping the eval index.

    With ``lcs_threshold`` set, an n-gram hit is only treated as contamination
    when the LCS overlap also clears the threshold; otherwise the hit is
    counted as rescued and the record is kept.
    """
    report = DecontaminationReport(
        input_records=len(records),
        eval_items=len(index),
        n=index.n,
        lcs_threshold=lcs_threshold,
    )
    kept: list[dict] = []
    for record in records:
        match = match_record(record, index)
        if match is None:
            kept.append(record)
            continue
        text, hit = match
        report.ngram_hits += 1
        if lcs_threshold is not None and lcs_ratio(text, index.items[hit], index.n) < lcs_threshold:
            report.lcs_rescued += 1
            kept.append(record)
            continue
        report.removed += 1
        if len(report.examples) < 5:
            report.examples.append(
                {"train": text[:120], "eval": index.items[hit][:120]}
            )
    return kept, report


def render(report: DecontaminationReport) -> str:
    lines = ["=== decontaminate"]
    lines.append(f"  eval items indexed: {report.eval_items} ({report.n}-gram)")
    lines.append(f"  input: {report.input_records} records")
    lines.append(f"  n-gram hits: {report.ngram_hits}")
    if report.lcs_threshold is not None:
        lines.append(
            f"  rescued by LCS < {report.lcs_threshold}: {report.lcs_rescued}"
        )
    lines.append(f"  removed: {report.removed}, kept: {report.kept}")
    for example in report.examples[:3]:
        lines.append(f"    train: {example['train']!r}")
        lines.append(f"    eval : {example['eval']!r}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove eval-contaminated training records")
    parser.add_argument("path")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--against", nargs="+", required=True, help="Eval/benchmark JSONL files")
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="n-gram size (SmolLM2 uses 13)")
    parser.add_argument(
        "--lcs_threshold",
        type=float,
        default=None,
        help="Require this LCS overlap ratio on top of an n-gram hit (SmolLM2 uses 0.6)",
    )
    parser.add_argument("--field", type=str, default="auto", help="Eval field, or 'auto'")
    parser.add_argument("--report", type=str, default=None)
    args = parser.parse_args()

    read_stats = ReadStats()
    records = [r.data for r in read_jsonl(args.path, read_stats)]
    index = load_eval_index(args.against, args.n, args.field)

    kept, report = decontaminate(records, index, args.lcs_threshold)
    write_jsonl(args.out, kept)
    print(render(report))
    print(f"  wrote -> {args.out}")

    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
        print(f"  report -> {args.report}")


if __name__ == "__main__":
    main()
