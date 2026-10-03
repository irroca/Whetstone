"""Build a training corpus from a mixture spec.

Mixture weights are in **tokens**, but corpora are published in documents and
bytes, so the budget can only be enforced by tokenizing as we go. The pipeline
is therefore streaming end to end: each source is pulled lazily, filtered,
token-counted, and written out the moment its share of the budget is met. At
10B tokens the corpus is ~30GB of text and nothing can be held in memory.

Stages, in this order:

1. **Pull** — HuggingFace ``streaming=True``, or a local JSONL (used by tests).
2. **Filter** — per-source thresholds from ``datatools.filters``; every
   rejection is attributed to the rule that caused it.
3. **Exact dedup** — a running digest set, which is the only dedup that scales
   in a single streaming pass (see the note on near-dedup below).
4. **Decontaminate** — 13-gram overlap against the eval sets.
5. **Split** — deterministic train/val/holdout by content hash.
6. **Manifest** — actual tokens and documents per source, rejection counts per
   rule, and the seeds, so two ablation runs can be told apart.

**Near-duplicate dedup is not part of this pass.** MinHash needs a signature
per surviving document (~1KB at 128 permutations), so it is bounded to roughly
1–2M documents in memory. Run ``datatools.dedup`` on a single source file when
it fits. In practice the yield is low here: FineWeb-Edu and FineWeb2-HQ are
already MinHash-deduplicated upstream, so the remaining near-duplicates come
from cross-source overlap and from our own synthetic data.

Every emitted record carries a ``source`` field naming the mixture slot it came
from, so the merged splits can still be broken down per source (per-language
validation loss is what the mixture ablations compare).

::

    python3 -m datatools.prepare configs/mixture_v1.json --probe 3
    python3 -m datatools.prepare configs/mixture_v1.json --dry_run
    python3 -m datatools.prepare configs/mixture_v1.json --out_dir datasets/prepared

``--probe`` pulls a few rows from every source and checks the spec against
them: a wrong config name, a gated repo or a renamed text field shows up for all
sources at once, instead of killing a long run at whichever source comes first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional, Sequence

from .decontaminate import DEFAULT_N, EvalIndex, load_eval_index
from .filters import FilterConfig, reject_reason
from .records import (
    PREFERENCE,
    SFT,
    TASK,
    detect_schema,
    normalize_text,
    read_jsonl,
    record_text,
    write_jsonl,
)
from .split import HOLDOUT, TRAIN, VAL, assign_split

TOKENIZE_BATCH = 256
PROJECTED_READ_AHEAD = 64 * 1024

# Fields kept when a row already has a non-pretrain schema; everything else is dropped.
SCHEMA_FIELDS = {
    SFT: ("conversations",),
    PREFERENCE: ("prompt", "chosen", "rejected"),
    TASK: ("question", "answer", "solution"),
}


@dataclass
class SourceSpec:
    """One component of the mixture."""

    name: str
    weight: float
    text_field: str = "text"
    jsonl: Optional[str] = None
    hf: Optional[dict] = None
    filters: FilterConfig = field(default_factory=FilterConfig)
    where: dict = field(default_factory=dict)
    max_records: Optional[int] = None
    note: str = ""

    @classmethod
    def from_dict(cls, values: dict) -> "SourceSpec":
        values = dict(values)
        if "filters" in values:
            values["filters"] = FilterConfig.from_dict(values["filters"])
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown source keys: {sorted(unknown)}")
        source = cls(**values)
        if (source.jsonl is None) == (source.hf is None):
            raise ValueError(f"source {source.name!r} needs exactly one of 'jsonl' or 'hf'")
        if source.weight <= 0:
            raise ValueError(f"source {source.name!r} must have weight > 0")
        columns = (source.hf or {}).get("columns")
        if columns is not None:
            needed = {source.text_field, *(key.split(".")[0] for key in source.where)}
            missing = sorted(needed - set(columns))
            if missing:
                # Every row would be rejected, and the run would only show 0% fill at the end.
                raise ValueError(f"source {source.name!r}: hf.columns {columns} omits {missing}")
        return source


@dataclass
class MixtureSpec:
    name: str
    total_tokens: int
    sources: list[SourceSpec]
    tokenizer: str = "./tokenizer/zh_6400"
    seed: int = 0
    val_fraction: float = 0.005
    holdout_fraction: float = 0.005
    decontaminate: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict) -> "MixtureSpec":
        values = dict(values)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown mixture keys: {sorted(unknown)}")
        values["sources"] = [SourceSpec.from_dict(s) for s in values.get("sources", [])]
        spec = cls(**values)
        if not spec.sources:
            raise ValueError("mixture needs at least one source")
        total_weight = sum(s.weight for s in spec.sources)
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"source weights must sum to 1.0, got {total_weight:.6f}")
        return spec

    @classmethod
    def load(cls, path: str) -> "MixtureSpec":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def token_budget(self, source: SourceSpec) -> int:
        return int(round(self.total_tokens * source.weight))


@dataclass
class SourceReport:
    name: str
    target_tokens: int = 0
    tokens: int = 0
    documents: int = 0
    seen: int = 0
    rejected: Counter = field(default_factory=Counter)
    exact_duplicates: int = 0
    exhausted: bool = False

    @property
    def fill(self) -> float:
        return self.tokens / max(self.target_tokens, 1)

    @property
    def keep_rate(self) -> float:
        """Share of records read that survived filtering and exact dedup."""
        return self.documents / max(self.seen, 1)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target_tokens": self.target_tokens,
            "tokens": self.tokens,
            "documents": self.documents,
            "records_read": self.seen,
            "fill": self.fill,
            "keep_rate": self.keep_rate,
            "tokens_per_doc": self.tokens / max(self.documents, 1),
            "rejected": dict(self.rejected),
            "exact_duplicates": self.exact_duplicates,
            "exhausted": self.exhausted,
        }


def hf_load_options(hf: dict) -> dict:
    """``load_dataset`` keyword arguments for an ``hf`` source spec.

    Column projection only saves bandwidth if each read stops at the end of
    the column chunk. fsspec reads 5MiB past every read by default, which lands
    in the next, dropped column: on FineWeb2-HQ that is the embedding, and one
    row group costs 8.7MB instead of 3.8MB (25.6MB with no projection at all).
    """
    options = dict(hf)
    options.setdefault("streaming", True)
    options.setdefault("split", "train")
    if "columns" in options:
        options.setdefault("storage_options", {"hf": {"block_size": PROJECTED_READ_AHEAD}})
    return options


def iter_raw(source: SourceSpec) -> Iterator[dict]:
    """Yield raw records from a source, lazily."""
    if source.jsonl is not None:
        for record in read_jsonl(source.jsonl):
            yield record.data
        return

    from datasets import load_dataset  # imported lazily: tests run offline

    options = hf_load_options(source.hf or {})
    path = options.pop("path")
    dataset = load_dataset(path, **options)
    for row in dataset:
        yield dict(row)


def where_mismatch(raw: dict, where: dict) -> Optional[str]:
    """The first ``where`` field a raw row fails, as a rejection reason.

    Keys are dotted paths into the row (``metadata.language``); a list value
    accepts any of its members. This is for upstream metadata the text filters
    cannot see, such as a book's declared language.
    """
    for key, expected in where.items():
        value: Any = raw
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        allowed = expected if isinstance(expected, list) else [expected]
        if value not in allowed:
            return f"where:{key}"
    return None


def to_record(raw: dict, source: SourceSpec) -> Optional[dict]:
    """Project a raw row onto one of the repo's four schemas, dropping the rest.

    Conversation, preference and task rows keep their schema's fields, so such
    a dataset can be mixed in without special-casing. Anything else becomes
    ``{"text": row[text_field]}``. Upstream columns are never carried along:
    FineWeb2-HQ ships a 768-float embedding with every document, and keeping it
    makes the Chinese slice 9.3x larger than its text (81MB for 8.8MB).
    """
    schema = detect_schema(raw)
    if schema in SCHEMA_FIELDS:
        return {key: raw[key] for key in SCHEMA_FIELDS[schema] if key in raw}
    value = raw.get(source.text_field)
    if value is None:
        return None
    return {"text": str(value)}


def _batched(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def count_tokens(tokenizer, texts: Sequence[str]) -> list[int]:
    """Token counts for a batch. Batched because this is the pipeline's hot path."""
    if not texts:
        return []
    # verbose=False: whole documents run past model_max_length by design, and
    # the warning about "indexing errors" only applies to feeding a model.
    encoded = tokenizer(list(texts), add_special_tokens=False, verbose=False)
    return [len(ids) for ids in encoded["input_ids"]]


def prepare_source(
    spec: MixtureSpec,
    source: SourceSpec,
    tokenizer,
    out_path: Optional[str],
    progress_every: int = 0,
) -> SourceReport:
    """Stream one source until its token budget is met, writing filtered records."""
    report = SourceReport(name=source.name, target_tokens=spec.token_budget(source))
    seen_digests: set[bytes] = set()
    # Each buffered record carries the counters as they stood when it was
    # accepted. When the budget fills at a record, the report rewinds to that
    # record's snapshot: rows read after it were never needed, and counting
    # them (and their rejections) makes a source of 70k-token books report an
    # 18% keep rate when 89% of its rows pass.
    buffer: list[tuple[dict, str, tuple[int, Counter, int]]] = []
    buffered_chars = 0
    handle = None
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        handle = open(out_path, "w", encoding="utf-8")

    def flush() -> bool:
        """Tokenize and emit the buffer. Returns True when the budget is met."""
        nonlocal buffer, buffered_chars
        if not buffer:
            return False
        counts = count_tokens(tokenizer, [text for _, text, _ in buffer])
        try:
            for (record, _, snapshot), n_tokens in zip(buffer, counts):
                if handle is not None:
                    tagged = {**record, "source": source.name}
                    handle.write(json.dumps(tagged, ensure_ascii=False) + "\n")
                report.tokens += n_tokens
                report.documents += 1
                if report.tokens >= report.target_tokens:
                    report.seen, report.rejected, report.exact_duplicates = snapshot
                    return True
            return False
        finally:
            buffer, buffered_chars = [], 0

    try:
        for raw in iter_raw(source):
            if source.max_records is not None and report.seen >= source.max_records:
                break
            report.seen += 1

            mismatch = where_mismatch(raw, source.where)
            if mismatch is not None:
                report.rejected[mismatch] += 1
                continue

            record = to_record(raw, source)
            if record is None:
                report.rejected["missing_text_field"] += 1
                continue

            text = record_text(record)
            reason = reject_reason(text, source.filters)
            if reason is not None:
                report.rejected[reason] += 1
                continue

            digest = hashlib.blake2b(
                normalize_text(text).encode("utf-8"), digest_size=16
            ).digest()
            if digest in seen_digests:
                report.exact_duplicates += 1
                continue
            seen_digests.add(digest)

            buffer.append((record, text, (report.seen, Counter(report.rejected), report.exact_duplicates)))
            buffered_chars += len(text)
            # Documents almost never have more tokens than characters, so the
            # budget cannot fill before the buffer holds as many characters as
            # tokens remain. Flushing from then on bounds over-reading to about
            # one document instead of a whole batch (256 books for a 7-book budget).
            remaining = report.target_tokens - report.tokens
            if (len(buffer) >= TOKENIZE_BATCH or buffered_chars >= remaining) and flush():
                return report
            if progress_every and report.documents and report.documents % progress_every == 0:
                print(
                    f"    {source.name}: {report.documents} docs, "
                    f"{report.tokens / 1e6:.1f}M/{report.target_tokens / 1e6:.1f}M tokens",
                    flush=True,
                )
        else:
            report.exhausted = True
        flush()
    finally:
        if handle is not None:
            handle.close()
    return report


def finalize(
    spec: MixtureSpec,
    source_paths: Sequence[str],
    out_dir: str,
    eval_index: Optional[EvalIndex],
    lcs_threshold: Optional[float] = None,
) -> dict:
    """Single streaming pass: decontaminate, assign splits, write the three files."""
    from .decontaminate import lcs_ratio, match_record

    handles = {
        name: open(os.path.join(out_dir, f"{name}.jsonl"), "w", encoding="utf-8")
        for name in (TRAIN, VAL, HOLDOUT)
    }
    counts = Counter()
    contaminated = 0
    rescued = 0
    hits: Counter = Counter()
    try:
        for path in source_paths:
            for record in read_jsonl(path):
                if eval_index is not None:
                    match = match_record(record.data, eval_index)
                    if match is not None:
                        part, hit = match
                        if lcs_threshold is not None and (
                            lcs_ratio(part, eval_index.items[hit], eval_index.n) < lcs_threshold
                        ):
                            rescued += 1
                        else:
                            contaminated += 1
                            hits[hit] += 1
                            continue
                name = assign_split(
                    record_text(record.data),
                    spec.val_fraction,
                    spec.holdout_fraction,
                    salt=str(spec.seed),
                )
                handles[name].write(json.dumps(record.data, ensure_ascii=False) + "\n")
                counts[name] += 1
    finally:
        for handle in handles.values():
            handle.close()
    # One eval item removing many documents is the signature of a false
    # positive (a generic phrase), not of a leak, which is why it is reported.
    top_matches = [
        {"eval_item": eval_index.items[item][:160], "documents": count}
        for item, count in hits.most_common(10)
    ]
    return {
        "counts": dict(counts),
        "contaminated_removed": contaminated,
        "lcs_rescued": rescued,
        "eval_items": len(eval_index) if eval_index is not None else 0,
        "top_matches": top_matches,
    }


@dataclass
class ProbeResult:
    name: str
    text_field: str
    rows: int = 0
    columns: list[str] = field(default_factory=list)
    missing_text: int = 0
    chars: int = 0
    rejected: Counter = field(default_factory=Counter)
    sample: str = ""
    error: Optional[str] = None

    @property
    def status(self) -> str:
        if self.error is not None:
            return "ERROR"
        if self.rows == 0:
            return "EMPTY"
        if self.missing_text:
            return "NO TEXT FIELD"
        return "ok"


def probe_source(source: SourceSpec, rows: int = 3) -> ProbeResult:
    """Pull the first few raw rows of a source and check the spec against them.

    Errors are captured, not raised: the point is to see every source's
    problem in one pass.
    """
    result = ProbeResult(name=source.name, text_field=source.text_field)
    try:
        for raw in iter_raw(source):
            result.rows += 1
            result.columns = sorted(raw)
            record = to_record(raw, source)
            if record is None:
                result.missing_text += 1
            else:
                text = record_text(record)
                result.chars += len(text)
                result.sample = result.sample or text
                reason = where_mismatch(raw, source.where) or reject_reason(text, source.filters)
                if reason is not None:
                    result.rejected[reason] += 1
            if result.rows >= rows:
                break
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def render_probe(results: Sequence[ProbeResult]) -> str:
    lines = []
    for result in results:
        lines.append(f"{result.name:<20} {result.status}")
        if result.error is not None:
            lines.append(f"    {result.error.strip().splitlines()[0][:300]}")
        if result.columns:
            lines.append(f"    columns: {', '.join(result.columns)}")
        if result.missing_text:
            lines.append(f"    text_field {result.text_field!r} missing in {result.missing_text}/{result.rows} rows")
        kept = result.rows - result.missing_text
        if kept:
            rejected = ", ".join(f"{k}={v}" for k, v in result.rejected.most_common()) or "none"
            lines.append(
                f"    {result.rows} rows, {result.chars // kept} chars/row; "
                f"filters would reject: {rejected}"
            )
        if result.sample:
            lines.append(f"    sample: {result.sample[:160]!r}")
    return "\n".join(lines)


def render_sources(reports: Sequence[SourceReport]) -> str:
    header = (
        f"{'source':<16} {'target':>12} {'tokens':>12} {'fill':>6} "
        f"{'docs':>9} {'read':>9} {'kept%':>6} {'tok/doc':>8}"
    )
    lines = [header, "-" * len(header)]
    for report in reports:
        lines.append(
            f"{report.name:<16} {report.target_tokens:>12} {report.tokens:>12} "
            f"{report.fill:>5.0%} {report.documents:>9} {report.seen:>9} "
            f"{report.keep_rate:>5.0%} {report.tokens / max(report.documents, 1):>8.0f}"
        )
        if report.exhausted and report.fill < 0.99:
            lines.append(
                f"  ! {report.name} ran out of data at {report.fill:.0%} of its budget"
            )
        for reason, count in report.rejected.most_common(4):
            lines.append(f"    -{count:>8} {reason}")
        if report.exact_duplicates:
            lines.append(f"    -{report.exact_duplicates:>8} exact_duplicate")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a corpus from a mixture spec")
    parser.add_argument("spec", help="Mixture spec JSON")
    parser.add_argument("--out_dir", type=str, default="datasets/prepared")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report what each source would contribute without writing the corpus",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply total_tokens, e.g. 0.01 for a quick pipeline check",
    )
    parser.add_argument("--progress_every", type=int, default=0, help="Docs between progress lines")
    parser.add_argument(
        "--probe",
        type=int,
        default=0,
        metavar="ROWS",
        help="Pull ROWS rows from every source, report spec problems, and exit",
    )
    args = parser.parse_args()

    spec = MixtureSpec.load(args.spec)
    if args.probe:
        results = [probe_source(source, args.probe) for source in spec.sources]
        print(render_probe(results))
        raise SystemExit(0 if all(r.status == "ok" for r in results) else 1)
    if args.scale != 1.0:
        spec.total_tokens = int(spec.total_tokens * args.scale)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.tokenizer)
    budget = (
        f"{spec.total_tokens / 1e9:.3f}B" if spec.total_tokens >= 1e8
        else f"{spec.total_tokens:,}"
    )
    print(f"mixture {spec.name!r}: {budget} tokens, tokenizer {spec.tokenizer}")

    os.makedirs(args.out_dir, exist_ok=True)
    source_dir = os.path.join(args.out_dir, "sources")
    reports, paths = [], []
    for source in spec.sources:
        out_path = None if args.dry_run else os.path.join(source_dir, f"{source.name}.jsonl")
        reports.append(prepare_source(spec, source, tokenizer, out_path, args.progress_every))
        if out_path:
            paths.append(out_path)

    print(render_sources(reports))

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    decon = spec.decontaminate or {}
    eval_index = None
    if decon.get("against"):
        eval_index = load_eval_index(
            decon["against"], decon.get("n", DEFAULT_N), decon.get("field", "auto")
        )
        print(f"\nindexed {len(eval_index)} eval items for decontamination")
    split_info = finalize(
        spec, paths, args.out_dir, eval_index, decon.get("lcs_threshold")
    )
    print(f"splits: {split_info['counts']}, contaminated removed: {split_info['contaminated_removed']}")
    for match in split_info["top_matches"][:5]:
        print(f"    -{match['documents']:>6} docs matched {match['eval_item'][:100]!r}")

    manifest = {
        "mixture": spec.name,
        "tokenizer": spec.tokenizer,
        "total_tokens_requested": spec.total_tokens,
        "total_tokens_collected": sum(r.tokens for r in reports),
        "seed": spec.seed,
        "val_fraction": spec.val_fraction,
        "holdout_fraction": spec.holdout_fraction,
        "sources": [r.to_dict() for r in reports],
        "split": split_info,
    }
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
