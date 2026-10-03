"""Pre-tokenize a prepared corpus into a flat binary that pretraining memory-maps.

Reading JSONL in the training loop costs twice: the whole file is parsed into a
Python list up front (a 30GB corpus does not fit), and every sample is
re-tokenized every epoch. This does both once, ahead of time, and writes:

* ``<out>.bin`` — every document's tokens back to back, each wrapped as
  ``bos + text + eos``: the wrapping ``dataset.PretrainDataset`` and
  ``eval_ppl.py`` apply, so switching formats changes the storage, not the
  input distribution.
* ``<out>.idx`` — ``uint64`` offset of each document's first token, plus one
  past the end, so document boundaries survive packing.
* ``<out>.meta.json`` — dtype, counts, tokens per source, and a fingerprint of
  the tokenizer. ``dataset.MemmapPretrainDataset`` refuses a file written by a
  different tokenizer: token ids mean nothing under another vocabulary, and no
  shape check would notice.

``uint16`` holds any vocabulary up to 65536 entries, so the planned 32k vocab
stores 10B tokens in 20GB. Files are written under a temporary name and renamed
at the end, so a killed run cannot leave a truncated corpus that looks finished.

::

    python3 -m datatools.tokenize_corpus datasets/prepared/train.jsonl \\
        --tokenizer ./tokenizer/zh_6400 --out datasets/prepared/train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from typing import Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from .records import PRETRAIN, ReadStats, detect_schema, read_jsonl

FORMAT = "whetstone-tokens/1"
DOC_TEMPLATE = "bos + text + eos"
ENCODE_BATCH = 512


def token_paths(path: str) -> tuple[str, str, str]:
    """``(bin, idx, meta)`` paths for an output prefix or an existing ``.bin``."""
    prefix = path[: -len(".bin")] if path.endswith(".bin") else path
    return f"{prefix}.bin", f"{prefix}.idx", f"{prefix}.meta.json"


def token_dtype(max_token_id: int) -> np.dtype:
    if max_token_id < 2**16:
        return np.dtype(np.uint16)
    if max_token_id < 2**32:
        return np.dtype(np.uint32)
    raise ValueError(f"token id {max_token_id} does not fit in uint32")


def tokenizer_fingerprint(tokenizer) -> str:
    """Identifies the id assignment, which is what the stored tokens depend on.

    Hashes the vocabulary in id order plus the bos/eos ids rather than the
    tokenizer files, whose serialization changes between library versions.
    """
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    payload = json.dumps(
        {"vocab": vocab, "bos": tokenizer.bos_token_id, "eos": tokenizer.eos_token_id},
        ensure_ascii=False,
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def encode_records(
    records: Iterable[tuple[str, str]],
    tokenizer,
    batch_size: int = ENCODE_BATCH,
    with_text: bool = False,
) -> Iterator[tuple]:
    """Batch-encode ``(source, text)`` pairs into ``(source, ids)``, in order.

    ``with_text`` yields ``(source, ids, text)``. Records are pulled a batch
    ahead of what has been yielded, which a consumer filtering on a running
    count has to allow for.
    """
    def encode(batch: list[tuple[str, str]]) -> Iterator[tuple]:
        # Whole documents exceed model_max_length by design; windows are cut later.
        encoded = tokenizer([text for _, text in batch], add_special_tokens=False, verbose=False)["input_ids"]
        for (source, text), ids in zip(batch, encoded):
            yield (source, ids, text) if with_text else (source, ids)

    batch: list[tuple[str, str]] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            yield from encode(batch)
            batch = []
    if batch:
        yield from encode(batch)


def write_token_corpus(
    encoded: Iterable[tuple[str, Sequence[int]]],
    tokenizer,
    out: str,
    tokenizer_path: str = "",
    inputs: Optional[list] = None,
    skipped: Optional[Counter] = None,
    progress_every: int = 0,
    check: Optional[Callable[[], None]] = None,
) -> dict:
    """Store ``(source, ids)`` documents as ``bos + ids + eos`` in ``<out>.bin`` and return the meta.

    ``inputs`` and ``skipped`` may still be filled while ``encoded`` is consumed;
    they are read once it is exhausted. ``check`` runs before anything is
    renamed into place, so raising from it leaves no corpus behind.
    """
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    if bos is None or eos is None:
        raise ValueError("tokenizer needs both a bos and an eos token")
    dtype = token_dtype(max(tokenizer.get_vocab().values()))
    bin_path, idx_path, meta_path = token_paths(out)
    os.makedirs(os.path.dirname(bin_path) or ".", exist_ok=True)
    inputs = inputs if inputs is not None else []
    skipped = skipped if skipped is not None else Counter()

    num_tokens = num_docs = 0
    per_source: dict[str, Counter] = {}
    started = time.time()
    next_progress = progress_every

    def write(batch: list[tuple[str, Sequence[int]]], bin_fh, idx_fh) -> None:
        nonlocal num_tokens, num_docs
        lengths = np.fromiter((len(ids) + 2 for _, ids in batch), dtype=np.int64, count=len(batch))
        flat = np.empty(int(lengths.sum()), dtype=dtype)
        starts = np.concatenate(([0], np.cumsum(lengths)[:-1]))
        for (source, ids), start, length in zip(batch, starts, lengths):
            flat[start] = bos
            flat[start + 1 : start + length - 1] = ids
            flat[start + length - 1] = eos
            counts = per_source.setdefault(source, Counter())
            counts["docs"] += 1
            counts["tokens"] += int(length)
        idx_fh.write((starts + num_tokens).astype(np.uint64).tobytes())
        bin_fh.write(flat.tobytes())
        num_tokens += int(lengths.sum())
        num_docs += len(batch)

    tmp_bin, tmp_idx, tmp_meta = (f"{p}.tmp" for p in (bin_path, idx_path, meta_path))
    try:
        with open(tmp_bin, "wb") as bin_fh, open(tmp_idx, "wb") as idx_fh:
            batch: list[tuple[str, Sequence[int]]] = []
            for document in encoded:
                batch.append(document)
                if len(batch) >= ENCODE_BATCH:
                    write(batch, bin_fh, idx_fh)
                    batch = []
                    if progress_every and num_docs >= next_progress:
                        rate = num_tokens / max(time.time() - started, 1e-9)
                        print(
                            f"    {num_docs} docs, {num_tokens / 1e6:.1f}M tokens "
                            f"({rate / 1e6:.2f}M tok/s)",
                            flush=True,
                        )
                        next_progress = (num_docs // progress_every + 1) * progress_every
            if batch:
                write(batch, bin_fh, idx_fh)
            idx_fh.write(np.array([num_tokens], dtype=np.uint64).tobytes())
        if check is not None:
            check()

        meta = {
            "format": FORMAT,
            "dtype": dtype.name,
            "num_tokens": num_tokens,
            "num_docs": num_docs,
            "doc_template": DOC_TEMPLATE,
            "tokenizer": {
                "path": tokenizer_path,
                "vocab_size": tokenizer.vocab_size,
                "bos_token_id": bos,
                "eos_token_id": eos,
                "fingerprint": tokenizer_fingerprint(tokenizer),
            },
            "sources": {name: dict(counts) for name, counts in sorted(per_source.items())},
            "skipped": {k: v for k, v in skipped.items() if v},
            "inputs": inputs,
            "seconds": round(time.time() - started, 2),
        }
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_bin, bin_path)
        os.replace(tmp_idx, idx_path)
        os.replace(tmp_meta, meta_path)
    finally:
        for tmp in (tmp_bin, tmp_idx, tmp_meta):
            if os.path.exists(tmp):
                os.remove(tmp)
    return meta


def tokenize_corpus(
    paths: Sequence[str],
    tokenizer,
    out: str,
    tokenizer_path: str = "",
    max_records: Optional[int] = None,
    batch_size: int = ENCODE_BATCH,
    progress_every: int = 0,
) -> dict:
    """Encode every pretrain record in ``paths`` into ``<out>.bin`` and return the meta."""
    skipped: Counter = Counter()
    inputs: list = []

    def records() -> Iterator[tuple[str, str]]:
        for path in paths:
            stats = ReadStats()
            default_source = os.path.splitext(os.path.basename(path))[0]
            for record in read_jsonl(path, stats, max_records=max_records):
                if detect_schema(record.data) != PRETRAIN:
                    skipped["not_pretrain"] += 1
                    continue
                text = str(record.data.get("text", ""))
                if not text:
                    skipped["empty_text"] += 1
                    continue
                yield str(record.data.get("source") or default_source), text
            skipped["malformed"] += stats.malformed
            inputs.append({
                "path": path,
                "bytes": os.path.getsize(path),
                "records_read": stats.parsed,
            })

    return write_token_corpus(
        encode_records(records(), tokenizer, batch_size), tokenizer, out,
        tokenizer_path=tokenizer_path, inputs=inputs, skipped=skipped, progress_every=progress_every,
    )


def load_token_meta(path: str) -> dict:
    """Read and sanity-check the meta of a tokenized corpus.

    The size check catches a partially copied ``.bin`` (an interrupted upload to
    a rented GPU box), which would otherwise train on whatever made it across.
    """
    bin_path, _, meta_path = token_paths(path)
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found; produce {bin_path} with datatools.tokenize_corpus"
        )
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("format") != FORMAT:
        raise ValueError(f"{meta_path}: unsupported format {meta.get('format')!r}, expected {FORMAT!r}")
    expected = meta["num_tokens"] * np.dtype(meta["dtype"]).itemsize
    actual = os.path.getsize(bin_path)
    if actual != expected:
        raise ValueError(
            f"{bin_path} is {actual} bytes but its meta describes {expected}: "
            f"the file is truncated or belongs to a different run"
        )
    return meta


def check_tokenizer(meta: dict, tokenizer, path: str = "") -> None:
    recorded = meta["tokenizer"]
    if recorded["fingerprint"] != tokenizer_fingerprint(tokenizer):
        raise ValueError(
            f"{path or 'corpus'} was tokenized with {recorded['path'] or 'another tokenizer'} "
            f"(vocab {recorded['vocab_size']}), not the tokenizer given (vocab "
            f"{tokenizer.vocab_size}); re-run datatools.tokenize_corpus with it"
        )


def render(meta: dict, bin_path: str) -> str:
    lines = [
        f"{bin_path}: {meta['num_tokens']:,} tokens in {meta['num_docs']:,} docs "
        f"({meta['dtype']}, {meta['seconds']}s)",
    ]
    total = max(meta["num_tokens"], 1)
    for name, counts in meta["sources"].items():
        lines.append(
            f"    {name:<20} {counts['tokens']:>14,} tokens {counts['tokens'] / total:>6.1%} "
            f"{counts['docs']:>10,} docs"
        )
    for reason, count in meta["skipped"].items():
        lines.append(f"    skipped {count} records: {reason}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-tokenize JSONL into a memory-mappable .bin")
    parser.add_argument("paths", nargs="+", help="Prepared JSONL files, concatenated in order")
    parser.add_argument("--out", required=True, help="Output prefix, e.g. datasets/prepared/train")
    parser.add_argument("--tokenizer", default="./tokenizer/zh_6400")
    parser.add_argument("--max_records", type=int, default=None, help="Per input file")
    parser.add_argument("--progress_every", type=int, default=0, help="Docs between progress lines")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    meta = tokenize_corpus(
        args.paths, tokenizer, args.out,
        tokenizer_path=args.tokenizer,
        max_records=args.max_records,
        progress_every=args.progress_every,
    )
    print(render(meta, token_paths(args.out)[0]))


if __name__ == "__main__":
    main()
