"""Probes for a pretrained base model, run after training.

They answer what a mixture or vocabulary ablation needs and loss cannot:

* ``bits_per_byte`` per source. Loss per token is not comparable across
  vocabularies (a tokenizer that spends fewer tokens on a page has more to
  predict per token), and a mixture average hides which language paid for a
  change. Dividing the summed NLL by the text's UTF-8 size fixes both.
* ``language_confusion``: given the start of a Chinese page, how often the
  model continues in another language, and the reverse. A bilingual small model
  that drifts into English on Chinese prompts fails in a way bpb does not show.
* ``arithmetic_probe``: few-shot addition, the verifiable task the project is
  about, scored by exact match on the greedy continuation.

As a CLI it runs all three on any checkpoint against a ``prepare`` holdout:

    python3 probes.py --checkpoint results/pretrain_final.pth \\
        --holdout datasets/mixture_v2/holdout.jsonl --tokenizer_path tokenizer/v1_32k \\
        --max_seq_len 2048 --out results/probes_pretrain.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from typing import Any, Optional, Sequence

import torch

from losses import token_logprobs

PROBE_DEFAULTS = {"max_tokens_per_source": 500_000, "confusion_prompts": 100, "arithmetic_items": 200}

CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN = re.compile(r"[A-Za-z]")
ANSWER = re.compile(r"\s*(-?\d+)")


def document_windows(ids: Sequence[int], max_seq_len: int) -> list[list[int]]:
    """Windows of ``max_seq_len`` tokens at stride ``max_seq_len - 1``.

    Each token after the first is a target exactly once, the same windowing
    ``MemmapPretrainDataset`` trains on.
    """
    step = max_seq_len - 1
    return [list(ids[start:start + max_seq_len]) for start in range(0, max(len(ids) - 1, 0), step)]


@torch.no_grad()
def bits_per_byte(
    model,
    tokenizer,
    texts: Sequence[str],
    max_seq_len: int,
    device: str,
    batch_size: int = 16,
    ctx: Any = None,
) -> dict:
    """Summed NLL of ``bos + text + eos`` over every document, in bits per UTF-8 byte."""
    ctx = ctx if ctx is not None else nullcontext()
    bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    encoded = tokenizer(list(texts), add_special_tokens=False, verbose=False)["input_ids"]
    windows = [w for ids in encoded for w in document_windows([bos, *ids, eos], max_seq_len)]
    was_training = model.training
    model.eval()
    nll, targets = 0.0, 0
    try:
        for start in range(0, len(windows), batch_size):
            batch = windows[start:start + batch_size]
            width = max(len(w) for w in batch) - 1
            X = torch.full((len(batch), width), pad, dtype=torch.long)
            Y = torch.full((len(batch), width), pad, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.bool)
            for row, window in enumerate(batch):
                n = len(window) - 1
                X[row, :n] = torch.tensor(window[:-1])
                Y[row, :n] = torch.tensor(window[1:])
                mask[row, :n] = True
            X, Y, mask = X.to(device), Y.to(device), mask.to(device)
            with ctx:
                logits = model(X).logits
            logprobs = token_logprobs(logits.float(), Y)
            nll -= float(logprobs[mask].sum())
            targets += int(mask.sum())
    finally:
        if was_training:
            model.train()
    num_bytes = sum(len(text.encode("utf-8")) for text in texts)
    return {
        "bpb": nll / math.log(2) / max(num_bytes, 1),
        "bits_per_token": nll / math.log(2) / max(targets, 1),
        "tokens": targets,
        "bytes": num_bytes,
        "docs": len(texts),
    }


def cjk_share(text: str) -> Optional[float]:
    """CJK characters over CJK plus Latin letters; ``None`` when there are neither."""
    cjk, latin = len(CJK.findall(text)), len(LATIN.findall(text))
    return cjk / (cjk + latin) if cjk + latin else None


def _prompts(tokenizer, texts: Sequence[str], prompt_tokens: int, limit: int) -> list[list[int]]:
    prompts = []
    for ids in tokenizer(list(texts), add_special_tokens=False, verbose=False)["input_ids"]:
        if len(ids) >= prompt_tokens:
            prompts.append([tokenizer.bos_token_id, *ids[:prompt_tokens]])
        if len(prompts) >= limit:
            break
    return prompts


@torch.no_grad()
def _continue(model, tokenizer, prompts: list[list[int]], new_tokens: int, device: str, **sampling) -> list[str]:
    """Continuations of equal-length prompts, each cut at its own first EOS."""
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    out = model.generate(
        torch.tensor(prompts, device=device), eos_token_id=eos, max_new_tokens=new_tokens,
        pad_token_id=pad, **sampling,
    )
    texts = []
    for row in out.tolist():
        if eos in row:
            row = row[:row.index(eos)]
        texts.append(tokenizer.decode(row, skip_special_tokens=True))
    return texts


def language_confusion(
    model,
    tokenizer,
    zh_texts: Sequence[str],
    en_texts: Sequence[str],
    device: str,
    prompts: int = 100,
    prompt_tokens: int = 32,
    new_tokens: int = 48,
    seed: int = 0,
    batch_size: int = 50,
) -> dict:
    """Share of continuations that leave the prompt's language.

    A Chinese prompt counts as confused when its continuation is less than half
    CJK (letters only, so digits and punctuation do not vote); an English prompt
    when it is more than half. Continuations with no letters are not scored.
    """
    model.eval()
    result: dict = {}
    for label, texts, confused in (
        ("zh", zh_texts, lambda share: share < 0.5),
        ("en", en_texts, lambda share: share > 0.5),
    ):
        batch_prompts = _prompts(tokenizer, texts, prompt_tokens, prompts)
        torch.manual_seed(seed)
        continuations: list[str] = []
        for start in range(0, len(batch_prompts), batch_size):
            continuations += _continue(
                model, tokenizer, batch_prompts[start:start + batch_size], new_tokens, device,
                temperature=0.7, top_p=0.9,
            )
        shares = [cjk_share(text) for text in continuations]
        scored = [share for share in shares if share is not None]
        result[label] = {
            "prompts": len(batch_prompts),
            "scored": len(scored),
            "confused": sum(confused(share) for share in scored) / max(len(scored), 1),
            "examples": continuations[:3],
        }
    return result


def addition_items(n: int, digits: int, shots: int, seed: int) -> list[tuple[str, int]]:
    """Few-shot prompts ``"a + b = c\\n" * shots + "a + b ="`` with their answers."""
    rng = random.Random(f"{seed}-{digits}")
    low, high = (0, 9) if digits == 1 else (10 ** (digits - 1), 10 ** digits - 1)
    items = []
    for _ in range(n):
        pairs = [(rng.randint(low, high), rng.randint(low, high)) for _ in range(shots + 1)]
        lines = [f"{a} + {b} = {a + b}" for a, b in pairs[:-1]]
        a, b = pairs[-1]
        items.append(("\n".join(lines + [f"{a} + {b} ="]), a + b))
    return items


def parse_answer(text: str) -> Optional[int]:
    match = ANSWER.match(text)
    return int(match.group(1)) if match else None


@torch.no_grad()
def arithmetic_probe(
    model,
    tokenizer,
    device: str,
    items: int = 200,
    digits: Sequence[int] = (1, 2),
    shots: int = 4,
    seed: int = 0,
) -> dict:
    """Exact-match accuracy of greedy few-shot addition, per operand length."""
    model.eval()
    result = {}
    per_length = max(1, items // len(digits))
    for length in digits:
        correct = 0
        for prompt, answer in addition_items(per_length, length, shots, seed):
            ids = [tokenizer.bos_token_id, *tokenizer(prompt, add_special_tokens=False)["input_ids"]]
            text = _continue(model, tokenizer, [ids], length + 3, device, temperature=0)[0]
            correct += parse_answer(text.split("\n")[0]) == answer
        result[f"add_{length}digit"] = correct / per_length
    result["items"] = per_length * len(digits)
    return result


def holdout_by_source(path: str) -> dict[str, list[str]]:
    """Pretraining texts of a ``prepare`` split, grouped by their ``source`` tag."""
    from datatools.records import read_jsonl

    texts: dict[str, list[str]] = {}
    for record in read_jsonl(path):
        texts.setdefault(record.data.get("source", "all"), []).append(str(record.data["text"]))
    return texts


def cap_tokens(tokenizer, texts: Sequence[str], max_tokens: int) -> list[str]:
    """Documents in order until ``max_tokens`` of them are covered."""
    kept, total = [], 0
    for text, ids in zip(texts, tokenizer(list(texts), add_special_tokens=False, verbose=False)["input_ids"]):
        if total >= max_tokens:
            break
        kept.append(text)
        total += len(ids) + 2
    return kept


def load_model(path: str, tokenizer, max_seq_len: int, device: str):
    """A checkpoint in eval mode, its architecture read from the checkpoint itself."""
    from model import Whetstone
    from train_utils import load_weights, resolve_model_config

    config = resolve_model_config(argparse.Namespace(), tokenizer.vocab_size, checkpoint_path=path, max_seq_len=max_seq_len)
    model = Whetstone(config).to(device)
    load_weights(path, model, device, strict=False)
    return model.eval()


def run_probes(model, tokenizer, holdout: dict[str, list[str]], max_seq_len: int, device: str,
               ctx: Any = None, **options) -> dict:
    """All three probes; ``holdout`` maps a source tag to its texts, as ``holdout_by_source`` reads them."""
    options = {**PROBE_DEFAULTS, **options}
    return {
        "bpb": {
            source: bits_per_byte(
                model, tokenizer, cap_tokens(tokenizer, texts, options["max_tokens_per_source"]),
                max_seq_len, device, ctx=ctx,
            )
            for source, texts in sorted(holdout.items())
        },
        "language_confusion": language_confusion(
            model, tokenizer, holdout.get("zh_web", []), holdout.get("en_web", []), device,
            prompts=options["confusion_prompts"],
        ),
        "arithmetic": arithmetic_probe(model, tokenizer, device, items=options["arithmetic_items"]),
    }


def main(argv=None) -> int:
    from transformers import AutoTokenizer

    from train_utils import build_autocast_scaler, resolve_device

    parser = argparse.ArgumentParser(description="bits per byte, language confusion and addition for one checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--holdout", required=True, help="a prepare split (holdout.jsonl), texts tagged by source")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--max_seq_len", type=int, default=2048, help="window length for bits per byte")
    parser.add_argument("--device", default=resolve_device())
    parser.add_argument(
        "--dtype", default="float32",
        help="autocast dtype for bits per byte; float32 keeps numbers comparable across runs",
    )
    for key, value in PROBE_DEFAULTS.items():
        parser.add_argument(f"--{key}", type=int, default=value)
    parser.add_argument("--out", default="", help="also write the results as JSON here")
    args = parser.parse_args(argv)

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args.checkpoint, tokenizer, args.max_seq_len, args.device)
    ctx, _ = build_autocast_scaler(args.device, args.dtype)
    result = run_probes(
        model, tokenizer, holdout_by_source(args.holdout), args.max_seq_len, args.device, ctx=ctx,
        **{key: getattr(args, key) for key in PROBE_DEFAULTS},
    )
    result = {
        "checkpoint": args.checkpoint, "holdout": args.holdout, "tokenizer": args.tokenizer_path,
        "max_seq_len": args.max_seq_len, "dtype": args.dtype, **result,
        "seconds": round(time.time() - started, 1),
    }

    for source, row in result["bpb"].items():
        print(f"bpb {source:>20}: {row['bpb']:.4f}  ({row['tokens']:,} tokens, {row['docs']} docs)")
    confusion = result["language_confusion"]
    print(f"language confusion: zh->other {confusion['zh']['confused']:.1%}, en->zh {confusion['en']['confused']:.1%}")
    print("arithmetic: " + ", ".join(f"{k} {v:.1%}" for k, v in result["arithmetic"].items() if k != "items"))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
