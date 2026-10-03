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
"""

from __future__ import annotations

import math
import random
import re
from contextlib import nullcontext
from typing import Any, Optional, Sequence

import torch

from losses import token_logprobs

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
