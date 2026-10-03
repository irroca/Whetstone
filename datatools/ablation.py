"""Data side of a mixture/tokenizer ablation: arm mixtures, the shared pool, per-arm corpora.

Arms of an ablation differ in one thing (the Chinese share, the vocabulary size)
and must agree on everything else, so they are all cut from one pool that
``datatools.prepare`` pulls once:

* An arm takes the **first** documents of each source in pool order. The 15%
  arm's Chinese pages are a prefix of the 30% arm's, so arms differ in how much
  of a source they see, not in which pages they happened to draw.
* Shares and the token budget are counted in the **arm's own tokenizer**, so
  arms train on the same number of tokens and a share is what the model sees.
  The pool is counted by prepare in the mixture's tokenizer (zh_6400), which
  compresses every source worse than the ablation vocabularies (1.2–1.6x on the
  smoke corpus), hence ``pool_margin``.
* Validation and holdout are the pool's splits, identical for every arm.

Pool documents come out of ``prepare``'s ``train.jsonl`` grouped by source, in
upstream order, so one sequential pass can cut every arm.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from typing import Iterator, Optional

from .records import read_jsonl

TOKENIZER_SAMPLE_NAME = "tokenizer_sample.jsonl"


@dataclass
class Arm:
    name: str
    vocab_size: int
    # Sources named here get exactly this share; the rest keep their mixture
    # proportions and fill whatever is left.
    shares: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict) -> "Arm":
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"arm {values.get('name')!r}: unknown keys {sorted(unknown)}")
        return cls(**values)


@dataclass
class AblationSpec:
    name: str
    mixture: str
    arms: list[Arm]
    tokens_per_arm: int
    data_dir: str
    results_dir: str
    pool_margin: float = 1.75
    tokenizer_sample_tokens: int = 100_000_000
    train: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    probes: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict) -> "AblationSpec":
        values = dict(values)
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown ablation keys: {sorted(unknown)}")
        values["arms"] = [Arm.from_dict(arm) for arm in values.get("arms", [])]
        spec = cls(**values)
        names = [arm.name for arm in spec.arms]
        if not names:
            raise ValueError("ablation needs at least one arm")
        if len(set(names)) != len(names):
            raise ValueError(f"arm names must be unique: {names}")
        if spec.pool_margin < 1.0:
            raise ValueError("pool_margin below 1 cannot cover even an identical tokenizer")
        return spec

    @classmethod
    def load(cls, path: str) -> "AblationSpec":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    @property
    def pool_dir(self) -> str:
        return os.path.join(self.data_dir, "pool")

    @property
    def pool_spec_path(self) -> str:
        return os.path.join(self.data_dir, "pool_spec.json")

    def tokenizer_dir(self, vocab_size: int) -> str:
        return os.path.join(self.data_dir, "tokenizers", vocab_name(vocab_size))

    def arm_corpus(self, arm: Arm) -> str:
        return os.path.join(self.data_dir, "arms", arm.name, "train")

    def split_corpus(self, split: str, vocab_size: int) -> str:
        return os.path.join(self.data_dir, f"{split}_{vocab_name(vocab_size)}")

    def vocab_sizes(self) -> list[int]:
        return sorted({arm.vocab_size for arm in self.arms})


def vocab_name(vocab_size: int) -> str:
    return f"v{vocab_size // 1024}k" if vocab_size % 1024 == 0 else f"v{vocab_size}"


def load_mixture(path: str) -> dict:
    """The mixture spec as JSON, validated by ``prepare.MixtureSpec``."""
    from .prepare import MixtureSpec

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    MixtureSpec.from_dict(raw)
    return raw


def base_weights(mixture: dict) -> dict[str, float]:
    return {source["name"]: float(source["weight"]) for source in mixture["sources"]}


def arm_shares(weights: dict[str, float], overrides: dict[str, float]) -> dict[str, float]:
    """Token shares of every source for one arm; they sum to 1."""
    unknown = sorted(set(overrides) - set(weights))
    if unknown:
        raise ValueError(f"shares name unknown sources {unknown}, known: {sorted(weights)}")
    if any(not 0.0 <= share <= 1.0 for share in overrides.values()):
        raise ValueError(f"shares must be within [0, 1], got {overrides}")
    fixed = sum(overrides.values())
    if fixed > 1.0 + 1e-9:
        raise ValueError(f"shares sum to {fixed:.4f} > 1")
    rest = {name: weight for name, weight in weights.items() if name not in overrides}
    rest_weight = sum(rest.values())
    left = max(0.0, 1.0 - fixed)
    if left > 1e-9 and rest_weight <= 0:
        raise ValueError(f"shares sum to {fixed:.4f} and no other source can fill the rest")
    shares = {name: float(overrides.get(name, 0.0)) for name in weights}
    for name, weight in rest.items():
        shares[name] = left * weight / rest_weight
    return shares


def arm_quotas(spec: AblationSpec, arm: Arm, weights: dict[str, float]) -> dict[str, int]:
    """Tokens of each source in the arm's corpus, in the arm's tokenizer, bos/eos included."""
    shares = arm_shares(weights, arm.shares)
    return {name: int(round(share * spec.tokens_per_arm)) for name, share in shares.items()}


def train_fraction(mixture: dict) -> float:
    from .prepare import MixtureSpec

    spec = MixtureSpec.from_dict(mixture)
    return 1.0 - spec.val_fraction - spec.holdout_fraction


def pool_targets(spec: AblationSpec, mixture: dict) -> dict[str, int]:
    """Tokens to pull per source, in the mixture tokenizer that prepare counts with.

    The largest quota any arm needs, times the margin for the arm tokenizers
    compressing better than the mixture tokenizer. The tokenizer sample is cut
    from the same documents, so it only matters if it is larger. Arms and the
    sample only read the train split, so the pull is scaled up by what
    validation and holdout take.
    """
    weights = base_weights(mixture)
    kept = train_fraction(mixture)
    targets = {}
    for name, weight in weights.items():
        need = max(arm_quotas(spec, arm, weights)[name] for arm in spec.arms)
        sample = weight * spec.tokenizer_sample_tokens
        targets[name] = int(round(max(need * spec.pool_margin, sample) / kept))
    return targets


def pool_mixture(spec: AblationSpec, mixture: dict) -> dict:
    """A copy of the mixture whose weights and total pull ``pool_targets``.

    Filters, cleaners, decontamination and split fractions are untouched, so the
    pool is exactly what the real mixture would produce, only more of some sources.
    Sources no arm uses are dropped (prepare requires positive weights).
    """
    targets = {name: tokens for name, tokens in pool_targets(spec, mixture).items() if tokens > 0}
    total = sum(targets.values())
    pool = copy.deepcopy(mixture)
    pool["name"] = f"{spec.name}-pool"
    pool["total_tokens"] = total
    pool["sources"] = [source for source in pool["sources"] if source["name"] in targets]
    for source in pool["sources"]:
        source["weight"] = targets[source["name"]] / total
    return pool


def write_pool_spec(spec: AblationSpec) -> dict:
    pool = pool_mixture(spec, load_mixture(spec.mixture))
    os.makedirs(spec.data_dir, exist_ok=True)
    with open(spec.pool_spec_path, "w", encoding="utf-8") as fh:
        json.dump(pool, fh, ensure_ascii=False, indent=2)
    return pool


def check_pool(spec: AblationSpec, pool: dict) -> dict:
    """The pool's manifest, after checking it was pulled for this pool spec."""
    path = os.path.join(spec.pool_dir, "manifest.json")
    with open(path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    wanted = {source["name"]: round(pool["total_tokens"] * source["weight"]) for source in pool["sources"]}
    got = {source["name"]: source["target_tokens"] for source in manifest["sources"]}
    if wanted != got:
        raise ValueError(
            f"{path} was pulled for targets {got}, but the ablation now needs {wanted}; "
            f"delete {spec.pool_dir} to pull it again"
        )
    return manifest


def iter_pool(path: str, open_quota: dict[str, int]) -> Iterator[tuple[str, str]]:
    """``(source, text)`` from a pool split, skipping sources whose quota is spent.

    ``open_quota`` is read on every record, so a caller that decrements it while
    consuming stops receiving a source as soon as it is full.
    """
    for record in read_jsonl(path):
        source = record.data.get("source")
        if open_quota.get(source, 0) > 0:
            yield source, str(record.data.get("text", ""))


def cut_corpus(
    pool_path: str,
    quotas: dict[str, int],
    tokenizer,
    out: str,
    tokenizer_path: str = "",
    count_special: bool = True,
) -> dict:
    """Write the first documents of each source until its quota of tokens is met.

    Counts what the corpus stores: ``bos + text + eos`` per document when
    ``count_special``. Raises if the pool runs out of any source, since a short
    source silently changes the mixture being measured.
    """
    from .tokenize_corpus import encode_records, write_token_corpus

    remaining = {name: quota for name, quota in quotas.items() if quota > 0}
    extra = 2 if count_special else 0

    def selected() -> Iterator[tuple[str, list[int]]]:
        for source, ids in encode_records(iter_pool(pool_path, remaining), tokenizer):
            if remaining.get(source, 0) <= 0:
                continue  # encoded in the batch that filled this source
            remaining[source] -= len(ids) + extra
            yield source, ids

    meta = write_token_corpus(
        selected(), tokenizer, out, tokenizer_path=tokenizer_path,
        inputs=[{"path": pool_path, "quotas": quotas}],
        check=lambda: _check_filled(remaining, quotas, pool_path),
    )
    return meta


def _check_filled(remaining: dict[str, int], quotas: dict[str, int], pool_path: str) -> None:
    short = {name: quotas[name] - left for name, left in remaining.items() if left > 0}
    if short:
        details = ", ".join(f"{name} {got:,}/{quotas[name]:,}" for name, got in short.items())
        raise ValueError(
            f"{pool_path} ran out of {sorted(short)} ({details} tokens); "
            f"raise pool_margin and pull the pool again"
        )


def write_tokenizer_sample(spec: AblationSpec, weights: dict[str, float], tokenizer, out_path: str) -> dict:
    """The base mixture's first documents, ``tokenizer_sample_tokens`` in all, as JSONL.

    Counted in the mixture tokenizer: the ablation vocabularies do not exist yet.
    """
    from .tokenize_corpus import encode_records

    quotas = {name: int(round(weight * spec.tokenizer_sample_tokens)) for name, weight in weights.items()}
    remaining = {name: quota for name, quota in quotas.items() if quota > 0}
    counts: dict[str, dict] = {}
    texts = iter_pool(os.path.join(spec.pool_dir, "train.jsonl"), remaining)
    tmp = f"{out_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for source, ids, text in encode_records(texts, tokenizer, with_text=True):
            if remaining.get(source, 0) <= 0:
                continue
            remaining[source] -= len(ids)
            fh.write(json.dumps({"text": text, "source": source}, ensure_ascii=False) + "\n")
            entry = counts.setdefault(source, {"docs": 0, "tokens": 0})
            entry["docs"] += 1
            entry["tokens"] += len(ids)
    _check_filled(remaining, quotas, os.path.join(spec.pool_dir, "train.jsonl"))
    os.replace(tmp, out_path)
    return counts


def describe_arms(spec: AblationSpec, weights: dict[str, float]) -> str:
    names = list(weights)
    header = f"{'arm':<14} {'vocab':>6} " + " ".join(f"{name[:9]:>9}" for name in names)
    lines = [header]
    for arm in spec.arms:
        shares = arm_shares(weights, arm.shares)
        lines.append(
            f"{arm.name:<14} {vocab_name(arm.vocab_size):>6} "
            + " ".join(f"{shares[name]:>9.1%}" for name in names)
        )
    return "\n".join(lines)


def describe_pool(spec: AblationSpec, mixture: dict) -> str:
    targets = pool_targets(spec, mixture)
    lines = [f"pool: {sum(targets.values()) / 1e6:,.0f}M tokens in the mixture tokenizer "
             f"(largest arm quota x {spec.pool_margin} margin / {train_fraction(mixture):.3f} train split)"]
    for name, tokens in targets.items():
        lines.append(f"    {name:<20} {tokens / 1e6:>9,.1f}M")
    return "\n".join(lines)


def required_paths(spec: AblationSpec) -> dict[str, Optional[str]]:
    """Every artifact the data stages produce, for status reporting."""
    paths = {"pool": os.path.join(spec.pool_dir, "manifest.json")}
    for vocab in spec.vocab_sizes():
        paths[f"tokenizer {vocab_name(vocab)}"] = os.path.join(spec.tokenizer_dir(vocab), "tokenizer.json")
        paths[f"val {vocab_name(vocab)}"] = f"{spec.split_corpus('val', vocab)}.bin"
    for arm in spec.arms:
        paths[f"corpus {arm.name}"] = f"{spec.arm_corpus(arm)}.bin"
    return paths
