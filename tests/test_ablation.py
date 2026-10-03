import json
import os

import numpy as np
import pytest
from transformers import AutoTokenizer

from datatools.ablation import (
    AblationSpec,
    arm_quotas,
    arm_shares,
    check_pool,
    cut_corpus,
    pool_mixture,
    pool_targets,
    write_tokenizer_sample,
)
from datatools.prepare import MixtureSpec
from datatools.tokenize_corpus import load_token_meta, token_paths

TOKENIZER = "./tokenizer/zh_6400"
WEIGHTS = {"zh_web": 0.3, "en_web": 0.28, "code": 0.2, "math": 0.14, "books": 0.05, "synthetic": 0.03}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def _spec(tmp_path, arms, **overrides):
    values = {
        "name": "t",
        "mixture": "configs/mixture_v1.json",
        "tokens_per_arm": 1_000_000,
        "data_dir": str(tmp_path / "data"),
        "results_dir": str(tmp_path / "results"),
        "arms": arms,
        **overrides,
    }
    return AblationSpec.from_dict(values)


def test_named_share_is_exact_and_the_rest_keep_their_proportions():
    shares = arm_shares(WEIGHTS, {"zh_web": 0.15})
    assert shares["zh_web"] == pytest.approx(0.15)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert shares["en_web"] / shares["code"] == pytest.approx(0.28 / 0.2)
    assert shares["en_web"] == pytest.approx(0.85 * 0.28 / 0.7)


def test_zero_share_drops_the_source():
    shares = arm_shares(WEIGHTS, {"zh_web": 0.0})
    assert shares["zh_web"] == 0.0
    assert shares["en_web"] == pytest.approx(0.4)


def test_no_override_is_the_base_mixture():
    assert arm_shares(WEIGHTS, {}) == pytest.approx(WEIGHTS)


@pytest.mark.parametrize(
    "overrides",
    [{"nope": 0.1}, {"zh_web": 1.2}, {"zh_web": -0.1}, {"zh_web": 0.7, "en_web": 0.5}],
)
def test_invalid_shares_raise(overrides):
    with pytest.raises(ValueError):
        arm_shares(WEIGHTS, overrides)


def test_shares_leaving_mass_with_nothing_to_fill_it_raise():
    with pytest.raises(ValueError):
        arm_shares({"a": 0.5, "b": 0.5}, {"a": 0.2, "b": 0.3})


def _mixture(val_fraction=0.0, holdout_fraction=0.0):
    sources = [{"name": name, "weight": weight, "jsonl": f"{name}.jsonl"} for name, weight in WEIGHTS.items()]
    return {"name": "m", "total_tokens": 1, "sources": sources,
            "val_fraction": val_fraction, "holdout_fraction": holdout_fraction}


def test_pool_pulls_the_largest_quota_any_arm_needs_times_the_margin(tmp_path):
    spec = _spec(tmp_path, [
        {"name": "zh0", "vocab_size": 1024, "shares": {"zh_web": 0.0}},
        {"name": "zh45", "vocab_size": 1024, "shares": {"zh_web": 0.45}},
    ], pool_margin=1.5, tokenizer_sample_tokens=0)
    targets = pool_targets(spec, _mixture())
    assert targets["zh_web"] == pytest.approx(0.45 * 1_000_000 * 1.5, abs=2)
    assert targets["en_web"] == pytest.approx(0.4 * 1_000_000 * 1.5, abs=2)


def test_pool_is_scaled_up_by_what_validation_and_holdout_take(tmp_path):
    spec = _spec(tmp_path, [{"name": "a", "vocab_size": 1024}], pool_margin=1.0, tokenizer_sample_tokens=0)
    targets = pool_targets(spec, _mixture(val_fraction=0.1, holdout_fraction=0.1))
    assert targets["zh_web"] == pytest.approx(0.3 * 1_000_000 / 0.8, abs=2)


def test_a_larger_tokenizer_sample_raises_the_pool(tmp_path):
    spec = _spec(tmp_path, [{"name": "a", "vocab_size": 1024}], pool_margin=1.0, tokenizer_sample_tokens=5_000_000)
    assert pool_targets(spec, _mixture())["books"] == pytest.approx(0.05 * 5_000_000, abs=2)


def test_pool_spec_is_the_mixture_with_only_the_amounts_changed(tmp_path):
    with open("configs/mixture_v1.json", encoding="utf-8") as fh:
        mixture = json.load(fh)
    spec = _spec(tmp_path, [
        {"name": "zh0", "vocab_size": 1024, "shares": {"zh_web": 0.0}},
        {"name": "zh45", "vocab_size": 1024, "shares": {"zh_web": 0.45}},
    ])
    pool = pool_mixture(spec, mixture)
    parsed = MixtureSpec.from_dict(pool)  # weights must still sum to 1
    targets = pool_targets(spec, mixture)
    for source in parsed.sources:
        assert parsed.token_budget(source) == pytest.approx(targets[source.name], abs=1)
    for original, pulled in zip(mixture["sources"], pool["sources"]):
        assert {k: v for k, v in pulled.items() if k != "weight"} == {k: v for k, v in original.items() if k != "weight"}
    assert pool["decontaminate"] == mixture["decontaminate"]
    assert pool["val_fraction"] == mixture["val_fraction"]


def test_a_pool_pulled_for_other_targets_is_refused(tmp_path):
    spec = _spec(tmp_path, [{"name": "a", "vocab_size": 1024}])
    with open("configs/mixture_v1.json", encoding="utf-8") as fh:
        pool = pool_mixture(spec, json.load(fh))
    os.makedirs(spec.pool_dir)
    manifest = {"sources": [{"name": s["name"], "target_tokens": 1} for s in pool["sources"]]}
    with open(os.path.join(spec.pool_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)
    with pytest.raises(ValueError, match="delete"):
        check_pool(spec, pool)


@pytest.mark.parametrize(
    "arms",
    [[], [{"name": "a", "vocab_size": 1024}, {"name": "a", "vocab_size": 2048}], [{"name": "a", "vocab_size": 1, "x": 1}]],
)
def test_invalid_ablation_specs_raise(tmp_path, arms):
    with pytest.raises(ValueError):
        _spec(tmp_path, arms)


def _pool(path, per_source=40):
    """A pool split grouped by source, like prepare's train.jsonl."""
    docs = {"a": [], "b": [], "c": []}
    with open(path, "w", encoding="utf-8") as fh:
        for source in docs:
            for i in range(per_source):
                text = f"{source} document {i}: " + " ".join(f"w{source}{i}x{j}" for j in range(20))
                docs[source].append(text)
                fh.write(json.dumps({"text": text, "source": source}) + "\n")
    return docs


def _stored(out, tokenizer):
    bin_path, idx_path, _ = token_paths(out)
    meta = load_token_meta(bin_path)
    tokens = np.fromfile(bin_path, dtype=meta["dtype"])
    offsets = np.fromfile(idx_path, dtype=np.uint64)
    return [tokenizer.decode(tokens[s + 1:e - 1].tolist()) for s, e in zip(offsets[:-1], offsets[1:])], meta


def test_cut_takes_each_sources_first_documents_until_its_quota(tmp_path, tokenizer):
    docs = _pool(tmp_path / "train.jsonl")
    quotas = {"a": 1500, "b": 600, "c": 0}
    out = str(tmp_path / "arm" / "train")
    cut_corpus(str(tmp_path / "train.jsonl"), quotas, tokenizer, out)
    stored, meta = _stored(out, tokenizer)
    longest = max(len(tokenizer(t, add_special_tokens=False)["input_ids"]) + 2 for d in docs.values() for t in d)
    for source in ("a", "b"):
        got = meta["sources"][source]["tokens"]
        assert quotas[source] <= got < quotas[source] + longest
    assert "c" not in meta["sources"]
    n_a = meta["sources"]["a"]["docs"]
    assert stored[:n_a] == docs["a"][:n_a]
    assert stored[n_a:] == docs["b"][:meta["sources"]["b"]["docs"]]


def test_a_pool_too_small_for_a_quota_raises_and_writes_nothing(tmp_path, tokenizer):
    _pool(tmp_path / "train.jsonl", per_source=3)
    out = str(tmp_path / "arm" / "train")
    with pytest.raises(ValueError, match="ran out of"):
        cut_corpus(str(tmp_path / "train.jsonl"), {"a": 100, "b": 10**6}, tokenizer, out)
    assert not any(os.path.exists(p) for p in token_paths(out))


def test_arm_quotas_add_up_to_the_budget(tmp_path):
    spec = _spec(tmp_path, [{"name": "a", "vocab_size": 1024, "shares": {"zh_web": 0.15}}])
    assert sum(arm_quotas(spec, spec.arms[0], WEIGHTS).values()) == pytest.approx(1_000_000, abs=len(WEIGHTS))


def test_tokenizer_sample_follows_the_base_mixture(tmp_path, tokenizer):
    spec = _spec(tmp_path, [{"name": "a", "vocab_size": 1024}], tokenizer_sample_tokens=2000)
    os.makedirs(spec.pool_dir)
    docs = _pool(os.path.join(spec.pool_dir, "train.jsonl"))
    out = str(tmp_path / "sample.jsonl")
    counts = write_tokenizer_sample(spec, {"a": 0.75, "b": 0.25, "c": 0.0}, tokenizer, out)
    assert counts["a"]["tokens"] >= 1500 and counts["b"]["tokens"] >= 500 and "c" not in counts
    with open(out, encoding="utf-8") as fh:
        texts = [json.loads(line)["text"] for line in fh]
    assert texts == docs["a"][:counts["a"]["docs"]] + docs["b"][:counts["b"]["docs"]]
