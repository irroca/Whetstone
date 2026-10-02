import json
import os
import pickle

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from dataset import MemmapPretrainDataset, PretrainDataset, build_pretrain_dataset
from datatools.records import read_jsonl, write_jsonl
from datatools.tokenize_corpus import (
    check_tokenizer,
    load_token_meta,
    token_dtype,
    token_paths,
    tokenize_corpus,
    tokenizer_fingerprint,
)

FIXTURE = "tests/fixtures/pretrain_tiny.jsonl"
TOKENIZER = "./tokenizer/zh_6400"


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


@pytest.fixture
def corpus(tmp_path, tokenizer):
    meta = tokenize_corpus([FIXTURE], tokenizer, str(tmp_path / "train"), tokenizer_path=TOKENIZER)
    return str(tmp_path / "train.bin"), meta


def _documents(bin_path, meta):
    bin_path, idx_path, _ = token_paths(bin_path)
    tokens = np.fromfile(bin_path, dtype=meta["dtype"])
    offsets = np.fromfile(idx_path, dtype=np.uint64)
    return [tokens[a:b].tolist() for a, b in zip(offsets[:-1], offsets[1:])]


def test_stored_documents_match_what_the_jsonl_reader_feeds_the_model(corpus, tokenizer):
    """Switching to the binary format must change the storage, not the inputs."""
    bin_path, meta = corpus
    texts = [r.data["text"] for r in read_jsonl(FIXTURE)]
    documents = _documents(bin_path, meta)

    assert len(documents) == len(texts) == meta["num_docs"]
    for document, text in zip(documents, texts):
        assert document == tokenizer(f"{tokenizer.bos_token}{text}{tokenizer.eos_token}").input_ids
    assert sum(map(len, documents)) == meta["num_tokens"]


def test_windows_make_every_token_a_target_exactly_once(corpus, tokenizer):
    bin_path, meta = corpus
    ds = MemmapPretrainDataset(bin_path, tokenizer, max_length=16)
    tokens = np.fromfile(bin_path, dtype=meta["dtype"]).astype(np.int64).tolist()

    assert len(ds) == (meta["num_tokens"] - 1) // 15
    inputs = torch.cat([ds[i][0] for i in range(len(ds))]).tolist()
    targets = torch.cat([ds[i][1] for i in range(len(ds))]).tolist()
    assert inputs == tokens[: len(ds) * 15]
    assert targets == tokens[1 : 1 + len(ds) * 15]
    assert torch.equal(ds[-1][0], ds[len(ds) - 1][0])
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_window_tensors_have_the_jsonl_readers_shapes_and_dtypes(corpus, tokenizer):
    bin_path, _ = corpus
    jsonl = PretrainDataset(FIXTURE, tokenizer, max_length=32)
    packed = MemmapPretrainDataset(bin_path, tokenizer, max_length=32)

    X, Y, mask = packed[0]
    for ours, theirs in zip(packed[0], jsonl[0]):
        assert ours.shape == theirs.shape == (31,)
        assert ours.dtype == theirs.dtype == torch.long
    assert torch.equal(X[1:], Y[:-1])
    assert bool(mask.all()), "packed windows have no padding to mask"


def test_factory_picks_the_reader_by_extension(corpus, tokenizer):
    bin_path, _ = corpus
    assert isinstance(build_pretrain_dataset(bin_path, tokenizer, 16), MemmapPretrainDataset)
    assert isinstance(build_pretrain_dataset(FIXTURE, tokenizer, 16), PretrainDataset)


def test_meta_counts_tokens_per_source(tmp_path, tokenizer):
    path = tmp_path / "mixed.jsonl"
    write_jsonl(str(path), [
        {"text": "中文网页文本。", "source": "zh_web"},
        {"text": "English web text.", "source": "en_web"},
        {"text": "更多的中文。", "source": "zh_web"},
        {"conversations": [{"role": "user", "content": "hi"}]},
        {"text": ""},
    ])

    meta = tokenize_corpus([str(path)], tokenizer, str(tmp_path / "mixed"))

    assert meta["sources"]["zh_web"]["docs"] == 2
    assert meta["sources"]["en_web"]["docs"] == 1
    assert sum(s["tokens"] for s in meta["sources"].values()) == meta["num_tokens"]
    assert meta["skipped"] == {"not_pretrain": 1, "empty_text": 1}


def test_records_without_a_source_are_attributed_to_their_file(corpus):
    _, meta = corpus
    assert list(meta["sources"]) == ["pretrain_tiny"]


def test_fingerprint_is_stable_across_loads(tokenizer):
    assert tokenizer_fingerprint(tokenizer) == tokenizer_fingerprint(AutoTokenizer.from_pretrained(TOKENIZER))


def test_a_corpus_from_another_tokenizer_is_refused(corpus, tokenizer):
    bin_path, meta = corpus
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "磨刀石": 3}
    other = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>", bos_token="<s>", eos_token="</s>",
    )

    check_tokenizer(meta, tokenizer)
    with pytest.raises(ValueError, match="re-run datatools.tokenize_corpus"):
        check_tokenizer(meta, other)
    with pytest.raises(ValueError, match="re-run datatools.tokenize_corpus"):
        MemmapPretrainDataset(bin_path, other, max_length=16)


def test_a_truncated_copy_is_refused(corpus, tokenizer):
    """An interrupted upload must not train on whatever made it across."""
    bin_path, _ = corpus
    with open(bin_path, "r+b") as fh:
        fh.truncate(os.path.getsize(bin_path) - 2)
    with pytest.raises(ValueError, match="truncated"):
        load_token_meta(bin_path)


def test_a_missing_meta_says_how_to_build_one(tmp_path, tokenizer):
    path = tmp_path / "orphan.bin"
    path.write_bytes(b"\0\0")
    with pytest.raises(FileNotFoundError, match="tokenize_corpus"):
        MemmapPretrainDataset(str(path), tokenizer)


def test_a_corpus_shorter_than_one_window_is_refused(corpus, tokenizer):
    bin_path, meta = corpus
    assert len(MemmapPretrainDataset(bin_path, tokenizer, max_length=meta["num_tokens"])) == 1
    with pytest.raises(ValueError, match="fewer than one window"):
        MemmapPretrainDataset(bin_path, tokenizer, max_length=meta["num_tokens"] + 1)


def test_pickling_for_dataloader_workers_leaves_the_memmap_behind(corpus, tokenizer):
    bin_path, _ = corpus
    ds = MemmapPretrainDataset(bin_path, tokenizer, max_length=16)
    ds[0]

    clone = pickle.loads(pickle.dumps(ds))

    assert clone._tokens is None
    assert torch.equal(clone[1][0], ds[1][0])


def test_a_failed_run_leaves_nothing_that_looks_finished(tmp_path, tokenizer):
    with pytest.raises(FileNotFoundError):
        tokenize_corpus([FIXTURE, str(tmp_path / "missing.jsonl")], tokenizer, str(tmp_path / "train"))
    assert list(tmp_path.iterdir()) == []


def test_meta_is_json_with_the_tokenizer_recorded(corpus, tokenizer):
    bin_path, meta = corpus
    with open(token_paths(bin_path)[2], encoding="utf-8") as fh:
        assert json.load(fh) == meta
    assert meta["dtype"] == "uint16"
    assert meta["tokenizer"]["path"] == TOKENIZER
    assert meta["tokenizer"]["vocab_size"] == tokenizer.vocab_size


def test_dtype_widens_past_uint16():
    assert token_dtype(65535) == np.uint16
    assert token_dtype(65536) == np.uint32
