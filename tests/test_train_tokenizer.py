import json
import random

import pytest
from transformers import AutoTokenizer

from train_tokenizer import SPECIAL_TOKENS, train_tokenizer


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    """Trained on text full of repeated numbers, which BPE would happily merge."""
    tmp = tmp_path_factory.mktemp("tok")
    rng = random.Random(0)
    lines = []
    for _ in range(300):
        a, b = rng.randint(0, 99), rng.choice([2024, 2025, 100, 1000])
        lines.append(f"In {b} there were {a} + {a} = {2 * a} items. 今年是{b}年，共{a}个。def f(x): return x * {a}")
    path = tmp / "corpus.jsonl"
    path.write_text("".join(json.dumps({"text": line}, ensure_ascii=False) + "\n" for line in lines), encoding="utf-8")
    out = train_tokenizer([str(path)], str(tmp / "out"), vocab_size=400)
    return AutoTokenizer.from_pretrained(out)


def test_every_digit_is_its_own_token(tokenizer):
    text = "In 2024 there were 37 + 37 = 74 items. 今年是2025年。12345"
    pieces = [tokenizer.decode([i]) for i in tokenizer(text, add_special_tokens=False)["input_ids"]]
    assert all(sum(ch.isdigit() for ch in piece) <= 1 for piece in pieces), pieces
    assert [p for p in pieces if p.isdigit()][-5:] == list("12345")


@pytest.mark.parametrize("text", ["2 + 2 = 4", "今年是2025年，共37个。", "def f(x):\n    return x * 3.14", "  spaces\tand\nnewlines  "])
def test_decoding_restores_the_text(tokenizer, text):
    assert tokenizer.decode(tokenizer(text, add_special_tokens=False)["input_ids"]) == text


def test_special_tokens_keep_the_ids_the_datasets_assume(tokenizer):
    assert [tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS] == [0, 1, 2]
    assert (tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) == (0, 1, 2)
