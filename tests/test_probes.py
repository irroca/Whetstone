import math

import pytest
import torch
from transformers import AutoTokenizer

from config import LLMConfig
from model import Whetstone
from probes import (
    addition_items,
    arithmetic_probe,
    bits_per_byte,
    cjk_share,
    document_windows,
    language_confusion,
    parse_answer,
)

TOKENIZER = "./tokenizer/zh_6400"
TEXTS = ["磨刀石是一块用来磨刀的石头。" * 6, "short", "A longer English page about whetstones. " * 12]


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(TOKENIZER)


def _model(tokenizer, seed=0):
    torch.manual_seed(seed)
    config = LLMConfig(dim=32, n_layers=1, n_heads=2, n_kv_heads=2, vocab_size=tokenizer.vocab_size, max_seq_len=64)
    return Whetstone(config).eval()


@pytest.mark.parametrize("length", [1, 2, 15, 16, 17, 100])
def test_windows_make_every_token_after_the_first_a_target_once(length):
    ids = list(range(length))
    windows = document_windows(ids, max_seq_len=16)
    targets = [t for w in windows for t in w[1:]]
    assert targets == ids[1:]
    assert all(len(w) <= 16 for w in windows)


def test_a_uniform_model_scores_log2_vocab_bits_per_token(tokenizer):
    """Zero output weights make every logit equal: each target costs exactly log2(V)."""
    model = _model(tokenizer)
    with torch.no_grad():
        model.output.weight.zero_()
    result = bits_per_byte(model, tokenizer, TEXTS, max_seq_len=16, device="cpu")
    assert result["bits_per_token"] == pytest.approx(math.log2(tokenizer.vocab_size), rel=1e-5)
    expected_targets = sum(len(tokenizer(t, add_special_tokens=False)["input_ids"]) + 1 for t in TEXTS)
    assert result["tokens"] == expected_targets
    assert result["bytes"] == sum(len(t.encode("utf-8")) for t in TEXTS)
    assert result["bpb"] == pytest.approx(result["bits_per_token"] * result["tokens"] / result["bytes"])


def test_bits_per_byte_does_not_depend_on_batching(tokenizer):
    """Padding a short window up to the batch width must not leak into the sum."""
    model = _model(tokenizer)
    one = bits_per_byte(model, tokenizer, TEXTS, max_seq_len=16, device="cpu", batch_size=1)
    many = bits_per_byte(model, tokenizer, TEXTS, max_seq_len=16, device="cpu", batch_size=7)
    assert many["bpb"] == pytest.approx(one["bpb"], rel=1e-5)


@pytest.mark.parametrize(
    "text, share",
    [("中文abc", 2 / 5), ("中文", 1.0), ("abc", 0.0), ("123 + 456 = ?", None), ("", None)],
)
def test_cjk_share_counts_letters_only(text, share):
    assert cjk_share(text) == (pytest.approx(share) if share is not None else None)


class _ScriptedModel:
    """Stands in for a model: ``generate`` answers every row with ``reply(prompt_text)``."""

    def __init__(self, tokenizer, reply):
        self.tokenizer, self.reply = tokenizer, reply

    def eval(self):
        return self

    def generate(self, input_ids, eos_token_id, max_new_tokens, pad_token_id, **_):
        rows = []
        for row in input_ids.tolist():
            prompt = self.tokenizer.decode(row, skip_special_tokens=True)
            ids = self.tokenizer(self.reply(prompt), add_special_tokens=False)["input_ids"] + [eos_token_id]
            rows.append(ids[:max_new_tokens])
        width = max(len(r) for r in rows)
        return torch.tensor([r + [pad_token_id] * (width - len(r)) for r in rows])


def test_confusion_counts_continuations_that_leave_the_prompts_language(tokenizer):
    zh = ["这是一个很长的中文段落，用来测试模型会不会在中文提示之后改说英文。" * 3] * 4
    en = ["This is a long English paragraph used to check which language the model continues in. " * 3] * 4
    always_english = _ScriptedModel(tokenizer, lambda prompt: " and then it goes on in English")
    result = language_confusion(always_english, tokenizer, zh, en, device="cpu", prompts=4, prompt_tokens=8)
    assert result["zh"]["confused"] == 1.0 and result["en"]["confused"] == 0.0
    assert result["zh"]["prompts"] == 4 and result["zh"]["scored"] == 4


def test_prompts_shorter_than_prompt_tokens_are_skipped(tokenizer):
    model = _ScriptedModel(tokenizer, lambda prompt: "继续说中文")
    result = language_confusion(model, tokenizer, ["短"], ["tiny"], device="cpu", prompts=4, prompt_tokens=8)
    assert result["zh"]["prompts"] == 0 and result["en"]["prompts"] == 0


def test_addition_items_are_deterministic_and_well_formed():
    items = addition_items(5, digits=2, shots=3, seed=0)
    assert items == addition_items(5, digits=2, shots=3, seed=0)
    prompt, answer = items[0]
    lines = prompt.split("\n")
    assert len(lines) == 4 and lines[-1].endswith("=")
    a, b = (int(x) for x in lines[-1][:-1].split("+"))
    assert answer == a + b and 10 <= a <= 99 and 10 <= b <= 99
    for line in lines[:-1]:
        left, right = line.split("=")
        assert sum(int(x) for x in left.split("+")) == int(right)


@pytest.mark.parametrize("text, value", [(" 47", 47), ("12 + 3", 12), (" -5", -5), (" x", None), ("", None)])
def test_parse_answer_reads_the_leading_integer(text, value):
    assert parse_answer(text) == value


def test_arithmetic_probe_scores_exact_answers(tokenizer):
    def solve(prompt):
        a, b = prompt.split("\n")[-1].rstrip("=").split("+")
        return f" {int(a) + int(b)}\n"

    result = arithmetic_probe(_ScriptedModel(tokenizer, solve), tokenizer, "cpu", items=10)
    assert result == {"add_1digit": 1.0, "add_2digit": 1.0, "items": 10}
    wrong = arithmetic_probe(_ScriptedModel(tokenizer, lambda p: " 1000\n"), tokenizer, "cpu", items=10)
    assert wrong["add_1digit"] == 0.0 and wrong["add_2digit"] == 0.0
