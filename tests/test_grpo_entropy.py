"""``entropy_all`` in grpo.py's metrics: the whole sampled batch, not just the groups the update used."""

import json
import sys

import pytest
import torch
from transformers import AutoTokenizer

import grpo
from config import LLMConfig
from model import Whetstone
from train_utils import save_final_weights

TOKENIZER = "./tokenizer/zh_6400"


def _run_grpo(tmp_path, monkeypatch, *extra):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(0)
    config = LLMConfig(dim=32, n_layers=1, n_heads=2, n_kv_heads=2, vocab_size=tokenizer.vocab_size, max_seq_len=128)
    policy = tmp_path / "sft_final.pth"
    save_final_weights(str(policy), Whetstone(config).eval(), config)
    save_dir = tmp_path / "out"
    argv = [
        "grpo.py", "--policy_path", str(policy), "--tokenizer_path", TOKENIZER, "--save_dir", str(save_dir),
        "--device", "cpu", "--rl_steps", "2", "--batch_size", "3", "--group_size", "4", "--max_new_tokens", "4",
        "--max_seq_len", "128", "--eval_size", "2", "--log_step", "1", *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    grpo.main()
    rows = [json.loads(line) for line in (save_dir / "grpo_metrics.jsonl").read_text("utf-8").splitlines()]
    return [row for row in rows if "entropy_all" in row]


def test_unfiltered_entropy_all_is_the_updates_own_entropy(tmp_path, monkeypatch):
    rows = _run_grpo(tmp_path, monkeypatch)

    assert len(rows) == 2
    for row in rows:
        assert row["groups_used"] == 3
        # one PPO epoch: the update's entropy is measured on the same weights and the same tokens
        assert row["entropy_all"] == pytest.approx(row["entropy"], rel=1e-3)


def test_filtering_every_group_still_reports_the_batchs_entropy(tmp_path, monkeypatch):
    # A random policy never answers correctly, so every group is silent and the filter drops all of them.
    rows = _run_grpo(tmp_path, monkeypatch, "--filter_zero_variance", "True")

    for row in rows:
        assert row["groups_used"] == 0
        assert row["entropy"] == 0.0
        assert row["entropy_all"] > 1.0
