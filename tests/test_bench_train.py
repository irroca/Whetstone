import json

import pytest

import bench_train
from config import LLMConfig
from model import Whetstone

TINY = ["--dim", "32", "--n_layers", "1", "--n_heads", "4", "--n_kv_heads", "2", "--hidden_dim", "64"]


def test_flops_per_token_counts_every_matmul_once_and_causal_attention():
    model = Whetstone(LLMConfig(dim=32, n_layers=1, n_heads=4, n_kv_heads=2, hidden_dim=64, vocab_size=64))
    embedding = 64 * 32  # tied: the LM head
    attention = 32 * 32 + 2 * (32 * 16) + 32 * 32  # wq, wk and wv at 2 kv heads of 8, wo
    mlp = 3 * 32 * 64
    assert bench_train.flops_per_token(model, 16) == 6 * (embedding + attention + mlp) + 6 * 1 * 16 * 32


def test_the_benchmark_runs_on_cpu_and_reports_mfu(tmp_path, capsys):
    out = tmp_path / "bench.json"
    argv = [*TINY, "--max_seq_len", "17", "--device", "cpu", "--dtype", "float32",
            "--batch_sizes", "1", "2", "--steps", "2", "--warmup", "1", "--peak_tflops", "1", "--out", str(out)]
    assert bench_train.main(argv) == 0

    result = json.loads(out.read_text())
    assert result["flash_attention"] == "n/a (not CUDA)"
    assert [row["batch_size"] for row in result["rows"]] == [1, 2]
    for row in result["rows"]:
        assert row["status"] == "ok"
        assert row["tokens_per_s"] > 0
        assert row["mfu"] == pytest.approx(row["tflops"] / 1.0)
    assert "tok/s" in capsys.readouterr().out
