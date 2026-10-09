"""Every DataLoader stage end to end on the fixtures: pretrain -> SFT -> distill / DPO."""

import os

import pytest

import distill
import dpo
import pretrain
import sft
from runlog import discover_runs, read_run

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
TINY = [
    "--dim", "32", "--n_layers", "1", "--n_heads", "4", "--n_kv_heads", "2", "--hidden_dim", "64",
    "--epochs", "1", "--batch_size", "2", "--max_seq_len", "64", "--device", "cpu",
    "--dtype", "float32", "--log_step", "1", "--val_every", "1",
]


def _run(monkeypatch, module, save_dir, *argv):
    monkeypatch.setattr("sys.argv", [f"{module.__name__}.py", *TINY, "--save_dir", str(save_dir), *argv])
    module.main()
    (run_dir,) = discover_runs(str(save_dir))
    run = read_run(run_dir)
    assert run["summary"]["status"] == "completed"
    return run


def _rows(run, split):
    return [row for row in run["metrics"] if row["split"] == split]


@pytest.fixture(scope="module")
def sft_weights(tmp_path_factory):
    """pretrain, then SFT from its weights: what distill and DPO start from."""
    root = tmp_path_factory.mktemp("stages")
    with pytest.MonkeyPatch.context() as monkeypatch:
        data = os.path.join(FIXTURES, "pretrain_tiny.jsonl")
        run = _run(monkeypatch, pretrain, root / "pretrain", "--data_path", data, "--val_data_path", data)
        assert [row["step"] for row in _rows(run, "train")] == [1, 2, 3]
        assert [row["step"] for row in _rows(run, "val")] == [1, 2, 3]

        data = os.path.join(FIXTURES, "sft_tiny.jsonl")
        run = _run(
            monkeypatch, sft, root / "sft", "--data_path", data, "--val_data_path", data,
            "--pretrained_path", str(root / "pretrain" / "pretrain_final.pth"),
        )
        assert _rows(run, "val")[-1]["loss"] > 0
    return root / "sft" / "sft_final.pth"


def test_pretrain_and_sft_write_their_final_weights(sft_weights):
    assert sft_weights.exists()
    assert (sft_weights.parent.parent / "pretrain" / "pretrain_final.config.json").exists()


def test_distill_logs_both_halves_of_its_loss(sft_weights, tmp_path, monkeypatch):
    run = _run(
        monkeypatch, distill, tmp_path, "--data_path", os.path.join(FIXTURES, "sft_tiny.jsonl"),
        "--teacher_path", str(sft_weights), "--student_path", str(sft_weights),
    )
    first = _rows(run, "train")[0]
    assert {"loss", "ce", "kd"} <= set(first)
    assert first["kd"] == pytest.approx(0.0, abs=1e-6)  # the student starts as the teacher
    assert (tmp_path / "distill_final.pth").exists()


def test_dpo_evaluates_preference_accuracy(sft_weights, tmp_path, monkeypatch):
    data = os.path.join(FIXTURES, "preference_tiny.jsonl")
    run = _run(
        monkeypatch, dpo, tmp_path, "--data_path", data, "--val_data_path", data,
        "--policy_path", str(sft_weights),
    )
    assert "dpo_loss" in _rows(run, "train")[0]
    assert 0.0 <= _rows(run, "val")[-1]["accuracy"] <= 1.0
    assert (tmp_path / "dpo_final.pth").exists()


def test_pretrain_resumes_from_its_epoch_checkpoint(tmp_path, monkeypatch):
    data = os.path.join(FIXTURES, "pretrain_tiny.jsonl")
    _run(monkeypatch, pretrain, tmp_path / "first", "--data_path", data)

    run = _run(
        monkeypatch, pretrain, tmp_path / "second", "--data_path", data, "--epochs", "2",
        "--resume_from", str(tmp_path / "first" / "epoch_1_checkpoint.pth"),
    )
    assert [row["step"] for row in _rows(run, "train")] == [4, 5, 6]
    assert {row["epoch"] for row in _rows(run, "train")} == {2}
