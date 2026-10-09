"""The loop shared by pretrain/SFT/distill/DPO, on a toy model so every property is exact."""

import argparse

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

import trainer
from train_utils import build_optimizer, get_lr, load_train_state, load_weights
from trainer import EpochSampler, TrainState, total_updates, train


class Recorder:
    def __init__(self):
        self.train, self.val, self.tokens = [], [], 0

    def add_tokens(self, count):
        self.tokens += count

    def log(self, step, **metrics):
        self.train.append((step, metrics))

    def log_eval(self, step, **metrics):
        self.val.append((step, metrics))


def _args(tmp_path, **overrides):
    values = dict(
        epochs=2, batch_size=2, accumulation_steps=2, learning_rate=1e-2, weight_decay=0.1,
        adam_beta1=0.9, adam_beta2=0.95, grad_clip=1.0, log_step=2, val_every=0, val_batches=0,
        save_step=0, save_dir=str(tmp_path), seed=7, num_workers=0, device="cpu", max_steps=0,
        lm_config={"toy": True},
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def _data(n):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(n, 4, generator=generator)
    return TensorDataset(x, x.sum(dim=1, keepdim=True))


def _model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 16), nn.Dropout(0.5), nn.Linear(16, 1))


def _step(model):
    def step(batch):
        x, y = batch
        loss = nn.functional.mse_loss(model(x), y)
        return loss, {"loss": loss}, x.shape[0]

    return step


def test_epoch_sampler_is_a_seeded_permutation_and_skips_a_prefix():
    full = list(EpochSampler(10, seed=3, epoch=0))
    assert sorted(full) == list(range(10))
    assert list(EpochSampler(10, seed=3, epoch=0)) == full
    assert list(EpochSampler(10, seed=3, epoch=0, skip=4)) == full[4:]
    assert len(EpochSampler(10, seed=3, epoch=0, skip=4)) == 6
    assert list(EpochSampler(10, seed=3, epoch=1)) != full


def test_a_partial_window_at_epoch_end_is_still_an_update(tmp_path):
    """11 batches with accumulation 2: five full windows plus one of one batch."""
    args = _args(tmp_path)
    data = _data(22)
    model = _model(0)
    state = train(model, build_optimizer(model, args), None, data, args, _step(model))
    assert total_updates(data, args) == 12
    assert state.global_step == 12
    assert (state.epoch, state.step) == (2, 0)


def test_logging_evaluation_and_saving_happen_once_per_update(tmp_path, monkeypatch):
    """With accumulation, the update counter sits still for several micro-batches.
    A check made on every micro-batch then fires once for each of them."""
    saved = []
    monkeypatch.setattr(trainer, "save_checkpoint", lambda path, *a: saved.append((path.rsplit("/", 1)[1], a[5])))
    args = _args(tmp_path, accumulation_steps=3, val_every=2, save_step=4)
    calls = []
    recorder = Recorder()
    model = _model(0)

    train(model, build_optimizer(model, args), None, _data(30), args, _step(model),
          evaluate=lambda: calls.append(1) or {"loss": 0.0}, recorder=recorder)

    assert [step for step, _ in recorder.val] == [2, 4, 6, 8, 10]
    assert len(calls) == 5
    assert [step for step, _ in recorder.train] == [1, 2, 4, 6, 8, 10]
    assert saved == [
        ("latest_checkpoint.pth", 4), ("epoch_1_checkpoint.pth", 5),
        ("latest_checkpoint.pth", 8), ("epoch_2_checkpoint.pth", 10),
    ]
    assert recorder.tokens == 2 * 30


def test_logged_loss_is_the_mean_over_the_micro_batches_since_the_last_log(tmp_path):
    args = _args(tmp_path, epochs=1, accumulation_steps=2, log_step=1)
    model = _model(0)
    recorder = Recorder()
    losses = []

    def step(batch):
        loss, metrics, tokens = _step(model)(batch)
        losses.append(float(loss.detach()))
        return loss, metrics, tokens

    train(model, build_optimizer(model, args), None, _data(8), args, step, recorder=recorder)

    logged = [metrics["loss"] for _, metrics in recorder.train]
    assert logged == pytest.approx([(losses[0] + losses[1]) / 2, (losses[2] + losses[3]) / 2])


def test_a_resumed_run_ends_with_the_same_weights_as_an_uninterrupted_one(tmp_path):
    """Same batches in the same order, same dropout draws: the result is bitwise equal."""
    data = _data(22)
    args = _args(tmp_path / "a", save_step=7)
    model = _model(0)
    train(model, build_optimizer(model, args), None, data, args, _step(model))
    expected = [p.detach().clone() for p in model.parameters()]

    # The checkpoint at update 7 is what a run killed after it leaves behind.
    resumed = _model(123)
    args_b = _args(tmp_path / "b")
    optimizer = build_optimizer(resumed, args_b)
    checkpoint = load_weights(str(tmp_path / "a" / "latest_checkpoint.pth"), resumed, "cpu", strict=True)
    state = TrainState(*load_train_state(checkpoint, optimizer, None))
    assert (state.epoch, state.step, state.global_step) == (1, 2, 7)

    state = train(resumed, optimizer, None, data, args_b, _step(resumed), state=state)

    assert state.global_step == 12
    for got, want in zip(resumed.parameters(), expected):
        assert torch.equal(got, want)


def test_max_steps_stops_early_and_the_schedule_spans_it(tmp_path):
    args = _args(tmp_path, max_steps=3, log_step=100)
    recorder = Recorder()
    model = _model(0)

    state = train(model, build_optimizer(model, args), None, _data(40), args, _step(model), recorder=recorder)

    assert state.global_step == 3
    assert args.total_steps == 3
    step, metrics = recorder.train[-1]
    assert step == 3
    assert metrics["lr"] == pytest.approx(get_lr(3, 3, args.learning_rate))
