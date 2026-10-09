"""The epoch loop shared by the DataLoader stages: pretrain, SFT, distill and DPO.

Each stage supplies how one batch becomes a loss. Everything around it lives
here, so three properties hold for all four stages at once:

* **Logging, evaluation and checkpoints happen right after an optimizer update.**
  Checked on every micro-batch instead, an evaluation due at update 1000 runs
  once for each micro-batch of the window that leaves the count at 1000.
* **A resumed run sees exactly the batches the interrupted one had left.** An
  epoch's order is a permutation seeded by ``seed + epoch``; a checkpoint records
  how many of its batches were consumed, the resumed sampler starts after them
  without loading them, and the RNG state comes back with the weights.
* **No host sync per micro-batch.** Losses and metrics are summed on the device
  and read when logged; token counts come from the mask before it is moved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader, Sampler

from losses import masked_cross_entropy
from train_utils import accelerator_type, get_lr, optimizer_step, save_checkpoint, should_evaluate

# step_fn(batch) -> (loss to back-propagate, {name: scalar tensor to log}, target tokens)
StepFn = Callable[[Any], "tuple[torch.Tensor, dict, int]"]


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0  # batches of ``epoch`` already consumed
    global_step: int = 0  # optimizer updates
    loss: float = float("nan")


class EpochSampler(Sampler):
    """The permutation of ``range(size)`` seeded by ``seed + epoch``, minus its first ``skip``."""

    def __init__(self, size: int, seed: int, epoch: int, skip: int = 0):
        self.size, self.seed, self.epoch, self.skip = size, seed, epoch, skip

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator)[self.skip:].tolist())

    def __len__(self) -> int:
        return max(0, self.size - self.skip)


def batches_per_epoch(dataset, args) -> int:
    return math.ceil(len(dataset) / args.batch_size)


def total_updates(dataset, args) -> int:
    """Optimizer updates in the run, which the LR schedule spans.

    An epoch's trailing partial window is an update too: it is applied at epoch end.
    """
    per_epoch = math.ceil(batches_per_epoch(dataset, args) / args.accumulation_steps)
    total = args.epochs * per_epoch
    max_steps = getattr(args, "max_steps", 0) or 0
    return max(1, min(total, max_steps) if max_steps else total)


class _Meter:
    """Sums since the last log, kept on the device until they are read."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.loss: Optional[torch.Tensor] = None
        self.metrics: dict[str, torch.Tensor] = {}
        self.count = 0

    def add(self, loss: torch.Tensor, metrics: dict) -> None:
        loss = loss.detach().float()
        self.loss = loss if self.loss is None else self.loss + loss
        for key, value in metrics.items():
            value = value.detach().float() if torch.is_tensor(value) else torch.tensor(float(value))
            self.metrics[key] = self.metrics[key] + value if key in self.metrics else value
        self.count += 1

    def read(self) -> tuple[float, dict[str, float]]:
        means = {key: float(value) / self.count for key, value in self.metrics.items()}
        return float(self.loss) / self.count, means


def lm_step(model, args, ctx) -> StepFn:
    """Masked next-token cross-entropy: the whole step for pretrain and SFT."""

    def step(batch):
        X, Y, mask = batch
        tokens = int(mask.sum())
        X, Y, mask = (t.to(args.device, non_blocking=True) for t in (X, Y, mask))
        with ctx:
            loss = masked_cross_entropy(model(X).logits, Y, mask)
        return loss, {"loss": loss}, tokens

    return step


def _fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def train(
    model,
    optimizer,
    scaler,
    dataset,
    args,
    step_fn: StepFn,
    *,
    state: Optional[TrainState] = None,
    evaluate: Optional[Callable[[], dict]] = None,
    recorder=None,
    wandb=None,
) -> TrainState:
    """Train from ``state`` to the end of ``args.epochs``, or ``args.max_steps`` updates.

    ``evaluate()`` returns held-out metrics and runs every ``args.val_every``
    updates. ``model`` is the module whose gradients are clipped and whose
    weights are checkpointed; ``step_fn`` may run a compiled wrapper of it.
    """
    state = state or TrainState()
    args.total_steps = getattr(args, "total_steps", None) or total_updates(dataset, args)
    max_steps = getattr(args, "max_steps", 0) or 0
    batches = batches_per_epoch(dataset, args)
    pin_memory = accelerator_type(args.device) == "cuda"
    meter = _Meter()
    last: dict = {}

    def log(epoch: int, consumed: int, lr: float, grad_norm: float) -> None:
        state.loss, metrics = meter.read()
        meter.reset()
        print(
            f"Epoch[{epoch + 1}/{args.epochs}] ({consumed}/{batches}) "
            + " ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            + f" lr={lr:.7f} global_step={state.global_step}"
        )
        if recorder is not None:
            recorder.log(state.global_step, epoch=epoch + 1, **metrics, lr=lr, grad_norm=grad_norm)
        if wandb is not None:
            wandb.log({**metrics, "lr": lr, "global_step": state.global_step})

    def update(epoch: int, consumed: int) -> bool:
        """Apply one optimizer update and what follows it. True when the run is over."""
        lr = get_lr(state.global_step + 1, args.total_steps, args.learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        grad_norm = optimizer_step(model, optimizer, scaler, args.grad_clip)
        state.global_step += 1
        state.step = consumed
        last.update(epoch=epoch, consumed=consumed, lr=lr, grad_norm=grad_norm)
        done = bool(max_steps) and state.global_step >= max_steps
        if state.global_step == 1 or (args.log_step and state.global_step % args.log_step == 0) or done:
            log(**last)
        if evaluate is not None and should_evaluate(state.global_step, args):
            stats = evaluate()
            print("  val: " + " ".join(f"{key}={_fmt(value)}" for key, value in stats.items()))
            if recorder is not None:
                recorder.log_eval(state.global_step, epoch=epoch + 1, **stats)
            if wandb is not None:
                wandb.log({**{f"val_{k}": v for k, v in stats.items()}, "global_step": state.global_step})
        if args.save_step and state.global_step % args.save_step == 0:
            save_checkpoint(
                f"{args.save_dir}/latest_checkpoint.pth", model, optimizer, scaler,
                epoch, consumed, state.global_step, state.loss, args.lm_config,
            )
        return done

    model.train()
    for epoch in range(state.epoch, args.epochs):
        skip = state.step if epoch == state.epoch else 0
        state.epoch, state.step = epoch, skip
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=EpochSampler(len(dataset), args.seed, epoch, skip * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            # Each iterator draws a worker seed; from the global RNG, that draw would
            # put a resumed run one step behind the dropout stream it restored.
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        pending = False
        for step, batch in enumerate(loader, start=skip):
            loss, metrics, tokens = step_fn(batch)
            scaled = loss / args.accumulation_steps
            (scaler.scale(scaled) if scaler is not None else scaled).backward()
            meter.add(loss, metrics)
            if recorder is not None:
                recorder.add_tokens(tokens)
            pending = (step + 1) % args.accumulation_steps != 0
            if not pending and update(epoch, step + 1):
                return state
        if pending and update(epoch, batches):
            return state
        state.epoch, state.step = epoch + 1, 0
        if meter.count and epoch + 1 == args.epochs:
            log(**last)
        save_checkpoint(
            f"{args.save_dir}/epoch_{epoch + 1}_checkpoint.pth", model, optimizer, scaler,
            epoch + 1, 0, state.global_step, state.loss, args.lm_config,
        )
    return state
