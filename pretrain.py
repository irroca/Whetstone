"""Causal LM pretraining."""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer

from dataset import build_pretrain_dataset
from evaluate import evaluate_lm
from model import Whetstone
from runlog import RunRecorder
from train_utils import (
    add_common_train_args,
    add_model_args,
    build_autocast_scaler,
    build_optimizer,
    build_val_loader,
    describe_model,
    init_wandb_if_needed,
    load_train_state,
    load_weights,
    resolve_model_config,
    save_final_weights,
    set_seed,
    should_evaluate,
    str2bool,
)
from trainer import TrainState, lm_step, total_updates, train


def main():
    parser = argparse.ArgumentParser()
    add_common_train_args(
        parser,
        learning_rate=5e-4,
        wandb_project="Whetstone-Pretrain",
        data_path="datasets/pretrain.jsonl",
    )
    add_model_args(parser)
    parser.add_argument(
        "--compile",
        type=str2bool,
        default=False,
        help="torch.compile the training forward: a one-off compile, then fused norm and "
             "activation kernels. Evaluation and checkpoints use the uncompiled module",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    args.lm_config = resolve_model_config(
        args, tokenizer.vocab_size, checkpoint_path=args.resume_from
    )
    model = Whetstone(args.lm_config).to(args.device)
    optimizer = build_optimizer(model, args)
    ctx, scaler = build_autocast_scaler(args.device, args.dtype)

    state = TrainState()
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = load_weights(args.resume_from, model, args.device, strict=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = TrainState(*load_train_state(ckpt, optimizer, scaler))

    print(describe_model(model, args.lm_config, "pretrain"))

    wandb = init_wandb_if_needed(args, run_name=f"pretrain-bs{args.batch_size}")

    ds = build_pretrain_dataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    val_loader = build_val_loader(build_pretrain_dataset, args, tokenizer)
    args.total_steps = total_updates(ds, args)
    forward = torch.compile(model) if args.compile else model

    def evaluate():
        return evaluate_lm(model, val_loader, args.device, ctx, args.val_batches or None)

    with RunRecorder.start(
        "pretrain", args, config=args.lm_config, model=model,
        data_paths=[args.data_path, args.val_data_path],
    ) as recorder:
        print(f"run: {recorder.run_dir}")
        state = train(
            model, optimizer, scaler, ds, args, lm_step(forward, args, ctx),
            state=state, evaluate=evaluate if val_loader is not None else None,
            recorder=recorder, wandb=wandb,
        )

        if val_loader is not None and not should_evaluate(state.global_step, args):
            stats = evaluate()
            print(f"final val: loss={stats['loss']:.4f} ppl={stats['ppl']:.2f}")
            recorder.log_eval(state.global_step, epoch=min(state.epoch + 1, args.epochs), **stats)

        save_final_weights(f"{args.save_dir}/pretrain_final.pth", model, args.lm_config)
        recorder.finish(status="completed", steps=state.global_step)
    print("Training completed!")


if __name__ == "__main__":
    main()
