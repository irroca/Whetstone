"""Direct Preference Optimization (DPO) training."""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer

from dataset import PreferenceDataset
from evaluate import evaluate_preference
from losses import dpo_loss, sequence_logprobs
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
)
from trainer import TrainState, total_updates, train


def dpo_step(policy, ref, args, ctx):
    def step(batch):
        tokens = int(batch[2].sum() + batch[5].sum())
        cX, cY, cM, rX, rY, rM = (t.to(args.device, non_blocking=True) for t in batch)
        with ctx:
            policy_chosen_logits = policy(cX).logits
            policy_rejected_logits = policy(rX).logits
            with torch.no_grad():
                ref_chosen_logits = ref(cX).logits
                ref_rejected_logits = ref(rX).logits
            loss = dpo_loss(
                sequence_logprobs(policy_chosen_logits, cY, cM),
                sequence_logprobs(policy_rejected_logits, rY, rM),
                sequence_logprobs(ref_chosen_logits, cY, cM),
                sequence_logprobs(ref_rejected_logits, rY, rM),
                beta=args.beta,
            )
        return loss, {"dpo_loss": loss}, tokens

    return step


def main():
    parser = argparse.ArgumentParser(description="DPO preference optimization")
    add_common_train_args(
        parser,
        batch_size=2,
        learning_rate=1e-5,
        wandb_project="Whetstone-DPO",
        log_step=1,
        max_seq_len=256,
        data_path="tests/fixtures/preference_tiny.jsonl",
    )
    add_model_args(parser)
    parser.add_argument("--policy_path", type=str, required=True, help="Init policy (usually SFT)")
    parser.add_argument("--ref_path", type=str, default=None, help="Frozen reference; default=policy_path")
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    if args.ref_path is None:
        args.ref_path = args.policy_path

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    cfg = resolve_model_config(
        args, tokenizer.vocab_size, checkpoint_path=args.resume_from or args.policy_path
    )
    args.lm_config = cfg

    policy = Whetstone(cfg).to(args.device)
    ref = Whetstone(cfg).to(args.device)
    print(describe_model(policy, cfg, "policy"))
    load_weights(args.policy_path, policy, args.device, strict=False)
    load_weights(args.ref_path, ref, args.device, strict=False)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    optimizer = build_optimizer(policy, args)
    ctx, scaler = build_autocast_scaler(args.device, args.dtype)

    state = TrainState()
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = load_weights(args.resume_from, policy, args.device, strict=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = TrainState(*load_train_state(ckpt, optimizer, scaler))

    wandb = init_wandb_if_needed(args, run_name=f"dpo-bs{args.batch_size}")

    ds = PreferenceDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    val_loader = build_val_loader(PreferenceDataset, args, tokenizer)
    args.total_steps = total_updates(ds, args)

    def evaluate():
        return evaluate_preference(policy, ref, val_loader, args.device, args.beta, ctx,
                                   args.val_batches or None)

    print(f"DPO: beta={args.beta} steps={args.total_steps}")
    with RunRecorder.start(
        "dpo", args, config=args.lm_config, model=policy,
        data_paths=[args.data_path, args.val_data_path],
        extra={"beta": args.beta, "ref_path": args.ref_path},
    ) as recorder:
        print(f"run: {recorder.run_dir}")
        state = train(
            policy, optimizer, scaler, ds, args, dpo_step(policy, ref, args, ctx),
            state=state, evaluate=evaluate if val_loader is not None else None,
            recorder=recorder, wandb=wandb,
        )

        if val_loader is not None and not should_evaluate(state.global_step, args):
            stats = evaluate()
            print(f"final val: accuracy={stats['accuracy']:.3f} margin={stats['margin']:+.4f}")
            recorder.log_eval(state.global_step, epoch=min(state.epoch + 1, args.epochs), **stats)

        final_path = f"{args.save_dir}/dpo_final.pth"
        save_final_weights(final_path, policy, args.lm_config)
        recorder.finish(status="completed", steps=state.global_step)
    print(f"Saved {final_path}; run -> {recorder.run_dir}")


if __name__ == "__main__":
    main()
