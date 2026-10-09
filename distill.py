"""Real knowledge distillation: frozen teacher + student CE + temperature KL."""

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoTokenizer

from dataset import SFTDataset
from evaluate import evaluate_lm
from losses import kd_loss, masked_cross_entropy
from model import Whetstone
from runlog import RunRecorder
from train_utils import (
    MODEL_ARCH_FIELDS,
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


def kd_step(student, teacher, args, ctx):
    def step(batch):
        X, Y, loss_mask = batch
        tokens = int(loss_mask.sum())
        X, Y, loss_mask = (t.to(args.device, non_blocking=True) for t in (X, Y, loss_mask))
        with ctx:
            student_logits = student(X).logits
            with torch.no_grad():
                teacher_logits = teacher(X).logits
            ce = masked_cross_entropy(student_logits, Y, loss_mask)
            kd = kd_loss(student_logits, teacher_logits, temperature=args.temperature, mask=loss_mask)
            loss = (1.0 - args.alpha) * ce + args.alpha * kd
        return loss, {"loss": loss, "ce": ce, "kd": kd}, tokens

    return step


def main():
    parser = argparse.ArgumentParser(description="Knowledge distillation (teacher -> student)")
    add_common_train_args(
        parser,
        batch_size=4,
        learning_rate=1e-4,
        wandb_project="Whetstone-Distill",
        log_step=1,
        max_seq_len=256,
        data_path="tests/fixtures/sft_tiny.jsonl",
    )
    add_model_args(parser)
    parser.add_argument("--teacher_path", type=str, required=True)
    parser.add_argument("--student_path", type=str, required=True)
    parser.add_argument("--alpha", type=float, default=0.5, help="KD mix weight")
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # Teacher and student get independent architectures resolved from their own
    # checkpoints; the --dim/--n_layers flags describe the *student* only, since
    # that is the model being trained. KD across sizes only needs a shared vocab,
    # which resolve_model_config enforces against the tokenizer.
    teacher_cfg = resolve_model_config(
        argparse.Namespace(**{f: None for f in MODEL_ARCH_FIELDS}, max_seq_len=args.max_seq_len),
        tokenizer.vocab_size,
        checkpoint_path=args.teacher_path,
    )
    args.lm_config = resolve_model_config(
        args, tokenizer.vocab_size, checkpoint_path=args.resume_from or args.student_path
    )

    teacher = Whetstone(teacher_cfg).to(args.device)
    student = Whetstone(args.lm_config).to(args.device)
    print(describe_model(teacher, teacher_cfg, "teacher"))
    print(describe_model(student, args.lm_config, "student"))

    load_weights(args.teacher_path, teacher, args.device, strict=False)
    load_weights(args.student_path, student, args.device, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = build_optimizer(student, args)
    ctx, scaler = build_autocast_scaler(args.device, args.dtype)

    state = TrainState()
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = load_weights(args.resume_from, student, args.device, strict=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = TrainState(*load_train_state(ckpt, optimizer, scaler))

    wandb = init_wandb_if_needed(args, run_name=f"distill-bs{args.batch_size}")

    ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    val_loader = build_val_loader(SFTDataset, args, tokenizer)
    args.total_steps = total_updates(ds, args)

    def evaluate():
        return evaluate_lm(student, val_loader, args.device, ctx, args.val_batches or None)

    print(f"KD: alpha={args.alpha} T={args.temperature} steps={args.total_steps}")
    with RunRecorder.start(
        "distill", args, config=args.lm_config, model=student,
        data_paths=[args.data_path, args.val_data_path],
        extra={"teacher": {"path": args.teacher_path, **{
            k: getattr(teacher_cfg, k) for k in ("dim", "n_layers", "n_heads", "n_kv_heads")
        }}, "kd": {"alpha": args.alpha, "temperature": args.temperature}},
    ) as recorder:
        print(f"run: {recorder.run_dir}")
        state = train(
            student, optimizer, scaler, ds, args, kd_step(student, teacher, args, ctx),
            state=state, evaluate=evaluate if val_loader is not None else None,
            recorder=recorder, wandb=wandb,
        )

        if val_loader is not None and not should_evaluate(state.global_step, args):
            stats = evaluate()
            print(f"final val: loss={stats['loss']:.4f} ppl={stats['ppl']:.2f}")
            recorder.log_eval(state.global_step, epoch=min(state.epoch + 1, args.epochs), **stats)

        final_path = f"{args.save_dir}/distill_final.pth"
        save_final_weights(final_path, student, args.lm_config)
        recorder.finish(status="completed", steps=state.global_step)
    print(f"Saved {final_path}; run -> {recorder.run_dir}")


if __name__ == "__main__":
    main()
