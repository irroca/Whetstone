"""Throughput, peak memory and MFU of the pretraining step, to size a GPU run.

Runs the step ``pretrain.py`` runs (model, AdamW, autocast, masked cross-entropy)
on random token ids, so it needs no data, over a grid of micro-batch sizes and
``torch.compile`` on/off. On CUDA it also checks that the training forward
reaches the flash-attention kernel.

    python3 bench_train.py --tokenizer_path tokenizer/v1_32k --dim 768 --n_layers 12 \\
        --n_heads 12 --n_kv_heads 3 --max_seq_len 2048 --batch_sizes 8 16 32 \\
        --compile False True --peak_tflops 312 --out results/bench.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
from transformers import AutoTokenizer

from losses import masked_cross_entropy
from model import Whetstone
from train_utils import (
    accelerator_type,
    add_model_args,
    build_autocast_scaler,
    build_optimizer,
    describe_model,
    optimizer_step,
    resolve_device,
    resolve_model_config,
    set_seed,
    str2bool,
)


def flops_per_token(model: Whetstone, seq_len: int) -> float:
    """Training FLOPs per token: 6 per matmul weight, plus causal attention.

    Every 2-D weight is a matmul, the tied embedding included since it is also
    the LM head. Scores and the weighted sum each cost ``2 * seq * dim`` per
    layer if every query saw every key; causal masking halves that, and the
    backward pass costs twice the forward: ``6 * layers * seq * dim`` in all.
    """
    matmul = sum(p.numel() for p in model.parameters() if p.ndim == 2)
    return 6 * matmul + 6 * model.params.n_layers * seq_len * model.params.dim


def _sync(device: str) -> None:
    kind = accelerator_type(device)
    if kind == "cuda":
        torch.cuda.synchronize()
    elif kind == "mps":
        torch.mps.synchronize()


def _peak_memory_gb(device: str):
    kind = accelerator_type(device)
    if kind == "cuda":
        return torch.cuda.max_memory_allocated() / 2**30
    if kind == "mps":
        return torch.mps.driver_allocated_memory() / 2**30  # current, not peak: MPS keeps no peak
    return None


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def flash_attention_check(config, args) -> str:
    """Run one training forward with every SDPA backend but flash disabled.

    Flash needs fp16/bf16 inputs, a supported head size and no explicit mask;
    the model passes ``attn_mask=None`` only for an unpadded full forward, so
    a regression there shows up as this check failing, not as a silent slowdown.
    """
    if accelerator_type(args.device) != "cuda":
        return "n/a (not CUDA)"
    from torch.nn.attention import SDPBackend, sdpa_kernel

    model = Whetstone(config).to(args.device)
    ctx, _ = build_autocast_scaler(args.device, args.dtype)
    x = torch.randint(config.vocab_size, (1, args.max_seq_len - 1), device=args.device)
    try:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION), ctx:
            model(x)
        return "ok"
    except RuntimeError as exc:
        return f"unavailable: {str(exc).strip().splitlines()[0]}"
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def bench(config, args, batch_size: int, compiled: bool) -> dict:
    set_seed(0)
    model = Whetstone(config).to(args.device)
    optimizer = build_optimizer(model, args)
    ctx, scaler = build_autocast_scaler(args.device, args.dtype)
    forward = torch.compile(model) if compiled else model
    model.train()

    # pretrain feeds max_seq_len - 1 positions: X and Y are one window shifted by one token
    seq = args.max_seq_len - 1
    generator = torch.Generator().manual_seed(0)
    x = torch.randint(config.vocab_size, (batch_size, seq), generator=generator).to(args.device)
    y = torch.randint(config.vocab_size, (batch_size, seq), generator=generator).to(args.device)
    mask = torch.ones(batch_size, seq, device=args.device)

    def update():
        for _ in range(args.accumulation_steps):
            with ctx:
                loss = masked_cross_entropy(forward(x).logits, y, mask) / args.accumulation_steps
            (scaler.scale(loss) if scaler is not None else loss).backward()
        optimizer_step(model, optimizer, scaler, args.grad_clip)

    row = {"batch_size": batch_size, "compile": compiled}
    try:
        for _ in range(args.warmup):
            update()
        _sync(args.device)
        if accelerator_type(args.device) == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(args.steps):
            update()
        _sync(args.device)
        elapsed = time.perf_counter() - start
    except RuntimeError as exc:
        if not _is_oom(exc):
            raise
        row["status"] = "OOM"
    else:
        tokens = batch_size * seq * args.accumulation_steps * args.steps
        row.update(
            status="ok",
            tokens_per_s=tokens / elapsed,
            update_s=elapsed / args.steps,
            peak_memory_gb=_peak_memory_gb(args.device),
        )
        row["tflops"] = row["tokens_per_s"] * flops_per_token(model, seq) / 1e12
        if args.peak_tflops:
            row["mfu"] = row["tflops"] / args.peak_tflops
    finally:
        del model, optimizer, forward
        gc.collect()
        if compiled:
            torch._dynamo.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def _cell(value, fmt):
    return "-" if value is None else format(value, fmt)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    add_model_args(parser)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--device", type=str, default=resolve_device())
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--compile", type=str2bool, nargs="+", default=[False])
    parser.add_argument(
        "--accumulation_steps", type=int, default=1,
        help="micro-batches per optimizer update, as in training: the update is amortized over them",
    )
    parser.add_argument("--steps", type=int, default=10, help="timed updates per configuration")
    parser.add_argument("--warmup", type=int, default=3, help="untimed updates first; compilation happens here")
    parser.add_argument(
        "--peak_tflops", type=float, default=0.0,
        help="the card's dense bf16 peak, for MFU (A100/A800 312, H100/H800 SXM 989); 0 = no MFU",
    )
    parser.add_argument("--out", type=str, default="", help="also write the results as JSON here")
    args = parser.parse_args(argv)
    args.learning_rate, args.weight_decay, args.adam_beta1, args.adam_beta2, args.grad_clip = 5e-4, 0.1, 0.9, 0.95, 1.0

    config = resolve_model_config(args, AutoTokenizer.from_pretrained(args.tokenizer_path).vocab_size)
    probe = Whetstone(config)
    per_token = flops_per_token(probe, args.max_seq_len - 1)
    print(describe_model(probe, config, "bench"))
    del probe
    device_name = torch.cuda.get_device_name() if accelerator_type(args.device) == "cuda" else args.device
    print(f"device: {device_name}  torch {torch.__version__}  dtype {args.dtype}  "
          f"{per_token / 1e9:.3f} GFLOP/token at seq {args.max_seq_len}")
    flash = flash_attention_check(config, args)
    print(f"flash attention: {flash}")

    rows = []
    print(f"{'batch':>5} {'compile':>7} {'tok/s':>10} {'update s':>9} {'mem GB':>7} {'TFLOPS':>7} {'MFU':>6}")
    for compiled in args.compile:
        for batch_size in args.batch_sizes:
            row = bench(config, args, batch_size, compiled)
            rows.append(row)
            if row["status"] != "ok":
                print(f"{batch_size:>5} {str(compiled):>7} {row['status']:>10}")
                continue
            print(
                f"{batch_size:>5} {str(compiled):>7} {row['tokens_per_s']:>10,.0f} {row['update_s']:>9.3f} "
                f"{_cell(row['peak_memory_gb'], '.1f'):>7} {row['tflops']:>7.1f} "
                f"{_cell(row.get('mfu'), '.1%'):>6}"
            )

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "device": device_name, "torch": torch.__version__, "dtype": args.dtype,
                "max_seq_len": args.max_seq_len, "accumulation_steps": args.accumulation_steps,
                "config": {k: getattr(config, k) for k in ("dim", "n_layers", "n_heads", "n_kv_heads",
                                                             "hidden_dim", "vocab_size")},
                "flops_per_token": per_token, "flash_attention": flash, "rows": rows,
            }, fh, indent=2)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
