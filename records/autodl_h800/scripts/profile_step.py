"""Ad-hoc: where does the GPU time of one compiled pretraining update go?"""
import argparse
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.getcwd())
import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoTokenizer

from model import Whetstone, set_attention_backend
from trainer import lm_loss_fn
from train_utils import (add_model_args, build_autocast_scaler, build_optimizer, optimizer_step,
                         resolve_model_config, set_seed)

parser = argparse.ArgumentParser()
add_model_args(parser)
parser.add_argument("--max_seq_len", type=int, default=2048)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--micro", type=int, default=4)
parser.add_argument("--compile", type=int, default=1)
parser.add_argument("--trace", default="")
parser.add_argument("--attention", default="cudnn")
args = parser.parse_args()
set_attention_backend(args.attention)
args.device, args.dtype = "cuda", "bfloat16"
args.learning_rate, args.weight_decay, args.adam_beta1, args.adam_beta2, args.grad_clip = 5e-4, 0.1, 0.9, 0.95, 1.0

config = resolve_model_config(args, AutoTokenizer.from_pretrained(args.tokenizer_path).vocab_size)
set_seed(0)
model = Whetstone(config).to("cuda")
optimizer = build_optimizer(model, args)
ctx, scaler = build_autocast_scaler("cuda", "bfloat16")
loss_fn = lm_loss_fn(model, bool(args.compile))
seq = args.max_seq_len - 1
x = torch.randint(config.vocab_size, (args.batch_size, seq), device="cuda")
y = torch.randint(config.vocab_size, (args.batch_size, seq), device="cuda")
mask = torch.ones(args.batch_size, seq, device="cuda")


def update():
    for _ in range(args.micro):
        with ctx:
            loss = loss_fn(x, y, mask) / args.micro
        loss.backward()
    optimizer_step(model, optimizer, scaler, args.grad_clip)


for _ in range(3):
    update()
torch.cuda.synchronize()
start = time.perf_counter()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    update()
    torch.cuda.synchronize()
wall_us = (time.perf_counter() - start) * 1e6

events = [e for e in prof.key_averages()]
def self_us(e):
    return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)

BUCKETS = [
    ("attention", ("flash", "fmha", "sdpa", "cudnn", "dot_do_o")),
    ("gemm", ("gemm", "xmma", "cutlass", "nvjet", "cublas", "_bf16_s", "splitk")),
    ("optimizer", ("multi_tensor", "adam", "fused_adam")),
    ("complex/rope", ("complex",)),
    ("cross_entropy", ("log_softmax", "nll_loss", "cross_entropy")),
    ("triton_fused", ("triton_",)),
    ("copy/cat", ("copy", "cat", "index", "gather", "scatter", "embedding")),
    ("elementwise", ("elementwise", "vectorized", "unrolled")),
    ("reduce", ("reduce",)),
]
totals = defaultdict(float)
rows = []
for e in events:
    t = self_us(e)
    if t <= 0:
        continue
    name = e.key
    low = name.lower()
    bucket = next((b for b, keys in BUCKETS if any(k in low for k in keys)), "other")
    totals[bucket] += t
    rows.append((t, e.count, bucket, name))
kernel_us = sum(totals.values())
tokens = args.batch_size * seq * args.micro
print(f"batch {args.batch_size} x {args.micro} micro, compile={bool(args.compile)}: wall {wall_us / 1e3:.1f} ms, "
      f"kernels {kernel_us / 1e3:.1f} ms ({kernel_us / wall_us:.0%} busy), {tokens / wall_us * 1e6:,.0f} tok/s")
for bucket, t in sorted(totals.items(), key=lambda kv: -kv[1]):
    print(f"  {bucket:<14} {t / 1e3:8.1f} ms  {t / kernel_us:6.1%}")
print("top kernels:")
for t, count, bucket, name in sorted(rows, reverse=True)[:40]:
    print(f"  {t / 1e3:7.1f} ms {t / kernel_us:6.1%} x{count:<4} [{bucket}] {name[:150]}")
if args.trace:
    prof.export_chrome_trace(args.trace)
