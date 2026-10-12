"""Ad-hoc: which ways of preferring cuDNN attention survive torch.compile?"""
import contextlib

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend as B
from torch.nn.attention import sdpa_kernel
from torch.profiler import ProfilerActivity, profile

ORDER = [B.CUDNN_ATTENTION, B.FLASH_ATTENTION, B.EFFICIENT_ATTENTION, B.MATH]
q = torch.randn(2, 12, 2047, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(2, 3, 2047, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(2, 3, 2047, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)


def attn(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


def attn_inner(q, k, v):
    with sdpa_kernel(ORDER, set_priority=True):
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)


def kernels(fn, ctx):
    with ctx:
        fn(q, k, v).sum().backward()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn(q, k, v).sum().backward()
            torch.cuda.synchronize()
    keys = [e.key for e in prof.key_averages()]
    if any("pytorch_flash" in x for x in keys):
        return "flash"
    attention = [x for x in keys if any(s in x.lower() for s in ("cudnn", "fmha", "sdpa", "attn", "sm90"))]
    return "cudnn: " + ", ".join(x[:60] for x in attention[:3]) if attention else "other: " + ", ".join(x[:40] for x in keys[:6])


def compiled(fn):
    torch._dynamo.reset()
    return torch.compile(fn)


cases = [
    ("A eager, priority ctx outside", lambda: attn, lambda: sdpa_kernel(ORDER, set_priority=True)),
    ("B compiled, cudnn-only ctx outside", lambda: compiled(attn), lambda: sdpa_kernel([B.CUDNN_ATTENTION])),
    ("C compiled, priority ctx inside fn", lambda: compiled(attn_inner), contextlib.nullcontext),
    ("D eager, priority ctx inside fn", lambda: attn_inner, contextlib.nullcontext),
]
for label, make_fn, make_ctx in cases:
    try:
        result = kernels(make_fn(), make_ctx())
    except Exception as exc:
        result = f"error {type(exc).__name__}: {str(exc).strip().splitlines()[0][:150]}"
    print(f"{label:<40} -> {result}")

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
for label, fn in (("E compiled, flash+efficient disabled", compiled(attn)), ("F eager, flash+efficient disabled", attn)):
    print(f"{label:<40} -> {kernels(fn, contextlib.nullcontext())}")
