"""Ad-hoc: SDPA backends at the formal run's attention shape, causal fwd+bwd in bf16."""
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

B, H, KV, T, D = 16, 12, 3, 2047, 64
flops = 6 * T * T * D * B * H  # causal: half of 4*T^2*D forward, backward twice the forward
torch.manual_seed(0)


def tensors(kv_heads):
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, kv_heads, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, kv_heads, T, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    return q, k, v


def run(backend, gqa, n=20):
    q, k, v = tensors(KV if gqa else H)
    grad = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
    kwargs = {"enable_gqa": True} if gqa else {}
    with sdpa_kernel(backend):
        for _ in range(3):
            F.scaled_dot_product_attention(q, k, v, is_causal=True, **kwargs).backward(grad)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n):
            F.scaled_dot_product_attention(q, k, v, is_causal=True, **kwargs).backward(grad)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n


ref = None
for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION):
    for gqa in (False, True):
        label = f"{backend.name:<20} gqa={gqa!s:<5}"
        try:
            dt = run(backend, gqa)
        except RuntimeError as exc:
            print(f"{label} unavailable: {str(exc).strip().splitlines()[0][:120]}")
            continue
        print(f"{label} {dt * 1e3:6.2f} ms fwd+bwd  {flops / dt / 1e12:6.1f} TFLOPS")

# numerics: cuDNN against flash on one forward/backward
q, k, v = tensors(H)
grad = torch.randn(B, H, T, D, device="cuda", dtype=torch.bfloat16)
outs = {}
for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION):
    try:
        with sdpa_kernel(backend):
            q.grad = k.grad = v.grad = None
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out.backward(grad)
            outs[backend.name] = (out.float(), q.grad.float(), k.grad.float(), v.grad.float())
    except RuntimeError:
        pass
if len(outs) == 2:
    a, b = outs.values()
    for name, x, y in zip(("out", "dq", "dk", "dv"), a, b):
        print(f"cudnn vs flash {name}: max abs diff {(x - y).abs().max().item():.3e}, "
              f"rel {((x - y).norm() / x.norm()).item():.3e}")
print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version())
