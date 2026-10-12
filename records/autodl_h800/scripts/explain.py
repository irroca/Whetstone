"""Ad-hoc: does the compiled forward+loss stay one graph with the cuDNN backend order?"""
import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
import torch
from transformers import AutoTokenizer

from losses import masked_cross_entropy
from model import Whetstone, set_attention_backend
from train_utils import add_model_args, resolve_model_config

parser = argparse.ArgumentParser()
add_model_args(parser)
parser.add_argument("--max_seq_len", type=int, default=2048)
args = parser.parse_args()
config = resolve_model_config(args, AutoTokenizer.from_pretrained(args.tokenizer_path).vocab_size)
model = Whetstone(config).cuda().train()


def loss_fn(X, Y, mask):
    return masked_cross_entropy(model(X).logits, Y, mask)


x = torch.randint(config.vocab_size, (2, 2047), device="cuda")
y = torch.randint(config.vocab_size, (2, 2047), device="cuda")
mask = torch.ones(2, 2047, device="cuda")
for backend in ("default", "cudnn"):
    set_attention_backend(backend)
    torch._dynamo.reset()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        explanation = torch._dynamo.explain(loss_fn)(x, y, mask)
    print(f"{backend}: graphs={explanation.graph_count} breaks={explanation.graph_break_count}")
    for reason in explanation.break_reasons[:3]:
        print("   ", str(reason.reason)[:200])
