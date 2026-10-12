"""How well does each checkpoint rank held-out preference pairs?

Two numbers per checkpoint, over the same pairs:

* ``implicit``: DPO's own accuracy, the share of pairs with beta * (policy - reference) log-ratio
  margin > 0, the reference being the SFT policy (``evaluate.evaluate_preference``);
* ``raw``: the share of pairs where the checkpoint gives the chosen completion a higher
  sequence log-probability than the rejected one, with no reference involved. This is what decides
  whether sampling prefers the gold answer.

The pairs are cut with ``envs.generate_data --split preference`` from a seed the training pairs did
not use, minus every dev / test question and every training prompt; see records/posttrain_v2/README.md::

    PYTHONPATH=. python3 records/posttrain_v2/pref_accuracy.py --pairs PAIRS.jsonl CKPT...
"""

from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader


def main() -> None:
    from transformers import AutoTokenizer

    from dataset import PreferenceDataset
    from evaluate import evaluate_preference
    from losses import sequence_logprobs
    from probes import load_model
    from train_utils import resolve_device

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--reference", default="results/posttrain_v1/sft/sft_final.pth")
    parser.add_argument("--tokenizer_path", default="tokenizer/v1_32k")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=25)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    loader = DataLoader(PreferenceDataset(args.pairs, tokenizer, max_length=args.max_seq_len), batch_size=args.batch_size)
    ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if device != "cpu" else None
    ref = load_model(args.reference, tokenizer, args.max_seq_len, device)
    ref.eval()

    def raw_accuracy(model) -> tuple[float, float]:
        right = pairs = 0
        margin = 0.0
        with torch.no_grad():
            for batch in loader:
                cX, cY, cM, rX, rY, rM = (t.to(device) for t in batch)
                with ctx or torch.autocast(device_type="cpu", enabled=False):
                    chosen = sequence_logprobs(model(cX).logits, cY, cM)
                    rejected = sequence_logprobs(model(rX).logits, rY, rM)
                right += int((chosen > rejected).sum())
                pairs += chosen.numel()
                margin += float((chosen - rejected).sum())
        return right / pairs, margin / pairs

    results = {}
    for path in [args.reference, *args.checkpoints]:
        model = ref if path == args.reference else load_model(path, tokenizer, args.max_seq_len, device)
        model.eval()
        implicit = evaluate_preference(model, ref, loader, device, args.beta, ctx)
        raw, raw_margin = raw_accuracy(model)
        results[path] = {
            "pairs": implicit["pairs"],
            "implicit_accuracy": implicit["accuracy"],
            "implicit_margin": implicit["margin"],
            "raw_accuracy": raw,
            "raw_logprob_margin": raw_margin,
        }
        print(f"{path}: implicit {implicit['accuracy']:.3f} (margin {implicit['margin']:+.3f})  "
              f"raw {raw:.3f} (log-prob margin {raw_margin:+.2f})  over {implicit['pairs']} pairs")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
