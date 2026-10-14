"""Does a policy still stop after its answer? P(EOS) right after the first ``</answer>`` of its own greedy completions.

GRPO runs on the 3-digit task sometimes ended up writing ``</answer>N</answer>N...`` to the token limit. This
measures the probability of the EOS token at the first place it should appear, teacher-forced on the policy's own
greedy completion cut after ``</answer>`` (taken from an ``eval_env.py`` result file, so no generation is needed):

    python3 probe_eos.py --tokenizer_path tokenizer/v1_32k --pairs sft=results/sft/test.json:results/sft/sft_final.pth ...
"""

from __future__ import annotations

import argparse
import json

import torch

CLOSE = "</answer>"


def eos_probability(model, tokenizer, env, tasks, completions, device, limit):
    from rollout import build_prompt_ids

    probs, eos_argmax, counted = [], 0, 0
    for task, text in list(zip(tasks, completions))[:limit]:
        cut = text.find(CLOSE)
        if cut < 0:
            continue
        prompt = build_prompt_ids(tokenizer, env, task, device)
        body = tokenizer(text[: cut + len(CLOSE)], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        ids = torch.cat([prompt, body[0]]).unsqueeze(0)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float()
        p = torch.softmax(logits, dim=-1)
        probs.append(float(p[tokenizer.eos_token_id]))
        eos_argmax += int(int(logits.argmax()) == tokenizer.eos_token_id)
        counted += 1
    return {"n": counted, "mean_p_eos": sum(probs) / max(counted, 1), "eos_is_argmax": eos_argmax / max(counted, 1)}


def main(argv=None) -> int:
    from transformers import AutoTokenizer

    from envs import load_tasks, make_env
    from probes import load_model
    from train_utils import resolve_device

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--pairs", nargs="+", required=True, help="name=result.json:checkpoint.pth")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--device", default=resolve_device())
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    env = make_env("arithmetic", seed=0)
    tasks = load_tasks(args.eval_path)
    out = {}
    for pair in args.pairs:
        name, rest = pair.split("=", 1)
        result_path, ckpt = rest.split(":", 1)
        with open(result_path, encoding="utf-8") as fh:
            completions = [row["completion"] for row in json.load(fh)["per_item"]]
        model = load_model(ckpt, tokenizer, 400, args.device)
        out[name] = eos_probability(model, tokenizer, env, tasks, completions, args.device, args.limit)
        print(f"{name:>14}: n {out[name]['n']}, mean P(EOS) {out[name]['mean_p_eos']:.4f}, "
              f"EOS is argmax {out[name]['eos_is_argmax']:.2f}")
        del model
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
