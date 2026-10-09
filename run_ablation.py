"""Run a data ablation end to end: pool, tokenizers, arm corpora, pretraining, probes, report.

::

    python3 run_ablation.py configs/ablation_v1.json plan     # arms, shares, pool size; touches nothing
    python3 run_ablation.py configs/ablation_v1.json run      # every stage below, in order
    python3 run_ablation.py configs/ablation_v1.json pool | tokenizers | data | train | probe | report

Every stage skips work whose output exists, and every output is written under a
temporary name first, so a killed run is resumed by running the same command
again. A partially trained arm resumes from its ``latest_checkpoint.pth``.
How arms are cut from the shared pool is in ``datatools/ablation.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Optional

from datatools.ablation import (
    TOKENIZER_SAMPLE_NAME,
    AblationSpec,
    Arm,
    arm_quotas,
    arm_shares,
    base_weights,
    check_pool,
    cut_corpus,
    describe_arms,
    describe_pool,
    load_mixture,
    required_paths,
    vocab_name,
    write_pool_spec,
    write_tokenizer_sample,
)

STAGES = ("pool", "tokenizers", "data", "train", "probe", "report")
REPO = os.path.dirname(os.path.abspath(__file__))


def _run(cmd: list[str], offline: Optional[bool]) -> None:
    # Spec paths are relative to the caller; the subprocess runs from the repo so
    # that `-m datatools.prepare` and the mixture's eval-set paths resolve.
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=_env(offline), cwd=REPO)


def _env(offline: Optional[bool]) -> dict:
    """``None`` keeps the caller's setting."""
    env = dict(os.environ)
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    elif offline is not None:
        env.pop("HF_HUB_OFFLINE", None)
    return env


def _write_json(path: str, payload) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def arm_save_dir(spec: AblationSpec, arm: Arm) -> str:
    return os.path.join(spec.results_dir, arm.name)


def stage_pool(spec: AblationSpec) -> None:
    pool = write_pool_spec(spec)
    if os.path.exists(os.path.join(spec.pool_dir, "manifest.json")):
        check_pool(spec, pool)
        print(f"pool: {spec.pool_dir} already pulled")
        return
    print(describe_pool(spec, load_mixture(spec.mixture)))
    from_hub = any(source.get("hf") for source in pool["sources"])
    _run([
        sys.executable, "-m", "datatools.prepare", os.path.abspath(spec.pool_spec_path),
        "--out_dir", os.path.abspath(spec.pool_dir), "--progress_every", "20000",
    ], offline=False if from_hub else None)
    check_pool(spec, pool)


def stage_tokenizers(spec: AblationSpec) -> None:
    from train_tokenizer import train_tokenizer

    missing = [v for v in spec.vocab_sizes() if not os.path.exists(os.path.join(spec.tokenizer_dir(v), "tokenizer.json"))]
    if not missing:
        print("tokenizers: all trained")
        return
    sample = os.path.join(spec.data_dir, TOKENIZER_SAMPLE_NAME)
    if not os.path.exists(sample):
        mixture = load_mixture(spec.mixture)
        counts = write_tokenizer_sample(spec, base_weights(mixture), _tokenizer(mixture["tokenizer"]), sample)
        print(f"tokenizer sample -> {sample}: {counts}")
    for vocab in missing:
        out = spec.tokenizer_dir(vocab)
        tmp = f"{out}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        started = time.time()
        train_tokenizer([sample], tmp, vocab)
        shutil.rmtree(out, ignore_errors=True)
        os.replace(tmp, out)
        print(f"tokenizer {vocab_name(vocab)} -> {out} ({time.time() - started:.0f}s)", flush=True)


def stage_data(spec: AblationSpec) -> None:
    from datatools.tokenize_corpus import render, token_paths, tokenize_corpus

    weights = base_weights(load_mixture(spec.mixture))
    pool_train = os.path.join(spec.pool_dir, "train.jsonl")
    for vocab in spec.vocab_sizes():
        tok_dir = spec.tokenizer_dir(vocab)
        tokenizer = _tokenizer(tok_dir)
        val = spec.split_corpus("val", vocab)
        if not os.path.exists(token_paths(val)[0]):
            meta = tokenize_corpus([os.path.join(spec.pool_dir, "val.jsonl")], tokenizer, val, tokenizer_path=tok_dir)
            print(render(meta, token_paths(val)[0]), flush=True)
        for arm in (a for a in spec.arms if a.vocab_size == vocab):
            out = spec.arm_corpus(arm)
            if os.path.exists(token_paths(out)[0]):
                continue
            meta = cut_corpus(pool_train, arm_quotas(spec, arm, weights), tokenizer, out, tokenizer_path=tok_dir)
            print(render(meta, token_paths(out)[0]), flush=True)


def train_command(spec: AblationSpec, arm: Arm) -> list[str]:
    save_dir = os.path.abspath(arm_save_dir(spec, arm))
    cmd = [
        sys.executable, os.path.join(REPO, "pretrain.py"),
        "--data_path", os.path.abspath(f"{spec.arm_corpus(arm)}.bin"),
        "--val_data_path", os.path.abspath(f"{spec.split_corpus('val', arm.vocab_size)}.bin"),
        "--tokenizer_path", os.path.abspath(spec.tokenizer_dir(arm.vocab_size)),
        "--save_dir", save_dir,
    ]
    for key, value in {**spec.train, **spec.model}.items():
        cmd += [f"--{key}", str(value)]
    latest = os.path.join(save_dir, "latest_checkpoint.pth")
    if os.path.exists(latest):
        cmd += ["--resume_from", latest]
    return cmd


def stage_train(spec: AblationSpec, only: Optional[list[str]] = None) -> None:
    for arm in spec.arms:
        if only and arm.name not in only:
            continue
        final = os.path.join(arm_save_dir(spec, arm), "pretrain_final.pth")
        if os.path.exists(final):
            print(f"train {arm.name}: done")
            continue
        started = time.time()
        _run(train_command(spec, arm), offline=True)
        print(f"train {arm.name}: {(time.time() - started) / 3600:.2f} h", flush=True)


def stage_probe(spec: AblationSpec, device: str) -> None:
    from probes import holdout_by_source, load_model, run_probes

    holdout = holdout_by_source(os.path.join(spec.pool_dir, "holdout.jsonl"))
    max_seq_len = int(spec.train.get("max_seq_len", 512))
    for arm in spec.arms:
        save_dir = arm_save_dir(spec, arm)
        out = os.path.join(save_dir, "probes.json")
        final = os.path.join(save_dir, "pretrain_final.pth")
        if os.path.exists(out) or not os.path.exists(final):
            continue
        started = time.time()
        tokenizer = _tokenizer(spec.tokenizer_dir(arm.vocab_size))
        model = load_model(final, tokenizer, max_seq_len, device)
        result = {
            "arm": arm.name,
            **run_probes(model, tokenizer, holdout, max_seq_len, device, **spec.probes),
            "seconds": round(time.time() - started, 1),
        }
        _write_json(out, result)
        print(f"probe {arm.name}: " + ", ".join(f"{s} {r['bpb']:.3f}" for s, r in result["bpb"].items())
              + f" ({result['seconds']}s)", flush=True)


def _final_val_loss(save_dir: str) -> Optional[float]:
    from runlog import discover_runs, read_run

    for run_dir in reversed(discover_runs(save_dir)):
        run = read_run(run_dir)
        if run["summary"].get("status") != "completed":
            continue
        evals = [row for row in run["metrics"] if row.get("split") == "val"]
        return evals[-1]["loss"] if evals else None
    return None


def stage_report(spec: AblationSpec) -> None:
    from datatools.tokenize_corpus import load_token_meta

    weights = base_weights(load_mixture(spec.mixture))
    rows = []
    for arm in spec.arms:
        save_dir = arm_save_dir(spec, arm)
        probes_path = os.path.join(save_dir, "probes.json")
        if not os.path.exists(probes_path):
            continue
        with open(probes_path, "r", encoding="utf-8") as fh:
            probes = json.load(fh)
        corpus = load_token_meta(f"{spec.arm_corpus(arm)}.bin")
        bpb = {source: r["bpb"] for source, r in probes["bpb"].items()}
        rows.append({
            "arm": arm.name,
            "vocab": vocab_name(arm.vocab_size),
            "shares": {name: corpus["sources"].get(name, {}).get("tokens", 0) / corpus["num_tokens"] for name in weights},
            "train_tokens": corpus["num_tokens"],
            "val_loss": _final_val_loss(save_dir),
            "bpb": bpb,
            # Weighted by the base mixture, so every arm is averaged the same way.
            "bpb_mixture": sum(weights[s] * bpb[s] for s in bpb if s in weights) / sum(weights[s] for s in bpb if s in weights),
            "zh_confused": probes["language_confusion"]["zh"]["confused"],
            "en_confused": probes["language_confusion"]["en"]["confused"],
            **{k: v for k, v in probes["arithmetic"].items() if k.startswith("add_")},
        })
    if not rows:
        print("report: no probed arms yet")
        return
    sources = [s for s in weights if any(s in row["bpb"] for row in rows)]
    adds = sorted({k for row in rows for k in row if k.startswith("add_")})
    header = (["arm", "vocab", "zh share", "val loss"] + [f"bpb {s}" for s in sources]
              + ["bpb mix", "zh→en", "en→zh"] + adds)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["arm"], row["vocab"], f"{row['shares'].get('zh_web', 0):.1%}",
                 "—" if row["val_loss"] is None else f"{row['val_loss']:.3f}"]
        cells += [f"{row['bpb'][s]:.3f}" if s in row["bpb"] else "—" for s in sources]
        cells += [f"{row['bpb_mixture']:.3f}", f"{row['zh_confused']:.0%}", f"{row['en_confused']:.0%}"]
        cells += [f"{row[k]:.0%}" for k in adds]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    os.makedirs(spec.results_dir, exist_ok=True)
    _write_json(os.path.join(spec.results_dir, "report.json"), rows)
    with open(os.path.join(spec.results_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# {spec.name}\n\n{spec.tokens_per_arm / 1e6:,.0f}M tokens per arm. "
                 f"bpb: holdout bits per UTF-8 byte (lower is better), comparable across vocabularies. "
                 f"val loss: per token, comparable only within one vocabulary.\n\n{table}\n")
    print(table)


def plan(spec: AblationSpec) -> None:
    mixture = load_mixture(spec.mixture)
    print(describe_arms(spec, base_weights(mixture)))
    print()
    print(describe_pool(spec, mixture))
    print()
    for name, path in required_paths(spec).items():
        print(f"    [{'x' if os.path.exists(path) else ' '}] {name:<22} {path}")
    for arm in spec.arms:
        final = os.path.join(arm_save_dir(spec, arm), "pretrain_final.pth")
        print(f"    [{'x' if os.path.exists(final) else ' '}] trained {arm.name:<14} {final}")


def main() -> int:
    from train_utils import resolve_device

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("spec", help="Ablation spec JSON, e.g. configs/ablation_v1.json")
    parser.add_argument("stage", choices=("plan", "run", *STAGES))
    parser.add_argument("--arms", nargs="+", default=None, help="train only these arms")
    parser.add_argument("--device", default=resolve_device(), help="device for the probes")
    args = parser.parse_args()

    spec = AblationSpec.load(args.spec)
    for arm in spec.arms:
        arm_shares(base_weights(load_mixture(spec.mixture)), arm.shares)  # fail on a bad arm before any work
    if args.stage == "plan":
        plan(spec)
        return 0
    stages = STAGES if args.stage == "run" else (args.stage,)
    for stage in stages:
        if stage == "pool":
            stage_pool(spec)
        elif stage == "tokenizers":
            stage_tokenizers(spec)
        elif stage == "data":
            stage_data(spec)
        elif stage == "train":
            stage_train(spec, args.arms)
        elif stage == "probe":
            stage_probe(spec, args.device)
        elif stage == "report":
            stage_report(spec)
    return 0


if __name__ == "__main__":
    sys.exit(main())
