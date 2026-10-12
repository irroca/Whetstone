import json
import random

import torch
from transformers import AutoTokenizer

import eval_env
from config import LLMConfig
from envs import ArithmeticEnv
from envs.generate_data import build_rows, write_jsonl
from model import Whetstone
from rollout import build_prompt_ids
from train_utils import save_final_weights

TOKENIZER = "./tokenizer/zh_6400"
EOS, PAD = 2, 0


def test_evaluate_tasks_breaks_accuracy_down_by_task_metadata():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    env = ArithmeticEnv(seed=0)
    tasks = env.sample(12)
    task_of = {tuple(build_prompt_ids(tokenizer, env, t).tolist()): t for t in tasks}

    class AddsOnlyPolicy:
        """Gold answers for sums, a wrong answer for every difference."""

        training = False

        def eval(self):
            return self

        def train(self):
            return self

        def generate(self, input_ids, **kwargs):
            rows = []
            for row in input_ids.tolist():
                task = task_of[tuple(row)]
                text = env.gold_completion(task) if task.meta["op"] == "+" else "<think></think><answer>-1</answer>"
                rows.append(tokenizer(text, add_special_tokens=False).input_ids + [EOS])
            width = max(map(len, rows))
            return torch.tensor([row + [PAD] * (width - len(row)) for row in rows])

    result = eval_env.evaluate_tasks(
        AddsOnlyPolicy(), tokenizer, env, tasks, max_new_tokens=64, device="cpu", group_by=["op"]
    )

    sums = sum(t.meta["op"] == "+" for t in tasks)
    assert 0 < sums < len(tasks), "the sample should mix operators"
    assert result["items"] == len(tasks)
    assert result["accuracy"] == sums / len(tasks)
    assert result["format_rate"] == 1.0
    assert result["by_group"] == {
        "op=+": {"items": sums, "accuracy": 1.0},
        "op=-": {"items": len(tasks) - sums, "accuracy": 0.0},
    }


def test_the_cli_scores_a_checkpoint_on_a_task_file(tmp_path):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    torch.manual_seed(0)
    config = LLMConfig(dim=32, n_layers=1, n_heads=2, n_kv_heads=2, vocab_size=tokenizer.vocab_size, max_seq_len=64)
    checkpoint = tmp_path / "sft_final.pth"
    save_final_weights(str(checkpoint), Whetstone(config).eval(), config)
    tasks = tmp_path / "test.jsonl"
    write_jsonl(str(tasks), build_rows(ArithmeticEnv(seed=1), "eval", 6, random.Random(1)))
    out = tmp_path / "result.json"

    assert eval_env.main([
        "--checkpoint", str(checkpoint), "--eval_path", str(tasks), "--tokenizer_path", TOKENIZER,
        "--max_new_tokens", "4", "--limit", "5", "--device", "cpu", "--out", str(out),
    ]) == 0

    result = json.loads(out.read_text("utf-8"))
    assert result["items"] == 5
    assert sum(row["items"] for row in result["by_group"].values()) == 5
    assert {"accuracy", "format_rate", "hack_rate", "examples", "seconds"} <= set(result)
