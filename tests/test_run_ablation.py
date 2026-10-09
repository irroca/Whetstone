import json
import os
import random

import pytest

import run_ablation
from datatools.ablation import AblationSpec
from datatools.tokenize_corpus import load_token_meta

ZH = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处府"
EN = "the of and to in is was for on that with as by at from his it an were are which this be or had not first one their its new after but who they has her she been other when there all also more two during into time can only up over"


def _docs(kind, n, rng):
    if kind == "zh":
        return ["".join(rng.choice(ZH) for _ in range(rng.randint(60, 90))) + "。" for _ in range(n)]
    words = EN.split()
    if kind == "en":
        return [" ".join(rng.choice(words) for _ in range(rng.randint(40, 60))) + "." for _ in range(n)]
    return [
        "\n".join(f"def f{i}_{j}(x):\n    return x * {j} + {rng.randint(0, 999)}" for j in range(rng.randint(4, 8)))
        for i in range(n)
    ]


@pytest.fixture()
def spec(tmp_path):
    rng = random.Random(0)
    sources = []
    for name, kind, weight in (("zh_web", "zh", 0.4), ("en_web", "en", 0.4), ("code", "code", 0.2)):
        path = tmp_path / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for text in _docs(kind, 400, rng):
                fh.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        sources.append({
            "name": name, "weight": weight, "jsonl": str(path),
            "filters": {"max_digit_ratio": None, "max_repetition": None, "max_duplicate_lines": None},
        })
    mixture = {
        "name": "tiny", "tokenizer": "./tokenizer/zh_6400", "total_tokens": 1,
        "val_fraction": 0.1, "holdout_fraction": 0.1, "sources": sources,
    }
    with open(tmp_path / "mixture.json", "w", encoding="utf-8") as fh:
        json.dump(mixture, fh)
    values = {
        "name": "tiny-ablation",
        "mixture": str(tmp_path / "mixture.json"),
        "tokens_per_arm": 4000,
        "pool_margin": 2.5,
        "tokenizer_sample_tokens": 3000,
        "data_dir": str(tmp_path / "data"),
        "results_dir": str(tmp_path / "results"),
        "model": {"dim": 32, "n_layers": 1, "n_heads": 2, "n_kv_heads": 2},
        "train": {
            "max_seq_len": 128, "batch_size": 4, "learning_rate": 0.001, "dtype": "float32",
            "epochs": 1, "val_every": 4, "val_batches": 2, "save_step": 1000, "log_step": 4,
            "device": "cpu", "seed": 1,
        },
        "probes": {"max_tokens_per_source": 300, "confusion_prompts": 3, "arithmetic_items": 4},
        "arms": [
            {"name": "zh0", "vocab_size": 300, "shares": {"zh_web": 0.0}},
            {"name": "zh50", "vocab_size": 320, "shares": {"zh_web": 0.5}},
        ],
    }
    path = tmp_path / "ablation.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(values, fh)
    return str(path)


def _main(monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", ["run_ablation.py", *argv])
    assert run_ablation.main() == 0


def test_every_stage_runs_and_a_second_run_redoes_nothing(spec, monkeypatch, capsys):
    _main(monkeypatch, spec, "run", "--device", "cpu")
    ablation = AblationSpec.load(spec)

    with open(os.path.join(ablation.results_dir, "report.json"), encoding="utf-8") as fh:
        rows = {row["arm"]: row for row in json.load(fh)}
    assert set(rows) == {"zh0", "zh50"}
    assert rows["zh0"]["shares"]["zh_web"] == 0.0
    assert rows["zh50"]["shares"]["zh_web"] == pytest.approx(0.5, abs=0.05)
    for arm in ablation.arms:
        meta = load_token_meta(f"{ablation.arm_corpus(arm)}.bin")
        assert meta["num_tokens"] >= ablation.tokens_per_arm
        assert meta["tokenizer"]["vocab_size"] == arm.vocab_size
        row = rows[arm.name]
        assert set(row["bpb"]) == {"zh_web", "en_web", "code"}
        assert row["val_loss"] is not None
    assert "| zh50 |" in open(os.path.join(ablation.results_dir, "report.md"), encoding="utf-8").read()

    finals = {arm.name: os.path.join(ablation.results_dir, arm.name, "pretrain_final.pth") for arm in ablation.arms}
    stamps = {name: os.path.getmtime(path) for name, path in finals.items()}
    capsys.readouterr()
    _main(monkeypatch, spec, "run", "--device", "cpu")
    assert {name: os.path.getmtime(path) for name, path in finals.items()} == stamps
    assert "pretrain.py" not in capsys.readouterr().out


def test_an_interrupted_arm_resumes_from_its_latest_checkpoint(spec):
    ablation = AblationSpec.load(spec)
    arm = ablation.arms[0]
    save_dir = os.path.join(ablation.results_dir, arm.name)
    assert "--resume_from" not in run_ablation.train_command(ablation, arm)
    os.makedirs(save_dir)
    open(os.path.join(save_dir, "latest_checkpoint.pth"), "w").close()
    cmd = run_ablation.train_command(ablation, arm)
    assert cmd[cmd.index("--resume_from") + 1] == os.path.abspath(os.path.join(save_dir, "latest_checkpoint.pth"))
    assert cmd[cmd.index("--n_kv_heads") + 1] == "2"
