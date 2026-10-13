import json

import pytest

import paired


def items(*correct):
    return [{"question": f"q{i}", "correct": bool(c)} for i, c in enumerate(correct)]


def test_mcnemar_matches_hand_computed_binomial_tails():
    assert paired.mcnemar_exact(0, 0) == 1.0
    assert paired.mcnemar_exact(5, 5) == 1.0
    # 0 of 10 and 10 of 10 are both ends of the same two-sided tail: 2 * 2**-10
    assert paired.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert paired.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    # P(X <= 2) for n = 8 is (1 + 8 + 28) / 256
    assert paired.mcnemar_exact(2, 6) == pytest.approx(2 * 37 / 256)


def test_mcnemar_survives_counts_where_two_to_the_minus_n_underflows():
    assert paired.mcnemar_exact(3000, 3000) == 1.0
    assert 0.0 <= paired.mcnemar_exact(0, 3000) < 1e-300


def test_paired_compare_counts_the_four_outcomes():
    a = items(1, 1, 0, 0, 1, 0)
    b = items(1, 0, 1, 1, 1, 0)
    result = paired.paired_compare(a, b)
    assert (result["both"], result["only_a"], result["only_b"], result["neither"]) == (2, 1, 2, 1)
    assert result["accuracy_a"] == 3 / 6 and result["accuracy_b"] == 4 / 6
    assert result["difference"] == pytest.approx(1 / 6)


def test_paired_compare_refuses_different_tasks():
    other = items(1, 1)
    other[1]["question"] = "something else"
    with pytest.raises(ValueError, match="different question"):
        paired.paired_compare(items(1, 1), other)
    with pytest.raises(ValueError, match="numbers of items"):
        paired.paired_compare(items(1, 1), items(1))


def test_the_cli_reads_two_result_files(tmp_path, capsys):
    paths = []
    for name, correct in (("a", (1, 1, 1, 0)), ("b", (1, 0, 0, 0))):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"per_item": items(*correct)}))
        paths.append(str(path))
    assert paired.main(paths) == 0
    out = capsys.readouterr().out
    assert "only A 2, only B 0" in out and "p = 0.5000" in out


def test_the_cli_asks_for_a_rerun_when_per_item_is_missing(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps({"accuracy": 0.5}))
    with pytest.raises(SystemExit, match="per_item"):
        paired.main([str(path), str(path)])
