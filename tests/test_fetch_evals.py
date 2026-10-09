"""Converters are tested offline against rows recorded from the HuggingFace API,
so the field names stay pinned without the suite needing network access."""

import json

import pytest

from datatools.fetch_evals import (
    DECONTAMINATION_SETS,
    EVAL_SOURCES,
    FetchReport,
    convert_big_math,
    convert_gsm8k,
    convert_humaneval,
    convert_math500,
    convert_mbpp,
    convert_mmlu,
    convert_tal_scq5k,
    render,
    update_spec_decontamination,
)
from datatools.decontaminate import load_eval_index, match_record
from datatools.records import detect_schema, record_parts, validate, write_jsonl

GSM8K_ROW = {
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half "
                "as many clips in May. How many clips did Natalia sell altogether?",
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
              "Natalia sold 48+24 = <<48+24=72>>72 clips altogether.\n#### 72",
}

MATH500_ROW = {
    "problem": "Convert the point $(0,3)$ in rectangular coordinates to polar coordinates.",
    "solution": "We have that $r = \\sqrt{0^2 + 3^2} = 3.$",
    "answer": "\\left( 3, \\frac{\\pi}{2} \\right)",
    "subject": "Precalculus",
    "level": "2",
    "unique_id": "test/precalculus/807.json",
}

TAL_ROW = {
    "qtype": "single_choice",
    "problem": "奶奶告诉小明：2006年共有53个星期日。小明立刻告诉奶奶：2007年的元旦一定是．",
    "answer_option_list": [
        [{"aoVal": "A", "content": "星期一 "}],
        [{"aoVal": "B", "content": "星期二 "}],
        [{"aoVal": "C", "content": "星期三 "}],
    ],
    "answer_analysis": ["2006年有365天，而365=7×52+1，所以2007年元旦是星期一。"],
    "answer_value": "B",
    "difficulty": "2",
}

MMLU_ROW = {
    "question": "What is the capital of France?",
    "subject": "geography",
    "choices": ["Berlin", "Paris", "Madrid", "Rome"],
    "answer": 1,
}

BIG_MATH_ROW = {
    "problem": "What is the remainder when 7^100 is divided by 5?",
    "answer": "1",
    "source": "olympiads",
    "domain": "number theory",
    "llama8b_solve_rate": 0.375,
}

HUMANEVAL_ROW = {
    "task_id": "HumanEval/0",
    "prompt": "from typing import List\n\n\n"
              "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
              '    """ Check if in given list of numbers, are any two numbers closer to each other than\n'
              "    given threshold.\n"
              "    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n"
              "    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n"
              '    """\n',
    "canonical_solution": "    for idx, elem in enumerate(numbers):\n"
                          "        for idx2, elem2 in enumerate(numbers):\n"
                          "            if idx != idx2:\n"
                          "                distance = abs(elem - elem2)\n"
                          "                if distance < threshold:\n"
                          "                    return True\n\n"
                          "    return False\n",
    "test": "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\n"
            "def check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n",
    "entry_point": "has_close_elements",
}

HUMANEVAL_IDIOM_ROW = {
    "task_id": "HumanEval/7",
    "prompt": "from typing import List\n\n\n"
              "def filter_by_substring(strings: List[str], substring: str) -> List[str]:\n"
              '    """ Filter an input list of strings only for ones that contain given substring\n'
              "    >>> filter_by_substring([], 'a')\n    []\n"
              "    >>> filter_by_substring(['abc', 'bacd', 'cde', 'array'], 'a')\n"
              "    ['abc', 'bacd', 'array']\n"
              '    """\n',
    "canonical_solution": "    return [x for x in strings if substring in x]\n",
    "test": "\n\ndef check(candidate):\n    assert candidate([], 'john') == []\n",
    "entry_point": "filter_by_substring",
}

MBPP_ROW = {
    "task_id": 11,
    "text": "Write a python function to remove first and last occurrence of a given character "
            "from the string.",
    "code": "def remove_Occ(s,ch): \r\n    for i in range(len(s)): \r\n        if (s[i] == ch): \r\n"
            "            s = s[0 : i] + s[i + 1:] \r\n            break\r\n"
            "    for i in range(len(s) - 1,-1,-1):  \r\n        if (s[i] == ch): \r\n"
            "            s = s[0 : i] + s[i + 1:] \r\n            break\r\n    return s ",
    "test_list": [
        'assert remove_Occ("hello","l") == "heo"',
        'assert remove_Occ("abcda","a") == "bcd"',
        'assert remove_Occ("PHP","P") == "H"',
    ],
    "test_setup_code": "",
    "challenge_test_list": [
        'assert remove_Occ("hellolloll","l") == "helollol"',
        'assert remove_Occ("","l") == ""',
    ],
}


def test_gsm8k_extracts_the_final_answer_after_the_marker():
    record = convert_gsm8k(GSM8K_ROW)
    assert record["answer"] == "72"
    assert record["question"].startswith("Natalia sold clips")
    assert "#### " not in record["solution"]
    assert "48/2" in record["solution"]  # chain of thought retained for decontamination


def test_gsm8k_strips_thousands_separators():
    record = convert_gsm8k({"question": "q", "answer": "reasoning\n#### 1,234"})
    assert record["answer"] == "1234"


def test_gsm8k_row_without_a_marker_is_skipped():
    assert convert_gsm8k({"question": "q", "answer": "no marker here"}) is None


def test_math500_keeps_problem_answer_and_solution():
    record = convert_math500(MATH500_ROW)
    assert record["answer"] == "\\left( 3, \\frac{\\pi}{2} \\right)"
    assert record["subject"] == "Precalculus"
    assert record["solution"].startswith("We have that")


def test_math500_row_missing_an_answer_is_skipped():
    assert convert_math500({"problem": "p", "answer": ""}) is None


def test_tal_resolves_the_letter_to_the_option_text():
    """A bare 'B' is useless as a verifiable target."""
    record = convert_tal_scq5k(TAL_ROW)
    assert record["answer"] == "星期二"
    assert record["answer_letter"] == "B"
    assert record["solution"].startswith("2006年有365天")


def test_tal_accepts_list_fields_serialized_as_repr():
    """The HF viewer (and some loaders) hand these back as Python reprs."""
    row = dict(TAL_ROW)
    row["answer_option_list"] = str(TAL_ROW["answer_option_list"])
    row["answer_analysis"] = str(TAL_ROW["answer_analysis"])

    record = convert_tal_scq5k(row)

    assert record["answer"] == "星期二"
    assert record["solution"].startswith("2006年")


def test_tal_row_with_an_unmatched_letter_is_skipped():
    row = dict(TAL_ROW, answer_value="Z")
    assert convert_tal_scq5k(row) is None


def test_tal_ignores_malformed_option_lists():
    assert convert_tal_scq5k(dict(TAL_ROW, answer_option_list="[not valid python")) is None


def test_a_question_without_text_is_skipped():
    """TAL-SCQ5K-EN has '?' stems with options a-d; the question was an image."""
    options = [[{"aoVal": letter, "content": letter.lower()}] for letter in "ABCD"]
    assert convert_tal_scq5k(dict(TAL_ROW, problem="?", answer_option_list=options, answer_value="B")) is None
    assert convert_mmlu(dict(MMLU_ROW, question=" ? ")) is None


def test_multiple_choice_questions_carry_their_options():
    tal = convert_tal_scq5k(TAL_ROW)
    assert tal["question"].splitlines() == [TAL_ROW["problem"], "A. 星期一", "B. 星期二", "C. 星期三"]
    mmlu = convert_mmlu(MMLU_ROW)
    assert mmlu["question"].splitlines() == [
        "What is the capital of France?", "A. Berlin", "B. Paris", "C. Madrid", "D. Rome",
    ]


def test_a_generic_stem_matches_only_together_with_its_options(tmp_path):
    """A stem such as '下列说法正确的是．' is generic; the stem with its options is the item."""
    row = dict(
        TAL_ROW,
        problem="下列说法正确的是．",
        answer_option_list=[
            [{"aoVal": "A", "content": "两个锐角的和一定是钝角"}],
            [{"aoVal": "B", "content": "平行四边形的对角线互相平分"}],
            [{"aoVal": "C", "content": "三角形的外角一定大于内角"}],
        ],
        answer_value="B",
        answer_analysis=["由平行四边形的判定可知。"],
    )
    record = convert_tal_scq5k(row)
    page = "第三课练习：下列说法正确的是．请逐条判断并说明理由，再完成课本上的例题。"
    leak = "期末复习第5题：" + record["question"] + "\n答案：B"

    bare = tmp_path / "bare.jsonl"
    write_jsonl(str(bare), [dict(record, question=row["problem"])])
    assert match_record({"text": page}, load_eval_index([str(bare)])) is not None

    path = tmp_path / "tal.jsonl"
    write_jsonl(str(path), [record])
    index = load_eval_index([str(path)])
    assert match_record({"text": page}, index) is None
    assert match_record({"text": leak}, index) is not None


def test_mmlu_resolves_the_answer_index():
    record = convert_mmlu(MMLU_ROW)
    assert record["answer"] == "Paris"
    assert record["choices"][0] == "Berlin"


@pytest.mark.parametrize("bad", [{"choices": [], "answer": 0}, {"choices": ["a"], "answer": 5},
                                 {"choices": ["a"], "answer": None}])
def test_mmlu_rejects_out_of_range_indices(bad):
    assert convert_mmlu({"question": "q", **bad}) is None


def test_big_math_keeps_the_solve_rate():
    """The per-problem pass rate is what makes a difficulty curriculum possible:
    a group where every rollout fails has no reward variance and no gradient."""
    record = convert_big_math(BIG_MATH_ROW)
    assert record["solve_rate"] == pytest.approx(0.375)
    assert record["answer"] == "1"


def test_humaneval_keeps_what_a_code_environment_needs_to_run_it():
    record = convert_humaneval(HUMANEVAL_ROW)
    assert record["question"].startswith("from typing import List")
    assert record["answer"].lstrip().startswith("for idx, elem in enumerate(numbers):")
    assert "def check(candidate):" in record["test"]
    assert record["entry_point"] == "has_close_elements"


def test_mbpp_normalizes_line_endings_and_keeps_its_tests():
    record = convert_mbpp(MBPP_ROW)
    assert "\r" not in record["answer"]
    assert record["answer"].startswith("def remove_Occ(s,ch):")
    assert record["test_list"] == MBPP_ROW["test_list"]


@pytest.mark.parametrize("missing", ["text", "code", "test_list"])
def test_mbpp_row_missing_a_field_is_skipped(missing):
    assert convert_mbpp({k: v for k, v in MBPP_ROW.items() if k != missing}) is None


def test_short_reference_code_only_matches_exactly(tmp_path):
    """A one-line body is an idiom, not a benchmark fingerprint; a long body is."""
    path = tmp_path / "humaneval.jsonl"
    write_jsonl(str(path), [convert_humaneval(HUMANEVAL_IDIOM_ROW), convert_humaneval(HUMANEVAL_ROW)])
    index = load_eval_index([str(path)])

    idiom = "def keep(strings, substring):\n    return [x for x in strings if substring in x]\n"
    leak = "import math\n\n\ndef close(numbers, threshold):\n" + HUMANEVAL_ROW["canonical_solution"]
    assert match_record({"text": idiom}, index) is None
    assert match_record({"text": leak}, index) is not None


@pytest.mark.parametrize(
    "converter,row",
    [
        (convert_gsm8k, GSM8K_ROW),
        (convert_math500, MATH500_ROW),
        (convert_tal_scq5k, TAL_ROW),
        (convert_mmlu, MMLU_ROW),
        (convert_big_math, BIG_MATH_ROW),
        (convert_humaneval, HUMANEVAL_ROW),
        (convert_mbpp, MBPP_ROW),
    ],
)
def test_every_converter_emits_a_valid_task_record(converter, row):
    """Output must load through envs.base.load_tasks and datatools.records alike."""
    record = converter(row)
    assert detect_schema(record) == "task"
    assert validate(record) == []


def test_solution_is_indexed_for_decontamination():
    """A page quoting only the worked solution is still contamination."""
    record = convert_gsm8k(GSM8K_ROW)
    parts = record_parts(record)
    assert record["question"] in parts
    assert record["answer"] in parts
    assert record["solution"] in parts


def test_registry_covers_the_decontamination_sets():
    assert set(DECONTAMINATION_SETS) <= set(EVAL_SOURCES)
    assert "big_math" not in DECONTAMINATION_SETS  # an RL prompt pool, not an eval set
    assert EVAL_SOURCES["big_math"].gated
    for source in EVAL_SOURCES.values():
        assert "path" in source.hf
        assert source.note


def test_update_spec_points_decontamination_at_the_files(tmp_path):
    spec_path = tmp_path / "mix.json"
    spec_path.write_text(json.dumps({"name": "t", "sources": []}), "utf-8")

    merged = update_spec_decontamination(str(spec_path), ["a/gsm8k.jsonl", "a/math500.jsonl"])

    written = json.loads(spec_path.read_text("utf-8"))
    assert merged == ["a/gsm8k.jsonl", "a/math500.jsonl"]
    assert written["decontaminate"]["against"] == merged
    assert written["decontaminate"]["n"] == 13
    assert written["decontaminate"]["lcs_threshold"] == 0.6


def test_update_spec_merges_without_duplicating(tmp_path):
    spec_path = tmp_path / "mix.json"
    spec_path.write_text(json.dumps({"decontaminate": {"against": ["a/gsm8k.jsonl"], "n": 13}}), "utf-8")

    merged = update_spec_decontamination(str(spec_path), ["a/gsm8k.jsonl", "a/mmlu.jsonl"])

    assert merged == ["a/gsm8k.jsonl", "a/mmlu.jsonl"]


def test_render_surfaces_failures_and_hints():
    reports = [
        FetchReport(name="gsm8k", written=1319, skipped=0, path="datasets/eval/gsm8k.jsonl"),
        FetchReport(name="big_math", error="GatedRepoError: access required",
                    notes=["dataset is gated; accept its terms and set HF_TOKEN"]),
    ]
    text = render(reports)
    assert "1319" in text
    assert "FAILED" in text
    assert "HF_TOKEN" in text
