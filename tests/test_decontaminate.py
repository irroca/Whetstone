import pytest

from datatools.decontaminate import (
    EvalIndex,
    build_eval_index,
    decontaminate,
    lcs_length,
    lcs_ratio,
    load_eval_index,
    ngrams,
    render,
    text_units,
)
from datatools.records import write_jsonl

GSM_STYLE = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as "
    "many clips in May. How many clips did Natalia sell altogether in April and May?"
)


def test_text_units_splits_cjk_per_character_and_latin_per_word():
    assert text_units("solve 12 + 7 计算结果") == [
        "solve", "12", "7", "计", "算", "结", "果",
    ]


def test_punctuation_and_symbols_are_not_units():
    """Each one used to be a unit, so a run of 13 dashes was a "phrase"."""
    assert text_units("|------|____|...|~~~~|，。！") == []
    assert text_units(r"\frac{3}{2}") == ["frac", "3", "2"]
    assert text_units("grpo_advantages café") == ["grpo", "advantages", "café"]


def test_ngrams_are_thirteen_units_long_by_default():
    grams = ngrams(GSM_STYLE)
    assert all(len(g) == 13 for g in grams)
    assert len(grams) > 1


def test_short_text_yields_a_single_short_gram():
    """Benchmark answers are often a couple of tokens; they still need indexing."""
    grams = ngrams("答案是 42")
    assert len(grams) == 1
    assert len(next(iter(grams))) < 13


def test_ngrams_are_case_and_whitespace_insensitive():
    assert ngrams("Hello   World Foo") == ngrams("hello world foo")


def test_ngrams_reject_invalid_n():
    with pytest.raises(ValueError):
        ngrams("abc", n=0)


def test_lcs_length_basics():
    assert lcs_length(list("abcde"), list("ace")) == 3
    assert lcs_length([], list("abc")) == 0


def test_lcs_ratio_normalizes_by_the_shorter_text():
    """A benchmark question buried in a long page must still score high."""
    question = "what is two plus two"
    page = "Here is some filler. " + question + " And a lot more filler text follows."
    assert lcs_ratio(question, page) == pytest.approx(1.0)


def test_lcs_ratio_of_unrelated_texts_is_low():
    assert lcs_ratio("完全不相关的一段中文", "an entirely unrelated english sentence") < 0.3


def test_index_matches_an_embedded_benchmark_question():
    index = build_eval_index([GSM_STYLE])
    page = "Math homework help!\n\n" + GSM_STYLE + "\n\nAnswer: 72 clips."
    assert index.match(page) is not None


def test_short_eval_item_still_matches_inside_a_longer_page():
    """Indexed at its own length, not as one whole-item gram: a 10-word question
    would otherwise never match a page that only produces 13-unit windows."""
    question = "what is the average airspeed velocity of an unladen swallow"
    index = build_eval_index([question])

    assert len(text_units(question)) < 13
    assert index.gram_sizes == {len(text_units(question))}
    assert index.match(f"Trivia night! {question}, anyway? Nobody agrees on this.") is not None


def test_blank_lines_and_table_rules_are_not_contamination():
    """An MMLU fill-in-the-blank question matched every markdown table in the
    smoke corpus through its run of underscores and dashes."""
    question = f"Fill in the blank: {'_' * 20} is the {'-' * 20} capital of France."
    index = build_eval_index([question])
    table = f"| city | country |\n|{'-' * 20}|{'-' * 20}|\n| {'_' * 20} | {'_' * 20} |"
    assert index.match(table) is None
    assert index.match("Quiz: " + question) is not None


def test_shared_latex_syntax_is_not_contamination():
    """Two formulas with the same brace structure used to share 13 units."""
    index = build_eval_index([r"so \left( \frac{1}{2} \right) + \left( \frac{1}{3} \right) holds"])
    page = r"we compute \left( \frac{1}{2} \right) + \left( \frac{1}{5} \right) here"
    assert index.match(page) is None


def test_generic_short_questions_only_match_exactly():
    """Below MIN_GRAM units a question is too generic to search for as a phrase."""
    index = build_eval_index(["What is the value of B?"])
    assert index.match("In this puzzle, what is the value of B? Nobody knows.") is None
    assert index.match("What is the value of B?") is not None


def test_index_does_not_match_unrelated_text():
    index = build_eval_index([GSM_STYLE])
    assert index.match("The capital of France is Paris, a city on the Seine.") is None


def test_index_matches_chinese_by_character_ngrams():
    """A 13-word gram does not exist in Chinese; 13 characters does."""
    question = "一个水池有两个进水管，甲管单独注满需要六小时，乙管单独注满需要四小时。"
    index = build_eval_index([question])

    assert index.match("练习题：" + question + " 求两管同时开需要多久？") is not None
    assert index.match("今天天气很好，我们一起去公园里散步吧，路上买点水果。") is None


def test_decontaminate_removes_contaminated_records():
    index = build_eval_index([GSM_STYLE])
    records = [
        {"text": "Clean document about gardening and soil pH levels."},
        {"text": "Solutions manual: " + GSM_STYLE + " The answer is 72."},
        {"text": "Another clean document about bicycle maintenance."},
    ]

    kept, report = decontaminate(records, index)

    assert len(kept) == 2
    assert report.removed == 1
    assert report.ngram_hits == 1
    assert report.kept == 2
    assert report.examples[0]["eval"].startswith("Natalia sold clips")


def test_lcs_threshold_rescues_coincidental_ngram_hits():
    """A shared 13-gram of boilerplate is not contamination. The eval item here
    is mostly its own content, so the shared stretch covers well under 60% of
    it: the LCS gate keeps the record, plain n-gram matching would drop it."""
    shared = "this is a long stretch of perfectly ordinary boilerplate text that appears everywhere"
    unique_eval = " ".join(f"evalword{i}" for i in range(200))
    index = build_eval_index([shared + " " + unique_eval])
    records = [{"text": shared + " " + " ".join(f"filler{i}" for i in range(200))}]

    kept_strict, strict = decontaminate(records, index)
    kept_gated, gated = decontaminate(records, index, lcs_threshold=0.6)

    assert strict.removed == 1 and kept_strict == []
    assert gated.removed == 0 and gated.lcs_rescued == 1
    assert len(kept_gated) == 1


def test_lcs_threshold_still_removes_real_contamination():
    index = build_eval_index([GSM_STYLE])
    records = [{"text": GSM_STYLE + " Answer: 72."}]

    kept, report = decontaminate(records, index, lcs_threshold=0.6)

    assert kept == []
    assert report.removed == 1 and report.lcs_rescued == 0


def test_lcs_gate_sees_a_leak_deep_inside_a_long_document():
    """Books run to tens of thousands of units; the leak can sit anywhere."""
    solution = " ".join(f"step{i}" for i in range(150))
    book = " ".join(f"chapter{i}" for i in range(2500)) + " " + solution + " the end"
    index = build_eval_index([solution])

    assert lcs_ratio(book, solution) == pytest.approx(1.0)
    kept, report = decontaminate([{"text": book}], index, lcs_threshold=0.6)
    assert kept == [] and report.removed == 1


def test_lcs_gate_ignores_item_words_scattered_across_a_page():
    """A shared phrase plus the item's other words strewn far apart down the
    page is a coincidence, not a copy: only the aligned stretch counts."""
    shared = "the area of the triangle with vertices x1 y1 x2 y2 x3 y3 is"
    rest = [f"term{i}" for i in range(20)]
    item = shared + " " + " ".join(rest)
    filler = " ".join(f"note{i}" for i in range(50))
    page = filler + " " + shared + " " + " ".join(f"{filler} {word}" for word in rest)
    index = build_eval_index([item])

    assert index.match(page) is not None
    assert lcs_ratio(page, item) < 0.6
    kept, report = decontaminate([{"text": page}], index, lcs_threshold=0.6)
    assert report.lcs_rescued == 1 and len(kept) == 1


def test_lcs_gate_sees_a_leak_deep_inside_a_long_document():
    """Books run to tens of thousands of units; the leak can sit anywhere."""
    solution = " ".join(f"step{i}" for i in range(150))
    book = " ".join(f"chapter{i}" for i in range(2500)) + " " + solution + " the end"
    index = build_eval_index([solution])

    assert lcs_ratio(book, solution) == pytest.approx(1.0)
    kept, report = decontaminate([{"text": book}], index, lcs_threshold=0.6)
    assert kept == [] and report.removed == 1


def test_lcs_gate_ignores_item_words_scattered_across_a_page():
    """A shared phrase plus the item's other words strewn far apart down the
    page is a coincidence, not a copy: only the aligned stretch counts."""
    shared = "the area of the triangle with vertices x1 y1 x2 y2 x3 y3 is"
    rest = [f"term{i}" for i in range(20)]
    item = shared + " " + " ".join(rest)
    filler = " ".join(f"note{i}" for i in range(50))
    page = filler + " " + shared + " " + " ".join(f"{filler} {word}" for word in rest)
    index = build_eval_index([item])

    assert index.match(page) is not None
    assert lcs_ratio(page, item) < 0.6
    kept, report = decontaminate([{"text": page}], index, lcs_threshold=0.6)
    assert report.lcs_rescued == 1 and len(kept) == 1


def test_decontaminate_covers_every_schema():
    question = "一个长方形的周长是三十六厘米，长比宽多两厘米，求这个长方形的面积。"
    index = build_eval_index([question])
    records = [
        {"question": question, "answer": "80 平方厘米"},
        {"question": "完全不同的另一道题：甲乙两车相向而行，求相遇时间。", "answer": "3 小时"},
    ]

    kept, report = decontaminate(records, index)

    assert len(kept) == 1 and kept[0]["answer"] == "3 小时"
    assert report.removed == 1


def test_decontaminate_catches_a_leaked_assistant_turn():
    """The conversation as a whole is a different string, but the assistant turn
    reproducing a benchmark answer is still contamination."""
    answer = "先求出长和宽分别是十厘米和八厘米，所以面积是八十平方厘米。"
    index = build_eval_index([answer])
    records = [{"conversations": [
        {"role": "user", "content": "帮我做一道几何题"},
        {"role": "assistant", "content": answer},
    ]}]

    kept, report = decontaminate(records, index)

    assert kept == [] and report.removed == 1


def test_empty_index_removes_nothing():
    records = [{"text": "anything at all"}]
    kept, report = decontaminate(records, EvalIndex())
    assert kept == records and report.removed == 0


def test_load_eval_index_indexes_each_field_separately(tmp_path):
    """Indexing the joined record would insert separators that never occur in
    natural text, so a page quoting only the question would slip through."""
    path = tmp_path / "eval.jsonl"
    write_jsonl(str(path), [
        {"question": "what is the average airspeed velocity of an unladen swallow", "answer": "11 m/s"}
    ])

    index = load_eval_index([str(path)])

    assert len(index) == 2  # question and answer are separate items
    quoted = "Trivia night! What is the average airspeed velocity of an unladen swallow, anyway?"
    assert index.match(quoted) is not None


def test_very_short_eval_answers_give_weak_protection(tmp_path):
    """Known limitation: a short answer becomes one short n-gram, which only
    matches a training part that is itself that string. Matching a bare "42"
    anywhere would delete the corpus, so the question carries the protection."""
    path = tmp_path / "eval.jsonl"
    write_jsonl(str(path), [{"question": "a sufficiently long and distinctive benchmark question here", "answer": "42"}])

    index = load_eval_index([str(path)])

    assert index.match("the answer to everything is 42, as everyone knows") is None
    assert index.match("a sufficiently long and distinctive benchmark question here") is not None


def test_short_answers_match_exactly_but_long_ones_as_phrases(tmp_path):
    """Short answers were most of the false positives on real data (MMLU's
    "1,2,3" matched 76 of 4.8k documents). A long answer is still distinctive."""
    long_answer = "petition the government for a redress of grievances and assemble peaceably in public"
    path = tmp_path / "eval.jsonl"
    write_jsonl(str(path), [
        {"question": "a sufficiently long and distinctive benchmark question here", "answer": "1,2,3"},
        {"question": "another sufficiently long and distinctive benchmark question", "answer": "The end of the Cold War"},
        {"question": "a third sufficiently long and distinctive benchmark question", "answer": long_answer},
    ])

    index = load_eval_index([str(path)])

    assert index.match("Steps 1,2,3 of the recipe are easy.") is None
    assert index.match("This essay is about the end of the Cold War and what followed.") is None
    assert index.match("The end of the Cold War") is not None
    assert index.match(f"The First Amendment lets people {long_answer}, it says.") is not None


def test_load_eval_index_can_target_one_field(tmp_path):
    path = tmp_path / "eval.jsonl"
    write_jsonl(str(path), [{"question": "unique question phrase here", "answer": "secret answer"}])

    index = load_eval_index([str(path)], field_name="question")

    assert index.match("secret answer") is None
    assert index.match("unique question phrase here") is not None


def test_render_reports_the_gate(tmp_path):
    index = build_eval_index([GSM_STYLE])
    _, report = decontaminate([{"text": GSM_STYLE}], index, lcs_threshold=0.6)
    text = render(report)
    assert "13-gram" in text
    assert "rescued by LCS < 0.6" in text
    assert "removed: 1" in text
