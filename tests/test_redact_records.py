import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "redact_records", Path(__file__).resolve().parent.parent / "scripts" / "redact_records.py"
)
redact_records = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(redact_records)


def test_hostname_and_absolute_paths_are_replaced_and_the_rest_is_kept():
    text = '{"hostname": "box-17", "path": "/work/repo/results/x.pth", "root": "/work/repo", "lr": 1e-05}'
    out = redact_records.redact_text(text, "box-17", "/work/repo")
    assert out == '{"hostname": "redacted", "path": "results/x.pth", "root": ".", "lr": 1e-05}'


def test_extra_words_are_replaced_too():
    out = redact_records.redact_text("run by alice on box-17", "box-17", "/work/repo", also=("alice",))
    assert out == "run by redacted on redacted"
