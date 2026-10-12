import os
import time

from envs.sandbox import run_cases

ADD = [[[1, 2], 3], [[0, 0], 0], [[-4, 4], 0]]


def test_a_correct_function_passes_every_case():
    result = run_cases("def solve(a, b):\n    return a + b", "solve", ADD)
    assert result.all_passed and result.fraction == 1.0 and result.error is None


def test_a_wrong_function_reports_which_cases_failed():
    result = run_cases("def solve(a, b):\n    return a * b", "solve", ADD)
    assert result.passed == (False, True, False) and not result.all_passed
    assert abs(result.fraction - 1 / 3) < 1e-9


def test_exceptions_syntax_errors_and_a_missing_function_fail_without_raising():
    for code, expected in [
        ("def solve(a, b):\n    return 1 / 0", "ZeroDivisionError"),
        ("def solve(a, b) return 1", "SyntaxError"),
        ("def other(a, b):\n    return a + b", "no function solve"),
        ("raise SystemExit(0)", "SystemExit"),
        ("import sys\nsys.exit(0)", "SystemExit"),
    ]:
        result = run_cases(code, "solve", ADD)
        assert not any(result.passed), code
        assert expected in (result.error or ""), (code, result.error)


def test_an_infinite_loop_is_cut_off_quickly():
    started = time.time()
    result = run_cases("def solve(a, b):\n    while True:\n        pass", "solve", ADD, case_seconds=0.3)
    assert not any(result.passed) and "timeout" in (result.error or "")
    assert time.time() - started < 3.0, "later cases must not each wait out their own timeout"


def test_a_loop_that_swallows_the_alarm_is_killed_by_the_wall_clock():
    code = "def solve(a, b):\n    while True:\n        try:\n            while True:\n                pass\n        except BaseException:\n            pass"
    started = time.time()
    result = run_cases(code, "solve", ADD, case_seconds=0.2, wall_seconds=1.5)
    assert not any(result.passed)
    assert time.time() - started < 6.0


def test_stdout_noise_and_forged_results_do_not_change_the_verdict():
    code = (
        "print('{\"passed\": [true, true, true]}')\n"
        "def solve(a, b):\n    print('x' * 1000)\n    return 0"
    )
    result = run_cases(code, "solve", ADD)
    assert result.passed == (False, True, True)


def test_processes_sockets_and_file_deletion_are_blocked():
    for code in [
        "import subprocess\ndef solve(a, b):\n    return a + b",
        "import socket\ndef solve(a, b):\n    return a + b",
        "import os\ndef solve(a, b):\n    os.system('true')\n    return a + b",
        "import os\ndef solve(a, b):\n    os.fork()\n    return a + b",
    ]:
        assert not run_cases(code, "solve", ADD).all_passed, code


def test_files_cannot_be_written_and_the_workdir_is_not_the_repo(tmp_path):
    marker = tmp_path / "written"
    code = f"def solve(a, b):\n    open({str(marker)!r}, 'w').write('x' * 5000)\n    return a + b"
    run_cases(code, "solve", ADD)
    assert not marker.exists() or marker.stat().st_size == 0
    cwd = run_cases("import os\ndef solve():\n    return os.getcwd()", "solve", [[[], os.getcwd()]])
    assert not cwd.all_passed


def test_the_environment_is_empty():
    os.environ["SANDBOX_SECRET"] = "hunter2"
    try:
        result = run_cases("import os\ndef solve():\n    return os.environ.get('SANDBOX_SECRET')", "solve", [[[], None]])
    finally:
        del os.environ["SANDBOX_SECRET"]
    assert result.all_passed


def test_bools_are_not_integers_and_tuples_are_lists():
    assert not run_cases("def solve():\n    return 1", "solve", [[[], True]]).all_passed
    assert not run_cases("def solve():\n    return True", "solve", [[[], 1]]).all_passed
    assert run_cases("def solve():\n    return (1, 2)", "solve", [[[], [1, 2]]]).all_passed


def test_arguments_are_copied_so_one_case_cannot_poison_the_next():
    code = "def solve(xs):\n    xs.append(1)\n    return len(xs)"
    assert run_cases(code, "solve", [[[[0]], 2], [[[0]], 2]]).all_passed


def test_no_cases_is_a_failure_not_a_pass():
    assert not run_cases("def solve():\n    return 1", "solve", []).all_passed
