"""Run model-written Python against test cases in a throwaway subprocess.

A code environment's reward is whether the program works, so the program has to run. The model's
output is arbitrary text, and the process that executes it must not be the trainer:

* **separate process, own session**: an infinite loop or a crash costs one child; a wall-clock
  timeout kills the whole process group
* **CPU and file-size limits** set inside the child (``RLIMIT_CPU``, ``RLIMIT_FSIZE`` = 0), plus an
  address-space cap where the OS honours it (Linux; macOS ignores ``RLIMIT_AS``)
* **per-case alarm**, so one slow case does not eat the others' budget
* **no way to start processes or open sockets**: ``subprocess``, ``socket``, ``ctypes`` and friends
  are made unimportable and ``os.system`` / ``os.fork`` / ``os.exec*`` raise
* empty environment, a temporary working directory, ``python -I`` (no user site, no env vars)

**This is not a security boundary.** Code that is determined to get out (introspecting the harness
frame, re-importing a blocked module through ``importlib`` internals) can. It is enough for a
100M-parameter policy sampling at temperature 1, which produces bugs, not exploits. Run the
trainer in a disposable container if the policy or the task source is ever untrusted.

Cases are ``[args, expected]`` pairs of JSON values; the function is called as ``entry(*args)``.
"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Runs in the child. It reads {"code", "entry", "cases", "nonce", "case_seconds", "cpu_seconds"} from
# stdin, so the candidate source never goes through a shell or argv.
_HARNESS = r'''
import copy, io, json, os, signal, sys

def main():
    spec = json.loads(sys.stdin.read())
    real_out = sys.stdout
    nonce, cases = spec["nonce"], spec["cases"]
    n = len(cases)
    try:
        import resource
        cpu = int(spec["cpu_seconds"]) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        if sys.platform.startswith("linux"):
            limit = 1 << 30
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception:
        pass

    def finish(passed, error=None):
        real_out.write(nonce + json.dumps({"passed": passed, "error": error}) + "\n")
        real_out.flush()
        os._exit(0)

    def refuse(*_a, **_k):
        raise PermissionError("blocked in the sandbox")

    for name in ("subprocess", "socket", "ctypes", "multiprocessing", "shutil", "urllib", "http", "ftplib",
                 "smtplib", "telnetlib", "pty", "pdb", "importlib.machinery"):
        sys.modules[name] = None
    for name in ("system", "popen", "fork", "forkpty", "execv", "execve", "execvp", "execl", "execle",
                 "execlp", "spawnv", "spawnl", "kill", "killpg", "remove", "unlink", "rmdir", "rename"):
        if hasattr(os, name):
            setattr(os, name, refuse)

    class CaseTimeout(BaseException):
        pass

    def on_alarm(_s, _f):
        raise CaseTimeout()

    signal.signal(signal.SIGALRM, on_alarm)
    namespace = {"__name__": "candidate"}
    sys.stdout = io.StringIO()
    try:
        signal.setitimer(signal.ITIMER_REAL, spec["case_seconds"])
        exec(compile(spec["code"], "<candidate>", "exec"), namespace)
        signal.setitimer(signal.ITIMER_REAL, 0)
    except CaseTimeout:
        sys.stdout = real_out
        finish([False] * n, "timeout while defining")
    except BaseException as exc:
        sys.stdout = real_out
        finish([False] * n, f"{type(exc).__name__}: {exc}"[:200])
    function = namespace.get(spec["entry"])
    if not callable(function):
        sys.stdout = real_out
        finish([False] * n, f"no function {spec['entry']}")

    passed, error = [], None
    for args, expected in cases:
        if error and error.startswith("timeout"):
            passed.append(False)
            continue
        try:
            signal.setitimer(signal.ITIMER_REAL, spec["case_seconds"])
            got = function(*copy.deepcopy(args))
            signal.setitimer(signal.ITIMER_REAL, 0)
            got = json.loads(json.dumps(got))
            ok = got == expected and isinstance(got, bool) == isinstance(expected, bool)
        except CaseTimeout:
            ok, error = False, "timeout"
        except BaseException as exc:
            signal.setitimer(signal.ITIMER_REAL, 0)
            ok, error = False, error or f"{type(exc).__name__}: {exc}"[:200]
        passed.append(bool(ok))
    sys.stdout = real_out
    finish(passed, error)

main()
'''


@dataclass(frozen=True)
class RunResult:
    """``passed[i]`` is whether case ``i`` matched. ``error`` is the first failure's reason, if any."""

    passed: tuple[bool, ...]
    error: str | None = None

    @property
    def all_passed(self) -> bool:
        return bool(self.passed) and all(self.passed)

    @property
    def fraction(self) -> float:
        return sum(self.passed) / len(self.passed) if self.passed else 0.0


def run_cases(
    code: str, entry: str, cases: list, *, case_seconds: float = 0.5, wall_seconds: float = 4.0,
) -> RunResult:
    """Execute ``code`` and check ``entry(*args) == expected`` for each ``[args, expected]`` case."""
    failed = RunResult(tuple(False for _ in cases))
    if not cases:
        return failed
    nonce = secrets.token_hex(8)
    payload = json.dumps({
        "code": code, "entry": entry, "cases": cases, "nonce": nonce,
        "case_seconds": case_seconds, "cpu_seconds": wall_seconds,
    })
    with tempfile.TemporaryDirectory(prefix="sandbox_") as workdir:
        try:
            proc = subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", _HARNESS], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, cwd=workdir, env={}, start_new_session=True, text=True,
            )
        except OSError as exc:
            return RunResult(failed.passed, f"could not start: {exc}")
        try:
            out, _ = proc.communicate(payload, timeout=wall_seconds)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            proc.communicate()
            return RunResult(failed.passed, "timeout")
        finally:
            _kill_group(proc)
    for line in reversed(out.splitlines()):
        if line.startswith(nonce):
            try:
                result = json.loads(line[len(nonce):])
                passed = tuple(bool(x) for x in result["passed"])
                if len(passed) == len(cases):
                    return RunResult(passed, result.get("error"))
            except (ValueError, KeyError, TypeError):
                break
    return RunResult(failed.passed, f"crashed (exit {proc.returncode})")


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, 9)
    except (ProcessLookupError, PermissionError, OSError):
        pass
