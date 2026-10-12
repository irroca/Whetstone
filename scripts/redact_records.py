"""Redact run records written on a machine that must not be named publicly, in place.

Replaces this machine's hostname with ``redacted`` and rewrites absolute paths under the repository
root as relative ones, in every text file below the given directories (``meta.json``, logs); nothing
else changes. Exits 1 if an identifier is still found afterwards, so it can gate a commit::

    python3 scripts/redact_records.py records/posttrain_v2 [--also WORD ...]
"""

from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def redact_text(text: str, hostname: str, root: str, also: tuple[str, ...] = ()) -> str:
    text = text.replace(root.rstrip("/") + "/", "").replace(root.rstrip("/"), ".")
    for word in (hostname, *also):
        if word:
            text = text.replace(word, "redacted")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="+", type=Path)
    parser.add_argument("--also", nargs="*", default=[], help="further words to replace, e.g. a user name")
    args = parser.parse_args(argv)

    hostname = socket.gethostname()
    watch = [hostname, str(ROOT), "/Users/", *args.also]
    changed, leaks = 0, []
    for directory in args.dirs:
        for path in sorted(p for p in directory.rglob("*") if p.is_file()):
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            new = redact_text(text, hostname, str(ROOT), tuple(args.also))
            if new != text:
                path.write_text(new, encoding="utf-8")
                changed += 1
            leaks += [str(path) for word in watch if word and word in new]
    print(f"redacted {changed} files; identifiers left in {len(set(leaks))}: {sorted(set(leaks))[:5]}")
    return 1 if leaks else 0


if __name__ == "__main__":
    sys.exit(main())
