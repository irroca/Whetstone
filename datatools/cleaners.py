"""Text rewrites applied to a source's documents before they are filtered.

Filters only accept or reject. A cleaner changes the text, so it runs first and
the filters, exact dedup and token counts all see the cleaned document. A source
opts in by name in the mixture spec (``"cleaners": ["starcoder_metadata"]``);
unknown names are rejected when the spec loads.
"""

from __future__ import annotations

import re
from typing import Callable, Sequence

# A marker inside the code (an argparse usage string '-f <filename>') is
# content, so only a first line made of nothing but metadata segments matches.
_STARCODER_HEADER = re.compile(r"\A(?:<(?:reponame|filename|gh_stars)>[^<\n]*)+\n")


def strip_starcoder_metadata(text: str) -> str:
    """Drop the ``<reponame>…<filename>…<gh_stars>…`` line StarCoder data starts with.

    StarCoder serialized repository metadata into the file content, each segment
    present at random. Our tokenizers have no special tokens for the markers, so
    they cost tokens, teach the model to emit them, and keep two forks of the
    same file from being exact duplicates.
    """
    return _STARCODER_HEADER.sub("", text, count=1)


CLEANERS: dict[str, Callable[[str], str]] = {
    "starcoder_metadata": strip_starcoder_metadata,
}


def clean(text: str, names: Sequence[str]) -> str:
    for name in names:
        text = CLEANERS[name](text)
    return text
