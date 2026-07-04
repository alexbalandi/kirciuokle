# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Deprecated wrapper for ALKSNIS-only dataset preparation."""

from __future__ import annotations

import sys
from typing import Iterable

from prep_corpus import main as prep_corpus_main


def main(argv: Iterable[str] | None = None) -> int:
    print(
        "prep_alksnis.py is deprecated; use prep_corpus.py --sources alksnis",
        file=sys.stderr,
    )
    forwarded = ["--sources", "alksnis"]
    if argv is not None:
        forwarded.extend(argv)
    else:
        forwarded.extend(sys.argv[1:])
    return prep_corpus_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
