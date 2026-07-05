# /// script
# requires-python = ">=3.11"
# dependencies = ["phonology_engine"]
# ///
"""Guesser tier: LIEPA accentuation for words no open source covers.

The BSD-licensed phonology_engine (native accentuation components of the
LIEPA speech synthesizer) answers arbitrary words. Benchmarked against the
VDU cache it agrees 87.9% exact-variant / 95.3% stress-position on exactly
the dictionary-gap words — good enough for a clearly-labelled lowest-
confidence tier, not good enough to enter the main artifact (which holds a
zero-disagreement gate).

Writes guesses.sqlite in the words schema with `liepa-guess` provenance for
every wordlist/VDU-key word the main artifact does not cover.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

try:  # pragma: no cover
    from ._common import (
        DATA_DIR,
        DEFAULT_GENERATED,
        DEFAULT_VDU_SQLITE,
        normalize_lt,
        normalize_notation,
        strip_accents,
    )
except ImportError:  # pragma: no cover
    from _common import (
        DATA_DIR,
        DEFAULT_GENERATED,
        DEFAULT_VDU_SQLITE,
        normalize_lt,
        normalize_notation,
        strip_accents,
    )

MARKS = {"0": "̀", "1": "́", "2": "̃"}
DEFAULT_GUESSES = DATA_DIR / "guesses.sqlite"


def engine_accent(pe, word: str) -> str | None:
    try:
        raw = pe.process_and_collapse(word, "word_with_all_numeric_stresses")
    except Exception:
        return None
    out = [MARKS.get(ch, ch) for ch in raw.lower()]
    form = normalize_notation(normalize_lt(unicodedata.normalize("NFC", "".join(out))))
    return form if strip_accents(form) == word else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_GUESSES)
    parser.add_argument("--wordlist", type=Path, default=DATA_DIR / "lt_50k.txt")
    args = parser.parse_args(argv)

    from phonology_engine import PhonologyEngine

    covered = set()
    if DEFAULT_GENERATED.exists():
        db = sqlite3.connect(DEFAULT_GENERATED)
        covered = {w for (w,) in db.execute("SELECT word FROM words")}
    candidates: set[str] = set()
    if args.wordlist.exists():
        candidates |= {
            line.split()[0]
            for line in args.wordlist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if DEFAULT_VDU_SQLITE.exists():
        vdu = sqlite3.connect(DEFAULT_VDU_SQLITE)
        candidates |= {w for (w,) in vdu.execute("SELECT word FROM words")}
    todo = sorted(w for w in candidates if w not in covered and w.isalpha())

    pe = PhonologyEngine()
    now = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    out = sqlite3.connect(args.output)
    out.executescript(
        """
        CREATE TABLE words (
          word TEXT PRIMARY KEY, variants TEXT NOT NULL, fetched_at TEXT NOT NULL,
          negative_until TEXT, default_form TEXT, accent_type TEXT,
          default_form_title TEXT, accent_type_title TEXT, provenance TEXT NOT NULL
        );
        """
    )
    rows = []
    for word in todo:
        form = engine_accent(pe, word)
        if not form:
            continue
        variants = json.dumps(
            [{"form": form, "info": "spėjimas", "mi": []}],
            ensure_ascii=False, separators=(",", ":"),
        )
        rows.append((word, variants, now, None, form, "ONE",
                     form[:1].upper() + form[1:], "ONE",
                     f"open-accentuator:liepa-guess:{word}"))
    out.executemany("INSERT OR IGNORE INTO words VALUES (?,?,?,?,?,?,?,?,?)", rows)
    out.commit()
    print(f"guessed {len(rows):,} of {len(todo):,} uncovered candidates -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
