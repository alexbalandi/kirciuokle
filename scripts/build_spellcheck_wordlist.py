# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Build the client-side spellcheck wordlist from the project's lexicon.

Emits `public/spellcheck-lt.txt` — one valid Lithuanian surface form per line
(the un-stressed spelling, keeping LT diacritics). The browser folds these to
ASCII to restore missing diacritics and runs edit-distance-1 typo lookups; no
model, no network. Cloudflare compresses the .txt over the wire (~0.3 MB).

    uv run scripts/build_spellcheck_wordlist.py
"""

from __future__ import annotations

import gzip
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LEXICON = REPO_ROOT / "local" / "accentuator" / "data" / "lexicon.sqlite"
OUTPUT = REPO_ROOT / "public" / "spellcheck-lt.txt"

LT_LETTERS = set("aąbcčdeęėfghiįyjklmnoprsštuųūvzž")
LT_LETTERS |= {c.upper() for c in LT_LETTERS}


def is_clean_form(form: str) -> bool:
    """Keep real surface forms: LT/Latin letters only, length >= 2, an internal
    hyphen allowed (compounds) but not leading/trailing (Wiktionary affixes)."""
    if len(form) < 2 or len(form) > 32:
        return False
    if form[0] == "-" or form[-1] == "-":
        return False
    for ch in form:
        if ch == "-":
            continue
        if ch not in LT_LETTERS:
            return False
    return True


def main() -> int:
    if not LEXICON.exists():
        raise SystemExit(f"lexicon not found: {LEXICON}")
    connection = sqlite3.connect(LEXICON)
    raw = {
        row[0]
        for row in connection.execute(
            "select distinct stripped from forms where stripped is not null"
        )
    }
    connection.close()

    forms = sorted(f for f in raw if is_clean_form(f))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    blob = "\n".join(forms) + "\n"
    OUTPUT.write_text(blob, encoding="utf-8")

    raw_bytes = len(blob.encode("utf-8"))
    gz_bytes = len(gzip.compress(blob.encode("utf-8"), 9))
    print(f"forms in lexicon: {len(raw)}")
    print(f"kept (clean):     {len(forms)}")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)}: {raw_bytes} bytes "
          f"(~{gz_bytes} gzipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
