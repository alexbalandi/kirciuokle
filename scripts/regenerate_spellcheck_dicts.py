# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Regenerate the client-side spellcheck dictionaries (BUILD ARTIFACTS).

The two files this produces are generated data, NOT source — like the ONNX model,
they are gitignored and must be regenerated locally before a build/deploy:

    uv run scripts/regenerate_spellcheck_dicts.py

Outputs (both under public/, copied into the site by `vite build`):
  * spellcheck-lt.txt      one line per surface form  "<form>\t<freq>"; the union
                           of lexicon.sqlite + generated.sqlite (~580k forms), with
                           corpus frequency from the hermitdave 2018 LT list.
  * spellcheck-bigrams.txt one line per adjacent word pair "<w1>\t<w2>\t<count>"
                           from the local corpora — used to break ties between
                           spelling candidates.

Requires the (gitignored) source data:
  local/accentuator/data/{lexicon.sqlite, generated.sqlite, eval/*.txt}
and network access (fetches the frequency list). `npm run build` fails fast if the
outputs are missing, pointing back here.
"""

from __future__ import annotations

import gzip
import re
import sqlite3
from collections import Counter
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "local" / "accentuator" / "data"
LEXICON = DATA_DIR / "lexicon.sqlite"
GENERATED = DATA_DIR / "generated.sqlite"
CORPUS_DIR = DATA_DIR / "eval"
CORPUS_FILES = [
    "literary-corpus.txt",
    "literary-corpus-2.txt",
    "lrt-corpus.txt",
    "chrestomatija-plain.txt",
]
WORDLIST_OUT = REPO_ROOT / "public" / "spellcheck-lt.txt"
BIGRAMS_OUT = REPO_ROOT / "public" / "spellcheck-bigrams.txt"
FREQ_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/"
    "content/2018/lt/lt_full.txt"
)
MIN_BIGRAM_COUNT = 2  # drop pairs seen once — noise, no ranking value
MAX_BIGRAMS = 220_000  # cap the shipped table
TOKEN = re.compile(r"[a-ząčęėįšųūž]+", re.IGNORECASE)

LT_LETTERS = set("aąbcčdeęėfghiįyjklmnoprsštuųūvzž")
LT_LETTERS |= {c.upper() for c in LT_LETTERS}


def is_clean_form(form: str) -> bool:
    """Real surface forms: LT/Latin letters only, length 2-32, an internal hyphen
    allowed (compounds) but not leading/trailing (Wiktionary affixes)."""
    if not form or len(form) < 2 or len(form) > 32:
        return False
    if form[0] == "-" or form[-1] == "-":
        return False
    return all(ch == "-" or ch in LT_LETTERS for ch in form)


def load_frequencies() -> dict[str, int]:
    """Corpus frequency per lowercased surface form (hermitdave 2018 lt_full)."""
    freq: dict[str, int] = {}
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        text = client.get(FREQ_URL).text
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            freq[parts[0].lower()] = int(parts[1])
    return freq


def build_wordlist() -> set[str]:
    """Write spellcheck-lt.txt and return the lowercased vocab (for the bigrams)."""
    if not LEXICON.exists():
        raise SystemExit(f"lexicon not found: {LEXICON}")
    if not GENERATED.exists():
        raise SystemExit(f"generated db not found: {GENERATED}")

    raw: set[str] = set()
    lexicon = sqlite3.connect(LEXICON)
    raw.update(
        row[0]
        for row in lexicon.execute(
            "select distinct stripped from forms where stripped is not null"
        )
    )
    lexicon.close()
    lexicon_count = len(raw)

    generated = sqlite3.connect(GENERATED)
    raw.update(row[0] for row in generated.execute("select word from words"))
    generated.close()

    freq = load_frequencies()
    forms = sorted(f for f in raw if is_clean_form(f))
    blob = "".join(f"{f}\t{freq.get(f.lower(), 0)}\n" for f in forms)
    WORDLIST_OUT.parent.mkdir(parents=True, exist_ok=True)
    WORDLIST_OUT.write_text(blob, encoding="utf-8")

    with_freq = sum(1 for f in forms if freq.get(f.lower(), 0) > 0)
    gz = len(gzip.compress(blob.encode("utf-8"), 9))
    print(f"lexicon forms: {lexicon_count} | union: {len(raw)} | kept: {len(forms)}")
    print(f"  with frequency: {with_freq}")
    print(f"  wrote {WORDLIST_OUT.relative_to(REPO_ROOT)}: ~{gz} bytes gzipped")
    return {f.lower() for f in forms}


def build_bigrams(vocab: set[str]) -> None:
    """Write spellcheck-bigrams.txt from the local corpora, both words in vocab."""
    pairs: Counter[tuple[str, str]] = Counter()
    total_tokens = 0
    for name in CORPUS_FILES:
        path = CORPUS_DIR / name
        if not path.exists():
            print(f"  skip (missing corpus): {name}")
            continue
        tokens = TOKEN.findall(path.read_text(encoding="utf-8", errors="ignore").lower())
        total_tokens += len(tokens)
        prev: str | None = None
        for tok in tokens:
            in_vocab = tok in vocab
            if prev is not None and in_vocab:
                pairs[(prev, tok)] += 1
            prev = tok if in_vocab else None

    kept = [(w1, w2, c) for (w1, w2), c in pairs.items() if c >= MIN_BIGRAM_COUNT]
    kept.sort(key=lambda r: r[2], reverse=True)
    kept = kept[:MAX_BIGRAMS]

    blob = "".join(f"{w1}\t{w2}\t{c}\n" for w1, w2, c in kept)
    BIGRAMS_OUT.write_text(blob, encoding="utf-8")
    gz = len(gzip.compress(blob.encode("utf-8"), 9))
    print(f"corpus tokens: {total_tokens} | distinct pairs: {len(pairs)} | kept: {len(kept)}")
    print(f"  wrote {BIGRAMS_OUT.relative_to(REPO_ROOT)}: ~{gz} bytes gzipped")


def main() -> int:
    vocab = build_wordlist()
    build_bigrams(vocab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
