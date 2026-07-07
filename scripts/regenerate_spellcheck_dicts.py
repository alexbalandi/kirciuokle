# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx"]
# ///
"""Regenerate the client-side spellcheck dictionaries (BUILD ARTIFACTS).

The files this produces are generated data, NOT source — like the ONNX model,
they are gitignored and must be regenerated locally before a build/deploy:

    uv run scripts/regenerate_spellcheck_dicts.py

Outputs (all under public/, copied into the site by `vite build`):
  * lt.aff, lt.dic         the BSD-3 Lithuanian hunspell dictionary (fetched from
                           wooorm/dictionaries). This is the ACCEPT vocabulary —
                           real hunspell applies its affix rules in the browser, so
                           every valid inflected form is recognised.
  * spellcheck-lt.txt      one line per surface form "<form>\t<freq>"; the
                           freq-bearing subset (freq >= ACCEPT_MIN_FREQ, ~162k) of
                           lexicon.sqlite + generated.sqlite. This drives suggestion
                           generation + ranking only (NOT acceptance — hunspell does
                           that), so it can stay small.
  * spellcheck-bigrams.txt one line per adjacent word pair "<w1>\t<w2>\t<count>"
                           from the local corpora — used to break ties between
                           spelling candidates.

Requires the (gitignored) source data:
  local/accentuator/data/{lexicon.sqlite, generated.sqlite, eval/*.txt}
and network access (fetches the frequency list + hunspell dictionary). `npm run
build` fails fast if any output is missing, pointing back here.
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
DIC_OUT = REPO_ROOT / "public" / "lt.dic"
AFF_OUT = REPO_ROOT / "public" / "lt.aff"
FREQ_URL = (
    "https://raw.githubusercontent.com/hermitdave/FrequencyWords/master/"
    "content/2018/lt/lt_full.txt"
)
# BSD-3-Clause Lithuanian hunspell dictionary (ispell-lt, via wooorm/dictionaries) —
# the comprehensive ACCEPT vocabulary. Lemmas + affix rules; real hunspell
# (hunspell-asm, compiled to wasm) applies them in the browser. See ATTRIBUTIONS.
HUNSPELL_BASE = "https://raw.githubusercontent.com/wooorm/dictionaries/main/dictionaries/lt"
ACCEPT_MIN_FREQ = 2  # freq-list words seen ≥ this become accept-only vocabulary
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
    own_count = len(raw)

    freq = load_frequencies()
    raw.update(w for w, f in freq.items() if f >= ACCEPT_MIN_FREQ and is_clean_form(w))

    # ACCEPT is handled by the hunspell dictionary (comprehensive morphology); this
    # wordlist only powers the browser's restore/typo SUGGESTIONS + frequency
    # ranking, so keep just the frequency-bearing forms (common words and their
    # diacritic spellings — the useful restore targets). Diacritic-dropped spellings
    # like "as" are here too (they have frequency), and the browser restores them.
    forms = sorted(
        f
        for f in raw
        if is_clean_form(f) and freq.get(f.lower(), 0) >= ACCEPT_MIN_FREQ
    )
    blob = "".join(f"{f}\t{freq.get(f.lower(), 0)}\n" for f in forms)
    WORDLIST_OUT.parent.mkdir(parents=True, exist_ok=True)
    WORDLIST_OUT.write_text(blob, encoding="utf-8")

    gz = len(gzip.compress(blob.encode("utf-8"), 9))
    print(f"suggestion wordlist (freq ≥ {ACCEPT_MIN_FREQ}, from {lexicon_count} lexicon / {own_count} own / freqlist): {len(forms)} forms")
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


def fetch_hunspell() -> None:
    """Fetch the BSD-3 Lithuanian hunspell dictionary (lemmas + affix rules) that is
    the browser's ACCEPT vocabulary (applied by real hunspell). Gitignored artifacts."""
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        DIC_OUT.write_bytes(client.get(f"{HUNSPELL_BASE}/index.dic").content)
        AFF_OUT.write_bytes(client.get(f"{HUNSPELL_BASE}/index.aff").content)
    dic_kb = DIC_OUT.stat().st_size / 1024
    aff_kb = AFF_OUT.stat().st_size / 1024
    print(f"hunspell: {DIC_OUT.name} {dic_kb:.0f} KB + {AFF_OUT.name} {aff_kb:.0f} KB (BSD-3, ispell-lt)")


def main() -> int:
    fetch_hunspell()
    vocab = build_wordlist()
    build_bigrams(vocab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
