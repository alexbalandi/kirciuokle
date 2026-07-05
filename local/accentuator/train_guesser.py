# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Own-trained stress guesser: letter-pattern suffix model over our dictionary.

Reimplements the spirit of Anbinderis & Kasparaitis (2010) — stress placement
predicted from pure letter patterns, no linguistic annotation — but trained on
our OWN generated dictionary (566k accented words, fully open provenance)
instead of Sakrament's proprietary corpus. A word's stress is encoded as
(offset of the stressed letter from the word end, mark); prediction walks the
longest attested suffix whose majority class yields a valid placement.

Evaluated on the VDU-cache words the dictionary does NOT cover — the same
slice where phonology_engine (LIEPA) scores 87.9% exact-variant / 95.3%
stress-position — plus an in-domain held-out split as a sanity bound.
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

try:  # pragma: no cover
    from ._common import DEFAULT_GENERATED, DEFAULT_VDU_SQLITE, normalize_notation, strip_accents
except ImportError:  # pragma: no cover
    from _common import DEFAULT_GENERATED, DEFAULT_VDU_SQLITE, normalize_notation, strip_accents

GRAVE, ACUTE, TILDE = "̀", "́", "̃"
STRESS_MARKS = {GRAVE, ACUTE, TILDE}
MAX_SUFFIX = 10
# Letters that can carry a mark, tightened by an audit of all 710k stressed
# variants in generated.sqlite (2026-07-06): long vowels never take the
# grave (0-3 noise rows vs tens of thousands acute/tilde); a bare short i
# takes only the grave (30,028 vs 1); mixed-diphthong first-element i/u
# takes only the grave (52,069 vs 5); a sonorant as the second diphthong
# element takes only the tilde (60,931 vs 0). Bans remove impossible
# (letter, mark) cells from every guesser's choice space.
VOWELS = set("aeiouyąęėįūų")
LONG_VOWELS = set("yąęėįūų")
SONORANTS = set("lmnr")


def stress_of(accented: str) -> tuple[int, str] | None:
    """Return (letter index in the plain word, mark) of the stress mark."""
    pos = mark = None
    cluster = -1
    for ch in unicodedata.normalize("NFD", accented):
        if not unicodedata.combining(ch):
            cluster += 1
        elif ch in STRESS_MARKS:
            pos, mark = cluster, ch
    if mark is None:
        return None
    return pos, mark


def apply_stress(word: str, pos: int, mark: str) -> str:
    # appending after the NFC letter keeps the mark after the cluster's own
    # combining marks (ė̃ ordering); normalize_notation then fixes placement
    raw = word[: pos + 1] + mark + word[pos + 1 :]
    return normalize_notation(unicodedata.normalize("NFC", raw))


def valid_target(word: str, pos: int, mark: str) -> bool:
    if not 0 <= pos < len(word):
        return False
    ch = word[pos]
    prev = word[pos - 1] if pos > 0 else ""
    nxt = word[pos + 1] if pos + 1 < len(word) else ""
    if ch in SONORANTS:
        return mark == TILDE and prev in VOWELS
    if ch not in VOWELS:
        return False
    if ch in LONG_VOWELS:
        return mark != GRAVE
    if ch in "iu" and prev not in VOWELS and nxt not in VOWELS:
        # not part of a vowel diphthong: bare short nucleus or the first
        # element of a mixed diphthong (il/ir/um/un...) — grave only,
        # except the attested u+tilde loan notation
        if nxt in SONORANTS:
            return mark == GRAVE
        if ch == "i":
            return mark == GRAVE
        return mark != ACUTE
    return True


class SuffixModel:
    def __init__(self) -> None:
        self.table: dict[str, Counter] = defaultdict(Counter)

    def train(self, pairs: list[tuple[str, int, str]]) -> None:
        for word, pos, mark in pairs:
            cls = (len(word) - pos, mark)
            for length in range(1, min(len(word), MAX_SUFFIX) + 1):
                self.table[word[-length:]][cls] += 1

    def predict(self, word: str, min_count: int = 2) -> tuple[int, str] | None:
        for length in range(min(len(word), MAX_SUFFIX), 0, -1):
            counter = self.table.get(word[-length:])
            if not counter:
                continue
            total = sum(counter.values())
            if total < min_count and length > 1:
                continue
            for (offset, mark), _n in counter.most_common():
                pos = len(word) - offset
                if valid_target(word, pos, mark):
                    return pos, mark
        return None


def load_training(path: Path) -> list[tuple[str, int, str]]:
    db = sqlite3.connect(path)
    pairs = []
    for word, form in db.execute("SELECT word, default_form FROM words"):
        if not word.isalpha():
            continue
        parsed = stress_of(form)
        if parsed is None:
            continue
        pos, mark = parsed
        if strip_accents(form) == word and valid_target(word, pos, mark):
            pairs.append((word, pos, mark))
    return pairs


def load_vdu_eval(path: Path, covered: set[str]) -> list[tuple[str, list[str]]]:
    db = sqlite3.connect(path)
    rows = []
    for word, variants in db.execute(
        "SELECT word, variants FROM words WHERE variants IS NOT NULL AND variants != '[]'"
    ):
        if not word.isalpha() or word in covered:
            continue
        forms = [v["form"] for v in json.loads(variants) if v.get("form")]
        if forms:
            rows.append((word, forms))
    return rows


def score(model: SuffixModel, rows: list[tuple[str, list[str]]], label: str) -> None:
    answered = exact = position = 0
    for word, forms in rows:
        pred = model.predict(word)
        if pred is None:
            continue
        answered += 1
        form = apply_stress(word, *pred)
        norm_forms = [unicodedata.normalize("NFC", f) for f in forms]
        if form in norm_forms:
            exact += 1
        gold = {stress_of(f)[0] for f in norm_forms if stress_of(f)}
        if stress_of(form) and stress_of(form)[0] in gold:
            position += 1
    n = len(rows)
    print(
        f"{label}: n={n:,} answered={answered/n:.1%} "
        f"exact={exact/answered:.1%} position={position/answered:.1%}"
        f"  (of answered; over all: exact={exact/n:.1%})"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=float, default=0.02)
    args = parser.parse_args(argv)

    pairs = load_training(DEFAULT_GENERATED)
    print(f"training words: {len(pairs):,}")
    rng = random.Random(20260705)
    rng.shuffle(pairs)
    cut = int(len(pairs) * args.holdout)
    held, train = pairs[:cut], pairs[cut:]

    model = SuffixModel()
    model.train(train)
    print(f"suffix table entries: {len(model.table):,}")

    held_rows = [(w, [apply_stress(w, p, m)]) for w, p, m in held]
    score(model, held_rows, "in-domain held-out")

    covered = {w for w, _p, _m in pairs}
    vdu_rows = load_vdu_eval(DEFAULT_VDU_SQLITE, covered)
    score(model, vdu_rows, "VDU-uncovered slice")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
