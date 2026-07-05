# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Shared utilities for the local open accentuator tooling."""

from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

ACCENTUATOR_DIR = Path(__file__).resolve().parent
LOCAL_DIR = ACCENTUATOR_DIR.parent
REPO_ROOT = LOCAL_DIR.parent
DATA_DIR = ACCENTUATOR_DIR / "data"
REPORTS_DIR = ACCENTUATOR_DIR / "reports"

DEFAULT_KAIKKI = LOCAL_DIR / "tagger-hf" / "data" / "raw" / "kaikki-lt.jsonl"
DEFAULT_MATAS = LOCAL_DIR / "tagger-hf" / "data" / "raw" / "MATAS3.conllu"
DEFAULT_LEXICON = DATA_DIR / "lexicon.sqlite"
DEFAULT_GENERATED = DATA_DIR / "generated.sqlite"
DEFAULT_TABLES = DATA_DIR / "paradigm_tables.json"
DEFAULT_CLOSED_DRAFT = DATA_DIR / "closed_draft.md"
DEFAULT_PARITY_REPORT = REPORTS_DIR / "parity-vdu.md"
DEFAULT_VDU_SQLITE = LOCAL_DIR / "data" / "words.sqlite"

COMBINING_GRAVE = "\u0300"
COMBINING_ACUTE = "\u0301"
COMBINING_TILDE = "\u0303"
COMBINING_DOT_ABOVE = "\u0307"
STRESS_MARKS = frozenset((COMBINING_GRAVE, COMBINING_ACUTE, COMBINING_TILDE))
STRESS_NAMES = {
    COMBINING_GRAVE: "grave",
    COMBINING_ACUTE: "acute",
    COMBINING_TILDE: "circumflex",
}
I_DOT_BASES = frozenset(("i", "I", "j", "J"))

PSEUDO_FORM_TAGS = frozenset(("table-tags", "inflection-template", "class"))
TABLE_SOURCES = frozenset(("declension", "conjugation", "inflection"))

CASE_TAGS = frozenset(
    ("nominative", "genitive", "dative", "accusative", "instrumental", "locative", "vocative")
)
NUMBER_TAGS = frozenset(("singular", "dual", "plural"))
FINITE_VERB_TAGS = frozenset(
    (
        "present",
        "past",
        "frequentative",
        "future",
        "conditional",
        "imperative",
        "infinitive",
    )
)
NONFINITE_VERB_TAGS = frozenset(
    ("participle", "adverbial", "active", "passive", "necessitative")
)

TAG_ORDER = {
    "canonical": 0,
    "infinitive": 1,
    "positive": 2,
    "comparative": 3,
    "superlative": 4,
    "definite": 5,
    "indefinite": 6,
    "masculine": 10,
    "feminine": 11,
    "neuter": 12,
    "singular": 20,
    "dual": 21,
    "plural": 22,
    "nominative": 30,
    "genitive": 31,
    "dative": 32,
    "accusative": 33,
    "instrumental": 34,
    "locative": 35,
    "vocative": 36,
    "present": 40,
    "past": 41,
    "frequentative": 42,
    "future": 43,
    "conditional": 44,
    "imperative": 45,
    "first-person": 50,
    "second-person": 51,
    "third-person": 52,
}

TAG_LABELS = {
    "canonical": "pagr.",
    "infinitive": "bendr.",
    "positive": "nelygin.",
    "comparative": "aukšt.",
    "superlative": "aukšč.",
    "definite": "įvardž.",
    "indefinite": "neįvardž.",
    "masculine": "vyr. g.",
    "feminine": "mot. g.",
    "neuter": "bev. g.",
    "singular": "vns.",
    "dual": "dvisk.",
    "plural": "dgs.",
    "nominative": "vard.",
    "genitive": "kilm.",
    "dative": "naud.",
    "accusative": "gal.",
    "instrumental": "įnag.",
    "locative": "viet.",
    "vocative": "šauksm.",
    "present": "es. l.",
    "past": "būt. k. l.",
    "frequentative": "būt. d. l.",
    "future": "būs. l.",
    "conditional": "tar.",
    "imperative": "liep.",
    "first-person": "1 asm.",
    "second-person": "2 asm.",
    "third-person": "3 asm.",
}

POS_LABELS = {
    "noun": "dkt.",
    "name": "dkt. tikr.",
    "adj": "bdv.",
    "pron": "įv.",
    "det": "įv.",
    "num": "sktv.",
    "verb": "vksm.",
    "adv": "prv.",
    "prep": "prl.",
    "conj": "jng.",
    "particle": "dll.",
    "intj": "jstk.",
}


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_lt(text: str | None) -> str:
    """Normalize Lithuanian accents while dropping Wiktionary's synthetic i-dot.

    Wiktionary often writes accented i as ``i + dot above + accent``. The local
    VDU cache uses compact forms such as ``ĩ`` and ``ì``. Removing dot-above only
    after i/j keeps real letters such as ``ė`` intact.
    """

    if not text:
        return ""
    out: list[str] = []
    last_base = ""
    for ch in unicodedata.normalize("NFD", html.unescape(str(text))):
        if unicodedata.combining(ch):
            if ch == COMBINING_DOT_ABOVE and last_base in I_DOT_BASES:
                continue
            out.append(ch)
        else:
            last_base = ch
            out.append(ch)
    return nfc("".join(out))


def strip_accents(text: str | None) -> str:
    if not text:
        return ""
    out: list[str] = []
    last_base = ""
    for ch in unicodedata.normalize("NFD", html.unescape(str(text))):
        if unicodedata.combining(ch):
            if ch in STRESS_MARKS:
                continue
            if ch == COMBINING_DOT_ABOVE and last_base in I_DOT_BASES:
                continue
            out.append(ch)
        else:
            last_base = ch
            out.append(ch)
    return nfc("".join(out))


def lower_key(text: str | None) -> str:
    return strip_accents(text).lower()


def has_stress(text: str | None) -> bool:
    if not text:
        return False
    return any(ch in STRESS_MARKS for ch in unicodedata.normalize("NFD", text))


def first_stress_mark(text: str | None) -> str | None:
    if not text:
        return None
    for ch in unicodedata.normalize("NFD", text):
        if ch in STRESS_MARKS:
            return STRESS_NAMES[ch]
    return None


def stressed_base_index(text: str | None) -> int | None:
    """Return the zero-based base-character index bearing the first stress mark."""

    if not text:
        return None
    base_index = -1
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch):
            if ch in STRESS_MARKS:
                return max(base_index, 0)
        else:
            base_index += 1
    return None


def sort_tags(tags: Iterable[str]) -> tuple[str, ...]:
    clean = [str(tag) for tag in tags if tag]
    return tuple(sorted(dict.fromkeys(clean), key=lambda tag: (TAG_ORDER.get(tag, 1000), tag)))


def cell_key(tags: str | Iterable[str]) -> str:
    if isinstance(tags, str):
        if tags.startswith("["):
            try:
                tags = json.loads(tags)
            except json.JSONDecodeError:
                tags = re.split(r"[|,\s]+", tags)
        else:
            tags = re.split(r"[|,\s]+", tags)
    return "|".join(sort_tags(tag for tag in tags if tag))


def tags_json(tags: Iterable[str]) -> str:
    return json.dumps(list(sort_tags(tags)), ensure_ascii=False, separators=(",", ":"))


def parse_tags(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return ()
        if raw.startswith("["):
            return sort_tags(json.loads(raw))
        return sort_tags(re.split(r"[|,\s]+", raw))
    return sort_tags(raw)


def morphology_label(pos: str | None, tags: Iterable[str]) -> str:
    parts: list[str] = []
    if pos and POS_LABELS.get(pos):
        parts.append(POS_LABELS[pos])
    for tag in sort_tags(tags):
        if tag == "error-unrecognized-form":
            continue
        parts.append(TAG_LABELS.get(tag, tag))
    return ", ".join(dict.fromkeys(parts))


def finite_verb_tags() -> dict[int, tuple[str, ...]]:
    persons = {
        1: ("first-person", "singular"),
        2: ("second-person", "singular"),
        3: ("third-person",),
        4: ("first-person", "plural"),
        5: ("second-person", "plural"),
        6: ("third-person", "plural"),
    }
    table: dict[int, tuple[str, ...]] = {}
    blocks = [
        ("present", 1),
        ("past", 7),
        ("past", 13, "frequentative"),
        ("future", 19),
    ]
    for block in blocks:
        tense = block[0]
        start = block[1]
        extra = block[2:] if len(block) > 2 else ()
        for offset, person_tags in persons.items():
            table[start + offset - 1] = sort_tags((*extra, tense, *person_tags))
    table[25] = sort_tags(("conditional", "first-person", "singular"))
    table[26] = sort_tags(("conditional", "second-person", "singular"))
    table[27] = sort_tags(("conditional", "third-person",))
    table[28] = sort_tags(("conditional", "first-person", "plural"))
    table[29] = sort_tags(("conditional", "second-person", "plural"))
    table[30] = sort_tags(("conditional", "third-person", "plural"))
    table[31] = sort_tags(("imperative", "second-person", "singular"))
    table[32] = sort_tags(("imperative", "third-person",))
    table[33] = sort_tags(("imperative", "first-person", "plural"))
    table[34] = sort_tags(("imperative", "second-person", "plural"))
    table[35] = sort_tags(("imperative", "third-person", "plural"))
    return table


FINITE_CONJ_TAGS = finite_verb_tags()


def split_template_forms(raw: str | None) -> list[str]:
    if not raw:
        return []
    text = html.unescape(str(raw))
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("[[", "").replace("]]", "")
    pieces = re.split(r"[,;\n]+", text)
    forms: list[str] = []
    for piece in pieces:
        cleaned = normalize_lt(piece.strip())
        if not cleaned or cleaned == "-" or "{" in cleaned or "}" in cleaned:
            continue
        forms.append(cleaned)
    return list(dict.fromkeys(forms))


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def safe_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)
