# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Self-accented suffix rules for words absent from the Wiktionary lexicon.

Implements the derivational fallback of the VDU accentuation algorithm
(Kazlauskienė, Raškinis, Norkevičius, Vaičiūnas 2010, "Automatinis lietuvių
kalbos žodžių skiemenavimas, kirčiavimas, transkribavimas", §3.2.4 and
Appendix C): certain noun/adjective suffixes fully determine the accent —
their own accented shape plus the kirčiuotė. Inflection endings per
(declension, kirčiuotė, cell) are induced from the classed Wiktionary
paradigms already in lexicon.sqlite, so nothing here copies a closed source.

The rules only ever fire for lemmas the observed lexicon does not know, and
only emit word keys the generated dictionary does not already contain.
"""

from __future__ import annotations

import sqlite3
import unicodedata
from collections import Counter
from typing import Iterable, Iterator

try:  # pragma: no cover
    from ._common import (
        CASE_TAGS,
        cell_key,
        normalize_lt,
        parse_tags,
        stressed_base_index,
        strip_accents,
    )
except ImportError:  # pragma: no cover
    from _common import (
        CASE_TAGS,
        cell_key,
        normalize_lt,
        parse_tags,
        stressed_base_index,
        strip_accents,
    )

# (accented nominative-singular suffix, kirčiuotė, pos) — VDU 2010 Appendix C,
# notation normalized. Suffixes whose accent needs the base are NOT listed
# (the monograph keeps those in the main dictionaries for the same reason).
APPENDIX_C = [
    ("ùkas", "2", "adj"), ("ùkė", "2", "adj"), ("ùtis", "2", "adj"), ("ùtė", "2", "adj"),
    ("ė̃lis", "2", "adj"), ("ė̃lė", "2", "adj"), ("ùlis", "2", "noun"), ("ùlė", "2", "noun"),
    ("ýtis", "1", "noun"), ("ýtė", "1", "noun"), ("áitis", "1", "noun"), ("áitė", "1", "noun"),
    ("ìngas", "1", "adj"), ("ìnga", "1", "adj"), ("úotas", "1", "adj"), ("úota", "1", "adj"),
    ("ókas", "1", "adj"), ("õkas", "2", "noun"),
    ("ùkas", "2", "noun"), ("ùkė", "2", "noun"), ("ùtis", "2", "noun"), ("ùtė", "2", "noun"),
    ("ùžis", "2", "noun"), ("ùžė", "2", "noun"), ("ū́kštis", "1", "noun"), ("ū́kštė", "1", "noun"),
    ("ẽlis", "2", "noun"), ("ẽlė", "2", "noun"), ("ė́jas", "1", "noun"), ("ė́ja", "1", "noun"),
    ("iẽtis", "2", "noun"), ("iẽtė", "2", "noun"), ("úotojas", "1", "noun"), ("úotoja", "1", "noun"),
    ("ãvimas", "1", "noun"), ("ùmas", "2", "noun"), ("ìzmas", "2", "noun"),
    ("ìškis", "2", "noun"), ("ìškė", "2", "noun"), ("iẽrius", "2", "noun"), ("iẽrė", "2", "noun"),
    ("áuskas", "1", "name"), ("áuskienė", "1", "name"), ("ẽvičius", "1", "name"),
    ("ẽvičienė", "1", "name"), ("ãvičius", "1", "name"), ("ãvičienė", "1", "name"),
    ("ū̃tė", "1", "noun"), ("ė́nas", "1", "noun"), ("ė́nienė", "1", "name"), ("ė́naitė", "1", "name"),
    ("ū́nas", "1", "noun"), ("ū́nienė", "1", "name"), ("ū́naitė", "1", "name"),
    ("ỹstė", "2", "noun"), ("ýbė", "1", "noun"),
    ("òbija", "1", "noun"), ("ãcija", "1", "noun"), ("ècija", "1", "noun"), ("ìcija", "1", "noun"),
    ("ãkcija", "1", "noun"), ("èkcija", "1", "noun"), ("ùkcija", "1", "noun"),
    ("áncija", "1", "noun"), ("eñcija", "1", "noun"), ("ùcija", "1", "noun"),
    ("ãfija", "1", "noun"), ("ògija", "1", "noun"), ("ãlija", "1", "noun"), ("ãmija", "1", "noun"),
    ("èmija", "1", "noun"), ("òmija", "1", "noun"), ("ãnija", "1", "noun"), ("ònija", "1", "noun"),
    ("ãpija", "1", "noun"), ("òpija", "1", "noun"), ("èrija", "1", "noun"), ("òrija", "1", "noun"),
    ("ãtrija", "1", "noun"), ("ètrija", "1", "noun"), ("èsija", "1", "noun"),
    ("ãtija", "1", "noun"), ("ãzija", "1", "noun"), ("èzija", "1", "noun"), ("ùzija", "1", "noun"),
    ("èzė", "2", "noun"), ("òzė", "2", "noun"), ("grãfas", "2", "noun"), ("ètras", "2", "noun"),
    ("ìstas", "2", "noun"), ("ỹvas", "2", "noun"),
]

# Curated additions for regular internationalisms not in the published excerpt
# (accents per Pakerys 2002 / standard usage: ageñtas, muzikañtas, advokãtas,
# kultūrà–kultū̃ros, psichològas, feminìstė, organizãtorius).
EXTRA_SUFFIXES = [
    ("eñtas", "2", "noun"), ("eñtė", "2", "noun"),
    ("ãtas", "2", "noun"), ("ū̃ra", "2", "noun"),
    ("ològas", "2", "noun"), ("ològė", "2", "noun"),
    ("ìstė", "2", "noun"), ("ãtorius", "1", "noun"), ("ãtorė", "1", "noun"),
]
# -antas is deliberately absent: VDU attests both seržántas and muzikañtas —
# the class is lexical there.

# Native derivational suffixes require an attested base noun/adjective and a
# consonant-final base; internationalisms (agentas: base "ag") do not.
NATIVE_SUFFIXES = frozenset((
    "ukas", "ukė", "utis", "utė", "ėlis", "ėlė", "elis", "elė", "ulis", "ulė",
    "ytis", "ytė", "aitis", "aitė", "ingas", "inga", "uotas", "uota", "okas",
    "užis", "užė", "ūkštis", "ūkštė", "ėjas", "ėja", "ietis", "ietė",
    "uotojas", "uotoja", "avimas", "umas", "iškis", "iškė", "ystė", "ybė", "ūtė",
))

# Prefixed bases retract stress to the prefix or root (núotrauka, nevỹkėlis) —
# outside what self-accented suffix rules can decide (Kushnir 2019 ch. 3).
BASE_PREFIXES = (
    "apsi", "atsi", "įsi", "išsi", "nusi", "pasi", "parsi", "persi", "prasi",
    "prisi", "susi", "užsi", "nebe", "tebe", "prie", "nuo", "api", "ati",
    "ap", "at", "iš", "nu", "pa", "par", "per", "pra", "pri", "su", "už",
    "ne", "be", "į", "są", "san",
)

# Rules that must emit a second reading (POS is not recoverable for -okas:
# naujõkas the noun vs gerókas the adjective).
COUNTERPARTS = {"õkas": ("ókas", "1", "adj"), "ókas": ("õkas", "2", "noun")}

NOMINAL_BASE_ENDINGS = ("as", "is", "ys", "us", "a", "ė")

# Pretonic suffixes: kirčiuotė 1, grave on the syllable right before the
# suffix (fìzika, matemàtika, klàsikas) — VDU 2010 §3.3.7 pattern.
PRETONIC_SUFFIXES = [("ika", "1", "noun"), ("ikas", "1", "noun")]

NOM_ENDINGS = ("ius", "ias", "jas", "as", "is", "ys", "us", "a", "ė", "ė̃")
VOWELS = "aeiouyąęėįųū"

MIN_BASE = 2          # shortest allowed segment before the suffix
MIN_CELL_EVIDENCE = 5  # observed lemmas backing an induced ending
MIN_CELL_RATIO = 0.75  # majority share required for an induced ending


def _nom_ending(lemma: str) -> str | None:
    for ending in NOM_ENDINGS:
        stripped = strip_accents(ending)
        if lemma.endswith(stripped) and len(lemma) > len(stripped):
            return stripped
    return None


def _accented_tail(accented: str, start_base: int) -> str:
    out: list[str] = []
    base = -1
    for ch in unicodedata.normalize("NFD", accented):
        if not unicodedata.combining(ch):
            base += 1
        if base >= start_base:
            out.append(ch)
    return normalize_lt("".join(out))


def build_class_tables(db: sqlite3.Connection) -> dict[tuple[str, str, str], dict[str, tuple[str, str]]]:
    """Induce (nom_ending, kirčiuotė, cell) -> (plain ending, site/accented ending).

    site "S": the stem keeps the stress; otherwise the value is the accented
    inflectional ending observed on ending-stressed cells (kl.2 Saussure
    cells, kl.1 has none). Majority-voted over all classed Wiktionary
    paradigms with evidence thresholds.
    """

    votes: dict[tuple[str, str, str], Counter[tuple[str, str, str]]] = {}
    query = """
        SELECT n.lemma, n.stress_class, f.accented, f.tags
        FROM nominals n JOIN forms f ON f.lemma = n.lemma AND f.pos = n.pos
        WHERE n.pos IN ('noun', 'adj')
    """
    for lemma, stress_class, accented, tags in db.execute(query):
        tag_tuple = parse_tags(tags)
        if not set(tag_tuple) & CASE_TAGS:
            continue
        nom_end = _nom_ending(lemma)
        if not nom_end:
            continue
        stem = lemma[: -len(nom_end)]
        stripped = strip_accents(accented)
        if not stripped.startswith(stem) or len(stripped) <= len(stem):
            continue
        ending = stripped[len(stem):]
        index = stressed_base_index(accented)
        if index is None:
            continue
        if index < len(stem):
            value = (ending, "S", "")
        else:
            value = (ending, "E", _accented_tail(accented, len(stem)))
        votes.setdefault((nom_end, str(stress_class), cell_key(tag_tuple)), Counter())[value] += 1

    # {(nom_ending, class): {cell: (plain ending, accented_ending_or_empty)}}
    shaped: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}
    for (nom_end, cls, cell), counter in votes.items():
        (ending, site, acc_ending), n = counter.most_common(1)[0]
        if n >= MIN_CELL_EVIDENCE and n / sum(counter.values()) >= MIN_CELL_RATIO:
            shaped.setdefault((nom_end, cls), {})[cell] = (
                ending,
                acc_ending if site == "E" else "",
            )
    return shaped


class SuffixRule:
    def __init__(self, accented: str, stress_class: str, pos: str, pretonic: bool = False):
        self.accented = normalize_lt(accented)
        self.plain = strip_accents(self.accented)
        self.stress_class = stress_class
        self.pos = pos
        self.pretonic = pretonic
        self.nom_ending = _nom_ending(self.plain) or ""
        self.stem_part = self.plain[: -len(self.nom_ending)] if self.nom_ending else self.plain
        self.accented_stem_part = _accented_head(self.accented, len(self.stem_part))


def _accented_head(accented: str, n_bases: int) -> str:
    out: list[str] = []
    base = -1
    for ch in unicodedata.normalize("NFD", accented):
        if not unicodedata.combining(ch):
            base += 1
            if base >= n_bases:
                break
        out.append(ch)
    return normalize_lt("".join(out))


def load_rules() -> list[SuffixRule]:
    rules = [SuffixRule(a, c, p) for a, c, p in APPENDIX_C + EXTRA_SUFFIXES]
    rules += [SuffixRule(a, c, p, pretonic=True) for a, c, p in PRETONIC_SUFFIXES]
    # longest suffix first so e.g. -ãkcija wins over -ìcija-shaped parses
    rules.sort(key=lambda r: -len(r.plain))
    return rules


def _pretonic_stem(base: str, suffix_stem: str) -> str | None:
    """Stress the last base vowel: fìzik-, lògik-, but pãnik-, matemãtik-.

    A stressed non-final a/e is long in native pronunciation, so it takes the
    circumflex (VDU cache: pãnika, matemãtika, gimnãstika); i/u and loanword
    e/o stay short with the grave (fìzika, lògika, tèchnika... e is kept
    grave because these are all internationalisms).
    """

    for i in range(len(base) - 1, -1, -1):
        if base[i] in VOWELS:
            mark = "̃" if base[i] == "a" else "̀"
            return normalize_lt(base[: i + 1] + mark + base[i + 1:]) + suffix_stem
    return None


# Deverbal agentive counterpart of the pretonic loanword suffix: apgavìkas,
# išdavìkas (kl.2, self-accented) vs fìzika, lògika (kl.1, pretonic).
AGENTIVE_IK = {
    "ika": ("ìka", "2", "noun"),
    "ikas": ("ìkas", "2", "noun"),
}


def _base_allowed(base: str, rule: SuffixRule, wordset: set[str]) -> bool:
    if not rule.pretonic and any(base.startswith(p) for p in BASE_PREFIXES):
        # prefixed bases retract stress (núotrauka, pàžeistas) — but pretonic
        # internationalisms may start with prefix lookalikes (pãnika, prãktika)
        return False
    if rule.pretonic and base[-1] in VOWELS:
        return False  # taĩka-type native words, not -ika derivatives
    if rule.plain in NATIVE_SUFFIXES:
        if base[-1] in VOWELS:
            return False  # kaũtis-type verb stems, not consonant-final bases
        if not any(base + e in wordset for e in NOMINAL_BASE_ENDINGS):
            return False  # diminutives need an attested base (namelis: namas)
    return True


def derive_lemmas(
    words: Iterable[str],
    rules: list[SuffixRule],
    tables: dict[tuple[str, str], dict[str, tuple[str, str]]],
    known_lemmas: set[str],
    verb_form_keys: set[str] | None = None,
) -> Iterator[tuple[str, SuffixRule, str]]:
    """Yield (plain lemma, rule, accented stem) for recognized unknown words."""

    agentive = {k: SuffixRule(*v) for k, v in AGENTIVE_IK.items()}
    wordset = {w for w in words if w.isalpha() and w == w.lower()}
    seen: set[tuple[str, str]] = set()
    for word in sorted(wordset):
        for rule in rules:
            cells = tables.get((rule.nom_ending, rule.stress_class))
            if not cells:
                continue
            for ending, _acc in set(cells.values()):
                if not word.endswith(ending):
                    continue
                stem_full = word[: len(word) - len(ending)]
                if not stem_full.endswith(rule.stem_part):
                    continue
                base = stem_full[: len(stem_full) - len(rule.stem_part)]
                if len(base) < MIN_BASE or not any(v in base for v in VOWELS):
                    continue
                lemma = stem_full + rule.nom_ending
                # The lemma itself must be attested in the wordlist — kills
                # verb forms and names accidentally parsing as derivatives.
                if lemma != word and lemma not in wordset:
                    continue
                if lemma in known_lemmas or (lemma, rule.plain) in seen:
                    continue
                if verb_form_keys and lemma in verb_form_keys:
                    continue
                if not _base_allowed(base, rule, wordset):
                    continue
                out_rule = rule
                if rule.pretonic:
                    deverbal = (
                        base.endswith("v") or base + "ti" in wordset or base + "yti" in wordset
                    )
                    if deverbal:
                        # agentive -ìkas (plėšìkas, apgavìkas), not pretonic
                        out_rule = agentive[rule.plain]
                        accented_stem = base + out_rule.accented_stem_part
                    elif len(base) < 3:
                        continue  # táika, láika are not -ika derivatives
                    else:
                        accented_stem = _pretonic_stem(base, rule.stem_part)
                        if not accented_stem:
                            continue
                else:
                    accented_stem = base + rule.accented_stem_part
                seen.add((lemma, rule.plain))
                yield lemma, out_rule, accented_stem
                counterpart = COUNTERPARTS.get(rule.accented)
                if counterpart:
                    twin = SuffixRule(*counterpart)
                    seen.add((lemma, twin.plain))
                    yield lemma, twin, base + twin.accented_stem_part
                break
            else:
                continue
            break


def paradigm_for(
    accented_stem: str,
    rule: SuffixRule,
    tables: dict[tuple[str, str], dict[str, tuple[str, str]]],
) -> Iterator[tuple[str, str]]:
    """Yield (cell, accented form) for the whole derived paradigm."""

    cells = tables.get((rule.nom_ending, rule.stress_class)) or {}
    plain_stem = strip_accents(accented_stem)
    for cell, (ending, acc_ending) in sorted(cells.items()):
        if acc_ending:
            yield cell, normalize_lt(plain_stem + acc_ending)
        else:
            yield cell, normalize_lt(accented_stem + ending)
