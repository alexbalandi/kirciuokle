# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate a standalone open-accentuator dictionary from lexicon.sqlite.

This is intentionally not wired into serving. It writes a parity-only artifact
with the replica words-table shape plus a provenance column.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

try:  # pragma: no cover
    from ._common import (
        CASE_TAGS,
        DATA_DIR,
        DEFAULT_GENERATED,
        DEFAULT_LEXICON,
        DEFAULT_VETOES,
        FINITE_VERB_TAGS,
        NONFINITE_VERB_TAGS,
        PURE_DIPHTHONGS,
        cell_key,
        count_stress_marks,
        first_stress_mark,
        has_stress,
        lower_key,
        morphology_label,
        normalize_lt,
        normalize_notation,
        parse_tags,
        safe_relative,
        stressed_base_index,
        strip_accents,
    )
    from .paradigm_engine import MARK_BY_NAME, _apply_stress, accent_nominal, accent_verb, build_forms_by_cell
    from .suffix_rules import _accented_tail, build_class_tables, derive_lemmas, load_rules, paradigm_for
except ImportError:  # pragma: no cover
    from _common import (
        CASE_TAGS,
        DATA_DIR,
        DEFAULT_GENERATED,
        DEFAULT_LEXICON,
        DEFAULT_VETOES,
        FINITE_VERB_TAGS,
        NONFINITE_VERB_TAGS,
        PURE_DIPHTHONGS,
        cell_key,
        count_stress_marks,
        first_stress_mark,
        has_stress,
        lower_key,
        morphology_label,
        normalize_lt,
        normalize_notation,
        parse_tags,
        safe_relative,
        stressed_base_index,
        strip_accents,
    )
    from paradigm_engine import MARK_BY_NAME, _apply_stress, accent_nominal, accent_verb, build_forms_by_cell
    from suffix_rules import _accented_tail, build_class_tables, derive_lemmas, load_rules, paradigm_for


Variant = dict[str, object]

VOWELS = "aeiouyąęėįųū"
LONG_VOWELS = "ąęėįųūyo"
SONORANTS = "lmnr"

# Longest match first: verbal prefixes (with optional reflexive -si-) whose base
# verb also exists in the lexicon mark cells where stress retraction applies.
VERB_PREFIXES = tuple(
    sorted(
        (
            "api", "ap", "apsi", "atsi", "ati", "at", "įsi", "į", "išsi", "iš",
            "nusi", "nu", "pasi", "parsi", "par", "pa", "persi", "per",
            "prasi", "pra", "prisi", "pri", "susi", "su", "užsi", "už",
            "nebe", "tebe", "be", "ne",
        ),
        key=len,
        reverse=True,
    )
)


def load_vetoes(path: Path = DEFAULT_VETOES) -> dict[str, dict]:
    if not path.exists():
        return {"lemmas": {}, "words": {}, "lemma_cells": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "lemmas": dict(raw.get("lemmas") or {}),
        "words": dict(raw.get("words") or {}),
        "lemma_cells": {
            k: v.get("tenses", v) if isinstance(v, dict) else v
            for k, v in (raw.get("lemma_cells") or {}).items()
        },
    }


def _cell_vetoed(lemma: str, tags: Iterable[str], vetoed_cells: dict | None) -> bool:
    if not vetoed_cells:
        return False
    for key, tenses in vetoed_cells.items():
        if lemma == key or lemma.endswith(key):
            if set(tags) & set(tenses):
                return True
    return False


def matched_verb_prefix(lemma: str) -> str | None:
    for prefix in VERB_PREFIXES:
        if lemma.startswith(prefix) and len(lemma) - len(prefix) >= 4:
            return prefix
    return None


def _ending_stressed_12sg(form: str) -> bool:
    stripped = strip_accents(form)
    index = stressed_base_index(form)
    if index is None:
        return False
    core = stripped[:-2] if stripped.endswith("si") else stripped
    return index >= len(core) - 2


def _future_third_metatony_risk(form: str) -> bool:
    """True when a future-3 form may be missing metatony (dìrbs → dir̃bs).

    The circumflex metatony of the future third person hits stems stressed on
    their final syllable; kaikki tables copy the infinitive accent instead. A
    grave on a genuinely short final nucleus (bùs) is fine and kept.
    """

    mark = first_stress_mark(form)
    if mark in (None, "circumflex"):
        return False
    stripped = strip_accents(form)
    index = stressed_base_index(form)
    if index is None or index >= len(stripped):
        return False
    after = index + 1
    is_diphthong = (
        after < len(stripped) and (stripped[index] + stripped[after]).lower() in PURE_DIPHTHONGS
    )
    if is_diphthong:
        after += 1
    if any(ch in VOWELS for ch in stripped[after:].lower()):
        return False  # stress is not on the final syllable — no metatony there
    if mark == "acute" or is_diphthong:
        return True
    if stripped[index].lower() in LONG_VOWELS:
        return True
    return after < len(stripped) and stripped[after].lower() in SONORANTS


def _metatonize_future(form: str) -> str | None:
    """dìrbs → dir̃bs, gáus → gaũs, kalbė́s → kalbė̃s (future-3 metatony).

    A final-syllable-stressed long stem takes the circumflex in the future
    third person (Stundžia; consistent across the VDU cache), placed by the
    standard notation: second pure-diphthong component, mixed-diphthong
    sonorant, or the long vowel itself.
    """

    stripped = strip_accents(form)
    index = stressed_base_index(form)
    if index is None or index >= len(stripped):
        return None
    tilde = MARK_BY_NAME["circumflex"]
    nxt = stripped[index + 1].lower() if index + 1 < len(stripped) else ""
    if (stripped[index] + nxt).lower() in PURE_DIPHTHONGS:
        return _apply_stress(stripped, index + 1, tilde)
    if nxt in SONORANTS:
        return _apply_stress(stripped, index + 1, tilde)
    if stripped[index].lower() in LONG_VOWELS:
        return _apply_stress(stripped, index, tilde)
    return None


def _retract_to_prefix(form: str, third_person: str) -> str | None:
    """Copy the third person's prefix accent onto a 1/2sg form (àtnešė → àtnešiau)."""

    index = stressed_base_index(third_person)
    mark = MARK_BY_NAME.get(first_stress_mark(third_person) or "")
    stripped = strip_accents(form)
    if index is None or not mark or index >= len(stripped):
        return None
    return _apply_stress(stripped, index, mark)


# Kushnir 2019 (125): gulėti and turėti are lexical exceptions — strong
# present roots despite matching the weak criteria. galėti patterns with
# them (VDU: nebegãli, nebegaliù), tikėti does not (VDU: nètikiu).
STRONG_PRESENT_EXCEPTIONS = frozenset(("gulėti", "turėti", "galėti"))

# (base lemma, tense) -> weak? — read off kaikki's real prefixed entries
# (àtnešė => nešti past weak; ištráukė => traukti past strong). Acute pasts
# split lexically (pàbaigė weak vs ištráukė strong), so evidence overrides
# the criteria wherever it exists.
_WEAK_EVIDENCE: dict[tuple[str, str], bool] = {}


def load_weak_evidence(
    db: sqlite3.Connection, vetoed_lemmas: dict[str, str] | None = None
) -> None:
    _WEAK_EVIDENCE.clear()
    votes: dict[tuple[str, str], Counter[bool]] = {}
    bases = {l for (l,) in db.execute("SELECT DISTINCT lemma FROM verbs")}
    for lemma, present_3, past_3 in db.execute(
        "SELECT DISTINCT lemma, present_3, past_3 FROM verbs"
    ):
        if vetoed_lemmas and lemma in vetoed_lemmas:
            continue  # known-bad entries (nekęsti) must not vote
        prefix = matched_verb_prefix(lemma)
        if not prefix or prefix in ("ne", "nebe", "tebe", "per", "persi"):
            # negation is not prefix evidence (negalė́ti vs nugalė́ti differ),
            # and per- is always stressed regardless of root strength
            continue
        base = lemma[len(prefix):]
        if base not in bases:
            continue
        for tense, principal in (("present", present_3), ("past", past_3)):
            form = normalize_lt(principal or "")
            index = stressed_base_index(form)
            if index is None:
                continue
            votes.setdefault((base, tense), Counter())[index < len(prefix)] += 1
    for key, counter in votes.items():
        _WEAK_EVIDENCE[key] = counter.most_common(1)[0][0]


def _weak_present_root(lemma: str, present_3: str, past_3: str) -> bool:
    """Kushnir 2019 §4.4.5 (123): which present roots are weak.

    Weak presents retract to a prefix (nè-, pá-): short-vowel stems (kìša,
    mèta, nẽša — surface grave, or lengthened ã/ẽ), ER~IR alternating stems
    (per̃ka/pir̃ko), and -aR- stems of -ėti verbs (kal̃ba). -o presents and
    acute stems are invariably strong.
    """

    if lemma in STRONG_PRESENT_EXCEPTIONS:
        # lexical exceptions outrank prefixed-entry evidence: nugalė́ti is
        # genuinely weak (nùgali) while negalė́ti stays strong (negãli)
        return False
    evidence = _WEAK_EVIDENCE.get((lemma, "present"))
    if evidence is not None:
        return evidence
    p3 = normalize_lt(present_3)
    key = lower_key(p3)
    if key.endswith("o"):
        return False
    if lemma.endswith(("yti", "oti", "uoti", "auti", "inti", "enti")):
        # §123: only primary and -ėti verbs can have weak presents (ketìna
        # and friends keep their accent: neketinù)
        return False
    mark = first_stress_mark(p3)
    if mark in (None, "acute"):
        return False
    stripped = strip_accents(p3)
    index = stressed_base_index(p3)
    if index is None or index >= len(stripped):
        return False
    # the mark may sit on a mixed-diphthong sonorant — shift to its vowel
    vowel = index - 1 if stripped[index] in SONORANTS and index > 0 else index
    if vowel > 0 and stripped[vowel - 1] in VOWELS:
        return False  # second component of a diphthong is long (miẽga, eĩna)
    nxt = stripped[vowel + 1] if vowel + 1 < len(stripped) else ""
    tautosyllabic_sonorant = nxt in SONORANTS and not (
        vowel + 2 < len(stripped) and stripped[vowel + 2] in VOWELS
    )
    if not tautosyllabic_sonorant:
        # plain short (or lengthened a/e) stem vowel: kìša, mèta, nẽša
        if stripped[vowel] in "ae":
            return True
        return stripped[vowel] in "iu" and mark == "grave"
    # mixed diphthong: weak on ER~IR alternation (per̃ka/pir̃ko) ...
    pst = strip_accents(normalize_lt(past_3))
    if (
        stripped[vowel] == "e"
        and vowel < len(pst)
        and pst[vowel] == "i"
        and vowel + 1 < len(pst)
        and pst[vowel + 1] == nxt
    ):
        return True
    # ... or an n-infix present (skreñda/skrìdo, rañda/rãdo — weak:
    # àtskrenda, sùrado; m-infix stays strong: sutam̃pa)
    if (
        nxt == "n"
        and vowel + 2 < len(stripped)
        and vowel + 1 < len(pst)
        and pst[vowel] in "aeiu"
        and pst[vowel + 1] == stripped[vowel + 2]
    ):
        return True
    # ... or -aR- stems of -ėti verbs (kal̃ba : kalbė́ti)
    return stripped[vowel] == "a" and lemma.endswith("ėti")


def negated_forms(
    form: str,
    tags: Iterable[str],
    lemma: str,
    prefix: str | None,
    present_3: str,
    past_3: str,
) -> list[str] | None:
    """Accented ne-/nebe- counterparts of a verb form, or None when unsafe.

    Kushnir 2019 §4.4.2: the negation behaves like the other weak prefixes —
    it takes the stress exactly when the tense's root allomorph is weak
    (nèkeitė, nèneša) and is unstressed otherwise (nežinaũ, nedìrba). With
    stacked prefixes the last one is stressed (nebè-, §4.4.4 (112)).
    """

    tag_set = set(tags)
    weak = False
    if not prefix and "frequentative" not in tag_set:
        present_stem = "present" in tag_set
        past_stem = "past" in tag_set
        if past_stem and not ("participle" in tag_set and "passive" in tag_set):
            # finite past + past active participles share the past root
            weak = _weak_root_possible(lemma, "past", present_3, past_3)
            if weak and ("participle" in tag_set or "adverbial" in tag_set):
                # past active participles are always root-strong (§4.6.2)
                weak = False
        elif present_stem:
            if _weak_present_root(lemma, present_3, past_3):
                if "participle" in tag_set or "adverbial" in tag_set:
                    return None  # weak present-stem non-finite: prefix wins — skip
                weak = True
    if weak:
        stripped = strip_accents(form)
        return [normalize_lt("nè" + stripped), normalize_lt("nebè" + stripped)]
    return ["ne" + form, "nebe" + form]


def _weak_root_possible(lemma: str, tense: str, present_3: str, past_3: str) -> bool:
    """Kushnir (2019 §4.4.5): which tense stems can carry a weak root at all.

    Past: theme must be -ė and the verb primary (no -yti verbs). Present:
    themes -a-/-ia-/-i- only — an -o present is invariably strong. Suffixal
    pasts in -o (kalbė́jo, mókino) fail the -ė test on their own.
    """

    evidence = _WEAK_EVIDENCE.get((lemma, tense))
    if evidence is not None:
        return evidence
    if tense == "past":
        if first_stress_mark(normalize_lt(past_3)) == "acute":
            # acute pasts split lexically (pàbaigė weak vs ištráukė strong);
            # without prefixed-entry evidence, default to strong
            return False
        return lower_key(past_3).endswith("ė") and not lemma.endswith("yti")
    return not lower_key(present_3).endswith("o")


# Participle declension: per Kushnir 2019 §4.6.1-4.6.2 the active participles
# keep the accent fixed on the head's syllable through the whole paradigm
# (pákeitė but pakeĩtęs -> pakeĩtusio; paválgiusiesiems), so declining an
# observed head is purely mechanical. Ending maps: (case cells) x (m/f, sg/pl).
_CASES = ("nominative", "genitive", "dative", "accusative", "instrumental", "locative")

# head ends in -antis (present/future active): stem = head minus "is"... the
# maps below replace the whole suffix portion after the invariant stem.
PARTICIPLE_ENDINGS = {
    "antis": {  # also matches -intis/-sintis heads via the trailing pattern
        ("masculine", "singular"): ("antis", "ančio", "ančiam", "antį", "ančiu", "ančiame"),
        ("masculine", "plural"): ("antys", "ančių", "antiems", "ančius", "ančiais", "ančiuose"),
        ("feminine", "singular"): ("anti", "ančios", "ančiai", "ančią", "ančia", "ančioje"),
        ("feminine", "plural"): ("ančios", "ančių", "ančioms", "ančias", "ančiomis", "ančiose"),
    },
    "ęs": {
        ("masculine", "singular"): ("ęs", "usio", "usiam", "usį", "usiu", "usiame"),
        ("masculine", "plural"): ("ę", "usių", "usiems", "usius", "usiais", "usiuose"),
        ("feminine", "singular"): ("usi", "usios", "usiai", "usią", "usia", "usioje"),
        ("feminine", "plural"): ("usios", "usių", "usioms", "usias", "usiomis", "usiose"),
    },
    "as": {  # passive participles (-tas, -mas) decline as fixed as-adjectives
        ("masculine", "singular"): ("as", "o", "am", "ą", "u", "ame"),
        ("masculine", "plural"): ("i", "ų", "iems", "us", "ais", "uose"),
        ("feminine", "singular"): ("a", "os", "ai", "ą", "a", "oje"),
        ("feminine", "plural"): ("os", "ų", "oms", "as", "omis", "ose"),
    },
}


def decline_participle(form: str, tags: set[str], present_3: str) -> Iterator[tuple[str, tuple[str, ...]]]:
    """Yield declined cells for a kept participle head, or nothing if unsafe.

    Active participles and frozen passives are fixed-stem (Kushnir §4.6-4.7);
    weak-stem present passives (kal̃biamas-type, non-o presents) go mobile at
    word level and are skipped.
    """

    stripped = strip_accents(form)
    if "active" in tags:
        key = "antis" if stripped.endswith("antis") else "ęs" if stripped.endswith("ęs") else None
    elif "passive" in tags or "necessitative" in tags:
        key = "as" if stripped.endswith("as") else None
        if key and "present" in tags and not lower_key(present_3).endswith("o"):
            return  # weak present-passive stem — mobile, not derivable here
    else:
        return
    if not key:
        return
    endings = PARTICIPLE_ENDINGS[key]
    head_len = len(stripped) - len(key)
    index = stressed_base_index(form)
    if index is None or index >= head_len:
        return  # suffix-accented head would need mobile handling
    stem = _accented_prefix(form, head_len)
    base_tags = tuple(t for t in tags if t not in ("masculine", "feminine", "singular", "plural"))
    for (gender, number), forms in endings.items():
        for case, ending in zip(_CASES, forms):
            yield stem + ending, (*base_tags, gender, number, case)


def _accented_prefix(accented: str, n_bases: int) -> str:
    out: list[str] = []
    base = -1
    for ch in unicodedata.normalize("NFD", accented):
        if not unicodedata.combining(ch):
            base += 1
            if base >= n_bases:
                break
        out.append(ch)
    return normalize_lt("".join(out))


# Root-extending suffixes whose -t- participles keep a frozen strong stem
# (Kushnir 2019 (204)/(208): matýtas, kalbė́tas, dainúotas ...).
EXTENDED_ROOT_INFINITIVES = ("yti", "ėti", "oti", "uoti", "auti", "inti", "enti")
CONVERB_ENDINGS = ("damas", "dama", "dami", "damos", "damasis", "damasi", "damiesi", "damosi")


def _resolve_nonfinite(
    prefix: str | None,
    tag_set: set[str],
    form: str,
    lemma: str,
) -> tuple[str | None, str | None]:
    stripped = strip_accents(form)
    if "adverbial" in tag_set and not tag_set & {"present", "past", "future", "participle"}:
        if stripped.endswith(CONVERB_ENDINGS):
            # Converb (§4.5): the prefix is never stressed except per-.
            index = stressed_base_index(form)
            if prefix and not prefix.startswith("per") and index is not None and index < len(prefix):
                return None, "converb with stressed prefix (invalid per Kushnir §4.5)"
            return form, None
        return None, "bare adverbial (būdinys) accent undecidable"
    if ("passive" in tag_set and "past" in tag_set) or "necessitative" in tag_set:
        # §4.7.2: the -t- participle copies the past stem's accent position and
        # is mobile at word level; kaikki copies the infinitive stem instead.
        # Only extended-root verbs are frozen-strong and safe to keep.
        if not lemma.endswith(EXTENDED_ROOT_INFINITIVES):
            return None, "primary-verb -t- participle mobile/retracted (Kushnir §4.7.2)"
    return form, None


def resolve_verb_form(
    prefix: str | None,
    tags: Iterable[str],
    form: str,
    lemma: str,
    present_3: str,
    past_3: str,
) -> tuple[str | None, str | None]:
    """Apply published accent rules to one observed verb form.

    Returns ``(form_to_emit, rule_tag)``; ``form_to_emit`` is None when the
    form must be skipped, ``rule_tag`` names the applied repair (for
    provenance) or carries the skip reason.
    """

    tag_set = set(tags)
    if tag_set & NONFINITE_VERB_TAGS:
        return _resolve_nonfinite(prefix, tag_set, form, lemma)
    if "future" in tag_set and "third-person" in tag_set:
        if _future_third_metatony_risk(form):
            fixed = _metatonize_future(form)
            if fixed:
                return fixed, "future-3-metatony"
            return None, "future-3 metatony unresolved"
        return form, None
    if (
        "singular" in tag_set
        and tag_set & {"first-person", "second-person"}
        and tag_set & {"past", "present"}
        and "frequentative" not in tag_set
        and _ending_stressed_12sg(form)
    ):
        if prefix:
            # Kushnir (2019: §4.4.2, §4.4.5): 1/2sg stress retracts to the
            # prefix exactly when the tense's root allomorph is weak — which
            # the verb's own third-person principal part already shows. The
            # eligibility gate also screens out prefix-lookalike lemmas
            # (prašyti is praš-, not pra- + šyti).
            tense = "past" if "past" in tag_set else "present"
            third = normalize_lt(past_3 if tense == "past" else present_3)
            index = stressed_base_index(third)
            if (
                _weak_root_possible(lemma, tense, present_3, past_3)
                and index is not None
                and index < len(prefix)
            ):
                retracted = _retract_to_prefix(form, third)
                if retracted:
                    return retracted, "prefix-retraction"
                return None, "prefixed-verb stress retraction unresolved"
        elif first_stress_mark(form) == "acute":
            return None, "acute on an ending-stressed 1/2sg cell (invalid notation)"
        ref3 = normalize_lt(past_3 if "past" in tag_set else present_3)
        if first_stress_mark(ref3) == "acute":
            # Saussure cannot shift off an acute stem (dū́rė, atsidū́rė) —
            # kaikki's mechanical dūriaũ repairs to the stem accent
            repaired = _retract_to_prefix(form, ref3)
            if repaired:
                return repaired, "acute-stem-1sg-repair"
            return None, "ending-stressed 1/2sg of an acute stem"
        # Strong non-acute root: Saussure's shift applies and the observed
        # ending-stressed form is correct (aptìko → aptikaũ).
    return form, None


def create_generated_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;

        CREATE TABLE words (
          word TEXT PRIMARY KEY,
          variants TEXT NOT NULL,
          fetched_at TEXT NOT NULL,
          negative_until TEXT,
          default_form TEXT,
          accent_type TEXT,
          default_form_title TEXT,
          accent_type_title TEXT,
          provenance TEXT NOT NULL
        );

        CREATE INDEX words_default_form ON words(default_form);
        """
    )


def title_case_form(form: str | None) -> str | None:
    if not form:
        return None
    return form[:1].upper() + form[1:]


_MARK_RANK = {"circumflex": 0, "acute": 1, "grave": 2, None: 3}


def default_form_for(variants: list[Variant]) -> str | None:
    """Pick the headword-like variant: leftmost stress, circumflex-first ties.

    Matches how dictionaries head their entries (the citation reading tends to
    carry the earliest stress; at the same syllable a circumflex reading heads
    the entry before acute/grave ones).
    """

    if not variants:
        return None
    forms = sorted(
        {str(v["form"]) for v in variants},
        key=lambda f: (
            index if (index := stressed_base_index(f)) is not None else 99,
            _MARK_RANK.get(first_stress_mark(f), 3),
            f,
        ),
    )
    return forms[0]


def accent_type_for(variants: list[Variant]) -> str:
    if not variants:
        return "NONE"
    forms = {str(v["form"]) for v in variants}
    return "ONE" if len(forms) == 1 else "MULTIPLE_VARIANT"


def add_variant(
    grouped: dict[str, dict[tuple[str, str], Variant]],
    *,
    form: str,
    pos: str,
    tags: Iterable[str],
    provenance: str,
) -> None:
    form = normalize_notation(normalize_lt(form))
    if not form or count_stress_marks(form) > 1:
        # Doubly-accented rows (trečiãdiẽnį) are template artifacts, never words.
        return
    word = lower_key(form)
    if not word:
        return
    tag_tuple = parse_tags(tags)
    label = morphology_label(pos, tag_tuple) or pos or "forma"
    key = (form, label)
    grouped[word][key] = {"form": form, "info": label, "mi": [label], "provenance": provenance}


def rows_for_lemma(db: sqlite3.Connection, lemma: str, pos: str) -> list[dict[str, str]]:
    return [
        {"form": accented, "tags": tags}
        for accented, tags in db.execute(
            "SELECT accented, tags FROM forms WHERE lemma = ? AND pos = ? ORDER BY tags, accented",
            (lemma, pos),
        )
    ]


def is_generation_nominal_cell(pos: str, tags: Iterable[str]) -> bool:
    tag_set = set(tags)
    if tag_set & CASE_TAGS:
        return True
    return pos == "adj" and "neuter" in tag_set


def generate_nominals(
    db: sqlite3.Connection,
    grouped: dict[str, dict[tuple[str, str], Variant]],
    limit: int | None,
    vetoed_lemmas: dict[str, str] | None = None,
    vlkk_name_lemmas: set[str] | None = None,
) -> int:
    query = """
        SELECT DISTINCT n.lemma, n.pos, n.accented_lemma, n.stress_class, m.declension_template
        FROM nominals n
        LEFT JOIN nominal_meta m
          ON m.lemma = n.lemma AND m.pos = n.pos AND m.stress_class = n.stress_class
        ORDER BY n.lemma, n.pos, n.stress_class
    """
    count = 0
    for lemma, pos, accented_lemma, stress_class, template in db.execute(query):
        if limit is not None and count >= limit:
            break
        if vetoed_lemmas and lemma in vetoed_lemmas:
            continue
        if pos == "name" and vlkk_name_lemmas and lower_key(lemma) in vlkk_name_lemmas:
            continue  # VLKK is authoritative for given names
        form_rows = rows_for_lemma(db, lemma, pos)
        forms_by_cell = {
            key: entries
            for key, entries in build_forms_by_cell(form_rows).items()
            if is_generation_nominal_cell(pos, parse_tags(key))
        }
        if not forms_by_cell:
            continue
        info = {
            "forms_by_cell": forms_by_cell,
            "stripped_lemma": lemma,
            "declension_template": template,
        }
        for key in sorted(forms_by_cell):
            tags = parse_tags(key)
            if not tags:
                continue
            for form, out_tags in accent_nominal(accented_lemma, stress_class, info, tags):
                add_variant(
                    grouped,
                    form=form,
                    pos=pos,
                    tags=out_tags,
                    provenance=f"open-accentuator:kaikki:{lemma}:{pos}:{stress_class}:{key}",
                )
        count += 1
    return count


def is_generation_verb_cell(tags: Iterable[str]) -> bool:
    # Non-finite head forms (participles, gerunds, converbs) are observed
    # kaikki facts like the finite cells; Kushnir (2019 §4.5-4.7) confirms
    # their accent derives from the same stem allomorphs.
    tag_set = set(tags)
    return bool(tag_set & (FINITE_VERB_TAGS | NONFINITE_VERB_TAGS))


def generate_verbs(
    db: sqlite3.Connection,
    grouped: dict[str, dict[tuple[str, str], Variant]],
    limit: int | None,
    vetoed_lemmas: dict[str, str] | None = None,
    vetoed_cells: dict | None = None,
) -> int:
    query = """
        SELECT DISTINCT v.lemma, v.accented_infinitive, v.present_3, v.past_3, m.conjugation_template
        FROM verbs v
        LEFT JOIN verb_meta m ON m.lemma = v.lemma
        ORDER BY v.lemma
    """
    count = 0
    for lemma, infinitive, present_3, past_3, template in db.execute(query):
        if limit is not None and count >= limit:
            break
        if vetoed_lemmas and lemma in vetoed_lemmas:
            continue
        form_rows = rows_for_lemma(db, lemma, "verb")
        forms_by_cell = {
            key: entries
            for key, entries in build_forms_by_cell(form_rows).items()
            if is_generation_verb_cell(parse_tags(key))
            and not _cell_vetoed(lemma, parse_tags(key), vetoed_cells)
        }
        if not forms_by_cell:
            continue
        _emit_verb_forms(
            grouped, lemma, infinitive, present_3, past_3, forms_by_cell, template,
        )
        count += 1
    return count


def _emit_verb_forms(
    grouped: dict[str, dict[tuple[str, str], Variant]],
    lemma: str,
    infinitive: str,
    present_3: str,
    past_3: str,
    forms_by_cell: dict[str, list[dict[str, object]]],
    template: str | None = None,
    source: str = "kaikki",
) -> None:
    prefix = matched_verb_prefix(lemma)
    info = {
        "forms_by_cell": forms_by_cell,
        "stripped_lemma": strip_accents(infinitive),
        "conjugation_template": template,
    }
    for key in sorted(forms_by_cell):
        tags = parse_tags(key)
        for form, out_tags in accent_verb(infinitive, present_3, past_3, tags, info):
            resolved, rule = resolve_verb_form(prefix, out_tags, form, lemma, present_3, past_3)
            if resolved is None:
                continue
            provenance = f"open-accentuator:{source}:{lemma}:verb:{key}"
            if rule:
                provenance += f":rule={rule}"
            add_variant(
                grouped,
                form=resolved,
                pos="verb",
                tags=out_tags,
                provenance=provenance,
            )
            tag_set = set(out_tags)
            if "participle" in tag_set or "necessitative" in tag_set:
                for decl_form, decl_tags in decline_participle(resolved, tag_set, present_3):
                    add_variant(
                        grouped,
                        form=decl_form,
                        pos="verb",
                        tags=decl_tags,
                        provenance=f"{provenance}:rule=participle-declension",
                    )
            if not lemma.startswith(("ne", "nebe")) and not tag_set & CASE_TAGS:
                negated = negated_forms(resolved, out_tags, lemma, prefix, present_3, past_3)
                for negated_form in negated or []:
                    add_variant(
                        grouped,
                        form=negated_form,
                        pos="verb",
                        tags=out_tags,
                        provenance=f"{provenance}:rule=negation",
                    )


# Prefixes safe for paradigm synthesis: short-vowel stress targets (pà-,
# atì-, ìš-), including the reflexive composites whose -si- carries the
# retraction (atsìnešė). per- (always stressed) and long į- need their own
# treatment.
SYNTH_PREFIXES = (
    "ap", "api", "at", "ati", "iš", "nu", "pa", "par", "pra", "pri", "su", "už",
    "apsi", "atsi", "išsi", "nusi", "pasi", "parsi", "prasi", "prisi", "susi", "užsi",
)


def _prefix_grave(prefix: str) -> str:
    for i in range(len(prefix) - 1, -1, -1):
        if prefix[i] in VOWELS:
            return normalize_lt(prefix[: i + 1] + "̀" + prefix[i + 1:])
    return prefix


def _prefixed_form(
    form: str,
    tags: Iterable[str],
    weak_present: bool,
    weak_past: bool,
    prefix: str,
    prefix_grave: str,
) -> str | None:
    """One base-verb form carried onto a prefixed verb (Kushnir §4.4.2).

    Strong tenses and all infinitive-stem forms keep the base accent with an
    unstressed prefix; weak finite tenses retract onto the prefix; weak-stem
    non-finite forms are skipped except the past active participle, whose
    root always keeps the accent (§4.6.2).
    """

    tag_set = set(tags)
    if "frequentative" in tag_set:
        weak = False
    elif "present" in tag_set:
        weak = weak_present
    elif "past" in tag_set:
        weak = weak_past
    else:
        weak = False
    if not weak:
        return prefix + form
    if tag_set & NONFINITE_VERB_TAGS:
        if "active" in tag_set and "past" in tag_set:
            return prefix + form
        return None
    return prefix_grave + strip_accents(form)


def generate_prefixed_verbs(
    db: sqlite3.Connection,
    grouped: dict[str, dict[tuple[str, str], Variant]],
    vetoed_lemmas: dict[str, str] | None = None,
    wordlist: Path | None = None,
    vetoed_cells: dict | None = None,
) -> int:
    """Synthesize prefixed-verb paradigms from unprefixed bases.

    Only combos attested in the frequency wordlist are generated, and real
    kaikki entries always win. The synthesized cells run through the same
    rule path as observed verbs, so metatony, participle declension, and
    negation all apply to them.
    """

    if wordlist is None:
        wordlist = DATA_DIR / DEFAULT_WORDLIST_NAME
    if not wordlist.exists():
        return 0
    words = {
        line.split()[0]
        for line in wordlist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    existing = {l for (l,) in db.execute("SELECT DISTINCT lemma FROM verbs")}
    count = 0
    for lemma, infinitive, present_3, past_3 in db.execute(
        "SELECT DISTINCT lemma, accented_infinitive, present_3, past_3 FROM verbs"
    ):
        if matched_verb_prefix(lemma) or lemma.endswith("tis"):
            continue  # bases only; reflexive morphology differs (pasi-)
        if vetoed_lemmas and lemma in vetoed_lemmas:
            continue
        if not (infinitive and present_3 and past_3):
            continue
        weak_present = _weak_present_root(lemma, present_3, past_3)
        weak_past = _weak_root_possible(lemma, "past", present_3, past_3)
        base_cells = {
            key: entries
            for key, entries in build_forms_by_cell(rows_for_lemma(db, lemma, "verb")).items()
            if is_generation_verb_cell(parse_tags(key))
            and not _cell_vetoed(lemma, parse_tags(key), vetoed_cells)
        }
        if not base_cells:
            continue
        base_keys = (
            strip_accents(normalize_lt(infinitive)),
            strip_accents(normalize_lt(present_3)),
            strip_accents(normalize_lt(past_3)),
        )
        for prefix in SYNTH_PREFIXES:
            combo = prefix + lemma
            if combo in existing or (vetoed_lemmas and combo in vetoed_lemmas):
                continue
            if not any(prefix + base in words for base in base_keys):
                continue
            grave = _prefix_grave(prefix)
            cells: dict[str, list[dict[str, object]]] = {}
            for key, entries in base_cells.items():
                tags = parse_tags(key)
                out = []
                for entry in entries:
                    moved = _prefixed_form(
                        str(entry["form"]), tags, weak_present, weak_past, prefix, grave,
                    )
                    if moved:
                        out.append({"form": moved, "tags": list(tags)})
                if out:
                    cells[key] = out
            if not cells:
                continue
            new_p3 = _prefixed_form(
                normalize_lt(present_3), ("present", "third-person"),
                weak_present, weak_past, prefix, grave,
            )
            new_pst3 = _prefixed_form(
                normalize_lt(past_3), ("past", "third-person"),
                weak_present, weak_past, prefix, grave,
            )
            _emit_verb_forms(
                grouped,
                combo,
                prefix + normalize_lt(infinitive),
                new_p3 or prefix + normalize_lt(present_3),
                new_pst3 or prefix + normalize_lt(past_3),
                cells,
                source="prefix-rule",
            )
            count += 1
    return count


OTHER_POS = ("adv", "intj", "prep", "conj", "particle")
ORPHAN_NOMINAL_POS = ("noun", "adj", "name", "pron", "num", "det")


def generate_other(
    db: sqlite3.Connection,
    grouped: dict[str, dict[tuple[str, str], Variant]],
    vetoed_lemmas: dict[str, str] | None = None,
) -> int:
    """Emit observed rows for POS with no paradigm engine.

    Covers adverbs, interjections, prepositions, conjunctions, particles, and
    nominal-POS lemmas that carry forms but no stress class (duals like abù,
    pronominal oddments) — pure observed kaikki facts, no generation.
    """

    placeholders = ",".join("?" for _ in OTHER_POS)
    nominal_placeholders = ",".join("?" for _ in ORPHAN_NOMINAL_POS)
    query = f"""
        SELECT lemma, pos, accented, tags FROM forms
        WHERE pos IN ({placeholders})
           OR (pos IN ({nominal_placeholders})
               AND NOT EXISTS (
                   SELECT 1 FROM nominals n WHERE n.lemma = forms.lemma AND n.pos = forms.pos
               ))
        ORDER BY lemma, pos, tags, accented
    """
    seen: set[tuple[str, str]] = set()
    for lemma, pos, accented, tags in db.execute(query, (*OTHER_POS, *ORPHAN_NOMINAL_POS)):
        if vetoed_lemmas and lemma in vetoed_lemmas:
            continue
        tag_tuple = parse_tags(tags)
        cell = cell_key(tag_tuple) or "canonical"
        add_variant(
            grouped,
            form=accented,
            pos=pos,
            tags=tag_tuple or ("canonical",),
            provenance=f"open-accentuator:kaikki:{lemma}:{pos}:{cell}",
        )
        seen.add((lemma, pos))
    return len(seen)


def generate_closed(db: sqlite3.Connection, grouped: dict[str, dict[tuple[str, str], Variant]]) -> int:
    count = 0
    for lemma, upos, accented_head in db.execute(
        "SELECT lemma, upos, accented_head FROM closed_draft WHERE accented_head IS NOT NULL ORDER BY lemma"
    ):
        pos = {
            "ADV": "adv",
            "ADP": "prep",
            "CCONJ": "conj",
            "SCONJ": "conj",
            "PART": "particle",
            "PRON": "pron",
            "DET": "det",
            "NUM": "num",
            "INTJ": "intj",
            "AUX": "verb",
        }.get(upos, upos.lower())
        add_variant(
            grouped,
            form=accented_head,
            pos=pos,
            tags=("canonical",),
            provenance=f"open-accentuator:closed-draft:{lemma}:{upos}",
        )
        count += 1
    return count


DEFAULT_WORDLIST_NAME = "lt_50k.txt"

# Adjectives with suppletive degree stems.
DEGREE_STEM_OVERRIDES = {"didelis": "did", "didis": "did"}
DEGREE_NOM_ENDINGS = ("ias", "as", "is", "ys", "us")


def _palatalize(stem: str, suffix: str) -> str:
    """d/t soften before the i-initial superlative suffix (saldžiáusias)."""

    if suffix.startswith("i"):
        if stem.endswith("d"):
            return stem[:-1] + "dž"
        if stem.endswith("t"):
            return stem[:-1] + "č"
    return stem


def generate_degrees(
    db: sqlite3.Connection,
    grouped: dict[str, dict[tuple[str, str], Variant]],
    vetoed_lemmas: dict[str, str] | None = None,
) -> int:
    """Synthesize comparative/superlative paradigms for adjectives lacking them.

    kaikki carries observed degree rows for most adjectives (they flow through
    generate_nominals); the accented suffixes are constant across the language
    (-èsnis/-esnì..., -iáusias), so per-cell majorities induced from those
    observed rows fill the gaps for the rest.
    """

    votes: dict[tuple[str, str], Counter[tuple[str, str]]] = {}
    have_degrees: set[str] = set()
    query = """
        SELECT f.lemma, f.accented, f.tags
        FROM forms f JOIN nominals n ON n.lemma = f.lemma AND n.pos = f.pos
        WHERE f.pos = 'adj' AND (f.tags LIKE '%comparative%' OR f.tags LIKE '%superlative%')
    """
    for lemma, accented, tags in db.execute(query):
        tag_tuple = parse_tags(tags)
        tag_set = set(tag_tuple)
        have_degrees.add(lemma)
        if not tag_set & CASE_TAGS:
            continue
        ending = next((e for e in DEGREE_NOM_ENDINGS if lemma.endswith(e)), None)
        if not ending:
            continue
        stem = lemma[: -len(ending)]
        stripped = strip_accents(normalize_lt(accented))
        if not stripped.startswith(stem) or len(stripped) <= len(stem):
            continue  # palatalized or irregular stems don't vote
        degree = "superlative" if "superlative" in tag_set else "comparative"
        suffix = stripped[len(stem):]
        acc_suffix = _accented_tail(normalize_lt(accented), len(stem))
        votes.setdefault((degree, cell_key(tag_tuple)), Counter())[(suffix, acc_suffix)] += 1

    table: dict[tuple[str, str], str] = {}
    for key, counter in votes.items():
        (suffix, acc_suffix), n = counter.most_common(1)[0]
        if n >= MIN_DEGREE_EVIDENCE and n / sum(counter.values()) >= 0.6:
            table[key] = acc_suffix

    count = 0
    for lemma, in db.execute("SELECT DISTINCT lemma FROM nominals WHERE pos = 'adj'"):
        if lemma in have_degrees or (vetoed_lemmas and lemma in vetoed_lemmas):
            continue
        ending = next((e for e in DEGREE_NOM_ENDINGS if lemma.endswith(e)), None)
        if not ending:
            continue
        stem = DEGREE_STEM_OVERRIDES.get(lemma, lemma[: -len(ending)])
        if len(stem) < 2:
            continue
        emitted = False
        for (degree, cell), acc_suffix in sorted(table.items()):
            form = _palatalize(stem, strip_accents(acc_suffix)) + acc_suffix
            add_variant(
                grouped,
                form=form,
                pos="adj",
                tags=parse_tags(cell),
                provenance=f"open-accentuator:degree-rule:{lemma}:adj:{degree}:{cell}",
            )
            emitted = True
        if emitted:
            count += 1
    return count


MIN_DEGREE_EVIDENCE = 25

ISKAS_BASE_ENDINGS = ("ius", "as", "is", "ys", "us", "a", "ė")

# Definite (pronominal) adjective forms — VDU 2010 §3.3.10, table 3.24.
# Mobile paradigm: constant accented endings per cell; None = stem cell
# (copies the simple form's stem accent). First member: -as adjectives,
# second: -us adjectives (with -i- glide).
DEFINITE_CELLS: dict[tuple[str, str, str], tuple[str | None, str, str | None, str]] = {
    # (gender, number, case): (as accented, as plain, us accented, us plain)
    ("masculine", "singular", "nominative"): ("àsis", "asis", "ùsis", "usis"),
    ("masculine", "singular", "genitive"): (None, "ojo", None, "iojo"),
    ("masculine", "singular", "dative"): ("ájam", "ajam", "iájam", "iajam"),
    ("masculine", "singular", "accusative"): (None, "ąjį", None, "ųjį"),
    ("masculine", "singular", "instrumental"): ("úoju", "uoju", "iúoju", "iuoju"),
    ("masculine", "singular", "locative"): ("ãjame", "ajame", "iãjame", "iajame"),
    ("masculine", "plural", "nominative"): ("íeji", "ieji", "íeji", "ieji"),
    ("masculine", "plural", "genitive"): ("ų̃jų", "ųjų", "ių̃jų", "iųjų"),
    ("masculine", "plural", "dative"): ("íesiems", "iesiems", "íesiems", "iesiems"),
    ("masculine", "plural", "accusative"): ("úosius", "uosius", "iúosius", "iuosius"),
    ("masculine", "plural", "instrumental"): ("aĩsiais", "aisiais", "iaĩsiais", "iaisiais"),
    ("masculine", "plural", "locative"): ("uõsiuose", "uosiuose", "iuõsiuose", "iuosiuose"),
    ("feminine", "singular", "nominative"): ("óji", "oji", "ióji", "ioji"),
    ("feminine", "singular", "genitive"): ("õsios", "osios", "iõsios", "iosios"),
    ("feminine", "singular", "dative"): (None, "ajai", None, "iajai"),
    ("feminine", "singular", "accusative"): (None, "ąją", None, "iąją"),
    ("feminine", "singular", "instrumental"): ("ą́ja", "ąja", "ią́ja", "iąja"),
    ("feminine", "singular", "locative"): ("õjoje", "ojoje", "iõjoje", "iojoje"),
    ("feminine", "plural", "nominative"): (None, "osios", None, "iosios"),
    ("feminine", "plural", "genitive"): ("ų̃jų", "ųjų", "ių̃jų", "iųjų"),
    ("feminine", "plural", "dative"): ("ósioms", "osioms", "iósioms", "iosioms"),
    ("feminine", "plural", "accusative"): ("ą́sias", "ąsias", "ią́sias", "iąsias"),
    ("feminine", "plural", "instrumental"): ("õsiomis", "osiomis", "iõsiomis", "iosiomis"),
    ("feminine", "plural", "locative"): ("õsiose", "osiose", "iõsiose", "iosiose"),
}


def generate_definite(
    db: sqlite3.Connection,
    grouped: dict[str, dict[tuple[str, str], Variant]],
    vetoed_lemmas: dict[str, str] | None = None,
) -> int:
    """Definite (pronominal) adjective forms per VDU 2010 §3.3.10.

    Class-1 adjectives keep the simple form's accent throughout; all others
    follow the mobile pattern of table 3.24 — constant accented endings with
    five stem cells copying the simple form's stem accent.
    """

    count = 0
    for lemma, stress_class in db.execute(
        "SELECT DISTINCT lemma, stress_class FROM nominals WHERE pos = 'adj'"
    ):
        if vetoed_lemmas and lemma in vetoed_lemmas:
            continue
        if lemma in DEGREE_STEM_OVERRIDES:
            stem, family = DEGREE_STEM_OVERRIDES[lemma], "is"  # didỹsis
        elif lemma.endswith("as"):
            stem, family = lemma[:-2], "as"
        elif lemma.endswith("us") and not lemma.endswith("ius"):
            stem, family = lemma[:-2], "us"
        elif lemma.endswith("is") and not lemma.endswith(("tis", "sis")):
            stem, family = lemma[:-2], "is"
        else:
            continue
        if len(stem) < 2:
            continue
        # stem accent from a stem-stressed simple row (nom or acc)
        accented_stem = None
        for (accented,) in db.execute(
            """SELECT accented FROM forms WHERE lemma = ? AND pos = 'adj'
               AND (tags LIKE '%accusative%' OR tags LIKE '%nominative%')""",
            (lemma,),
        ):
            form = normalize_lt(accented)
            index = stressed_base_index(form)
            if index is not None and index < len(stem) and strip_accents(form).startswith(stem):
                accented_stem = _accented_prefix(form, len(stem))
                break
        if not accented_stem:
            continue
        fixed = str(stress_class) == "1"
        emitted = False
        for (gender, number, case), (acc_as, plain_as, acc_us, plain_us) in DEFINITE_CELLS.items():
            if family == "as":
                accented_ending, plain_ending = acc_as, plain_as
            else:
                accented_ending, plain_ending = acc_us, plain_us
            if family == "is":
                # -is adjectives (didis -> didỹsis): us-type endings with
                # the nom/acc masculine singular replaced, stems palatalized
                if (gender, number, case) == ("masculine", "singular", "nominative"):
                    accented_ending, plain_ending = "ỹsis", "ysis"
                elif (gender, number, case) == ("masculine", "singular", "accusative"):
                    accented_ending, plain_ending = None, "įjį"
            if fixed or accented_ending is None:
                stem_part, ending = accented_stem, plain_ending
            else:
                stem_part, ending = stem, accented_ending
            if family == "is":
                stem_part = _palatalize(stem_part, strip_accents(ending))
            form = stem_part + ending
            add_variant(
                grouped,
                form=form,
                pos="adj",
                tags=("definite", gender, number, case),
                provenance=f"open-accentuator:definite-rule:{lemma}:adj:{stress_class}:{gender}|{number}|{case}",
            )
            emitted = True
        if emitted:
            count += 1
    return count


def generate_iskas(
    db: sqlite3.Connection,
    grouped: dict[str, dict[tuple[str, str], Variant]],
    vetoed_lemmas: dict[str, str] | None = None,
    wordlist: Path | None = None,
) -> int:
    """-iškas adjectives and -iškai adverbs: fixed (class 1) paradigms whose
    stem accent copies the base noun (vaĩkiškas ← vaĩkas, móteriškas ←
    móteris); international bases without a Lithuanian source fall back to
    the pretonic rule (dramãtiškas, idiòtiškas). Only wordlist-attested
    derivatives are generated, and real kaikki entries win.
    """

    if wordlist is None:
        wordlist = DATA_DIR / DEFAULT_WORDLIST_NAME
    if not wordlist.exists():
        return 0
    words = {
        line.split()[0]
        for line in wordlist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    known = {l for (l,) in db.execute("SELECT DISTINCT lemma FROM nominals")}
    known |= {l for (l,) in db.execute("SELECT DISTINCT lemma FROM forms")}

    # accented stems of potential bases (stress must sit inside the stem);
    # oblique-stem nouns register via the accusative (asmuo -> ãsmen-), and
    # -ias/-is stems also register their i-elided variant (vélnias -> véln-)
    base_stems: dict[str, str] = {}

    def register(stem: str, form: str) -> None:
        index = stressed_base_index(form)
        if (
            len(stem) >= 2
            and stem not in base_stems
            and index is not None
            and index < len(stem)
            and strip_accents(form).startswith(stem)
        ):
            base_stems[stem] = _accented_prefix(form, len(stem))
            if stem.endswith("i"):
                base_stems.setdefault(stem[:-1], _accented_prefix(form, len(stem) - 1))

    for lemma, accented, tags in db.execute(
        """SELECT n.lemma, f.accented, f.tags FROM nominals n
           JOIN forms f ON f.lemma = n.lemma AND f.pos = n.pos
           WHERE n.pos IN ('noun', 'name', 'adj')
           AND (f.tags LIKE '%accusative%' OR f.tags LIKE '%nominative%')"""
    ):
        form = normalize_lt(accented)
        ending = next((e for e in ISKAS_BASE_ENDINGS if lemma.endswith(e)), None)
        if ending:
            register(lemma[: -len(ending)], form)
        if "accusative" in tags and strip_accents(form)[-1:] in "ąįęų":
            register(strip_accents(form)[:-1], form)

    tables = build_class_tables(db)
    adj_cells = {
        cell: ending
        for cell, (ending, acc) in (tables.get(("as", "1")) or {}).items()
        if not acc and ("masculine" in cell or "feminine" in cell)
    }
    if not adj_cells:
        return 0

    # candidate -iškas lemmas attested in the wordlist
    candidates = sorted({
        w[: -len(suffix)] for w in words
        for suffix in ("iškas", "iškai")
        if w.endswith(suffix) and len(w) > len(suffix) + 2
    })
    count = 0
    for stem in candidates:
        lemma = stem + "iškas"
        if lemma in known or (vetoed_lemmas and lemma in vetoed_lemmas):
            continue
        accented_stem = base_stems.get(stem)
        if not accented_stem:
            # no attested base — the pretonic guess proved disjoint-prone
            # for native stems (milžìniškas vs mil̃žiniškas), so skip
            continue
        emitted = False
        for cell, ending in sorted(adj_cells.items()):
            add_variant(
                grouped,
                form=accented_stem + "išk" + ending,
                pos="adj",
                tags=parse_tags(cell),
                provenance=f"open-accentuator:iskas-rule:{lemma}:adj:1:{cell}",
            )
            emitted = True
        if emitted:
            add_variant(
                grouped,
                form=accented_stem + "iškai",
                pos="adv",
                tags=("canonical",),
                provenance=f"open-accentuator:iskas-rule:{lemma}:adv:canonical",
            )
            count += 1
    return count

# Deverbal -imas nouns: only the suffixal families whose accent is a plain
# copy of the past-tense stem (mãtė -> mãtymas, kalbė́jo -> kalbė́jimas,
# kirčiãvo -> kirčiãvimas). Primary verbs are lexically split (gė́rimas vs
# pylìmas) and are left to the dictionary sources.
IMAS_FAMILIES = ("yti", "ėti", "oti", "auti", "uoti")


def generate_deverbal_imas(
    db: sqlite3.Connection,
    grouped: dict[str, dict[tuple[str, str], Variant]],
    vetoed_lemmas: dict[str, str] | None = None,
) -> int:
    tables = build_class_tables(db)
    cells = tables.get(("as", "1")) or {}
    if not cells:
        return 0
    noun_lemmas = {l for (l,) in db.execute("SELECT DISTINCT lemma FROM nominals WHERE pos = 'noun'")}
    count = 0
    for lemma, past_3 in db.execute("SELECT DISTINCT lemma, past_3 FROM verbs"):
        if vetoed_lemmas and lemma in vetoed_lemmas:
            continue
        if not lemma.endswith(IMAS_FAMILIES) or lemma.startswith(("ne", "nebe")):
            continue
        past = normalize_lt(past_3 or "")
        stripped_past = strip_accents(past)
        if len(stripped_past) < 3 or not has_stress(past):
            continue
        # stem = past-3 minus its final theme vowel (mãtė -> mãt, kalbė́jo -> kalbė́j)
        stem = _accented_prefix(past, len(stripped_past) - 1)
        suffix = "ymas" if lemma.endswith("yti") else "imas"
        noun = strip_accents(stem) + suffix
        if noun in noun_lemmas:
            continue  # the dictionary knows better
        emitted = False
        for cell, (ending, acc_ending) in sorted(cells.items()):
            if acc_ending or "masculine" in cell or "feminine" in cell:
                continue  # class 1 is fixed; adjective-gendered cells don't apply
            add_variant(
                grouped,
                form=stem + suffix[:-2] + ending,
                pos="noun",
                tags=parse_tags(cell),
                provenance=f"open-accentuator:imas-rule:{lemma}:noun:1:{cell}",
            )
            emitted = True
        if emitted:
            count += 1
    return count
VLKK_NAMES_FILE = "vlkk_names.json"


def generate_vlkk_names(
    db: sqlite3.Connection,
    grouped: dict[str, dict[tuple[str, str], Variant]],
    names_path: Path | None = None,
) -> tuple[int, set[str]]:
    """Emit given names from the VLKK recommended-names database.

    VLKK is the project's normative authority, so these paradigms take
    precedence over kaikki name entries (returned set = lemmas the kaikki
    name generator must skip). Names with a fetched kirčiuotė + singular
    paradigm additionally get plural cells from the induced class tables;
    names with only an accented nominative emit just that form.
    """

    if names_path is None:
        names_path = DATA_DIR / VLKK_NAMES_FILE
    if not names_path.exists():
        return 0, set()
    data = json.loads(names_path.read_text(encoding="utf-8"))
    tables = build_class_tables(db)
    count = 0
    covered_lemmas: set[str] = set()
    for name, entry in sorted(data.items()):
        accented_nom = entry.get("accented")
        # Only names with a fetched kirčiuotė + paradigm are trustworthy:
        # letter-page nominatives alone miss variant sets of mobile names
        # (Márkas/Markàs) and collide with common words (Ròjus vs rõjus).
        if not accented_nom or not entry.get("cells") or not entry.get("class"):
            continue
        lemma = lower_key(name)
        cells: dict[str, str] = dict(entry.get("cells") or {})
        klass = entry.get("class")
        emitted = False
        if cells and klass:
            # plural cells induced from the classed Wiktionary paradigms
            nom_ending = next((e for e in ("ius", "as", "is", "ys", "us", "a", "ė") if lemma.endswith(e)), None)
            table = tables.get((nom_ending, klass)) if nom_ending else None
            if table:
                stem_len = len(lemma) - len(nom_ending)
                accented_gen = normalize_lt(cells.get("genitive") or "")
                for cell, (ending, acc_ending) in table.items():
                    if "plural" not in cell or cell in cells:
                        continue
                    gen_index = stressed_base_index(accented_gen) if accented_gen else None
                    if acc_ending:
                        cells[cell] = lemma[:stem_len] + acc_ending
                    elif (
                        accented_gen
                        and gen_index is not None
                        and gen_index < stem_len
                        and strip_accents(accented_gen)[:stem_len] == lemma[:stem_len]
                    ):
                        # copy the stem accent of the stem-stressed genitive
                        cells[cell] = _accented_prefix(accented_gen, stem_len) + ending
        for cell, form in sorted(cells.items()):
            if not has_stress(normalize_lt(form)):
                continue
            add_variant(
                grouped,
                form=form,
                pos="name",
                tags=parse_tags(cell) or ("canonical",),
                provenance=f"open-accentuator:vlkk-vardai:{name}:{klass or '?'}:{cell}",
            )
            emitted = True
        if emitted:
            covered_lemmas.add(lemma)
            count += 1
    return count, covered_lemmas




def generate_derived(
    db: sqlite3.Connection,
    grouped: dict[str, dict[tuple[str, str], Variant]],
    vetoed_lemmas: dict[str, str] | None = None,
    wordlist: Path | None = None,
) -> int:
    """Suffix-rule fallback for lemmas the observed lexicon does not know.

    Candidate word forms come from the hermitdave frequency list; a word is
    only derived when it parses as (base + self-accented suffix + induced
    inflection ending), and derived paradigms never overwrite word keys the
    observed generators already produced.
    """

    if wordlist is None:
        wordlist = DATA_DIR / DEFAULT_WORDLIST_NAME
    if not wordlist.exists():
        return 0
    words = [
        line.split()[0]
        for line in wordlist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tables = build_class_tables(db)
    rules = load_rules()
    known = {l for (l,) in db.execute("SELECT DISTINCT lemma FROM nominals")}
    known |= {l for (l,) in db.execute("SELECT DISTINCT lemma FROM forms")}
    verb_form_keys = {
        lower_key(accented) for (accented,) in db.execute("SELECT accented FROM forms WHERE pos = 'verb'")
    }
    count = 0
    derived_words: set[str] = set()
    for lemma, rule, accented_stem in derive_lemmas(words, rules, tables, known, verb_form_keys):
        if vetoed_lemmas and lemma in vetoed_lemmas:
            continue
        emitted = False
        for cell, form in paradigm_for(accented_stem, rule, tables):
            word = lower_key(form)
            if not word or (word in grouped and word not in derived_words):
                continue
            derived_words.add(word)
            add_variant(
                grouped,
                form=form,
                pos=rule.pos,
                tags=parse_tags(cell),
                provenance=f"open-accentuator:vdu2010-suffix:{lemma}:{rule.pos}:{rule.stress_class}:{rule.plain}:{cell}",
            )
            emitted = True
        if emitted:
            count += 1
    return count


def write_generated(
    output: Path,
    grouped: dict[str, dict[tuple[str, str], Variant]],
    vetoed_words: dict[str, str] | None = None,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    now = datetime.now(timezone.utc).isoformat()
    written = 0
    try:
        create_generated_schema(db)
        rows = []
        for word in sorted(grouped):
            if vetoed_words and word in vetoed_words:
                continue
            written += 1
            variants = list(grouped[word].values())
            variants.sort(key=lambda v: (str(v["form"]), str(v["info"])))
            public_variants = [
                {"form": v["form"], "info": v["info"], "mi": v["mi"]}
                for v in variants
            ]
            default_form = default_form_for(variants)
            accent_type = accent_type_for(public_variants)
            provenance = ";".join(sorted({str(v["provenance"]) for v in variants}))
            rows.append(
                (
                    word,
                    json.dumps(public_variants, ensure_ascii=False, separators=(",", ":")),
                    now,
                    None,
                    default_form,
                    accent_type,
                    title_case_form(default_form),
                    accent_type,
                    provenance,
                )
            )
        db.executemany("INSERT INTO words VALUES (?,?,?,?,?,?,?,?,?)", rows)
        db.commit()
    finally:
        db.close()
    os.replace(tmp, output)
    return written


def generate_dictionary(
    *,
    lexicon: Path = DEFAULT_LEXICON,
    output: Path = DEFAULT_GENERATED,
    limit: int | None = None,
    vetoes_path: Path = DEFAULT_VETOES,
) -> dict[str, int]:
    vetoes = load_vetoes(vetoes_path)
    source = sqlite3.connect(lexicon)
    grouped: dict[str, dict[tuple[str, str], Variant]] = defaultdict(dict)
    try:
        load_weak_evidence(source, vetoes["lemmas"])
        vlkk_count, vlkk_lemmas = generate_vlkk_names(source, grouped)
        nominal_count = generate_nominals(source, grouped, limit, vetoes["lemmas"], vlkk_lemmas)
        verb_count = generate_verbs(source, grouped, limit, vetoes["lemmas"], vetoes["lemma_cells"])
        prefixed_count = generate_prefixed_verbs(
            source, grouped, vetoes["lemmas"], vetoed_cells=vetoes["lemma_cells"],
        )
        other_count = generate_other(source, grouped, vetoes["lemmas"])
        closed_count = generate_closed(source, grouped)
        degree_count = generate_degrees(source, grouped, vetoes["lemmas"])
        imas_count = generate_deverbal_imas(source, grouped, vetoes["lemmas"])
        iskas_count = generate_iskas(source, grouped, vetoes["lemmas"])
        definite_count = generate_definite(source, grouped, vetoes["lemmas"])
        derived_count = generate_derived(source, grouped, vetoes["lemmas"])
    finally:
        source.close()
    words = write_generated(output, grouped, vetoes["words"])
    return {
        "vlkk_names": vlkk_count,
        "nominal_lemmas": nominal_count,
        "verb_lemmas": verb_count,
        "prefixed_verbs": prefixed_count,
        "other_lemmas": other_count,
        "closed_rows": closed_count,
        "degree_lemmas": degree_count,
        "imas_lemmas": imas_count,
        "iskas_lemmas": iskas_count,
        "definite_lemmas": definite_count,
        "derived_lemmas": derived_count,
        "vetoed_lemmas": len(vetoes["lemmas"]),
        "vetoed_words": len(vetoes["words"]),
        "words": words,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate standalone open accentuator dictionary SQLite.")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON, help="Input lexicon.sqlite path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_GENERATED, help="Output generated.sqlite path.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-POS lemma limit for smoke tests. Omit for full generation.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress summary output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = generate_dictionary(lexicon=args.lexicon, output=args.output, limit=args.limit)
    if not args.quiet:
        for key, value in summary.items():
            print(f"{key}: {value:,}")
        print(f"wrote {safe_relative(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
