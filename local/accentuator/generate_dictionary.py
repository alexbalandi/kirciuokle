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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover
    from ._common import (
        CASE_TAGS,
        DEFAULT_GENERATED,
        DEFAULT_LEXICON,
        DEFAULT_VETOES,
        FINITE_VERB_TAGS,
        NONFINITE_VERB_TAGS,
        PURE_DIPHTHONGS,
        cell_key,
        count_stress_marks,
        first_stress_mark,
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
except ImportError:  # pragma: no cover
    from _common import (
        CASE_TAGS,
        DEFAULT_GENERATED,
        DEFAULT_LEXICON,
        DEFAULT_VETOES,
        FINITE_VERB_TAGS,
        NONFINITE_VERB_TAGS,
        PURE_DIPHTHONGS,
        cell_key,
        count_stress_marks,
        first_stress_mark,
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


def load_vetoes(path: Path = DEFAULT_VETOES) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {"lemmas": {}, "words": {}}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "lemmas": dict(raw.get("lemmas") or {}),
        "words": dict(raw.get("words") or {}),
    }


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


def _weak_root_possible(lemma: str, tense: str, present_3: str, past_3: str) -> bool:
    """Kushnir (2019 §4.4.5): which tense stems can carry a weak root at all.

    Past: theme must be -ė and the verb primary (no -yti verbs). Present:
    themes -a-/-ia-/-i- only — an -o present is invariably strong. Suffixal
    pasts in -o (kalbė́jo, mókino) fail the -ė test on their own.
    """

    if tense == "past":
        return lower_key(past_3).endswith("ė") and not lemma.endswith("yti")
    return not lower_key(present_3).endswith("o")


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
            # Strong root: Saussure's shift applies and the observed
            # ending-stressed form is correct (aptìko → aptikaũ).
            return form, None
        if first_stress_mark(form) == "acute":
            return None, "acute on an ending-stressed 1/2sg cell (invalid notation)"
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
        prefix = matched_verb_prefix(lemma)
        form_rows = rows_for_lemma(db, lemma, "verb")
        forms_by_cell = {
            key: entries
            for key, entries in build_forms_by_cell(form_rows).items()
            if is_generation_verb_cell(parse_tags(key))
        }
        if not forms_by_cell:
            continue
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
                provenance = f"open-accentuator:kaikki:{lemma}:verb:{key}"
                if rule:
                    provenance += f":rule={rule}"
                add_variant(
                    grouped,
                    form=resolved,
                    pos="verb",
                    tags=out_tags,
                    provenance=provenance,
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
        nominal_count = generate_nominals(source, grouped, limit, vetoes["lemmas"])
        verb_count = generate_verbs(source, grouped, limit, vetoes["lemmas"])
        other_count = generate_other(source, grouped, vetoes["lemmas"])
        closed_count = generate_closed(source, grouped)
    finally:
        source.close()
    words = write_generated(output, grouped, vetoes["words"])
    return {
        "nominal_lemmas": nominal_count,
        "verb_lemmas": verb_count,
        "other_lemmas": other_count,
        "closed_rows": closed_count,
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
