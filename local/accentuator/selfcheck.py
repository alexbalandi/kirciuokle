# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Selfcheck for the W2 open accentuator core."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover
    from ._common import (
        CASE_TAGS,
        DEFAULT_CLOSED_DRAFT,
        DEFAULT_KAIKKI,
        DEFAULT_LEXICON,
        DEFAULT_MATAS,
        DEFAULT_TABLES,
        FINITE_VERB_TAGS,
        NONFINITE_VERB_TAGS,
        cell_key,
        has_stress,
        normalize_lt,
        parse_tags,
        safe_relative,
        strip_accents,
    )
    from ._common import nfc, normalize_notation
    from .extract_lexicon import build_lexicon
    from .generate_dictionary import matched_verb_prefix, resolve_verb_form
    from .paradigm_engine import accent_nominal, accent_verb, build_forms_by_cell, normalize_cell
except ImportError:  # pragma: no cover
    from _common import (
        CASE_TAGS,
        DEFAULT_CLOSED_DRAFT,
        DEFAULT_KAIKKI,
        DEFAULT_LEXICON,
        DEFAULT_MATAS,
        DEFAULT_TABLES,
        FINITE_VERB_TAGS,
        NONFINITE_VERB_TAGS,
        cell_key,
        has_stress,
        normalize_lt,
        parse_tags,
        safe_relative,
        strip_accents,
    )
    from _common import nfc, normalize_notation
    from extract_lexicon import build_lexicon
    from generate_dictionary import matched_verb_prefix, resolve_verb_form
    from paradigm_engine import accent_nominal, accent_verb, build_forms_by_cell, normalize_cell


def ensure_lexicon(args: argparse.Namespace) -> None:
    if args.rebuild or not args.lexicon.exists() or not args.tables.exists():
        print("building lexicon data from Kaikki for selfcheck...")
        build_lexicon(
            kaikki=args.kaikki,
            matas=args.matas,
            output=args.lexicon,
            tables=args.tables,
            closed_markdown=args.closed_markdown,
            include_closed=True,
            quiet=args.quiet,
        )


def rows_for(db: sqlite3.Connection, lemma: str, pos: str) -> list[dict[str, str]]:
    return [
        {"form": accented, "tags": tags}
        for accented, tags in db.execute(
            "SELECT accented, tags FROM forms WHERE lemma = ? AND pos = ? ORDER BY tags, accented",
            (lemma, pos),
        )
    ]


def nominal_rows_for(db: sqlite3.Connection, lemma: str, pos: str) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for row in rows_for(db, lemma, pos):
        tags = parse_tags(row["tags"])
        if set(tags) & CASE_TAGS:
            result.append(row)
    return result


def assert_equal_sets(label: str, expected: set[str], actual: set[str]) -> None:
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise AssertionError(f"{label}: expected {sorted(expected)}, got {sorted(actual)}; missing={missing}, extra={extra}")


def check_nominal(
    db: sqlite3.Connection,
    *,
    lemma: str,
    pos: str,
    stress_class: str,
    min_cells: int,
) -> int:
    row = db.execute(
        "SELECT accented_lemma FROM nominals WHERE lemma = ? AND pos = ? AND stress_class = ? LIMIT 1",
        (lemma, pos, stress_class),
    ).fetchone()
    if not row:
        raise AssertionError(f"missing nominal row for {lemma}/{pos}/{stress_class}")
    accented_lemma = row[0]
    form_rows = nominal_rows_for(db, lemma, pos)
    forms_by_cell = build_forms_by_cell(form_rows)
    if len(forms_by_cell) < min_cells:
        raise AssertionError(f"{lemma}: expected at least {min_cells} cells, found {len(forms_by_cell)}")
    info = {"forms_by_cell": forms_by_cell, "stripped_lemma": lemma}
    checked = 0
    for key, entries in sorted(forms_by_cell.items()):
        expected = {normalize_lt(entry["form"]) for entry in entries}
        actual = {form for form, _tags in accent_nominal(accented_lemma, stress_class, info, parse_tags(key))}
        assert_equal_sets(f"{lemma} {key}", expected, actual)
        checked += 1
    return checked


def check_namas(db: sqlite3.Connection) -> None:
    cells = check_nominal(db, lemma="namas", pos="noun", stress_class="4", min_cells=14)
    expected = {
        "nãmas",
        "nãmo",
        "nãmui",
        "nãmą",
        "namù",
        "namè",
        "nãme",
        "namaĩ",
        "namų̃",
        "namáms",
        "namùs",
        "namaĩs",
        "namuosè",
    }
    observed = {
        normalize_lt(accented)
        for accented, in db.execute(
            "SELECT accented FROM forms WHERE lemma = 'namas' AND pos = 'noun'"
        )
    }
    if not expected <= observed:
        raise AssertionError(f"namas table missing expected forms: {sorted(expected - observed)}")
    if cells < 14:
        raise AssertionError("namas full paradigm did not cover 14 case/number cells")


def check_class_examples(db: sqlite3.Connection) -> None:
    check_nominal(db, lemma="varna", pos="noun", stress_class="1", min_cells=12)
    check_nominal(db, lemma="upė", pos="noun", stress_class="2", min_cells=12)
    check_nominal(db, lemma="galva", pos="noun", stress_class="3", min_cells=12)
    check_nominal(db, lemma="geras", pos="adj", stress_class="4", min_cells=20)

    upe_acc_pl = {
        form
        for form, _tags in accent_nominal(
            "ùpė",
            "2",
            {"forms_by_cell": build_forms_by_cell(nominal_rows_for(db, "upė", "noun")), "stripped_lemma": "upė"},
            ("accusative", "plural"),
        )
    }
    if "upès" not in upe_acc_pl:
        raise AssertionError(f"class-2 Saussure cell upė acc.pl expected upès, got {sorted(upe_acc_pl)}")


def is_generation_verb_cell(tags: Iterable[str]) -> bool:
    tag_set = set(tags)
    if "infinitive" in tag_set:
        return True
    if tag_set & NONFINITE_VERB_TAGS:
        return False
    return bool(tag_set & FINITE_VERB_TAGS)


def check_verb(db: sqlite3.Connection) -> None:
    row = db.execute(
        "SELECT accented_infinitive, present_3, past_3 FROM verbs WHERE lemma = 'daryti' LIMIT 1"
    ).fetchone()
    if not row:
        raise AssertionError("missing verb row for daryti")
    infinitive, present_3, past_3 = row
    form_rows = [
        row
        for row in rows_for(db, "daryti", "verb")
        if is_generation_verb_cell(parse_tags(row["tags"]))
    ]
    forms_by_cell = build_forms_by_cell(form_rows)
    info = {"forms_by_cell": forms_by_cell, "stripped_lemma": "daryti"}
    aliases = {
        "present_3": "dãro",
        "past_3": "dãrė",
        "future_3": None,
        "conditional_3": "darýtų",
        "imperative_2sg": "darýk",
    }
    for alias, must_include in aliases.items():
        key = normalize_cell(alias)
        expected = {entry["form"] for entry in forms_by_cell.get(key, [])}
        actual = {form for form, _tags in accent_verb(infinitive, present_3, past_3, alias, info)}
        if not expected:
            raise AssertionError(f"daryti missing expected cell {alias}")
        assert_equal_sets(f"daryti {alias}", expected, actual)
        if must_include and must_include not in actual:
            raise AssertionError(f"daryti {alias} expected {must_include}, got {sorted(actual)}")
        if alias == "future_3":
            if not any(strip_accents(form) == "darys" and has_stress(form) for form in actual):
                raise AssertionError(f"daryti future_3 expected stressed darys, got {sorted(actual)}")


def check_notation() -> None:
    cases = {
        # circumflex moves to the second component of a pure diphthong
        "ãusys": "aũsys",
        "ãusį": "aũsį",
        # circumflex moves to the sonorant of a mixed diphthong
        "ĩlgas": "il̃gas",
        # acute cannot sit on a sonorant — repaired to the circumflex
        "giŕdite": "gir̃dite",
        # morpheme-boundary hiatus: the i opens the ie diphthong — no move
        "pãieškai": "pãieškai",
        # long o + sonorant is not a mixed diphthong
        "kõl": "kõl",
        # already-standard notation is left alone
        "aũsis": "aũsis",
        "muĩlas": "muĩlas",
        "vil̃kas": "vil̃kas",
        "tiñka": "tiñka",
        "kur̃": "kur̃",
        "septỹni": "septỹni",
        "gãlios": "gãlios",
        # grave and acute on vowels are never converted (priegaidė is lexical)
        "apkabìnti": "apkabìnti",
        "pìlnas": "pìlnas",
        "dúona": "dúona",
        # sonorant before a vowel starts the next syllable — no move
        "mẽnas": "mẽnas",
    }
    for raw, expected in cases.items():
        actual = normalize_notation(normalize_lt(raw))
        if actual != nfc(expected):
            raise AssertionError(f"normalize_notation({raw!r}) = {actual!r}, expected {expected!r}")


def check_verb_rules() -> None:
    if matched_verb_prefix("atnešti") != "at":
        raise AssertionError("atnešti should match prefix at-")
    if matched_verb_prefix("pasielgti") != "pasi":
        raise AssertionError("pasielgti should match prefix pasi-")
    if matched_verb_prefix("atiduoti") != "ati":
        raise AssertionError("atiduoti should match prefix ati-")
    if matched_verb_prefix("paruošti") is None:
        raise AssertionError("paruošti should count as prefixed")
    if matched_verb_prefix("nešti") is not None:
        raise AssertionError("nešti is not a prefixed verb")

    future_3 = ("future", "third-person")
    past_1sg = ("singular", "past", "first-person")
    prs_1sg = ("singular", "present", "first-person")
    # (prefix, tags, observed form, lemma, present_3, past_3) -> expected emitted form
    expect = {
        # future-3 metatony: long final-syllable stems take the circumflex
        (None, future_3, "dìrbs", "dirbti", "dìrba", "dìrbo"): "dir̃bs",
        (None, future_3, "gáus", "gauti", "gáuna", "gãvo"): "gaũs",
        (None, future_3, "kalbė́s", "kalbėti", "kal̃ba", "kalbė́jo"): "kalbė̃s",
        (None, future_3, "kalbė̃s", "kalbėti", "kal̃ba", "kalbė́jo"): "kalbė̃s",
        (None, future_3, "darỹs", "daryti", "dãro", "dãrė"): "darỹs",
        (None, future_3, "mókys", "mokyti", "móko", "mókė"): "mókys",
        # short final nucleus keeps the grave (Kushnir 2019: (17a) kás)
        (None, future_3, "bùs", "būti", "yrà", "bùvo"): "bùs",
        # weak past root (Kushnir 2019 §4.4.5: -ė past, primary verb) — the
        # third person shows prefix stress, and 1/2sg retracts with it
        ("at", past_1sg, "atnešiaũ", "atnešti", "àtneša", "àtnešė"): "àtnešiau",
        ("ati", past_1sg, "atidaviaũ", "atiduoti", "atìduoda", "atìdavė"): "atìdaviau",
        # strong past root (-o past): Saussure shift stands (aptikaũ)
        ("ap", past_1sg, "aptikaũ", "aptikti", "aptiñka", "aptìko"): "aptikaũ",
        # prefix-lookalike lemma: praš- is the root, -yti pasts are never weak
        ("pra", past_1sg, "prašiaũ", "prašyti", "prãšo", "prãšė"): "prašiaũ",
        ("pra", prs_1sg, "prašaũ", "prašyti", "prãšo", "prãšė"): "prašaũ",
        # stem-stressed observed forms pass through untouched
        ("pa", past_1sg, "pàdirbau", "padirbti", "pàdirba", "pàdirbo"): "pàdirbau",
        (None, past_1sg, "nešiaũ", "nešti", "nẽša", "nẽšė"): "nešiaũ",
        # acute on an ending-stressed 1sg is invalid notation — skipped
        (None, past_1sg, "elgiaúsi", "elgtis", "el̃giasi", "el̃gėsi"): None,
    }
    for (prefix, tags, form, lemma, prs3, pst3), expected in expect.items():
        resolved, rule = resolve_verb_form(prefix, tags, normalize_lt(form), lemma, prs3, pst3)
        want = nfc(expected) if expected else None
        if resolved != want:
            raise AssertionError(
                f"resolve_verb_form({prefix}, {tags}, {form!r}) = {resolved!r} ({rule}), expected {expected!r}"
            )


def run_selfcheck(args: argparse.Namespace) -> int:
    check_notation()
    check_verb_rules()
    ensure_lexicon(args)
    db = sqlite3.connect(args.lexicon)
    try:
        check_namas(db)
        check_class_examples(db)
        check_verb(db)
    finally:
        db.close()
    if not args.quiet:
        print("selfcheck passed")
        print(f"lexicon: {safe_relative(args.lexicon)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run W2 open accentuator selfcheck.")
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON, help="Input/output lexicon.sqlite path.")
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES, help="Input/output paradigm_tables.json path.")
    parser.add_argument("--kaikki", type=Path, default=DEFAULT_KAIKKI, help="Kaikki dump used if extraction is needed.")
    parser.add_argument("--matas", type=Path, default=DEFAULT_MATAS, help="MATAS corpus used if extraction is needed.")
    parser.add_argument(
        "--closed-markdown",
        type=Path,
        default=DEFAULT_CLOSED_DRAFT,
        help="Closed-class markdown path used if extraction is needed.",
    )
    parser.add_argument("--rebuild", action="store_true", help="Rebuild lexicon data before checking.")
    parser.add_argument("--quiet", action="store_true", help="Suppress successful summary output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_selfcheck(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
