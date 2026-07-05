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
        FINITE_VERB_TAGS,
        NONFINITE_VERB_TAGS,
        cell_key,
        lower_key,
        morphology_label,
        normalize_lt,
        parse_tags,
        safe_relative,
        strip_accents,
    )
    from .paradigm_engine import accent_nominal, accent_verb, build_forms_by_cell
except ImportError:  # pragma: no cover
    from _common import (
        CASE_TAGS,
        DEFAULT_GENERATED,
        DEFAULT_LEXICON,
        FINITE_VERB_TAGS,
        NONFINITE_VERB_TAGS,
        cell_key,
        lower_key,
        morphology_label,
        normalize_lt,
        parse_tags,
        safe_relative,
        strip_accents,
    )
    from paradigm_engine import accent_nominal, accent_verb, build_forms_by_cell


Variant = dict[str, object]


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
    form = normalize_lt(form)
    if not form:
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


def generate_nominals(db: sqlite3.Connection, grouped: dict[str, dict[tuple[str, str], Variant]], limit: int | None) -> int:
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
    tag_set = set(tags)
    if "infinitive" in tag_set:
        return True
    if tag_set & NONFINITE_VERB_TAGS:
        return False
    return bool(tag_set & FINITE_VERB_TAGS)


def generate_verbs(db: sqlite3.Connection, grouped: dict[str, dict[tuple[str, str], Variant]], limit: int | None) -> int:
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
                add_variant(
                    grouped,
                    form=form,
                    pos="verb",
                    tags=out_tags,
                    provenance=f"open-accentuator:kaikki:{lemma}:verb:{key}",
                )
        count += 1
    return count


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


def write_generated(output: Path, grouped: dict[str, dict[tuple[str, str], Variant]]) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    db = sqlite3.connect(tmp)
    now = datetime.now(timezone.utc).isoformat()
    try:
        create_generated_schema(db)
        rows = []
        for word in sorted(grouped):
            variants = list(grouped[word].values())
            variants.sort(key=lambda v: (str(v["form"]), str(v["info"])))
            public_variants = [
                {"form": v["form"], "info": v["info"], "mi": v["mi"]}
                for v in variants
            ]
            default_form = str(public_variants[0]["form"]) if public_variants else None
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
    return len(grouped)


def generate_dictionary(
    *,
    lexicon: Path = DEFAULT_LEXICON,
    output: Path = DEFAULT_GENERATED,
    limit: int | None = None,
) -> dict[str, int]:
    source = sqlite3.connect(lexicon)
    grouped: dict[str, dict[tuple[str, str], Variant]] = defaultdict(dict)
    try:
        nominal_count = generate_nominals(source, grouped, limit)
        verb_count = generate_verbs(source, grouped, limit)
        closed_count = generate_closed(source, grouped)
    finally:
        source.close()
    words = write_generated(output, grouped)
    return {
        "nominal_lemmas": nominal_count,
        "verb_lemmas": verb_count,
        "closed_rows": closed_count,
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
