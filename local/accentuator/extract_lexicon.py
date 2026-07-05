# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Extract the open accentuator lexicon from the Lithuanian Kaikki dump."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

try:  # pragma: no cover
    from ._common import (
        CASE_TAGS,
        DEFAULT_CLOSED_DRAFT,
        DEFAULT_KAIKKI,
        DEFAULT_LEXICON,
        DEFAULT_MATAS,
        DEFAULT_TABLES,
        FINITE_CONJ_TAGS,
        NONFINITE_VERB_TAGS,
        PSEUDO_FORM_TAGS,
        TABLE_SOURCES,
        cell_key,
        common_prefix_len,
        first_stress_mark,
        has_stress,
        lower_key,
        normalize_lt,
        safe_relative,
        sort_tags,
        split_template_forms,
        strip_accents,
        stressed_base_index,
        tags_json,
    )
except ImportError:  # pragma: no cover
    from _common import (
        CASE_TAGS,
        DEFAULT_CLOSED_DRAFT,
        DEFAULT_KAIKKI,
        DEFAULT_LEXICON,
        DEFAULT_MATAS,
        DEFAULT_TABLES,
        FINITE_CONJ_TAGS,
        NONFINITE_VERB_TAGS,
        PSEUDO_FORM_TAGS,
        TABLE_SOURCES,
        cell_key,
        common_prefix_len,
        first_stress_mark,
        has_stress,
        lower_key,
        normalize_lt,
        safe_relative,
        sort_tags,
        split_template_forms,
        strip_accents,
        stressed_base_index,
        tags_json,
    )

NOMINAL_POS = frozenset(("noun", "name", "adj", "pron", "num", "det"))
VALID_STRESS_CLASSES = frozenset(("1", "2", "3", "3a", "3b", "4"))
STRESS_TAG_RE = re.compile(r"^stress-pattern-(.+)$")
STRESS_EXPANSION_RE = re.compile(r"stress pattern\s+([0-9]+[abᵃᵇ]?(?:[-/][0-9]+[abᵃᵇ]?)?)", re.I)
CLOSED_UPOS = frozenset(("CCONJ", "SCONJ", "ADP", "PART", "PRON", "DET", "INTJ", "NUM", "ADV", "AUX"))


@dataclass
class InductionBucket:
    metadata: dict[str, str]
    counters: dict[str, Counter[tuple[str, str, str, str, str]]] = field(default_factory=lambda: defaultdict(Counter))
    examples: dict[tuple[str, tuple[str, str, str, str, str]], list[dict[str, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )


class ParadigmInducer:
    def __init__(self) -> None:
        self.nominals: dict[str, InductionBucket] = {}
        self.verbs: dict[str, InductionBucket] = {}

    def add_nominal(
        self,
        *,
        pos: str,
        template: str,
        stress_class: str,
        lemma: str,
        accented_lemma: str,
        form: str,
        tags: Iterable[str],
    ) -> None:
        if not template or not stress_class:
            return
        key = f"{pos}:{template}:{stress_class}"
        bucket = self.nominals.setdefault(
            key,
            InductionBucket({"pos": pos, "declension": template, "stress_class": stress_class}),
        )
        self._add(bucket, lemma, accented_lemma, form, tags)

    def add_verb(
        self,
        *,
        template: str,
        lemma: str,
        accented_lemma: str,
        form: str,
        tags: Iterable[str],
    ) -> None:
        if not template:
            return
        key = template
        bucket = self.verbs.setdefault(key, InductionBucket({"conjugation": template}))
        self._add(bucket, lemma, accented_lemma, form, tags)

    def _add(
        self,
        bucket: InductionBucket,
        lemma: str,
        accented_lemma: str,
        form: str,
        tags: Iterable[str],
    ) -> None:
        tag_tuple = sort_tags(tags)
        if not tag_tuple:
            return
        stripped_lemma = lower_key(accented_lemma or lemma)
        stripped_form = lower_key(form)
        if not stripped_lemma or not stripped_form:
            return
        common = common_prefix_len(stripped_lemma, stripped_form)
        stress_index = stressed_base_index(form)
        stress_site = "none"
        if stress_index is not None:
            stress_site = "ending" if stress_index >= common else "stem"
        shape = (
            stripped_lemma[common:],
            stripped_form[common:],
            _accented_suffix_by_base_index(form, common),
            stress_site,
            first_stress_mark(form) or "",
        )
        key = cell_key(tag_tuple)
        bucket.counters[key][shape] += 1
        example_key = (key, shape)
        if len(bucket.examples[example_key]) < 5:
            bucket.examples[example_key].append(
                {"lemma": normalize_lt(accented_lemma or lemma), "form": normalize_lt(form)}
            )

    def to_json(self, source: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        conflicts: list[dict[str, Any]] = []
        data = {
            "version": 1,
            "source": safe_relative(source),
            "note": "Empirically induced from Kaikki rendered forms; majority shapes are selected per cell.",
            "nominals": self._finalize_group(self.nominals, "nominal", conflicts),
            "verbs": self._finalize_group(self.verbs, "verb", conflicts),
        }
        return data, conflicts

    def _finalize_group(
        self,
        buckets: dict[str, InductionBucket],
        kind: str,
        conflicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key in sorted(buckets):
            bucket = buckets[key]
            cells: dict[str, Any] = {}
            for cell in sorted(bucket.counters):
                counter = bucket.counters[cell]
                alternatives = []
                for shape, count in counter.most_common(8):
                    shape_data = _shape_to_dict(shape)
                    shape_data["count"] = count
                    shape_data["examples"] = bucket.examples.get((cell, shape), [])
                    alternatives.append(shape_data)
                majority = dict(alternatives[0]) if alternatives else {}
                majority.pop("count", None)
                majority.pop("examples", None)
                cells[cell] = {
                    "count": sum(counter.values()),
                    "majority": majority,
                    "alternatives": alternatives,
                    "conflict": len(counter) > 1,
                }
                if len(counter) > 1:
                    conflicts.append(
                        {
                            "kind": kind,
                            "paradigm": key,
                            "cell": cell,
                            "alternatives": alternatives[:5],
                        }
                    )
            out[key] = {**bucket.metadata, "cells": cells}
        return out


def _shape_to_dict(shape: tuple[str, str, str, str, str]) -> dict[str, str]:
    return {
        "lemma_suffix": shape[0],
        "form_suffix": shape[1],
        "accented_form_suffix": shape[2],
        "stress_site": shape[3],
        "accent_mark": shape[4],
    }


def _accented_suffix_by_base_index(text: str, start_index: int) -> str:
    out: list[str] = []
    base_index = -1
    include = False
    for ch in unicodedata.normalize("NFD", normalize_lt(text)):
        if unicodedata.combining(ch):
            if include:
                out.append(ch)
        else:
            base_index += 1
            include = base_index >= start_index
            if include:
                out.append(ch)
    return normalize_lt("".join(out))


def iter_kaikki(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("lang_code") == "lt":
                yield row


def normalize_stress_classes(raw: str | None) -> list[str]:
    if not raw:
        return []
    text = str(raw).lower().strip()
    text = text.replace("stress-pattern-", "")
    text = text.replace("ᵃ", "a").replace("ᵇ", "b")
    text = re.sub(r"[^0-9ab/\-]+", "", text)
    classes: list[str] = []
    for part in re.split(r"[-/]+", text):
        if part in VALID_STRESS_CLASSES:
            classes.append(part)
    return list(dict.fromkeys(classes))


def stress_classes(row: dict[str, Any]) -> list[str]:
    classes: list[str] = []
    for form in row.get("forms") or []:
        for tag in form.get("tags") or []:
            match = STRESS_TAG_RE.match(str(tag))
            if match:
                classes.extend(normalize_stress_classes(match.group(1)))
    for tpl in row.get("head_templates") or []:
        args = tpl.get("args") or {}
        for key in ("2", "3", "4", "stress", "accent", "ap"):
            classes.extend(normalize_stress_classes(args.get(key)))
        expansion = tpl.get("expansion") or ""
        for match in STRESS_EXPANSION_RE.finditer(expansion):
            classes.extend(normalize_stress_classes(match.group(1)))
    for marker in inflection_markers(row, source="declension"):
        for part in marker.split("-"):
            classes.extend(normalize_stress_classes(part))
    return [cls for cls in dict.fromkeys(classes) if cls in VALID_STRESS_CLASSES]


def extract_head(row: dict[str, Any]) -> str:
    for preferred in ("lt-noun", "lt-proper noun", "lt-adj", "lt-verb", "head"):
        for tpl in row.get("head_templates") or []:
            if tpl.get("name") != preferred:
                continue
            args = tpl.get("args") or {}
            for key in ("head", "head1"):
                forms = split_template_forms(args.get(key))
                if forms:
                    return forms[0]
    for form in row.get("forms") or []:
        tags = set(form.get("tags") or ())
        if "canonical" in tags:
            text = normalize_lt(form.get("form") or "")
            if text and " stress pattern " not in text and "\n" not in text:
                if len(text.split()) == 1:
                    return text
    return normalize_lt(row.get("word") or "")


def extract_gender(row: dict[str, Any]) -> str | None:
    for tpl in row.get("head_templates") or []:
        args = tpl.get("args") or {}
        for key in ("g", "1"):
            value = args.get(key)
            if value in ("m", "masculine"):
                return "masculine"
            if value in ("f", "feminine"):
                return "feminine"
            if value in ("n", "neuter"):
                return "neuter"
    for form in row.get("forms") or []:
        tags = set(form.get("tags") or ())
        for gender in ("masculine", "feminine", "neuter"):
            if gender in tags:
                return gender
    return None


def extract_plural(row: dict[str, Any]) -> str | None:
    best: str | None = None
    for form in row.get("forms") or []:
        tags = set(form.get("tags") or ())
        if "plural" not in tags:
            continue
        text = normalize_lt(form.get("form") or "")
        if not text or text.startswith("no-table"):
            continue
        if form.get("source") == "declension" and "nominative" in tags and has_stress(text):
            return text
        if has_stress(text) and best is None:
            best = text
    return best


def inflection_markers(row: dict[str, Any], source: str | None = None) -> list[str]:
    markers: list[str] = []
    for form in row.get("forms") or []:
        tags = set(form.get("tags") or ())
        if "inflection-template" not in tags:
            continue
        if source and form.get("source") != source:
            continue
        marker = normalize_lt(form.get("form") or "")
        if marker:
            markers.append(marker)
    return markers


def nominal_template(row: dict[str, Any]) -> str | None:
    markers = inflection_markers(row, source="declension") or inflection_markers(row, source="inflection")
    for marker in markers:
        if marker.startswith(("lt-noun", "lt-proper", "lt-adj", "lt-3rd", "l", "small")):
            return marker
    return markers[0] if markers else None


def verb_template(row: dict[str, Any]) -> str | None:
    for tpl in row.get("inflection_templates") or []:
        name = tpl.get("name") or ""
        if name.startswith("lt-conj") and name != "lt-conj":
            return name
    for marker in inflection_markers(row, source="conjugation"):
        if marker.startswith("lt-conj") and marker != "lt-conj":
            return marker
    return "lt-conj" if any((tpl.get("name") == "lt-conj") for tpl in row.get("inflection_templates") or []) else None


def clean_tags(tags: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    for tag in tags:
        tag = str(tag)
        if tag in PSEUDO_FORM_TAGS or tag == "error-unrecognized-form":
            continue
        if tag.startswith("stress-pattern-"):
            continue
        result.append(tag)
    return sort_tags(result)


def usable_form(text: str | None) -> str:
    normalized = normalize_lt(text or "")
    if not normalized:
        return ""
    if normalized == "-" or normalized.startswith("no-table") or "{" in normalized or "}" in normalized:
        return ""
    if normalized.startswith("lt-") and not has_stress(normalized):
        return ""
    return normalized


def generic_form_rows(row: dict[str, Any], lemma: str, pos: str) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    for form in row.get("forms") or []:
        tags = set(form.get("tags") or ())
        if tags & PSEUDO_FORM_TAGS:
            continue
        source = form.get("source") or ""
        if source and source not in TABLE_SOURCES:
            continue
        cleaned_tags = clean_tags(tags)
        if not cleaned_tags:
            continue
        text = usable_form(form.get("form"))
        if not text or not has_stress(text):
            continue
        if source and source not in TABLE_SOURCES:
            continue
        rows.append((lower_key(text), text, lemma, pos, tags_json(cleaned_tags)))
    return rows


def verb_form_rows(row: dict[str, Any], lemma: str, pos: str) -> list[tuple[str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(text: str, tags: Iterable[str]) -> None:
        form = usable_form(text)
        cleaned = clean_tags(tags)
        if not form or not has_stress(form) or not cleaned:
            return
        key = (form, tags_json(cleaned))
        if key in seen:
            return
        seen.add(key)
        rows.append((lower_key(form), form, lemma, pos, tags_json(cleaned)))

    for form in row.get("forms") or []:
        tags = clean_tags(form.get("tags") or ())
        tag_set = set(tags)
        if "canonical" in tag_set:
            add(form.get("form") or "", ("infinitive",))
        elif ({"present", "past"} & tag_set) and "third-person" in tag_set:
            add(form.get("form") or "", tags)
        elif (tag_set & NONFINITE_VERB_TAGS) and form.get("source") == "conjugation":
            add(form.get("form") or "", tags)

    for tpl in row.get("inflection_templates") or []:
        if tpl.get("name") != "lt-conj":
            continue
        args = tpl.get("args") or {}
        for index, tags in FINITE_CONJ_TAGS.items():
            for form in split_template_forms(args.get(str(index))):
                add(form, tags)
    return rows


def verb_principal_parts(row: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    infinitive = extract_head(row)
    present_3 = None
    past_3 = None
    for tpl in row.get("head_templates") or []:
        if tpl.get("name") != "lt-verb":
            continue
        args = tpl.get("args") or {}
        present = split_template_forms(args.get("1"))
        past = split_template_forms(args.get("2"))
        if present:
            present_3 = present[0]
        if past:
            past_3 = past[0]
    for form in row.get("forms") or []:
        tags = set(form.get("tags") or ())
        text = usable_form(form.get("form"))
        if not text:
            continue
        if "canonical" in tags:
            infinitive = text
        elif {"present", "third-person"} <= tags:
            present_3 = text
        elif {"past", "third-person"} <= tags:
            past_3 = text
    if not (has_stress(infinitive) and has_stress(present_3) and has_stress(past_3)):
        return None, None, None
    return infinitive, present_3, past_3


def create_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;

        CREATE TABLE nominals(
          lemma TEXT NOT NULL,
          pos TEXT NOT NULL,
          accented_lemma TEXT NOT NULL,
          stress_class TEXT NOT NULL,
          plural_accented TEXT,
          gender TEXT,
          source TEXT NOT NULL
        );

        CREATE TABLE verbs(
          lemma TEXT NOT NULL,
          accented_infinitive TEXT NOT NULL,
          present_3 TEXT NOT NULL,
          past_3 TEXT NOT NULL,
          source TEXT NOT NULL
        );

        CREATE TABLE forms(
          stripped TEXT NOT NULL,
          accented TEXT NOT NULL,
          lemma TEXT NOT NULL,
          pos TEXT NOT NULL,
          tags TEXT NOT NULL
        );

        CREATE TABLE closed_draft(
          lemma TEXT NOT NULL,
          upos TEXT NOT NULL,
          accented_head TEXT,
          frequency INTEGER NOT NULL,
          verified INTEGER NOT NULL
        );

        CREATE TABLE nominal_meta(
          lemma TEXT NOT NULL,
          pos TEXT NOT NULL,
          stress_class TEXT NOT NULL,
          declension_template TEXT
        );

        CREATE TABLE verb_meta(
          lemma TEXT NOT NULL,
          conjugation_template TEXT
        );

        CREATE UNIQUE INDEX forms_unique ON forms(stripped, accented, lemma, pos, tags);
        CREATE INDEX forms_lookup ON forms(stripped);
        CREATE INDEX forms_lemma ON forms(lemma, pos);
        CREATE INDEX nominals_lookup ON nominals(lemma, pos, stress_class);
        CREATE INDEX verbs_lookup ON verbs(lemma);
        """
    )


def matas_closed_rows(matas: Path, accented_heads: dict[str, str]) -> list[tuple[str, str, str | None, int, int]]:
    if not matas.exists():
        return []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    with matas.open(encoding="utf-8") as fh:
        for line in fh:
            if not line or not line[0].isdigit():
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 4 or not cols[0].isdigit():
                continue
            lemma = strip_accents(cols[2]).lower()
            upos = cols[3]
            if lemma and any(ch.isalpha() for ch in lemma):
                counts[lemma][upos] += 1

    rows: list[tuple[str, str, str | None, int, int]] = []
    for lemma, upos_counts in counts.items():
        frequency = sum(upos_counts.values())
        upos, _ = upos_counts.most_common(1)[0]
        if frequency < 50 or upos not in CLOSED_UPOS:
            continue
        rows.append((lemma, upos, accented_heads.get(lemma), frequency, 0))
    rows.sort(key=lambda row: (-row[3], row[0]))
    return rows


def write_closed_markdown(path: Path, rows: list[tuple[str, str, str | None, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Closed-Class Draft",
        "",
        "| lemma | UPOS | frequency | accented_head | verified |",
        "|---|---:|---:|---|---:|",
    ]
    for lemma, upos, accented, freq, verified in rows:
        lines.append(f"| {lemma} | {upos} | {freq} | {accented or ''} | {verified} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_conflicts(path: Path, conflicts: list[dict[str, Any]]) -> None:
    if not conflicts:
        return
    conflict_path = path.with_name("paradigm_conflicts.jsonl")
    with conflict_path.open("w", encoding="utf-8") as fh:
        for row in conflicts:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build_lexicon(
    *,
    kaikki: Path = DEFAULT_KAIKKI,
    matas: Path = DEFAULT_MATAS,
    output: Path = DEFAULT_LEXICON,
    tables: Path = DEFAULT_TABLES,
    closed_markdown: Path = DEFAULT_CLOSED_DRAFT,
    include_closed: bool = True,
    quiet: bool = False,
) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    tables.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    nominal_rows: list[tuple[str, str, str, str, str | None, str | None, str]] = []
    nominal_meta_rows: list[tuple[str, str, str, str | None]] = []
    verb_rows: list[tuple[str, str, str, str, str]] = []
    verb_meta_rows: list[tuple[str, str | None]] = []
    form_rows: list[tuple[str, str, str, str, str]] = []
    accented_heads: dict[str, str] = {}
    inducer = ParadigmInducer()

    entries = 0
    for row in iter_kaikki(kaikki):
        entries += 1
        pos = str(row.get("pos") or "")
        lemma = lower_key(row.get("word") or "")
        if not lemma:
            continue
        head = extract_head(row)
        if has_stress(head):
            accented_heads.setdefault(lower_key(head), head)
            accented_heads.setdefault(lemma, head)

        if pos == "verb":
            rows = verb_form_rows(row, lemma, pos)
            form_rows.extend(rows)
            infinitive, present_3, past_3 = verb_principal_parts(row)
            conjugation = verb_template(row)
            if infinitive and present_3 and past_3:
                verb_rows.append((lemma, infinitive, present_3, past_3, f"kaikki:{conjugation or 'verb'}"))
                verb_meta_rows.append((lemma, conjugation))
                for _, form, _, _, raw_tags in rows:
                    tags = json.loads(raw_tags)
                    if set(tags) & NONFINITE_VERB_TAGS:
                        continue
                    inducer.add_verb(
                        template=conjugation or "lt-conj",
                        lemma=lemma,
                        accented_lemma=infinitive,
                        form=form,
                        tags=tags,
                    )
            continue

        rows = generic_form_rows(row, lemma, pos)
        form_rows.extend(rows)

        if pos not in NOMINAL_POS:
            continue
        classes = stress_classes(row)
        if not classes or not has_stress(head):
            continue
        template = nominal_template(row)
        plural = extract_plural(row)
        gender = extract_gender(row)
        for stress_class in classes:
            nominal_rows.append(
                (
                    lemma,
                    pos,
                    head,
                    stress_class,
                    plural,
                    gender,
                    f"kaikki:{template or pos}",
                )
            )
            nominal_meta_rows.append((lemma, pos, stress_class, template))
            for _, form, _, _, raw_tags in rows:
                tags = json.loads(raw_tags)
                if not (set(tags) & CASE_TAGS):
                    continue
                inducer.add_nominal(
                    pos=pos,
                    template=template or pos,
                    stress_class=stress_class,
                    lemma=lemma,
                    accented_lemma=head,
                    form=form,
                    tags=tags,
                )

    closed_rows = matas_closed_rows(matas, accented_heads) if include_closed else []

    db = sqlite3.connect(tmp)
    try:
        create_schema(db)
        db.executemany("INSERT INTO nominals VALUES (?,?,?,?,?,?,?)", nominal_rows)
        db.executemany("INSERT INTO verbs VALUES (?,?,?,?,?)", verb_rows)
        db.executemany("INSERT OR IGNORE INTO forms VALUES (?,?,?,?,?)", form_rows)
        db.executemany("INSERT INTO closed_draft VALUES (?,?,?,?,?)", closed_rows)
        db.executemany("INSERT INTO nominal_meta VALUES (?,?,?,?)", nominal_meta_rows)
        db.executemany("INSERT INTO verb_meta VALUES (?,?)", verb_meta_rows)
        db.commit()
    finally:
        db.close()

    os.replace(tmp, output)
    table_data, conflicts = inducer.to_json(kaikki)
    tables.write_text(json.dumps(table_data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_conflicts(tables, conflicts)
    if include_closed:
        write_closed_markdown(closed_markdown, closed_rows)

    summary = {
        "entries": entries,
        "nominals": len(nominal_rows),
        "verbs": len(verb_rows),
        "forms": len(set(form_rows)),
        "closed_draft": len(closed_rows),
        "nominal_tables": len(table_data["nominals"]),
        "verb_tables": len(table_data["verbs"]),
        "conflicts": len(conflicts),
    }
    if not quiet:
        for key, value in summary.items():
            print(f"{key}: {value:,}")
        print(f"wrote {safe_relative(output)}")
        print(f"wrote {safe_relative(tables)}")
        if include_closed:
            print(f"wrote {safe_relative(closed_markdown)}")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract open accentuator lexicon data from Kaikki.")
    parser.add_argument("--kaikki", type=Path, default=DEFAULT_KAIKKI, help="Path to kaikki-lt.jsonl.")
    parser.add_argument("--matas", type=Path, default=DEFAULT_MATAS, help="Path to MATAS3.conllu for closed draft.")
    parser.add_argument("--output", type=Path, default=DEFAULT_LEXICON, help="Output lexicon SQLite path.")
    parser.add_argument("--tables", type=Path, default=DEFAULT_TABLES, help="Output induced paradigm JSON path.")
    parser.add_argument("--closed-markdown", type=Path, default=DEFAULT_CLOSED_DRAFT, help="Output closed draft markdown.")
    parser.add_argument("--no-closed-draft", action="store_true", help="Skip MATAS closed-class draft extraction.")
    parser.add_argument("--quiet", action="store_true", help="Suppress summary output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    build_lexicon(
        kaikki=args.kaikki,
        matas=args.matas,
        output=args.output,
        tables=args.tables,
        closed_markdown=args.closed_markdown,
        include_closed=not args.no_closed_draft,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
