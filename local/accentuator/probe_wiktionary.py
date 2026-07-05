# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""W1 probe: what does Wiktionary (kaikki extract) give the open accentuator?

Measures, per docs/PLAN-open-accentuator.md:
  1. lemma inventory with accent classes / accented tables, by POS;
  2. token-mass coverage against MATAS gold lemma frequencies;
  3. form-level agreement against the D1/SQLite VDU cache (QA ground truth).

Usage:
    uv run local/accentuator/probe_wiktionary.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
KAIKKI = BASE / "tagger-hf" / "data" / "raw" / "kaikki-lt.jsonl"
MATAS = BASE / "tagger-hf" / "data" / "raw" / "MATAS3.conllu"
D1_SQLITE = BASE / "data" / "words.sqlite"

STRESS_MARKS = ("̀", "́", "̃")
STRESS_TAG = re.compile(r"^stress-pattern-(\S+)$")
PSEUDO_FORM_TAGS = {"table-tags", "inflection-template", "class"}


def strip_accents(s: str) -> str:
    decomposed = unicodedata.normalize("NFD", s)
    return unicodedata.normalize(
        "NFC", "".join(c for c in decomposed if c not in STRESS_MARKS)
    )


def norm(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def main() -> None:
    # ---- 1. parse kaikki -------------------------------------------------
    entries = 0
    by_pos: Counter[str] = Counter()
    with_class: Counter[str] = Counter()
    with_table: Counter[str] = Counter()
    lemma_class: dict[tuple[str, str], set[str]] = defaultdict(set)
    # stripped form -> set of accented options (for D1 comparison)
    form_index: dict[str, set[str]] = defaultdict(set)
    lemmas_any: set[str] = set()
    lemmas_with_class: set[str] = set()

    for line in KAIKKI.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("lang_code") != "lt":
            continue
        word = norm(row.get("word", ""))
        pos = row.get("pos", "?")
        entries += 1
        by_pos[pos] += 1
        lemmas_any.add(word.lower())

        stress = None
        table_forms = 0
        for form in row.get("forms", []):
            tags = set(form.get("tags", ()))
            if tags & PSEUDO_FORM_TAGS:
                continue
            text = norm(form.get("form", ""))
            if not text or "{" in text or text.startswith("no-table"):
                continue
            for tag in tags:
                m = STRESS_TAG.match(tag)
                if m:
                    stress = m.group(1)
            if tags and text:
                table_forms += 1
                form_index[strip_accents(text).lower()].add(text)
        # accent class can also live in the lt-* head template args (position 3/4)
        if stress is None:
            for tpl in row.get("head_templates", []):
                expansion = tpl.get("expansion", "")
                m = re.search(r"stress pattern (\S+)", expansion)
                if m:
                    stress = m.group(1)
                    break

        if stress is not None:
            with_class[pos] += 1
            lemma_class[(word.lower(), pos)].add(stress)
            lemmas_with_class.add(word.lower())
        if table_forms >= 4:
            with_table[pos] += 1

    print(f"entries: {entries:,}")
    print(f"{'pos':12} {'entries':>8} {'w/class':>8} {'w/table':>8}")
    for pos, n in by_pos.most_common(10):
        print(f"{pos:12} {n:8,} {with_class[pos]:8,} {with_table[pos]:8,}")
    print(f"distinct stripped forms in tables: {len(form_index):,}")
    multi_class = sum(1 for v in lemma_class.values() if len(v) > 1)
    print(f"lemma+pos with class: {len(lemma_class):,} (multi-class: {multi_class:,})")

    # ---- 2. MATAS token-mass coverage ------------------------------------
    lemma_freq: Counter[str] = Counter()
    for line in MATAS.open(encoding="utf-8"):
        if line[:1].isdigit():
            cols = line.split("\t")
            if len(cols) >= 4 and cols[0].isdigit():
                lemma = cols[2].lower()
                if any(ch.isalpha() for ch in lemma):
                    lemma_freq[lemma] += 1
    total = sum(lemma_freq.values())
    cov_any = sum(n for l, n in lemma_freq.items() if l in lemmas_any)
    cov_class = sum(n for l, n in lemma_freq.items() if l in lemmas_with_class)
    print(f"\nMATAS lemma tokens: {total:,}")
    print(f"  token mass with lemma in kaikki (any): {100 * cov_any / total:.1f}%")
    print(f"  token mass with lemma in kaikki (with class): {100 * cov_class / total:.1f}%")
    missing = [(l, n) for l, n in lemma_freq.most_common(400) if l not in lemmas_with_class]
    print("  top frequent lemmas missing class info:", [l for l, _ in missing[:15]])

    # ---- 3. D1 cache agreement -------------------------------------------
    db = sqlite3.connect(D1_SQLITE)
    rows = db.execute(
        "SELECT word, variants, default_form FROM words WHERE accent_type IS NOT NULL"
        " AND variants != '[]'"
    ).fetchall()
    covered = agree_default = agree_all = disjoint = 0
    samples: list[str] = []
    for word, variants_json, default_form in rows:
        options = form_index.get(word)
        if not options:
            continue
        covered += 1
        vdu_forms = {norm(v["form"]).lower() for v in json.loads(variants_json)}
        wik = {norm(o).lower() for o in options}
        if default_form and norm(default_form).lower() in wik:
            agree_default += 1
        if vdu_forms & wik:
            agree_all += 1
        else:
            disjoint += 1
            if len(samples) < 12:
                samples.append(f"{word}: VDU={sorted(vdu_forms)} WIK={sorted(wik)}")
    print(f"\nD1 positive entries: {len(rows):,}; covered by kaikki tables: {covered:,} "
          f"({100 * covered / len(rows):.1f}%)")
    if covered:
        print(f"  VDU default form present in wiktionary options: {100 * agree_default / covered:.1f}%")
        print(f"  any variant overlap: {100 * agree_all / covered:.1f}% | disjoint: {disjoint}")
    print("  disjoint samples:")
    for s in samples:
        print("   ", s)


if __name__ == "__main__":
    main()
