# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Empirical Lithuanian accent paradigm engine.

The core functions are pure: callers provide observed/induced paradigm data and
receive accented forms plus their morphology tags. The command-line interface is
only a smoke-test/help surface for W2 tooling.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable

try:  # pragma: no cover - import style depends on script/package execution.
    from ._common import (
        COMBINING_ACUTE,
        COMBINING_GRAVE,
        COMBINING_TILDE,
        DEFAULT_TABLES,
        STRESS_NAMES,
        cell_key,
        first_stress_mark,
        has_stress,
        lower_key,
        morphology_label,
        normalize_lt,
        parse_tags,
        sort_tags,
        split_template_forms,
        strip_accents,
        stressed_base_index,
    )
except ImportError:  # pragma: no cover
    from _common import (
        COMBINING_ACUTE,
        COMBINING_GRAVE,
        COMBINING_TILDE,
        DEFAULT_TABLES,
        STRESS_NAMES,
        cell_key,
        first_stress_mark,
        has_stress,
        lower_key,
        morphology_label,
        normalize_lt,
        parse_tags,
        sort_tags,
        split_template_forms,
        strip_accents,
        stressed_base_index,
    )

GeneratedForm = tuple[str, tuple[str, ...]]

MARK_BY_NAME = {name: mark for mark, name in STRESS_NAMES.items()}
MARK_BY_NAME.update({"acute": COMBINING_ACUTE, "grave": COMBINING_GRAVE, "tilde": COMBINING_TILDE})

VERB_ALIAS_CELLS = {
    "infinitive": ("infinitive",),
    "present_1sg": ("present", "first-person", "singular"),
    "present_2sg": ("present", "second-person", "singular"),
    "present_3": ("present", "third-person"),
    "present_3sg": ("present", "third-person"),
    "present_3pl": ("present", "third-person", "plural"),
    "past_1sg": ("past", "first-person", "singular"),
    "past_2sg": ("past", "second-person", "singular"),
    "past_3": ("past", "third-person"),
    "past_3sg": ("past", "third-person"),
    "past_3pl": ("past", "third-person", "plural"),
    "future_1sg": ("future", "first-person", "singular"),
    "future_2sg": ("future", "second-person", "singular"),
    "future_3": ("future", "third-person"),
    "future_3sg": ("future", "third-person"),
    "future_3pl": ("future", "third-person", "plural"),
    "conditional_3": ("conditional", "third-person"),
    "conditional_3sg": ("conditional", "third-person"),
    "conditional_3pl": ("conditional", "third-person", "plural"),
    "imperative_2sg": ("imperative", "second-person", "singular"),
    "imperative_1pl": ("imperative", "first-person", "plural"),
    "imperative_2pl": ("imperative", "second-person", "plural"),
}


def normalize_cell(cell: str | Iterable[str]) -> str:
    if isinstance(cell, str) and cell in VERB_ALIAS_CELLS:
        return cell_key(VERB_ALIAS_CELLS[cell])
    return cell_key(cell)


def accent_nominal(
    accented_lemma: str,
    stress_class: str,
    declension_info: dict[str, Any] | None,
    cell: str | Iterable[str],
) -> list[GeneratedForm]:
    """Return accented nominal forms for one paradigm cell.

    ``declension_info`` may contain either observed cells:

    ``{"forms_by_cell": {"nominative|singular": [{"form": "...", "tags": [...]}]}}``

    or an induced suffix table under ``"paradigm_table"``. The observed path is
    what the W2 selfcheck and standalone generator use; the suffix-table fallback
    keeps the engine usable for held-out lemmas that share a regular table.
    """

    info = declension_info or {}
    key = normalize_cell(cell)
    exact = _forms_from_info(info, key)
    if exact:
        return exact

    if key in (cell_key(("canonical",)), cell_key(("nominative", "singular"))):
        return [(normalize_lt(accented_lemma), parse_tags(cell))]

    table = info.get("paradigm_table")
    stripped_lemma = info.get("stripped_lemma") or lower_key(accented_lemma)
    if isinstance(table, dict):
        generated = _generate_from_induced_table(
            normalize_lt(accented_lemma), stripped_lemma, table, key, parse_tags(cell)
        )
        if generated:
            return generated

    return []


def accent_verb(
    accented_infinitive: str,
    present_3: str,
    past_3: str,
    cell: str | Iterable[str],
    conjugation_info: dict[str, Any] | None = None,
) -> list[GeneratedForm]:
    """Return accented verb forms for one finite/infinitive cell."""

    key = normalize_cell(cell)
    info = conjugation_info or {}
    exact = _forms_from_info(info, key)
    if exact:
        return exact

    tags = parse_tags(VERB_ALIAS_CELLS.get(str(cell), cell if not isinstance(cell, str) else key.split("|")))
    accented_infinitive = normalize_lt(accented_infinitive)
    present_3 = normalize_lt(present_3)
    past_3 = normalize_lt(past_3)

    if key == cell_key(("infinitive",)):
        return [(accented_infinitive, ("infinitive",))]
    if key in (cell_key(("present", "third-person")), cell_key(("present", "third-person", "plural"))):
        return [(present_3, parse_tags(key))]
    if key in (cell_key(("past", "third-person")), cell_key(("past", "third-person", "plural"))):
        return [(past_3, parse_tags(key))]

    stripped = strip_accents(accented_infinitive)
    stem = stripped[:-2] if stripped.endswith("ti") else stripped
    accented_stem = accented_infinitive[:-2] if strip_accents(accented_infinitive).endswith("ti") else accented_infinitive

    if key in (cell_key(("future", "third-person")), cell_key(("future", "third-person", "plural"))):
        form = _future_third_from_infinitive(accented_stem, stem)
        return [(form, parse_tags(key))]
    if key in (cell_key(("conditional", "third-person")), cell_key(("conditional", "third-person", "plural"))):
        return [(accented_stem + "tų", parse_tags(key))]
    if key == cell_key(("imperative", "second-person", "singular")):
        return [(accented_stem + "k", parse_tags(key))]

    table = info.get("paradigm_table")
    if isinstance(table, dict):
        generated = _generate_from_induced_table(accented_infinitive, stripped, table, key, tags)
        if generated:
            return generated
    return []


def load_paradigm_tables(path: str | Path = DEFAULT_TABLES) -> dict[str, Any]:
    """Load an induced ``paradigm_tables.json`` artifact."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_forms_by_cell(rows: Iterable[tuple[str, str] | dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    forms_by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict):
            form = normalize_lt(row.get("form") or row.get("accented") or "")
            tags = parse_tags(row.get("tags") or ())
        else:
            form = normalize_lt(row[0])
            tags = parse_tags(row[1])
        if not form or not tags:
            continue
        forms_by_cell.setdefault(cell_key(tags), []).append({"form": form, "tags": list(tags)})
    return forms_by_cell


def _forms_from_info(info: dict[str, Any], key: str) -> list[GeneratedForm]:
    by_cell = info.get("forms_by_cell") or info.get("cells")
    if not isinstance(by_cell, dict):
        return []
    entries = by_cell.get(key)
    if not entries:
        return []

    results: list[GeneratedForm] = []
    for entry in entries:
        if isinstance(entry, str):
            results.append((normalize_lt(entry), tuple(key.split("|"))))
        elif isinstance(entry, (tuple, list)) and entry:
            form = normalize_lt(str(entry[0]))
            tags = parse_tags(entry[1] if len(entry) > 1 else key.split("|"))
            results.append((form, tags))
        elif isinstance(entry, dict):
            form = normalize_lt(entry.get("form") or entry.get("accented") or "")
            tags = parse_tags(entry.get("tags") or key.split("|"))
            if form:
                results.append((form, tags))
    return _dedupe_generated(results)


def _generate_from_induced_table(
    accented_lemma: str,
    stripped_lemma: str,
    table: dict[str, Any],
    key: str,
    fallback_tags: tuple[str, ...],
) -> list[GeneratedForm]:
    cell = (table.get("cells") or {}).get(key)
    if not isinstance(cell, dict):
        return []
    majority = cell.get("majority") or {}
    if not isinstance(majority, dict):
        return []
    lemma_suffix = majority.get("lemma_suffix") or ""
    form_suffix = majority.get("form_suffix") or ""
    accented_form_suffix = majority.get("accented_form_suffix") or form_suffix
    stress_site = majority.get("stress_site")
    accent_mark = majority.get("accent_mark")

    if lemma_suffix and not stripped_lemma.endswith(lemma_suffix):
        return []
    stripped_stem = stripped_lemma[: len(stripped_lemma) - len(lemma_suffix)] if lemma_suffix else stripped_lemma
    stripped_form = stripped_stem + form_suffix

    if stress_site == "ending":
        return [(normalize_lt(stripped_stem + accented_form_suffix), fallback_tags)]
    if stress_site == "stem":
        return [(_copy_lemma_accent(accented_lemma, stripped_form, accent_mark), fallback_tags)]
    if has_stress(accented_lemma):
        return [(_copy_lemma_accent(accented_lemma, stripped_form, accent_mark), fallback_tags)]
    return [(normalize_lt(stripped_form), fallback_tags)]


def _copy_lemma_accent(accented_lemma: str, stripped_form: str, accent_mark: str | None) -> str:
    index = stressed_base_index(accented_lemma)
    if index is None or index >= len(stripped_form):
        return normalize_lt(stripped_form)
    mark = MARK_BY_NAME.get(accent_mark or first_stress_mark(accented_lemma) or "")
    if not mark:
        return normalize_lt(stripped_form)
    return _apply_stress(stripped_form, index, mark)


def _apply_stress(stripped: str, base_index: int, mark: str) -> str:
    # The stress mark must follow the cluster's own combining marks (ė is
    # e + dot above; both marks share a combining class, so NFC cannot
    # reorder a mis-ordered sequence).
    out: list[str] = []
    current = -1
    pending = False
    for ch in unicodedata.normalize("NFD", stripped):
        if not unicodedata.combining(ch):
            if pending:
                out.append(mark)
                pending = False
            current += 1
            if current == base_index:
                pending = True
        out.append(ch)
    if pending:
        out.append(mark)
    return normalize_lt("".join(out))


def _future_third_from_infinitive(accented_stem: str, stripped_stem: str) -> str:
    # The productive future 3rd person often carries circumflex on final long y,
    # e.g. darýti -> darỹs. If the infinitive stem is accented elsewhere, keep it.
    if stripped_stem.endswith("y"):
        return normalize_lt(stripped_stem[:-1] + "ỹs")
    return normalize_lt(accented_stem + "s")


def _dedupe_generated(forms: Iterable[GeneratedForm]) -> list[GeneratedForm]:
    seen: set[tuple[str, tuple[str, ...]]] = set()
    result: list[GeneratedForm] = []
    for form, tags in forms:
        key = (normalize_lt(form), sort_tags(tags))
        if key in seen:
            continue
        seen.add(key)
        result.append(key)
    return result


def _demo(args: argparse.Namespace) -> int:
    forms = accent_verb(args.infinitive, args.present3, args.past3, args.cell)
    for form, tags in forms:
        print(f"{form}\t{morphology_label('verb', tags)}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure empirical accent paradigm helpers.")
    parser.add_argument("--cell", default="future_3", help="Cell alias or pipe-separated tag set to demo.")
    parser.add_argument("--infinitive", default="darýti", help="Accented infinitive for the verb demo.")
    parser.add_argument("--present3", default="dãro", help="Accented present third-person for the verb demo.")
    parser.add_argument("--past3", default="dãrė", help="Accented past third-person for the verb demo.")
    parser.add_argument(
        "--show-demo",
        action="store_true",
        help="Print a tiny verb demo. Without this flag the command only validates imports.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.show_demo:
        return _demo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
