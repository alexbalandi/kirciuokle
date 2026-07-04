from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from kirciuokle.dictionary import WordDictionary
from kirciuokle.dictionary_types import WordDictionaryPutEntry
from kirciuokle.disambiguate import AccentVariant


def make_dictionary(tmp_path: Path) -> WordDictionary:
    return WordDictionary(tmp_path / "words.sqlite")


def put_word(
    dictionary: WordDictionary,
    word: str,
    variants: list[AccentVariant],
    *,
    default_form: str | None = None,
    accent_type: str | None = None,
    default_form_title: str | None = None,
    accent_type_title: str | None = None,
) -> None:
    entry: WordDictionaryPutEntry = {
        "word": word,
        "variants": variants,
        "defaultForm": default_form if default_form is not None else variants[0]["form"] if variants else None,
        "accentType": accent_type if accent_type is not None else "ONE" if variants else "NONE",
        "defaultFormTitle": default_form_title
        if default_form_title is not None
        else title_case(default_form if default_form is not None else variants[0]["form"] if variants else None),
        "accentTypeTitle": accent_type_title
        if accent_type_title is not None
        else accent_type if accent_type is not None else "ONE" if variants else "NONE",
    }
    dictionary.put_words([entry])


def insert_raw_word(dictionary: WordDictionary, **row: Any) -> None:
    with sqlite3.connect(dictionary.path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO words "
            "(word, variants, fetched_at, negative_until, default_form, accent_type, "
            "default_form_title, accent_type_title) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["word"],
                row["variants"],
                row.get("fetched_at", "2026-07-02T00:00:00.000Z"),
                row.get("negative_until"),
                row.get("default_form"),
                row.get("accent_type"),
                row.get("default_form_title"),
                row.get("accent_type_title"),
            ),
        )
        conn.commit()


def title_case(form: str | None) -> str | None:
    if not form:
        return None
    return f"{form[0].upper()}{form[1:]}"
