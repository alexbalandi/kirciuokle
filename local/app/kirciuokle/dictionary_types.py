from __future__ import annotations

from typing import TypedDict

from .disambiguate import AccentVariant


class WordAccentEntry(TypedDict):
    variants: list[AccentVariant]
    defaultForm: str | None
    accentType: str | None
    defaultFormTitle: str | None
    accentTypeTitle: str | None


class WordDictionaryPutEntry(WordAccentEntry):
    word: str
