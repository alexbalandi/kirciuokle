from __future__ import annotations

import asyncio
import unicodedata
from typing import Any, Literal

import regex

from .dictionary import FallbackMode, WordDictionary, normalize_word_key
from .dictionary_types import WordAccentEntry, WordDictionaryPutEntry
from .disambiguate import match_case
from .vdu import accent_text, accent_text_parts, lookup_word_entries_concurrently


WORD_RE = regex.compile(r"[\p{L}\p{M}]+")
LT_LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzĄČĘĖĮŠŲŪŽąčęėįšųūž")
ROMAN_NUMERAL_RE = regex.compile(r"^[IVXLCDM]+$")
MISS_BUDGET = 10
ABBREVIATIONS = {
    "a",
    "akad",
    "adr",
    "angl",
    "aps",
    "apskr",
    "dab",
    "dir",
    "doc",
    "dr",
    "egz",
    "est",
    "etc",
    "gen",
    "gr",
    "gyv",
    "habil",
    "insp",
    "isp",
    "it",
    "jaun",
    "kan",
    "kpt",
    "kt",
    "latv",
    "lenk",
    "liet",
    "lot",
    "m",
    "min",
    "mjr",
    "mln",
    "mlrd",
    "mstl",
    "nr",
    "pan",
    "pav",
    "pgl",
    "pirm",
    "plg",
    "plk",
    "pers",
    "pr",
    "pranc",
    "proc",
    "prof",
    "psn",
    "pvz",
    "rus",
    "sav",
    "sek",
    "sen",
    "sk",
    "str",
    "šnek",
    "šv",
    "tarm",
    "tel",
    "tūkst",
    "ukr",
    "val",
    "vok",
    "vyr",
    "žr",
}


BackgroundTasksLike = Any


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


async def accent_text_local_first(
    text: str,
    dictionary: WordDictionary,
    *,
    fallback: FallbackMode = "vdu",
    use_tagger: bool = True,
    background_tasks: BackgroundTasksLike | None = None,
) -> dict[str, Any]:
    normalized_text = nfc(text)
    tokenized = tokenize_like_vdu(normalized_text)
    distinct_words = list(dict.fromkeys(word["key"] for word in tokenized["lookupWords"]))
    entries_by_word = dictionary.get_words(distinct_words)
    misses = [word for word in distinct_words if entries_by_word.get(word) is None]

    if fallback == "vdu" and len(misses) > MISS_BUDGET:
        schedule_seed_misses(misses[:MISS_BUDGET], dictionary, background_tasks)
        return await accent_text(
            normalized_text,
            lookup_variants=lambda word: lookup_word_variants_from_dictionary(
                dictionary,
                word,
                fallback=fallback,
            ),
            use_tagger=use_tagger,
        )

    if fallback == "vdu" and len(misses) > 0:
        fetched = await lookup_word_entries_concurrently(misses)
        for word, entry in fetched.items():
            entries_by_word[word] = entry
        dictionary.put_words([{"word": word, **entry} for word, entry in fetched.items()])

    apply_dictionary_results(tokenized, entries_by_word)
    parts = await accent_text_parts(
        normalized_text,
        tokenized["textParts"],
        lookup_variants=lambda word: lookup_variants_from_entries(entries_by_word, word),
        use_tagger=use_tagger,
    )
    return {**parts, "source": "local"}


def tokenize_like_vdu(text: str) -> dict[str, list[dict[str, Any]]]:
    text_parts: list[dict[str, Any]] = []
    lookup_words: list[dict[str, Any]] = []
    last_index = 0

    for match in WORD_RE.finditer(text):
        word = nfc(match.group(0))
        index = match.start()

        if index > last_index:
            text_parts.append(
                {
                    "string": nfc(text[last_index:index]),
                    "type": "SEPARATOR",
                }
            )

        part_index = len(text_parts)
        word_end = match.end()
        if is_abbreviation(word, text, word_end):
            text_parts.append(
                {
                    "string": nfc(text[index : word_end + 1]),
                    "type": "WORD",
                    "accentType": "NONE",
                }
            )
            last_index = word_end + 1
            continue

        if ROMAN_NUMERAL_RE.fullmatch(word):
            text_parts.append({"string": word, "type": "WITH_NUMBER"})
        elif is_lt_word(word):
            key = normalize_word_key(word)
            text_parts.append({"string": word, "type": "WORD"})
            lookup_words.append({"partIndex": part_index, "text": word, "key": key})
        else:
            text_parts.append({"string": word, "type": "NON_LT"})

        last_index = word_end

    if last_index < len(text):
        text_parts.append({"string": nfc(text[last_index:]), "type": "SEPARATOR"})

    return {"textParts": text_parts, "lookupWords": lookup_words}


def apply_dictionary_results(
    tokenized: dict[str, list[dict[str, Any]]],
    entries_by_word: dict[str, WordAccentEntry | None],
) -> None:
    for word in tokenized["lookupWords"]:
        part_index = word["partIndex"]
        part = tokenized["textParts"][part_index] if part_index < len(tokenized["textParts"]) else None
        if not part:
            continue

        entry = entries_by_word.get(word["key"])
        side = select_canonical_side(entry, word["text"]) if entry else None

        if not side or side["type"] == "NONE" or not side["form"]:
            part["accentType"] = "NONE"
            part.pop("accented", None)
            continue

        part["accented"] = nfc(match_case(side["form"], word["text"]))
        part["accentType"] = side["type"] or "ONE"


def is_abbreviation(word: str, text: str, word_end: int) -> bool:
    if word_end >= len(text) or text[word_end] != ".":
        return False
    return len(list(word)) == 1 or normalize_word_key(word) in ABBREVIATIONS


def is_lt_word(word: str) -> bool:
    return all(char in LT_LETTERS for char in word)


def select_canonical_side(
    entry: WordAccentEntry,
    word: str,
) -> dict[str, str | None]:
    if starts_with_uppercase(word):
        return {
            "form": entry["defaultFormTitle"],
            "type": entry["accentTypeTitle"],
        }
    return {
        "form": entry["defaultForm"],
        "type": entry["accentType"],
    }


def starts_with_uppercase(word: str) -> bool:
    first = next(iter(word), "")
    return bool(first and first.upper() == first and first.lower() != first)


async def lookup_variants_from_entries(
    entries_by_word: dict[str, WordAccentEntry | None],
    word: str,
) -> list[dict[str, Any]]:
    entry = entries_by_word.get(normalize_word_key(word))
    return entry["variants"] if entry else []


async def lookup_word_variants_from_dictionary(
    dictionary: WordDictionary,
    word: str,
    *,
    fallback: FallbackMode,
) -> list[dict[str, Any]]:
    from .dictionary import lookup_word_variants

    return await lookup_word_variants(dictionary, word, fallback=fallback)


def schedule_seed_misses(
    words: list[str],
    dictionary: WordDictionary,
    background_tasks: BackgroundTasksLike | None,
) -> None:
    if len(words) == 0:
        return

    async def seed() -> None:
        try:
            fetched = await lookup_word_entries_concurrently(words, swallow_errors=True)
            entries: list[WordDictionaryPutEntry] = [
                {"word": word, **entry} for word, entry in fetched.items()
            ]
            dictionary.put_words(entries)
        except Exception:
            return

    if background_tasks is not None:
        background_tasks.add_task(seed)
    else:
        asyncio.create_task(seed())
