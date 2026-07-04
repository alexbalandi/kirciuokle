from __future__ import annotations

import asyncio
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal, TypeVar

import httpx

from . import tagger as tagger_client
from .dictionary_types import WordAccentEntry
from .disambiguate import (
    AccentVariant,
    Token,
    align_tokens,
    match_case,
    pick_variant,
    to_public_variants,
)


NONCE_URL = "https://kalbu.vdu.lt/mokymosi-priemones/kirciuoklis/"
AJAX_URL = "https://kalbu.vdu.lt/ajax-call"
NONCE_TTL_SECONDS = 6 * 60 * 60
DEFAULT_CHUNK_SIZE = 4500
WORD_CACHE_SECONDS = 7 * 24 * 60 * 60


class UpstreamError(Exception):
    def __init__(self, message: str = "VDU kirčiuoklė laikinai nepasiekiama."):
        super().__init__(message)


class RetryableVduError(Exception):
    pass


@dataclass(slots=True)
class NonceCache:
    value: str
    expires_at: float


nonce_cache: NonceCache | None = None


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def clear_nonce_cache() -> None:
    global nonce_cache
    nonce_cache = None


def extract_nonce(html: str) -> str | None:
    match = re.search(r'"NONCE":"([0-9a-f]+)"', html)
    return match.group(1) if match else None


async def fetch_nonce() -> str:
    global nonce_cache
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            NONCE_URL,
            headers={"accept": "text/html,application/xhtml+xml"},
        )

    if response.status_code >= 400:
        raise UpstreamError()

    nonce = extract_nonce(response.text)
    if not nonce:
        raise UpstreamError()

    nonce_cache = NonceCache(
        value=nonce,
        expires_at=time.time() + NONCE_TTL_SECONDS,
    )
    return nonce


async def get_nonce(force_refresh: bool = False) -> str:
    if not force_refresh and nonce_cache and nonce_cache.expires_at > time.time():
        return nonce_cache.value
    return await fetch_nonce()


async def post_vdu(
    action: Literal["text_accents", "word_accent"],
    fields: dict[str, str],
) -> dict[str, Any]:
    global nonce_cache

    for attempt in range(2):
        try:
            nonce = await get_nonce(force_refresh=attempt > 0)
            data = {"action": action, "nonce": nonce, **fields}
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    AJAX_URL,
                    data=data,
                    headers={
                        "accept": "application/json",
                        "content-type": "application/x-www-form-urlencoded",
                    },
                )

            if response.status_code >= 400:
                raise RetryableVduError()

            envelope = response.json()
            if envelope.get("code") != 200:
                raise RetryableVduError()

            if envelope.get("message") is False:
                return {}

            message = envelope.get("message")
            if not isinstance(message, str):
                raise RetryableVduError()

            parsed = json.loads(message)
            return parsed if isinstance(parsed, dict) else {}
        except Exception as error:
            if attempt == 0:
                nonce_cache = None
                continue
            if isinstance(error, UpstreamError):
                raise
            raise UpstreamError() from error

    raise UpstreamError()


def split_text_into_chunks(text: str, max_length: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    if len(text) <= max_length:
        return [text] if text else []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        remaining = len(text) - start
        if remaining <= max_length:
            chunks.append(text[start:])
            break

        segment = text[start : start + max_length]
        cut = find_sentence_boundary(segment)

        if cut <= 0:
            last_space = segment.rfind(" ")
            cut = last_space + 1 if last_space >= 0 else 0

        if cut <= 0:
            cut = max_length

        chunks.append(text[start : start + cut])
        start += cut

    return chunks


def find_sentence_boundary(segment: str) -> int:
    for index in range(len(segment) - 1, -1, -1):
        if segment[index] in ".!?\n":
            return index + 1
    return -1


def normalize_text_parts(text_parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []

    for part in text_parts:
        original = nfc(str(part.get("string") or ""))

        if part.get("type") == "SEPARATOR":
            parts.append({"text": original, "type": "sep"})
            continue

        normalized: dict[str, Any] = {"text": original, "type": "word"}

        accented = part.get("accented")
        if accented:
            normalized["accented"] = nfc(str(accented))

        if part.get("accentType") == "MULTIPLE_MEANING":
            normalized["ambiguous"] = True

        if part.get("type") == "NON_LT" or part.get("accentType") == "NONE":
            normalized["unknown"] = True
            normalized.pop("accented", None)

        parts.append(normalized)

    return parts


def flatten_variants(response: dict[str, Any]) -> list[AccentVariant]:
    variants: list[AccentVariant] = []
    accent_info = response.get("accentInfo") or []
    if not isinstance(accent_info, list):
        return variants

    for entry in accent_info:
        if not isinstance(entry, dict):
            continue
        information = entry.get("information") or []
        if not isinstance(information, list):
            information = []
        info = format_information(information)
        mi = [
            item.get("mi")
            for item in information
            if isinstance(item, dict) and isinstance(item.get("mi"), str)
        ]

        accented = entry.get("accented") or []
        if not isinstance(accented, list):
            continue
        for form in accented:
            if isinstance(form, str):
                variants.append({"form": nfc(form), "info": info, "mi": list(mi)})

    return variants


def format_information(information: list[Any]) -> str:
    formatted: list[str] = []
    for item in information:
        if not isinstance(item, dict):
            continue
        values = [value for value in (item.get("mi"), item.get("meaning")) if value]
        if values:
            formatted.append(" - ".join(str(value) for value in values))
    return "; ".join(formatted)


LookupVariants = Callable[[str], Awaitable[list[AccentVariant]]]


async def accent_text(
    text: str,
    *,
    lookup_variants: LookupVariants | None = None,
    use_tagger: bool = True,
) -> dict[str, Any]:
    parts = await accent_text_parts(
        text,
        await fetch_text_accent_parts(text),
        lookup_variants=lookup_variants,
        use_tagger=use_tagger,
    )
    return {**parts, "source": "vdu"}


async def accent_text_parts(
    text: str,
    text_parts: list[dict[str, Any]],
    *,
    lookup_variants: LookupVariants | None = None,
    use_tagger: bool = True,
) -> dict[str, Any]:
    tagger_result = await get_tagger_result(text, use_tagger)
    parts = normalize_text_parts(text_parts)
    word_parts = [part for part in text_parts if is_word_part(part)]
    aligned = (
        align_tokens(text_parts, tagger_result["tokens"])
        if tagger_result["tagger"] == "ok"
        else [None] * len(word_parts)
    )
    ambiguous_words = distinct_ambiguous_words(word_parts)
    variants_by_word = await fetch_ambiguous_variants(
        ambiguous_words,
        lookup_variants or lookup_word_variants,
    )

    word_index = 0
    disambiguated_parts: list[dict[str, Any]] = []

    for index, part in enumerate(parts):
        original = text_parts[index] if index < len(text_parts) else None
        if not original or not is_word_part(original):
            disambiguated_parts.append(part)
            continue

        token = aligned[word_index] if word_index < len(aligned) else None
        word_index += 1

        if original.get("accentType") != "MULTIPLE_MEANING":
            disambiguated_parts.append(part)
            continue

        default_form = nfc(str(original.get("accented") or part.get("accented") or ""))
        if not default_form:
            default_form = None
        variants = variants_by_word.get(normalize_word_key(str(original.get("string") or "")), [])
        choice = pick_variant(part["text"], variants, token, default_form)
        choice_index = choice.get("index")
        selected_variant = variants[choice_index] if choice_index is not None else None
        accented = nfc(
            selected_variant["form"]
            if selected_variant
            else default_form or part.get("accented") or part["text"]
        )

        chosen_part = {
            **part,
            "ambiguous": True,
            "accented": nfc(match_case(accented, part["text"])),
            "variants": to_public_variants(variants),
        }
        if choice_index is not None:
            chosen_part["chosen"] = choice_index
        if choice.get("resolvedBy"):
            chosen_part["resolvedBy"] = choice["resolvedBy"]

        disambiguated_parts.append(chosen_part)

    return {"parts": disambiguated_parts, "tagger": tagger_result["tagger"]}


async def get_tagger_result(text: str, use_tagger: bool) -> dict[str, Any]:
    if not use_tagger:
        return {"tagger": "unavailable", "tokens": []}

    try:
        tokens = await tagger_client.tag_text(text)
        return {"tagger": "ok", "tokens": tokens}
    except Exception:
        return {"tagger": "unavailable", "tokens": []}


async def fetch_text_accent_parts(text: str) -> list[dict[str, Any]]:
    text_parts: list[dict[str, Any]] = []
    for chunk in split_text_into_chunks(text):
        response = await post_vdu("text_accents", {"body": chunk})
        chunk_parts = response.get("textParts") or []
        if isinstance(chunk_parts, list):
            text_parts.extend(part for part in chunk_parts if isinstance(part, dict))
    return text_parts


def is_word_part(part: dict[str, Any]) -> bool:
    return part.get("type") in ("WORD", "NON_LT")


def distinct_ambiguous_words(word_parts: list[dict[str, Any]]) -> list[str]:
    words = {
        normalize_word_key(str(part.get("string")))
        for part in word_parts
        if part.get("accentType") == "MULTIPLE_MEANING" and part.get("string")
    }
    return sorted(words)


def normalize_word_key(word: str) -> str:
    return nfc(word).lower()


async def fetch_ambiguous_variants(
    words: list[str],
    lookup_variants: LookupVariants,
) -> dict[str, list[AccentVariant]]:
    return await lookup_variants_concurrently(
        words,
        lookup_variants=lookup_variants,
        swallow_errors=True,
    )


T = TypeVar("T")


async def lookup_concurrently(
    words: list[str],
    lookup: Callable[[str], Awaitable[T]],
    *,
    swallow_errors: bool,
    fallback: Callable[[], T],
) -> dict[str, T]:
    results_by_word: dict[str, T] = {}
    next_index = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal next_index
        while True:
            async with lock:
                if next_index >= len(words):
                    return
                word = words[next_index]
                next_index += 1

            try:
                results_by_word[word] = await lookup(word)
            except Exception as error:
                if not swallow_errors:
                    raise UpstreamError() from error
                results_by_word[word] = fallback()

    await asyncio.gather(*(worker() for _ in range(min(6, len(words)))))
    return results_by_word


async def lookup_variants_concurrently(
    words: list[str],
    *,
    lookup_variants: LookupVariants | None = None,
    swallow_errors: bool = False,
) -> dict[str, list[AccentVariant]]:
    return await lookup_concurrently(
        words,
        lookup_variants or lookup_word_variants,
        swallow_errors=swallow_errors,
        fallback=list,
    )


LookupEntry = Callable[[str], Awaitable[WordAccentEntry]]


async def lookup_word_entries_concurrently(
    words: list[str],
    *,
    lookup_entry: LookupEntry | None = None,
    swallow_errors: bool = False,
) -> dict[str, WordAccentEntry]:
    return await lookup_concurrently(
        words,
        lookup_entry or fetch_word_entry,
        swallow_errors=swallow_errors,
        fallback=lambda: {
            "variants": [],
            "defaultForm": None,
            "accentType": "NONE",
            "defaultFormTitle": None,
            "accentTypeTitle": "NONE",
        },
    )


async def lookup_word_variants(word: str) -> list[AccentVariant]:
    response = await post_vdu("word_accent", {"word": word})
    return flatten_variants(response)


async def fetch_word_entry(word: str) -> WordAccentEntry:
    variants = await lookup_word_variants(word)
    lower = await fetch_canonical_word_side(word)
    title = await fetch_canonical_word_side(to_title_case(word))
    return {
        "variants": variants,
        "defaultForm": lower["form"],
        "accentType": lower["type"],
        "defaultFormTitle": title["form"],
        "accentTypeTitle": title["type"],
    }


async def fetch_canonical_word_side(word: str) -> dict[str, str | None]:
    text_parts = await fetch_text_accent_parts(word)
    word_part = next((part for part in text_parts if part.get("type") == "WORD"), None)
    form = nfc(str(word_part.get("accented"))) if word_part and word_part.get("accented") else None

    if not word_part or not form:
        return {"form": None, "type": "NONE"}

    return {
        "form": form,
        "type": str(word_part.get("accentType") or "ONE"),
    }


def to_title_case(word: str) -> str:
    letters = list(nfc(word))
    if not letters:
        return word
    return f"{letters[0].upper()}{''.join(letters[1:])}"
