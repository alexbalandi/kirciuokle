from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from kirciuokle.dictionary import NEGATIVE_WORD_TTL, lookup_word_variants

from .helpers import insert_raw_word, make_dictionary, put_word


YRA_VARIANTS = [{"form": "ỹra", "info": "vksm.", "mi": ["vksm."]}]


def test_get_words_returns_hits_absent_rows_and_normalizes(tmp_path) -> None:
    dictionary = make_dictionary(tmp_path)
    put_word(dictionary, "yra", YRA_VARIANTS)

    result = dictionary.get_words(["YRA", "nėra"])

    assert result["yra"] == {
        "variants": YRA_VARIANTS,
        "defaultForm": "ỹra",
        "accentType": "ONE",
        "defaultFormTitle": "Ỹra",
        "accentTypeTitle": "ONE",
    }
    assert result["nėra"] is None


def test_put_words_stores_valid_negatives_for_thirty_days(tmp_path) -> None:
    dictionary = make_dictionary(tmp_path)
    before = datetime.now(UTC) - timedelta(seconds=1)
    put_word(
        dictionary,
        "velvet",
        [],
        default_form=None,
        accent_type="NONE",
        default_form_title=None,
        accent_type_title="NONE",
    )
    row = dictionary._conn.execute(
        "SELECT variants, negative_until FROM words WHERE word = ?",
        ("velvet",),
    ).fetchone()

    assert json.loads(row["variants"]) == []
    negative_until = datetime.fromisoformat(row["negative_until"].replace("Z", "+00:00"))
    assert before + NEGATIVE_WORD_TTL <= negative_until <= datetime.now(UTC) + timedelta(days=30)


def test_expired_negative_and_legacy_null_title_are_misses(tmp_path) -> None:
    dictionary = make_dictionary(tmp_path)
    insert_raw_word(
        dictionary,
        word="old",
        variants="[]",
        negative_until="2026-07-01T00:00:00.000Z",
        accent_type="NONE",
        accent_type_title="NONE",
    )
    insert_raw_word(
        dictionary,
        word="legacy",
        variants=json.dumps(YRA_VARIANTS, ensure_ascii=False),
        default_form="ỹra",
        accent_type="ONE",
        default_form_title=None,
        accent_type_title=None,
    )

    result = dictionary.get_words(["old", "legacy"])

    assert result["old"] is None
    assert result["legacy"] is None


@pytest.mark.asyncio
async def test_lookup_word_variants_fallback_none_returns_empty_without_fetch(
    tmp_path,
    monkeypatch,
) -> None:
    dictionary = make_dictionary(tmp_path)

    async def fail_fetch(word: str):
        raise AssertionError(f"unexpected fetch for {word}")

    monkeypatch.setattr("kirciuokle.dictionary.fetch_word_entry", fail_fetch)

    assert await lookup_word_variants(dictionary, "velvet", fallback="none") == []
