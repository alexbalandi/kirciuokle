from __future__ import annotations

import unicodedata

import pytest

from kirciuokle.accent import accent_text_local_first, tokenize_like_vdu
from kirciuokle.disambiguate import parse_conllu

from .helpers import make_dictionary, put_word


CIA_VARIANTS = [{"form": "čià", "info": "prv.", "mi": ["prv."]}]
YRA_VARIANTS = [
    {"form": "ỹra", "info": "vksm., es. l., 3 asm.", "mi": ["vksm., es. l., 3 asm."]},
    {"form": "yrà", "info": "vksm., es. l., 3 asm.", "mi": ["vksm., es. l., 3 asm."]},
]
YRA_CONLLU = (
    "1\tyra\tbūti\tAUX\t_\tMood=Ind|Person=3|Tense=Pres|VerbForm=Fin"
)


@pytest.mark.asyncio
async def test_alyta_case_sensitive_canonical_sides(tmp_path) -> None:
    dictionary = make_dictionary(tmp_path)
    put_word(
        dictionary,
        "alyta",
        [{"form": "Alytà", "info": "dkt.", "mi": ["dkt."]}],
        default_form=None,
        accent_type="NONE",
        default_form_title="Alytà",
        accent_type_title="MULTIPLE_MEANING",
    )

    response = await accent_text_local_first(
        "alyta Alyta",
        dictionary,
        fallback="none",
        use_tagger=False,
    )

    assert response["parts"] == [
        {"text": "alyta", "type": "word", "unknown": True},
        {"text": " ", "type": "sep"},
        {
            "text": "Alyta",
            "accented": "Alytà",
            "type": "word",
            "ambiguous": True,
            "variants": [{"form": "Alytà", "info": "dkt."}],
            "chosen": 0,
        },
    ]


@pytest.mark.asyncio
async def test_vilnius_variant_vs_meaning_sides(tmp_path) -> None:
    dictionary = make_dictionary(tmp_path)
    put_word(
        dictionary,
        "vilnius",
        [
            {"form": "vìlnius", "info": "dkt.", "mi": ["dkt."]},
            {"form": "vil̃nius", "info": "dkt.", "mi": ["dkt."]},
        ],
        default_form="vìlnius",
        accent_type="MULTIPLE_VARIANT",
        default_form_title="Vìlnius",
        accent_type_title="MULTIPLE_MEANING",
    )

    response = await accent_text_local_first(
        "vilnius Vilnius",
        dictionary,
        fallback="none",
        use_tagger=False,
    )

    assert response["parts"] == [
        {"text": "vilnius", "accented": "vìlnius", "type": "word"},
        {"text": " ", "type": "sep"},
        {
            "text": "Vilnius",
            "accented": "Vìlnius",
            "type": "word",
            "ambiguous": True,
            "variants": [
                {"form": "vìlnius", "info": "dkt."},
                {"form": "vil̃nius", "info": "dkt."},
            ],
            "chosen": 0,
        },
    ]


@pytest.mark.asyncio
async def test_one_hit_suppresses_extra_readings(tmp_path) -> None:
    dictionary = make_dictionary(tmp_path)
    put_word(
        dictionary,
        "kas",
        [
            {"form": "kàs", "info": "įv.", "mi": ["įv."]},
            {"form": "kas", "info": "dll.", "mi": ["dll."]},
        ],
        default_form="kàs",
        accent_type="ONE",
    )

    response = await accent_text_local_first(
        "Kas kas",
        dictionary,
        fallback="none",
        use_tagger=False,
    )

    assert response["parts"] == [
        {"text": "Kas", "accented": "Kàs", "type": "word"},
        {"text": " ", "type": "sep"},
        {"text": "kas", "accented": "kàs", "type": "word"},
    ]


@pytest.mark.asyncio
async def test_abbreviations_and_uppercase_roman_numerals_skip_lookup(tmp_path) -> None:
    dictionary = make_dictionary(tmp_path)
    put_word(
        dictionary,
        "kalba",
        [{"form": "kalbà", "info": "dkt.", "mi": ["dkt."]}],
        default_form="kalbà",
        accent_type="ONE",
        default_form_title="Kalbà",
        accent_type_title="ONE",
    )

    response = await accent_text_local_first(
        "m. rus. V. kalba XX a.",
        dictionary,
        fallback="none",
        use_tagger=False,
    )

    assert response["parts"] == [
        {"text": "m.", "type": "word", "unknown": True},
        {"text": " ", "type": "sep"},
        {"text": "rus.", "type": "word", "unknown": True},
        {"text": " ", "type": "sep"},
        {"text": "V.", "type": "word", "unknown": True},
        {"text": " ", "type": "sep"},
        {"text": "kalba", "accented": "kalbà", "type": "word"},
        {"text": " ", "type": "sep"},
        {"text": "XX", "type": "word"},
        {"text": " ", "type": "sep"},
        {"text": "a.", "type": "word", "unknown": True},
    ]


def test_preaccented_words_stay_whole_and_non_lt() -> None:
    tokenized = tokenize_like_vdu("Tas mė́nuo baigėsi.")

    assert [(part["string"], part["type"]) for part in tokenized["textParts"]] == [
        ("Tas", "WORD"),
        (" ", "SEPARATOR"),
        (unicodedata.normalize("NFC", "mė́nuo"), "NON_LT"),
        (" ", "SEPARATOR"),
        ("baigėsi", "WORD"),
        (".", "SEPARATOR"),
    ]
    assert [word["key"] for word in tokenized["lookupWords"]] == ["tas", "baigėsi"]


@pytest.mark.asyncio
async def test_yra_lemma_exception_uses_mocked_tagger(tmp_path, monkeypatch) -> None:
    dictionary = make_dictionary(tmp_path)
    put_word(
        dictionary,
        "yra",
        YRA_VARIANTS,
        default_form="ỹra",
        accent_type="MULTIPLE_MEANING",
        default_form_title="Ỹra",
        accent_type_title="MULTIPLE_MEANING",
    )

    async def fake_tag_text(text: str):
        assert text == "yra"
        return parse_conllu(YRA_CONLLU)

    monkeypatch.setattr("kirciuokle.tagger.tag_text", fake_tag_text)

    response = await accent_text_local_first("yra", dictionary, fallback="none")

    assert response == {
        "source": "local",
        "tagger": "ok",
        "parts": [
            {
                "text": "yra",
                "accented": "yrà",
                "type": "word",
                "ambiguous": True,
                "resolvedBy": "lemma",
                "variants": [
                    {"form": "ỹra", "info": "vksm., es. l., 3 asm."},
                    {"form": "yrà", "info": "vksm., es. l., 3 asm."},
                ],
                "chosen": 1,
            }
        ],
    }


@pytest.mark.asyncio
async def test_fallback_none_miss_is_unknown_without_vdu(tmp_path, monkeypatch) -> None:
    dictionary = make_dictionary(tmp_path)

    async def fail_lookup(*args, **kwargs):
        raise AssertionError("VDU lookup must not run with FALLBACK=none")

    monkeypatch.setattr("kirciuokle.accent.lookup_word_entries_concurrently", fail_lookup)

    response = await accent_text_local_first(
        "Velvet",
        dictionary,
        fallback="none",
        use_tagger=False,
    )

    assert response["parts"] == [{"text": "Velvet", "type": "word", "unknown": True}]


@pytest.mark.asyncio
async def test_output_is_nfc(tmp_path) -> None:
    dictionary = make_dictionary(tmp_path)
    put_word(
        dictionary,
        "čia",
        CIA_VARIANTS,
        default_form="Čia\u0300",
        accent_type="ONE",
        default_form_title="Čia\u0300",
        accent_type_title="ONE",
    )

    response = await accent_text_local_first(
        "C\u030Cia",
        dictionary,
        fallback="none",
        use_tagger=False,
    )

    part = response["parts"][0]
    assert part == {"text": "Čia", "accented": "Čià", "type": "word"}
    assert part["text"] == unicodedata.normalize("NFC", part["text"])
    assert part["accented"] == unicodedata.normalize("NFC", part["accented"])
