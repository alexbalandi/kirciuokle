from __future__ import annotations

from kirciuokle.disambiguate import (
    Token,
    align_tokens,
    parse_conllu,
    parse_mi,
    pick_variant,
    score_tags,
    token_tags,
)


def token(
    form: str,
    lemma: str,
    upos: str,
    feats: dict[str, str] | None = None,
) -> Token:
    return Token(form=form, lemma=lemma, upos=upos, xpos="_", feats=feats or {})


def test_parse_conllu_skips_comments_ranges_and_empty_nodes() -> None:
    tokens = parse_conllu(
        """# sent_id = 1
1-2\tČia_yra\t_\t_\t_\t_
1\tČia\tčia\tADV\t_\tDegree=Pos
2\tyra\tbūti\tAUX\t_\tMood=Ind|Person=3|Tense=Pres|VerbForm=Fin
2.1\tpraleisti\tpraleisti\tVERB\t_\t_
3\t.\t.\tPUNCT\t_\t_"""
    )

    assert tokens == [
        token("Čia", "čia", "ADV", {"Degree": "Pos"}),
        token(
            "yra",
            "būti",
            "AUX",
            {"Mood": "Ind", "Person": "3", "Tense": "Pres", "VerbForm": "Fin"},
        ),
        token(".", ".", "PUNCT"),
    ]


def test_align_tokens_filters_number_and_punctuation_tokens() -> None:
    yra = token("yra", "būti", "AUX")
    tokens = [
        token("Karas", "karas", "NOUN"),
        *[token(form, form, "PUNCT") for form in ["(", "1918", ".", "02", "16", ")"]],
        yra,
    ]

    assert align_tokens(
        [
            {"string": "Karas", "type": "WORD"},
            {"string": " ", "type": "SEPARATOR"},
            {"string": "yra", "type": "WORD"},
        ],
        tokens,
    ) == [tokens[0], yra]


def test_parse_mi_longest_first_and_score_weights() -> None:
    assert parse_mi("vksm., būt. d. l., 3 asm.") == {
        "pos": "VERB",
        "tense": "PastIter",
        "person": "3",
    }

    context = token_tags(
        token(
            "geras",
            "geras",
            "ADJ",
            {"Case": "Nom", "Degree": "Pos", "Gender": "Masc", "Number": "Sing"},
        )
    )
    assert score_tags(parse_mi("bdv., vyr. g., vns. vard."), context) == 10
    assert score_tags(parse_mi("bdv., mot. g., vns. gal."), context) == 2


def test_det_tokens_score_as_pron_against_vdu_pronoun_variants() -> None:
    context = token_tags(token("tas", "tas", "DET"))

    assert score_tags(parse_mi("įv."), context) == 4


def test_interjection_mi_uses_vdu_jstk_abbreviation() -> None:
    # VDU writes interjections as "jstk." (jaustukas). Before this mapping existed,
    # "jstk., pagr." parsed to {} and scored 0 against an INTJ context — which made
    # the joint-dataset projector mask ALL interjections (prašom, ačiū, …) out of
    # stress supervision (SPEC59).
    assert parse_mi("jstk., pagr.") == {"pos": "INTJ"}

    context = token_tags(token("prašom", "prašom", "INTJ"))
    assert score_tags(parse_mi("jstk., pagr."), context) == 4
    # the verb reading of prašom still loses against an INTJ context
    assert score_tags(parse_mi("vksm., dgs., es. l., 1 asm."), context) == -3


def test_pick_variant_uses_lemma_exception_before_scoring() -> None:
    variants = [
        {"form": "ỹra", "info": "vksm., es. l., 3 asm.", "mi": ["vksm., es. l., 3 asm."]},
        {"form": "yrà", "info": "vksm., es. l., 3 asm.", "mi": ["vksm., es. l., 3 asm."]},
    ]

    assert pick_variant("yra", variants, token("yra", "būti", "AUX"), "ỹra") == {
        "index": 1,
        "resolvedBy": "lemma",
    }


def test_pick_variant_keeps_default_on_ties() -> None:
    variants = [
        {"form": "ỹra", "info": "vksm., es. l., 3 asm.", "mi": ["vksm., es. l., 3 asm."]},
        {"form": "yrà", "info": "vksm., es. l., 3 asm.", "mi": ["vksm., es. l., 3 asm."]},
    ]

    assert pick_variant("yra", variants, token("yra", "nebūti", "AUX"), "yrà") == {
        "index": 1,
    }
