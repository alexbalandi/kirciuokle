from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Literal, TypedDict


Slot = Literal[
    "pos",
    "gender",
    "number",
    "case",
    "tense",
    "person",
    "voice",
    "degree",
]


class Variant(TypedDict):
    form: str
    info: str


class AccentVariant(Variant):
    mi: list[str]


class PickResult(TypedDict, total=False):
    index: int | None
    resolvedBy: Literal["lemma", "context"]


@dataclass(slots=True)
class Token:
    form: str
    lemma: str
    upos: str
    xpos: str = "_"
    feats: dict[str, str] = field(default_factory=dict)


MI_TAGS: dict[str, tuple[Slot, str]] = {
    "dkt.": ("pos", "NOUN"),
    "bdv.": ("pos", "ADJ"),
    "vksm.": ("pos", "VERB"),
    "dlv.": ("pos", "PART_VERB"),
    "psdlv.": ("pos", "PART_VERB"),
    "padlv.": ("pos", "PART_VERB"),
    "prv.": ("pos", "ADV"),
    "įv.": ("pos", "PRON"),
    "sktv.": ("pos", "NUM"),
    "jng.": ("pos", "CCONJ"),
    "prl.": ("pos", "ADP"),
    "dll.": ("pos", "PART"),
    "jst.": ("pos", "INTJ"),
    "vyr. g.": ("gender", "Masc"),
    "mot. g.": ("gender", "Fem"),
    "bev. g.": ("gender", "Neut"),
    "vns.": ("number", "Sing"),
    "dgs.": ("number", "Plur"),
    "vard.": ("case", "Nom"),
    "kilm.": ("case", "Gen"),
    "naud.": ("case", "Dat"),
    "gal.": ("case", "Acc"),
    "įnag.": ("case", "Ins"),
    "viet.": ("case", "Loc"),
    "šauksm.": ("case", "Voc"),
    "es. l.": ("tense", "Pres"),
    "būt. l.": ("tense", "Past"),
    "būt. k. l.": ("tense", "Past"),
    "būt. d. l.": ("tense", "PastIter"),
    "būs. l.": ("tense", "Fut"),
    "1 asm.": ("person", "1"),
    "2 asm.": ("person", "2"),
    "3 asm.": ("person", "3"),
    "veik. r.": ("voice", "Act"),
    "neveik. r.": ("voice", "Pass"),
    "aukšt. l.": ("degree", "Cmp"),
    "aukšč. l.": ("degree", "Sup"),
}


LEMMA_EXCEPTIONS = {
    "yra\0būti": "yrà",
    "yra\0irti": "ỹra",
}

MI_TAG_KEYS = sorted(MI_TAGS, key=len, reverse=True)
SCORING_SLOTS: tuple[Slot, ...] = (
    "case",
    "gender",
    "number",
    "tense",
    "person",
    "voice",
    "degree",
)


def nfc(value: str) -> str:
    return unicodedata.normalize("NFC", value)


def parse_conllu(conllu: str) -> list[Token]:
    tokens: list[Token] = []

    for line in conllu.splitlines():
        if not line or line.startswith("#"):
            continue

        columns = line.split("\t")
        if len(columns) < 6 or not columns[0].isdigit():
            continue

        tokens.append(
            Token(
                form=columns[1] or "",
                lemma=columns[2] or "",
                upos=columns[3] or "",
                xpos=columns[4] or "",
                feats=parse_feats(columns[5] or "_"),
            )
        )

    return tokens


def parse_feats(raw: str) -> dict[str, str]:
    if raw == "_":
        return {}

    feats: dict[str, str] = {}
    for feature in raw.split("|"):
        separator = feature.find("=")
        if separator <= 0:
            continue
        feats[feature[:separator]] = feature[separator + 1 :]
    return feats


def align_tokens(parts: list[dict], tokens: list[Token]) -> list[Token | None]:
    tokens = [token for token in tokens if has_letter(token.form)]
    aligned: list[Token | None] = []
    token_index = 0

    for part in parts:
        if part.get("type") not in ("WORD", "NON_LT"):
            continue

        found: Token | None = None
        original = part.get("string") or ""
        scan_end = min(token_index + 8, len(tokens))

        for index in range(token_index, scan_end):
            if tokens[index].form.lower() == original.lower():
                found = tokens[index]
                token_index = index + 1
                break

        aligned.append(found)

    return aligned


def parse_mi(mi: str) -> dict[Slot, str]:
    tags: dict[Slot, str] = {}
    remaining = mi.strip()

    for abbreviation in MI_TAG_KEYS:
        if abbreviation not in remaining:
            continue

        slot, value = MI_TAGS[abbreviation]
        tags.setdefault(slot, value)
        remaining = remaining.replace(abbreviation, " ")

    return tags


def token_tags(token: Token) -> dict[Slot, str]:
    tags: dict[Slot, str] = {}

    if token.upos in ("VERB", "AUX"):
        tags["pos"] = "PART_VERB" if token.feats.get("VerbForm") == "Part" else "VERB"
    elif token.upos in ("NOUN", "PROPN"):
        tags["pos"] = "NOUN"
    elif token.upos == "DET":
        # POS family follows VDU conventions: no DET in Lithuanian traditional grammar; see docs/SPEC13.md.
        tags["pos"] = "PRON"
    elif token.upos in ("CCONJ", "SCONJ"):
        tags["pos"] = "CCONJ"
    else:
        tags["pos"] = token.upos

    copy_feature(tags, token.feats, "gender", "Gender")
    copy_feature(tags, token.feats, "number", "Number")
    copy_feature(tags, token.feats, "case", "Case")
    copy_feature(tags, token.feats, "tense", "Tense")
    copy_feature(tags, token.feats, "person", "Person")
    copy_feature(tags, token.feats, "voice", "Voice")

    degree = token.feats.get("Degree")
    if degree and degree != "Pos":
        tags["degree"] = degree

    return tags


def copy_feature(
    tags: dict[Slot, str],
    feats: dict[str, str],
    slot: Slot,
    feature: str,
) -> None:
    value = feats.get(feature)
    if value:
        tags[slot] = value


def score_tags(variant_tags: dict[Slot, str], context_tags: dict[Slot, str]) -> int:
    score = 0

    if variant_tags.get("pos") and context_tags.get("pos"):
        score += 4 if variant_tags["pos"] == context_tags["pos"] else -3

    for slot in SCORING_SLOTS:
        variant_value = variant_tags.get(slot)
        context_value = context_tags.get(slot)
        if not variant_value or not context_value:
            continue
        score += 2 if variant_value == context_value else -2

    return score


def score_variant(variant: AccentVariant, context_tags: dict[Slot, str]) -> int:
    if len(variant["mi"]) == 0:
        return 0
    return max(score_tags(parse_mi(label), context_tags) for label in variant["mi"])


def pick_variant(
    word: str,
    variants: list[AccentVariant],
    token: Token | None,
    default_form: str | None = None,
) -> PickResult:
    if len(variants) == 0:
        return {"index": None}

    default_index = find_variant_index(variants, default_form)
    fallback_index = default_index if default_index >= 0 else 0

    if token is None:
        return {"index": fallback_index}

    exception_form = LEMMA_EXCEPTIONS.get(f"{word.lower()}\0{token.lemma}")
    if exception_form:
        exception_index = find_variant_index(variants, exception_form)
        if exception_index >= 0:
            return {"index": exception_index, "resolvedBy": "lemma"}

    context_tags = token_tags(token)
    scored = sorted(
        (
            {"index": index, "score": score_variant(variant, context_tags)}
            for index, variant in enumerate(variants)
        ),
        key=lambda item: (-item["score"], item["index"]),
    )

    if len(scored) > 1 and scored[0]["score"] > scored[1]["score"]:
        return {"index": scored[0]["index"], "resolvedBy": "context"}

    return {"index": fallback_index}


def find_variant_index(variants: list[AccentVariant], form: str | None) -> int:
    if not form:
        return -1

    normalized = nfc(form)
    for index, variant in enumerate(variants):
        if nfc(variant["form"]) == normalized:
            return index
    return -1


def match_case(accented: str, original: str) -> str:
    if len(original) > 1 and original.upper() == original:
        return accented.upper()

    if original and original[0].upper() == original[0]:
        return f"{accented[0].upper()}{accented[1:]}" if accented else accented

    return accented


def to_public_variants(variants: list[AccentVariant]) -> list[Variant]:
    return [{"form": variant["form"], "info": variant["info"]} for variant in variants]


def has_letter(value: str) -> bool:
    return any(unicodedata.category(char).startswith("L") for char in value)
