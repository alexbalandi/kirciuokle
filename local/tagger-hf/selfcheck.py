# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Dependency-light sanity checks for tagger head and pooling helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from head_config import (
    UPOS_SLOT,
    assemble_label,
    assemble_label_from_ids,
    build_slots_from_labels,
    label_token_positions,
    labels_from_file,
    slot_ids_for_label,
    slot_values_for_label,
    word_piece_spans,
)
from lemma_scripts import apply_lemma_script, make_lemma_script
from metrics import canonicalize_feats, combined_label, feats_string, split_label
from prep_corpus import multext_ud_feats, xpos_ud_feats


BASE_DIR = Path(__file__).resolve().parent
SYNTHETIC_LABELS = [
    "NOUN|Case=Nom|Gender=Fem|Number=Sing",
    "VERB|Mood=Ind|Number=Plur|Person=3|Tense=Past|VerbForm=Fin",
    "ADV|Degree=Cmp",
    "PUNCT|_",
]


def fail(message: str) -> None:
    raise AssertionError(message)


def assert_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        fail(f"{message}: expected {expected!r}, got {actual!r}")


def load_sample_labels(labels_path: Path, limit: int) -> list[str]:
    if labels_path.exists():
        return labels_from_file(labels_path)[:limit]
    return list(SYNTHETIC_LABELS)


def canonical_label(label: str) -> str:
    upos, feats = split_label(label)
    return combined_label(upos, feats_string(feats))


def check_canonical_feats() -> None:
    assert_equal(
        canonicalize_feats("Number=Sing|Case=Nom|Gender=Fem"),
        "Case=Nom|Gender=Fem|Number=Sing",
        "canonical FEATS ordering",
    )
    assembled = assemble_label(
        {
            UPOS_SLOT: "NOUN",
            "Number": "Sing",
            "Case": "Gen",
            "Gender": "Masc",
        }
    )
    assert_equal(
        assembled,
        "NOUN|Case=Gen|Gender=Masc|Number=Sing",
        "factored assembly ordering",
    )


def check_factored_roundtrip(labels: list[str]) -> None:
    slots = build_slots_from_labels(labels)
    for label in labels:
        values = slot_values_for_label(label, slots)
        assert_equal(
            assemble_label(values),
            canonical_label(label),
            f"slot value assembly for {label}",
        )
        ids = slot_ids_for_label(label, slots)
        assert_equal(
            assemble_label_from_ids(ids, slots),
            canonical_label(label),
            f"slot id assembly for {label}",
        )


def check_pooling_indices() -> None:
    toy_word_ids = [None, 0, 0, 1, 2, 2, None]
    first, last = word_piece_spans(toy_word_ids, 3)
    assert_equal(first, [1, 3, 4], "first subword indices")
    assert_equal(last, [2, 3, 5], "last subword indices")
    assert_equal(
        label_token_positions(toy_word_ids, 3, "first"),
        [1, 3, 4],
        "first-pooling label positions",
    )
    assert_equal(
        label_token_positions(toy_word_ids, 3, "last"),
        [2, 3, 5],
        "last-pooling label positions",
    )


def check_multext_decoder() -> None:
    assert_equal(
        multext_ud_feats("Agpmsgn"),
        {
            "Case": "Gen",
            "Definite": "Ind",
            "Degree": "Pos",
            "Gender": "Masc",
            "Number": "Sing",
        },
        "MULTEXT adjective decode",
    )
    assert_equal(
        multext_ud_feats("Pgmsin", lemma="šis"),
        {
            "Case": "Ins",
            "Definite": "Ind",
            "Gender": "Masc",
            "Number": "Sing",
            "PronType": "Dem",
        },
        "MULTEXT pronoun decode",
    )
    assert_equal(
        multext_ud_feats("Momsaly"),
        {
            "Case": "Acc",
            "Definite": "Def",
            "Gender": "Masc",
            "NumForm": "Word",
            "NumType": "Ord",
            "Number": "Sing",
        },
        "MULTEXT numeral decode",
    )
    assert_equal(
        multext_ud_feats("Vgmp3---y--ni-"),
        {
            "Mood": "Ind",
            "Person": "3",
            "Polarity": "Neg",
            "Tense": "Pres",
            "VerbForm": "Fin",
        },
        "MULTEXT verb decode",
    )
    assert_equal(
        xpos_ud_feats("bdv.nelygin.įvardž.vyr.vns.K.", None).get("Definite"),
        "Def",
        "XPOS definiteness repair",
    )
    assert_equal(
        xpos_ud_feats("vksm.asm.neig.tiesiog.es.3.", None).get("Polarity"),
        "Neg",
        "XPOS polarity repair",
    )
    assert_equal(
        xpos_ud_feats("įv.savyb.vns.K.", None).get("Poss"),
        "Yes",
        "XPOS possessive repair",
    )


def check_lemma_scripts() -> None:
    examples = [
        ("namai", "namas"),
        ("yra", "būti"),
        ("Vilniaus", "Vilnius"),
        ("Namai", "namas"),
    ]
    for form, lemma in examples:
        script = make_lemma_script(form, lemma)
        assert_equal(
            apply_lemma_script(form, script),
            lemma,
            f"lemma script round-trip for {form}->{lemma}",
        )
    assert_equal(
        make_lemma_script("yra", "būti").startswith("W|"),
        True,
        "irregular lemma fallback",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels",
        type=Path,
        default=BASE_DIR / "data" / "combined" / "labels.json",
        help="optional labels.json sample source",
    )
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args(list(argv) if argv is not None else None)

    labels = load_sample_labels(args.labels, args.limit)
    check_canonical_feats()
    check_factored_roundtrip(labels)
    check_pooling_indices()
    check_multext_decoder()
    check_lemma_scripts()
    print(f"selfcheck ok: {len(labels)} labels checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
