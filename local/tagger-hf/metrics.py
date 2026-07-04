# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Shared label normalization and scoring helpers for the HF tagger."""

from __future__ import annotations

from typing import Iterable


SCORING_SLOTS = ("case", "gender", "number", "tense", "person", "voice", "degree")


def canonicalize_feats(raw: str | None) -> str:
    """Return a stable CoNLL-U FEATS string sorted by feature key."""
    if not raw or raw == "_":
        return "_"

    pairs: list[tuple[str, str, str]] = []
    for feature in raw.split("|"):
        if not feature or feature == "_":
            continue
        separator = feature.find("=")
        if separator <= 0:
            pairs.append((feature, "", feature))
            continue
        key = feature[:separator]
        value = feature[separator + 1 :]
        pairs.append((key, value, f"{key}={value}"))

    if not pairs:
        return "_"
    return "|".join(item[2] for item in sorted(pairs, key=lambda item: (item[0], item[1])))


def combined_label(upos: str, feats: str | None) -> str:
    return f"{upos}|{canonicalize_feats(feats)}"


def parse_feats(raw: str | None) -> dict[str, str]:
    if not raw or raw == "_":
        return {}
    feats: dict[str, str] = {}
    for feature in canonicalize_feats(raw).split("|"):
        separator = feature.find("=")
        if separator <= 0:
            continue
        feats[feature[:separator]] = feature[separator + 1 :]
    return feats


def feats_string(feats: dict[str, str]) -> str:
    if not feats:
        return "_"
    return "|".join(f"{key}={feats[key]}" for key in sorted(feats))


def split_label(label: str) -> tuple[str, dict[str, str]]:
    if "|" not in label:
        return label, {}
    upos, feats = label.split("|", 1)
    return upos, parse_feats(feats)


def token_tags_from_parts(upos: str, feats: dict[str, str]) -> dict[str, str]:
    """Faithful port of tokenTags in src/worker/disambiguation.ts."""
    tags: dict[str, str] = {}

    if upos in ("VERB", "AUX"):
        tags["pos"] = "PART_VERB" if feats.get("VerbForm") == "Part" else "VERB"
    elif upos in ("NOUN", "PROPN"):
        tags["pos"] = "NOUN"
    elif upos in ("CCONJ", "SCONJ"):
        tags["pos"] = "CCONJ"
    else:
        tags["pos"] = upos

    for slot, feature in (
        ("gender", "Gender"),
        ("number", "Number"),
        ("case", "Case"),
        ("tense", "Tense"),
        ("person", "Person"),
        ("voice", "Voice"),
    ):
        value = feats.get(feature)
        if value:
            tags[slot] = value

    degree = feats.get("Degree")
    if degree and degree != "Pos":
        tags["degree"] = degree

    return tags


def token_tags_for_label(label: str) -> dict[str, str]:
    return token_tags_from_parts(*split_label(label))


def safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def evaluate_label_pairs(
    predicted_labels: Iterable[str],
    gold_labels: Iterable[str],
) -> dict[str, float | int]:
    """Score token-level label predictions with the benchmark metric family."""
    predicted = list(predicted_labels)
    gold = list(gold_labels)
    total = len(gold)

    label_ok = 0
    upos_ok = 0
    feats_ok = 0
    slots_ok = 0
    aux_verb_total = 0
    aux_verb_ok = 0

    for index, gold_label in enumerate(gold):
        pred_label = predicted[index] if index < len(predicted) else "_|_"
        if pred_label == gold_label:
            label_ok += 1

        pred_upos, pred_feats = split_label(pred_label)
        gold_upos, gold_feats = split_label(gold_label)
        if pred_upos == gold_upos:
            upos_ok += 1
        if pred_feats == gold_feats:
            feats_ok += 1
        if token_tags_from_parts(pred_upos, pred_feats) == token_tags_from_parts(
            gold_upos, gold_feats
        ):
            slots_ok += 1

        if gold_upos in ("AUX", "VERB"):
            aux_verb_total += 1
            if pred_upos == gold_upos:
                aux_verb_ok += 1

    return {
        "tokens": total,
        "label_correct": label_ok,
        "upos_correct": upos_ok,
        "feats_exact_correct": feats_ok,
        "slot_correct": slots_ok,
        "aux_verb_correct": aux_verb_ok,
        "aux_verb_total": aux_verb_total,
        "label_accuracy": safe_rate(label_ok, total),
        "upos_accuracy": safe_rate(upos_ok, total),
        "feats_exact_accuracy": safe_rate(feats_ok, total),
        "slot_accuracy": safe_rate(slots_ok, total),
        "aux_verb_accuracy": safe_rate(aux_verb_ok, aux_verb_total),
    }
