"""Shared head, slot, and subword-pooling helpers for the HF tagger."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from metrics import combined_label, feats_string, split_label


VALID_HEADS = ("combined", "factored")
VALID_POOLINGS = ("first", "last", "first_last")
NONE_CLASS = "__none__"
UPOS_SLOT = "UPOS"
DEFAULT_LABEL = "X|_"


def model_short_name(model_name: str) -> str:
    candidate = model_name.rstrip("/").split("/")[-1] or "model"
    candidate = re.sub(r"[^A-Za-z0-9._-]+", "-", candidate).strip(".-_").lower()
    return candidate or "model"


def derive_run_name(
    model_name: str,
    head: str = "combined",
    pooling: str = "first",
) -> str:
    return f"{model_short_name(model_name)}__{head}__{pooling}"


def validate_head(value: str) -> str:
    if value not in VALID_HEADS:
        raise ValueError(f"unknown head {value!r}; expected one of {VALID_HEADS}")
    return value


def validate_pooling(value: str) -> str:
    if value not in VALID_POOLINGS:
        raise ValueError(f"unknown pooling {value!r}; expected one of {VALID_POOLINGS}")
    return value


def slot_output_name(slot: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", slot).strip("_") or "slot"
    return f"logits__{safe}"


def output_names_for_config(head_config: dict) -> list[str]:
    if head_config["head"] == "combined":
        return ["logits"]
    return [slot_output_name(slot) for slot in head_config["slots"]]


def labels_from_file(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["labels"])


def normalize_slots(slots: dict[str, Iterable[str]]) -> dict[str, list[str]]:
    ordered: dict[str, list[str]] = {}
    slot_names = [UPOS_SLOT] + sorted(slot for slot in slots if slot != UPOS_SLOT)
    for slot in slot_names:
        seen = [str(value) for value in slots.get(slot, [])]
        values = [value for value in sorted(set(seen)) if value != NONE_CLASS]
        ordered[slot] = [NONE_CLASS] + values
    return ordered


def build_slots_from_labels(labels: Iterable[str]) -> dict[str, list[str]]:
    raw_slots: dict[str, set[str]] = {UPOS_SLOT: set()}
    for label in labels:
        upos, feats = split_label(label)
        if upos:
            raw_slots[UPOS_SLOT].add(upos)
        for key, value in feats.items():
            raw_slots.setdefault(key, set()).add(value)
    return normalize_slots(raw_slots)


def slot_values_for_label(label: str, slot_names: Iterable[str]) -> dict[str, str]:
    upos, feats = split_label(label)
    values: dict[str, str] = {}
    for slot in slot_names:
        if slot == UPOS_SLOT:
            values[slot] = upos or NONE_CLASS
        else:
            values[slot] = feats.get(slot, NONE_CLASS)
    return values


def slot_ids_for_label(label: str, slots: dict[str, list[str]]) -> list[int]:
    values = slot_values_for_label(label, slots)
    ids: list[int] = []
    for slot, slot_values in slots.items():
        value = values.get(slot, NONE_CLASS)
        if value not in slot_values:
            value = NONE_CLASS
        ids.append(slot_values.index(value))
    return ids


def assemble_label(slot_values: dict[str, str]) -> str:
    upos = slot_values.get(UPOS_SLOT)
    if not upos or upos == NONE_CLASS:
        upos = "X"
    feats = {
        key: value
        for key, value in slot_values.items()
        if key != UPOS_SLOT and value and value != NONE_CLASS
    }
    return combined_label(upos, feats_string(feats))


def assemble_label_from_ids(ids: Iterable[int], slots: dict[str, list[str]]) -> str:
    values: dict[str, str] = {}
    for slot, index in zip(slots, ids):
        slot_values = slots[slot]
        if 0 <= int(index) < len(slot_values):
            values[slot] = slot_values[int(index)]
        else:
            values[slot] = NONE_CLASS
    return assemble_label(values)


def word_piece_spans(
    word_ids: Iterable[int | None],
    word_count: int,
) -> tuple[list[int], list[int]]:
    first = [-1] * word_count
    last = [-1] * word_count
    for token_index, word_id in enumerate(word_ids):
        if word_id is None or word_id < 0 or word_id >= word_count:
            continue
        if first[word_id] == -1:
            first[word_id] = token_index
        last[word_id] = token_index
    return first, last


def label_token_positions(
    word_ids: Iterable[int | None],
    word_count: int,
    pooling: str,
) -> list[int]:
    first, last = word_piece_spans(word_ids, word_count)
    if pooling == "first":
        return first
    if pooling == "last":
        return last
    raise ValueError("first_last pooling uses word-level labels, not token positions")


def represented_word_indices(word_ids: Iterable[int | None], word_count: int) -> list[int]:
    first, last = word_piece_spans(word_ids, word_count)
    return [index for index in range(word_count) if first[index] != -1 and last[index] != -1]


def build_head_config(
    *,
    head: str,
    pooling: str,
    base_model: str,
    hidden_size: int,
    max_length: int,
    labels: list[str] | None = None,
    slots: dict[str, list[str]] | None = None,
) -> dict:
    validate_head(head)
    validate_pooling(pooling)
    payload: dict[str, object] = {
        "base_model": base_model,
        "head": head,
        "hidden_size": int(hidden_size),
        "max_length": int(max_length),
        "pooling": pooling,
    }
    if head == "combined":
        if labels is None:
            raise ValueError("combined head_config requires labels")
        payload["labels"] = list(labels)
    else:
        if slots is None:
            raise ValueError("factored head_config requires slots")
        payload["slots"] = normalize_slots(slots)
    return payload


def load_head_config(model_dir: Path) -> dict:
    path = model_dir / "head_config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing required head_config.json in {model_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_head(str(payload.get("head")))
    validate_pooling(str(payload.get("pooling")))
    if payload["head"] == "combined":
        payload["labels"] = [str(label) for label in payload["labels"]]
    else:
        payload["slots"] = normalize_slots(payload["slots"])
    payload["hidden_size"] = int(payload["hidden_size"])
    payload["max_length"] = int(payload["max_length"])
    payload["base_model"] = str(payload["base_model"])
    return payload


def write_head_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
