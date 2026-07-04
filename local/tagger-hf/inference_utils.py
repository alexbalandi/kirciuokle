"""Runtime decoding helpers shared by export, comparison, and serving."""

from __future__ import annotations

from typing import Any, Iterable

from head_config import (
    DEFAULT_LABEL,
    assemble_label_from_ids,
    slot_output_name,
    word_piece_spans,
)


def outputs_to_labels(
    *,
    outputs: dict[str, Any],
    word_ids: Iterable[int | None],
    word_count: int,
    head_config: dict,
    batch_index: int = 0,
    default_label: str = DEFAULT_LABEL,
) -> list[str]:
    first, last = word_piece_spans(word_ids, word_count)
    labels: list[str] = []
    for word_index in range(word_count):
        first_index = first[word_index]
        last_index = last[word_index]
        if first_index == -1 or last_index == -1:
            labels.append(default_label)
            continue
        if head_config["head"] == "combined":
            combined_logits = (
                outputs["logits"]
                if "logits" in outputs
                else next(iter(outputs.values()))
            )
            labels.append(
                _combined_label(
                    combined_logits,
                    head_config["labels"],
                    head_config["pooling"],
                    first_index,
                    last_index,
                    batch_index,
                    default_label,
                )
            )
        else:
            labels.append(
                _factored_label(
                    outputs,
                    head_config["slots"],
                    head_config["pooling"],
                    first_index,
                    last_index,
                    batch_index,
                )
            )
    return labels


def _combined_label(
    logits: Any,
    labels: list[str],
    pooling: str,
    first_index: int,
    last_index: int,
    batch_index: int,
    default_label: str,
) -> str:
    word_logits = _word_logits(logits, pooling, first_index, last_index, batch_index)
    label_id = int(word_logits.argmax(axis=-1))
    if 0 <= label_id < len(labels):
        return labels[label_id]
    return default_label


def _factored_label(
    outputs: dict[str, Any],
    slots: dict[str, list[str]],
    pooling: str,
    first_index: int,
    last_index: int,
    batch_index: int,
) -> str:
    ids: list[int] = []
    for slot in slots:
        logits = outputs[slot_output_name(slot)]
        word_logits = _word_logits(logits, pooling, first_index, last_index, batch_index)
        ids.append(int(word_logits.argmax(axis=-1)))
    return assemble_label_from_ids(ids, slots)


def _word_logits(
    logits: Any,
    pooling: str,
    first_index: int,
    last_index: int,
    batch_index: int,
) -> Any:
    item = logits[batch_index]
    if pooling == "first":
        return item[first_index]
    if pooling == "last":
        return item[last_index]
    if pooling == "first_last":
        return item[first_index, 0] + item[last_index, 1]
    raise ValueError(f"unknown pooling: {pooling}")
