"""Lemma edit-script helpers shared by prep, training, and serving."""

from __future__ import annotations

from collections import Counter
from typing import Iterable


REGULAR_PREFIX = "R"
WHOLE_PREFIX = "W"


def lower_first(value: str) -> str:
    if not value:
        return value
    return value[0].lower() + value[1:]


def _common_prefix_len(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index].casefold() == right[index].casefold():
        index += 1
    return index


def make_lemma_script(form: str, lemma: str) -> str:
    """Return a compact edit script that turns FORM into LEMMA."""
    form = str(form)
    lemma = str(lemma)
    candidates: list[tuple[int, int, int, str]] = []
    for lowercase_first in (0, 1):
        base_form = lower_first(form) if lowercase_first else form
        prefix_len = _common_prefix_len(base_form, lemma)
        strip_count = len(base_form) - prefix_len
        suffix = lemma[prefix_len:]
        if prefix_len == 0 and base_form != lemma:
            continue
        candidate = base_form[: len(base_form) - strip_count] + suffix
        if candidate == lemma:
            cost = strip_count + len(suffix)
            candidates.append(
                (
                    cost,
                    lowercase_first,
                    strip_count,
                    f"{REGULAR_PREFIX}|{lowercase_first}|{strip_count}|{suffix}",
                )
            )
    if not candidates:
        return f"{WHOLE_PREFIX}|{lemma}"
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return candidates[0][3]


def apply_lemma_script(form: str, script: str) -> str:
    """Apply a script produced by make_lemma_script to FORM."""
    form = str(form)
    if script.startswith(f"{WHOLE_PREFIX}|"):
        return script.split("|", 1)[1]

    parts = script.split("|", 3)
    if len(parts) != 4 or parts[0] != REGULAR_PREFIX:
        return form.lower()
    try:
        lowercase_first = int(parts[1])
        strip_count = int(parts[2])
    except ValueError:
        return form.lower()

    base_form = lower_first(form) if lowercase_first else form
    if strip_count < 0:
        return base_form
    if strip_count > len(base_form):
        prefix = ""
    elif strip_count == 0:
        prefix = base_form
    else:
        prefix = base_form[:-strip_count]
    return prefix + parts[3]


def script_inventory(scripts: Iterable[str]) -> list[str]:
    return sorted(set(scripts))


def top_script_coverage(scripts: Iterable[str], top_n: int) -> tuple[int, int, float]:
    counts = Counter(scripts)
    total = sum(counts.values())
    covered = sum(count for _, count in counts.most_common(max(0, top_n)))
    return covered, total, (covered / total if total else 0.0)
