# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Compare per-UPOS FEATS-key coverage between CoNLL-U or JSONL corpora."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from metrics import parse_feats, split_label


SLOT_FEATS_KEYS = (
    "Case",
    "Gender",
    "Number",
    "Tense",
    "Person",
    "Voice",
    "Degree",
    "VerbForm",
    "Mood",
    "Reflex",
)


@dataclass(frozen=True)
class FeatsToken:
    upos: str
    feats: dict[str, str]


@dataclass(frozen=True)
class CoverageRow:
    upos: str
    key: str
    left_total: int
    right_total: int
    left_coverage: float
    right_coverage: float

    @property
    def delta(self) -> float:
        return self.left_coverage - self.right_coverage


def key_filter(mode: str) -> tuple[str, ...] | None:
    if mode == "slots":
        return SLOT_FEATS_KEYS
    if mode == "all":
        return None
    raise ValueError(f"unsupported FEATS key mode: {mode}")


def filter_feats_keys(
    feats: dict[str, str],
    keys: Iterable[str] | None,
) -> dict[str, str]:
    if keys is None:
        return dict(feats)
    allowed = set(keys)
    return {key: value for key, value in feats.items() if key in allowed}


def tokens_from_jsonl(path: Path) -> Iterator[FeatsToken]:
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            for label in row.get("labels", []):
                upos, feats = split_label(label)
                yield FeatsToken(upos, feats)


def tokens_from_conllu(path: Path) -> Iterator[FeatsToken]:
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            columns = line.split("\t")
            if len(columns) < 6 or not columns[0].isdigit():
                continue
            yield FeatsToken(columns[3], parse_feats(columns[5]))


def tokens_from_path(path: Path) -> list[FeatsToken]:
    if path.suffix.lower() == ".jsonl":
        return list(tokens_from_jsonl(path))
    return list(tokens_from_conllu(path))


def normalize_tokens(
    tokens: Iterable[FeatsToken],
    keys: Iterable[str] | None,
) -> list[FeatsToken]:
    return [
        FeatsToken(token.upos, filter_feats_keys(token.feats, keys))
        for token in tokens
    ]


def coverage_counts(tokens: Iterable[FeatsToken]) -> tuple[Counter[str], Counter[tuple[str, str]]]:
    upos_totals: Counter[str] = Counter()
    key_totals: Counter[tuple[str, str]] = Counter()
    for token in tokens:
        upos_totals[token.upos] += 1
        for key in token.feats:
            key_totals[(token.upos, key)] += 1
    return upos_totals, key_totals


def coverage_rows(
    left_tokens: Iterable[FeatsToken],
    right_tokens: Iterable[FeatsToken],
    keys: Iterable[str] | None = None,
) -> list[CoverageRow]:
    left = normalize_tokens(left_tokens, keys)
    right = normalize_tokens(right_tokens, keys)
    left_upos, left_keys = coverage_counts(left)
    right_upos, right_keys = coverage_counts(right)
    upos_values = sorted(set(left_upos) | set(right_upos))

    if keys is None:
        per_upos_keys: dict[str, set[str]] = defaultdict(set)
        for upos, key in set(left_keys) | set(right_keys):
            per_upos_keys[upos].add(key)
    else:
        key_set = set(keys)
        per_upos_keys = {upos: set(key_set) for upos in upos_values}

    rows: list[CoverageRow] = []
    for upos in upos_values:
        for key in sorted(per_upos_keys.get(upos, ())):
            left_total = left_upos[upos]
            right_total = right_upos[upos]
            left_present = left_keys[(upos, key)]
            right_present = right_keys[(upos, key)]
            rows.append(
                CoverageRow(
                    upos=upos,
                    key=key,
                    left_total=left_total,
                    right_total=right_total,
                    left_coverage=left_present / left_total if left_total else 0.0,
                    right_coverage=right_present / right_total if right_total else 0.0,
                )
            )
    return rows


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def format_pp(value: float) -> str:
    return f"{value * 100:+.1f}pp"


def format_coverage_table(
    rows: Iterable[CoverageRow],
    left_name: str,
    right_name: str,
    min_delta: float,
) -> str:
    filtered = [
        row
        for row in rows
        if abs(row.delta) >= min_delta
        and (row.left_total > 0 or row.right_total > 0)
    ]
    filtered.sort(key=lambda row: (row.upos, row.key))

    if not filtered:
        return f"no FEATS-key coverage deltas >= {format_pp(min_delta)[1:]}"

    lines = [
        f"| UPOS | key | {left_name} | {right_name} | delta | tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in filtered:
        lines.append(
            "| "
            f"{row.upos} | {row.key} | "
            f"{format_percent(row.left_coverage)} | "
            f"{format_percent(row.right_coverage)} | "
            f"{format_pp(row.delta)} | "
            f"{row.left_total:,}/{row.right_total:,} |"
        )
    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path, help="left CoNLL-U or JSONL input")
    parser.add_argument("right", type=Path, help="right CoNLL-U or JSONL input")
    parser.add_argument("--left-name", default="left")
    parser.add_argument("--right-name", default="right")
    parser.add_argument(
        "--feats-keys",
        choices=("slots", "all"),
        default="all",
        help="compare only scoring slots or all observed FEATS keys",
    )
    parser.add_argument(
        "--min-delta-pct",
        type=float,
        default=10.0,
        help="minimum absolute coverage delta in percentage points",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    keys = key_filter(args.feats_keys)
    rows = coverage_rows(
        tokens_from_path(args.left),
        tokens_from_path(args.right),
        keys=keys,
    )
    print(
        format_coverage_table(
            rows,
            args.left_name,
            args.right_name,
            min_delta=args.min_delta_pct / 100.0,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
