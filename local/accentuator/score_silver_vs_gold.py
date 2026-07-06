# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Score VDU+UDPipe silver tokens against the Chrestomatija gold set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import eval_chrestomatija as chrest  # noqa: E402
from _common import safe_relative  # noqa: E402


SYSTEM_NAME = "vdu-udpipe (online)"
DEFAULT_SILVER = SCRIPT_DIR / "data" / "eval" / "chrestomatija-vdu-silver.jsonl"
MAX_UNALIGNED_RATE = 0.01


@dataclass(frozen=True)
class SilverToken:
    word: str
    accented: str
    mi: str | None
    ambiguous: bool
    line_number: int


class SilverNotReady(RuntimeError):
    """Raised when the silver JSONL is missing or visibly incomplete."""


def load_silver(path: Path) -> list[SilverToken]:
    if not path.exists():
        raise SilverNotReady(
            f"silver JSONL is not ready: missing {safe_relative(path)}"
        )

    tokens: list[SilverToken] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                word = chrest.word_key(raw.get("word"))
                accented = chrest.norm_form(raw.get("accented"))
                if not word or not accented:
                    raise ValueError(f"bad silver row at {path}:{line_number}")
                tokens.append(
                    SilverToken(
                        word=word,
                        accented=accented,
                        mi=raw.get("mi") or None,
                        ambiguous=bool(raw.get("ambiguous")),
                        line_number=line_number,
                    )
                )
    except json.JSONDecodeError as exc:
        raise SilverNotReady(
            f"silver JSONL is not ready: partial JSON at {safe_relative(path)}:"
            f"{exc.lineno}:{exc.colno}"
        ) from exc

    if not tokens:
        raise SilverNotReady(f"silver JSONL is not ready: empty {safe_relative(path)}")
    return tokens


def align_silver_to_gold(
    gold: list[chrest.GoldToken],
    silver: list[SilverToken],
    window: int = 8,
) -> tuple[list[tuple[int, int]], int, int]:
    aligned: list[tuple[int, int]] = []
    skipped_gold = 0
    skipped_silver = 0
    gold_index = 0
    silver_index = 0

    while gold_index < len(gold):
        target = gold[gold_index].word
        if silver_index < len(silver) and silver[silver_index].word == target:
            aligned.append((gold_index, silver_index))
            gold_index += 1
            silver_index += 1
            continue

        found_silver = None
        for lookahead in range(
            silver_index + 1,
            min(len(silver), silver_index + window + 1),
        ):
            if silver[lookahead].word == target:
                found_silver = lookahead
                break

        found_gold = None
        if silver_index < len(silver):
            current = silver[silver_index].word
            for lookahead in range(
                gold_index + 1,
                min(len(gold), gold_index + window + 1),
            ):
                if gold[lookahead].word == current:
                    found_gold = lookahead
                    break

        if found_silver is not None and (
            found_gold is None or found_silver - silver_index <= found_gold - gold_index
        ):
            skipped_silver += found_silver - silver_index
            silver_index = found_silver
            continue
        if found_gold is not None:
            skipped_gold += found_gold - gold_index
            gold_index = found_gold
            continue

        skipped_gold += 1
        gold_index += 1

    skipped_silver += max(0, len(silver) - silver_index)
    return aligned, skipped_gold, skipped_silver


def sentence_positions(
    sentences: list[chrest.GoldSentence],
) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for sentence_index, sentence in enumerate(sentences):
        for token_index, _token in enumerate(sentence.tokens):
            positions.append((sentence_index, token_index))
    return positions


def predictions_by_sentence(
    sentences: list[chrest.GoldSentence],
    silver: list[SilverToken],
    aligned: list[tuple[int, int]],
) -> list[list[str | object]]:
    predictions: list[list[str | object]] = [
        [chrest.MISSING] * len(sentence.tokens) for sentence in sentences
    ]
    positions = sentence_positions(sentences)
    for gold_index, silver_index in aligned:
        sentence_index, token_index = positions[gold_index]
        predictions[sentence_index][token_index] = silver[silver_index].accented
    return predictions


def silver_answered(token: chrest.GoldToken, predicted: str | object) -> bool:
    if predicted is chrest.MISSING:
        return False
    if isinstance(predicted, str) and chrest.count_stress_marks(predicted) > 0:
        return True
    return not chrest.gold_has_stress(token)


def score_predictions(
    sentences: list[chrest.GoldSentence],
    predictions: list[list[str | object]],
    elapsed_seconds: float,
    skipped_gold_tokens: int,
    skipped_silver_tokens: int,
) -> chrest.ChrestomatijaMetrics:
    total_tokens = answered = exact = position = 0
    sentence_exact = 0
    for sentence, sentence_predictions in zip(sentences, predictions):
        if len(sentence_predictions) != len(sentence.tokens):
            sentence_predictions = [chrest.MISSING] * len(sentence.tokens)
        sentence_ok = True
        for token, predicted in zip(sentence.tokens, sentence_predictions):
            total_tokens += 1
            answered += int(silver_answered(token, predicted))
            is_exact = chrest.token_exact(token, predicted)
            exact += int(is_exact)
            position += int(chrest.token_position(token, predicted))
            sentence_ok = sentence_ok and is_exact
        sentence_exact += int(sentence_ok)

    return chrest.ChrestomatijaMetrics(
        system=SYSTEM_NAME,
        status="ok",
        total_tokens=total_tokens,
        answered_tokens=answered,
        token_exact=exact,
        token_position=position,
        total_sentences=len(sentences),
        sentence_exact=sentence_exact,
        elapsed_seconds=elapsed_seconds,
        skipped_gold_tokens=skipped_gold_tokens,
        skipped_model_tokens=skipped_silver_tokens,
    )


def metric_row(metrics: chrest.ChrestomatijaMetrics) -> str:
    return chrest.metric_rows([metrics])[2]


def replace_table_row(
    report: str,
    header: str,
    system: str,
    new_row: str,
) -> str:
    lines = report.splitlines()
    try:
        header_index = lines.index(header)
    except ValueError as exc:
        raise RuntimeError(f"report table header not found: {header}") from exc

    row_start = header_index + 2
    row_end = row_start
    while row_end < len(lines) and lines[row_end].startswith("| "):
        row_end += 1

    prefix = f"| {system} |"
    rows = [line for line in lines[row_start:row_end] if not line.startswith(prefix)]
    lines[row_start:row_end] = [*rows, new_row]
    return "\n".join(lines) + "\n"


def update_report(path: Path, metrics: chrest.ChrestomatijaMetrics) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    report = path.read_text(encoding="utf-8")
    report = replace_table_row(
        report,
        "| system | status | answered | token exact | token position | sentence sequence | time |",
        SYSTEM_NAME,
        metric_row(metrics),
    )
    report = replace_table_row(
        report,
        "| system | skipped gold tokens | skipped model tokens |",
        SYSTEM_NAME,
        (
            f"| {SYSTEM_NAME} | {metrics.skipped_gold_tokens:,} "
            f"| {metrics.skipped_model_tokens:,} |"
        ),
    )
    path.write_text(report, encoding="utf-8", newline="\n")


def fail_if_visibly_partial(
    silver: list[SilverToken],
    total_gold_tokens: int,
    silver_path: Path,
) -> None:
    if len(silver) < total_gold_tokens:
        raise SilverNotReady(
            "silver JSONL appears partial: "
            f"{len(silver):,} rows in {safe_relative(silver_path)} for "
            f"{total_gold_tokens:,} gold tokens; wait for build_silver_truth.py to finish"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=chrest.DEFAULT_GOLD)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--report", type=Path, default=chrest.DEFAULT_REPORT)
    parser.add_argument("--alignment-window", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    started = time.perf_counter()

    if not args.gold.exists():
        parser.error(f"missing gold JSONL: {args.gold}")

    try:
        sentences = chrest.load_gold(args.gold)
        gold = chrest.flatten_tokens(sentences)
        silver = load_silver(args.silver)
        fail_if_visibly_partial(silver, len(gold), args.silver)

        aligned, skipped_gold, skipped_silver = align_silver_to_gold(
            gold,
            silver,
            window=args.alignment_window,
        )
        unaligned = skipped_gold + skipped_silver
        unaligned_rate = unaligned / max(len(gold), len(silver), 1)
        print(
            "alignment: "
            f"{len(aligned):,} aligned; "
            f"{skipped_gold:,} skipped gold; "
            f"{skipped_silver:,} skipped silver; "
            f"{unaligned_rate:.2%} unaligned"
        )
        if unaligned_rate >= MAX_UNALIGNED_RATE:
            raise RuntimeError(
                "alignment unaligned rate too high: "
                f"{unaligned_rate:.2%} ({unaligned:,} unaligned tokens; "
                f"{skipped_gold:,} gold, {skipped_silver:,} silver)"
            )

        predictions = predictions_by_sentence(sentences, silver, aligned)
        metrics = score_predictions(
            sentences=sentences,
            predictions=predictions,
            elapsed_seconds=time.perf_counter() - started,
            skipped_gold_tokens=skipped_gold,
            skipped_silver_tokens=skipped_silver,
        )
        update_report(args.report, metrics)
        print(metric_row(metrics))
        print(f"report written: {safe_relative(args.report)}")
        return 0
    except SilverNotReady as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
