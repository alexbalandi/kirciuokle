# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Fill narrow MATAS FEATS gaps from the UDPipe 2 ALKSNIS teacher.

The script writes a temp file next to the requested output and replaces the
final file only after a complete run. Reruns therefore restart from the
beginning rather than resuming a partial output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from metrics import feats_string, parse_feats


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_INPUT = DEFAULT_RAW_DIR / "MATAS3.conllu"
DEFAULT_OUTPUT = DEFAULT_RAW_DIR / "MATAS3.teacher.conllu"
DEFAULT_UDPIPE_URL = "https://lindat.mff.cuni.cz/services/udpipe/api"
UDPIPE_MODEL = "lithuanian-alksnis"
BATCH_SIZE = 80
DEFAULT_RPS = 1.0
DEFAULT_PROGRESS_EVERY = 5_000
FILL_KEYS = ("Number", "Person", "Reflex")


@dataclass
class TokenLine:
    line_index: int
    columns: list[str]


@dataclass
class Sentence:
    index: int
    lines: list[str]
    tokens: list[TokenLine]
    has_gap: bool


@dataclass
class Stats:
    processed: int = 0
    selected: int = 0
    filled: int = 0
    failed: int = 0
    fill_counts: dict[str, int] = field(
        default_factory=lambda: {key: 0 for key in FILL_KEYS}
    )


class UdpPipeClient:
    def __init__(self, base_url: str, rps: float) -> None:
        self.process_url = base_url.rstrip("/") + "/process"
        self.min_interval = 1.0 / rps
        self.last_request_at: float | None = None

    def process(self, conllu: str) -> str:
        self._pace()
        payload = urllib.parse.urlencode(
            {
                "input": "conllu",
                "tagger": "",
                "model": UDPIPE_MODEL,
                "data": conllu,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.process_url,
            data=payload,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "accentuation-lt-teacher-fill/1.0",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            raw_payload = response.read().decode("utf-8")

        payload_obj = json.loads(raw_payload)
        if payload_obj.get("error"):
            raise RuntimeError(str(payload_obj["error"]))
        result = payload_obj.get("result")
        if not isinstance(result, str):
            raise RuntimeError("UDPipe response did not contain a string result")
        return result

    def _pace(self) -> None:
        if self.last_request_at is not None:
            elapsed = time.monotonic() - self.last_request_at
            remaining = self.min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_at = time.monotonic()


def gap_keys(upos: str, feats: dict[str, str]) -> tuple[str, ...]:
    if upos in {"VERB", "AUX"} and "Number" not in feats:
        return ("Number",)
    if upos == "PRON":
        missing = [key for key in ("Person", "Reflex") if key not in feats]
        return tuple(missing)
    return ()


def parse_sentence(index: int, lines: list[str]) -> Sentence:
    tokens: list[TokenLine] = []
    has_gap = False
    for line_index, line in enumerate(lines):
        if line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) != 10:
            continue
        token = TokenLine(line_index, columns)
        tokens.append(token)
        if columns[0].isdigit() and gap_keys(columns[3], parse_feats(columns[5])):
            has_gap = True
    return Sentence(index=index, lines=lines, tokens=tokens, has_gap=has_gap)


def iter_sentences(path: Path, limit: int | None = None) -> Iterable[Sentence]:
    lines: list[str] = []
    index = 0

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if line.endswith("\r"):
                line = line[:-1]
            if line:
                lines.append(line)
                continue

            if lines:
                index += 1
                yield parse_sentence(index, lines)
                if limit is not None and index >= limit:
                    return
                lines = []

    if lines and (limit is None or index < limit):
        index += 1
        yield parse_sentence(index, lines)


def skeleton_sentence(sentence: Sentence) -> str:
    lines = []
    for token in sentence.tokens:
        columns = [token.columns[0], token.columns[1], *("_" for _ in range(8))]
        lines.append("\t".join(columns))
    return "\n".join(lines)


def skeleton_chunk(sentences: list[Sentence]) -> str:
    return "\n\n".join(skeleton_sentence(sentence) for sentence in sentences) + "\n\n"


def parse_teacher_sentences(conllu: str) -> list[list[list[str]]]:
    sentences: list[list[list[str]]] = []
    rows: list[list[str]] = []

    for raw_line in conllu.splitlines():
        line = raw_line.strip("\r")
        if not line:
            if rows:
                sentences.append(rows)
                rows = []
            continue
        if line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) == 10:
            rows.append(columns)

    if rows:
        sentences.append(rows)
    return sentences


def fetch_teacher_batch(
    client: UdpPipeClient, sentences: list[Sentence]
) -> list[list[list[str]] | None]:
    chunk = skeleton_chunk(sentences)
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            teacher_sentences = parse_teacher_sentences(client.process(chunk))
            if len(teacher_sentences) != len(sentences):
                raise RuntimeError(
                    "UDPipe returned "
                    f"{len(teacher_sentences)} sentences for {len(sentences)} inputs"
                )
            return teacher_sentences
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt == 0:
                continue

    print(
        "warning: teacher batch failed after retry; "
        f"leaving {len(sentences)} selected sentences unfilled ({last_error})",
        file=sys.stderr,
    )
    return [None for _ in sentences]


def apply_teacher(sentence: Sentence, teacher_rows: list[list[str]]) -> dict[str, int] | None:
    if len(sentence.tokens) != len(teacher_rows):
        return None

    for token, teacher_columns in zip(sentence.tokens, teacher_rows):
        if token.columns[0] != teacher_columns[0] or token.columns[1] != teacher_columns[1]:
            return None

    fill_counts = {key: 0 for key in FILL_KEYS}
    for token, teacher_columns in zip(sentence.tokens, teacher_rows):
        if not token.columns[0].isdigit():
            continue

        gold_feats = parse_feats(token.columns[5])
        keys = gap_keys(token.columns[3], gold_feats)
        if not keys:
            continue

        teacher_feats = parse_feats(teacher_columns[5])
        changed = False
        for key in keys:
            value = teacher_feats.get(key)
            if value:
                gold_feats[key] = value
                fill_counts[key] += 1
                changed = True

        if changed:
            token.columns[5] = feats_string(gold_feats)
            sentence.lines[token.line_index] = "\t".join(token.columns)

    return fill_counts


def process_pending(
    pending: list[Sentence],
    client: UdpPipeClient,
    stats: Stats,
) -> None:
    selected = [sentence for sentence in pending if sentence.has_gap]
    if not selected:
        return

    teacher_batch = fetch_teacher_batch(client, selected)
    for sentence, teacher_rows in zip(selected, teacher_batch):
        if teacher_rows is None:
            stats.failed += 1
            continue

        fill_counts = apply_teacher(sentence, teacher_rows)
        if fill_counts is None:
            stats.failed += 1
            continue

        sentence_filled = False
        for key, count in fill_counts.items():
            if count:
                stats.fill_counts[key] += count
                sentence_filled = True
        if sentence_filled:
            stats.filled += 1


def write_sentence(handle, sentence: Sentence) -> None:
    for line in sentence.lines:
        handle.write(line)
        handle.write("\n")
    handle.write("\n")


def report_progress(stats: Stats, progress_every: int) -> None:
    if stats.processed and stats.processed % progress_every == 0:
        print(
            "progress: "
            f"{stats.processed:,} sentences processed; "
            f"{stats.filled:,} filled; "
            f"{stats.failed:,} failed",
            flush=True,
        )


def flush_pending(
    pending: list[Sentence],
    output_handle,
    client: UdpPipeClient,
    stats: Stats,
    progress_every: int,
) -> None:
    process_pending(pending, client, stats)
    for sentence in pending:
        write_sentence(output_handle, sentence)
        stats.processed += 1
        report_progress(stats, progress_every)
    pending.clear()


def run_fill(
    input_path: Path,
    output_path: Path,
    udpipe_url: str,
    rps: float,
    limit: int | None,
    progress_every: int,
) -> Stats:
    if not input_path.exists():
        raise FileNotFoundError(f"missing input file: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    client = UdpPipeClient(udpipe_url, rps)
    stats = Stats()
    pending: list[Sentence] = []
    pending_selected = 0

    with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sentence in iter_sentences(input_path, limit=limit):
            if sentence.has_gap:
                stats.selected += 1
                pending.append(sentence)
                pending_selected += 1
            elif pending:
                pending.append(sentence)
            else:
                write_sentence(handle, sentence)
                stats.processed += 1
                report_progress(stats, progress_every)
                continue

            if pending_selected >= BATCH_SIZE or len(pending) >= progress_every:
                flush_pending(pending, handle, client, stats, progress_every)
                pending_selected = 0

        if pending:
            flush_pending(pending, handle, client, stats, progress_every)

    temp_path.replace(output_path)
    return stats


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fill missing MATAS Number/Person/Reflex FEATS from UDPipe 2. "
            "Output is written to a temp file and renamed at completion; "
            "reruns restart from the beginning."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--udpipe-url",
        default=os.environ.get("UDPIPE_URL", DEFAULT_UDPIPE_URL),
        help="UDPipe REST API base URL; env UDPIPE_URL overrides the default",
    )
    parser.add_argument(
        "--rps",
        type=positive_float,
        default=DEFAULT_RPS,
        help="maximum teacher requests per second",
    )
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="process only the first N sentences for a trial run",
    )
    parser.add_argument(
        "--progress-every",
        type=positive_int,
        default=DEFAULT_PROGRESS_EVERY,
        help="print progress after this many processed sentences",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    stats = run_fill(
        input_path=args.input,
        output_path=args.output,
        udpipe_url=args.udpipe_url,
        rps=args.rps,
        limit=args.limit,
        progress_every=args.progress_every,
    )

    print(f"sentences processed: {stats.processed:,}")
    print(f"sentences selected: {stats.selected:,}")
    print(f"sentences filled: {stats.filled:,}")
    print(f"sentences failed: {stats.failed:,}")
    print("per-key fill counts:")
    for key in FILL_KEYS:
        print(f"  {key}: {stats.fill_counts[key]:,}")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
