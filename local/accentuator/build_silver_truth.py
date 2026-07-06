# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""Build silver token truth from the production VDU + UDPipe accenting path."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import unicodedata
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_INPUT = SCRIPT_DIR / "data" / "eval" / "lrt-corpus.txt"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "eval" / "lrt-silver.jsonl"
MIN_REQUEST_INTERVAL = 1.05
T = TypeVar("T")

sys.path.insert(0, str(SCRIPTS_DIR))
import accent_text  # noqa: E402
import eval_accenter  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR))
from _common import normalize_lt, strip_accents  # noqa: E402


class AsyncThrottle:
    def __init__(self, interval: float = MIN_REQUEST_INTERVAL) -> None:
        self.interval = interval
        self.last_start = 0.0

    async def call(self, factory: Callable[[], Awaitable[T]]) -> T:
        now = time.monotonic()
        if self.last_start:
            delay = self.interval - (now - self.last_start)
            if delay > 0:
                await asyncio.sleep(delay)
        self.last_start = time.monotonic()
        return await factory()


def token_count(text: str) -> int:
    return len(eval_accenter.tokenize(text))


def count_jsonl_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def lower_plain(text: str) -> str:
    return strip_accents(normalize_lt(text)).lower()


def lower_form(text: str) -> str:
    return normalize_lt(text).lower()


def has_stress(text: str) -> bool:
    return any(mark in unicodedata.normalize("NFD", text) for mark in eval_accenter.STRESS_MARKS)


def part_tokens(text: str) -> list[str]:
    return eval_accenter.tokenize(text)


def ud_payload(tok: accent_text.Token | None) -> dict[str, str] | None:
    if tok is None:
        return None
    feats = "|".join(f"{key}={value}" for key, value in tok.feats.items()) if tok.feats else "_"
    return {"upos": str(tok.upos or "X"), "feats": feats}


def append_token_records(
    records: list[dict[str, Any]],
    original_text: str,
    accented_text: str,
    mi: str | None,
    ambiguous: bool,
    ud: accent_text.Token | None = None,
) -> None:
    original_tokens = part_tokens(original_text)
    accented_tokens = part_tokens(accented_text)
    if len(accented_tokens) != len(original_tokens):
        accented_tokens = original_tokens
    for index, original in enumerate(original_tokens):
        records.append(
            {
                "word": lower_plain(original),
                "accented": lower_form(accented_tokens[index]),
                "mi": mi if index == 0 else None,
                "ambiguous": ambiguous if index == 0 else False,
                "ud": ud_payload(ud) if index == 0 else None,
            }
        )


def selected_mi(selected_form: str, variants: list[dict[str, Any]], tok: accent_text.Token | None) -> str | None:
    selected = lower_form(selected_form)
    matches = [
        variant
        for variant in variants
        if lower_form(str(variant.get("form", ""))) == selected
    ]
    if not matches:
        return None
    if tok is not None:
        ctx = accent_text.token_tags(tok)
        scored: list[tuple[int, str]] = []
        for variant in matches:
            for mi in variant.get("mi", []):
                scored.append((accent_text.score(accent_text.parse_mi(mi), ctx), mi))
        if scored:
            scored.sort(key=lambda item: item[0], reverse=True)
            return scored[0][1]
    for variant in matches:
        for mi in variant.get("mi", []):
            return mi
    return None


async def records_for_chunk(
    client: httpx.AsyncClient,
    throttle: AsyncThrottle,
    nonce: str,
    chunk: str,
    variant_cache: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    parts = await throttle.call(lambda: accent_text.text_accents(client, nonce, chunk))
    try:
        tokens = await throttle.call(lambda: accent_text.udpipe_tag(client, chunk))
    except (httpx.HTTPError, KeyError, json.JSONDecodeError) as exc:
        print(f"[tagger unavailable for chunk, falling back to defaults: {exc}]", file=sys.stderr)
        tokens = []

    word_parts = [part for part in parts if part.get("type") in ("WORD", "NON_LT")]
    aligned = accent_text.align(parts, tokens) if tokens else [None] * len(word_parts)

    ambiguous_words = {
        lower_plain(part["string"])
        for part in word_parts
        if part.get("accentType") == "MULTIPLE_MEANING"
    }
    for word in sorted(ambiguous_words - variant_cache.keys()):
        variant_cache[word] = await throttle.call(lambda word=word: accent_text.word_variants(client, nonce, word))

    records: list[dict[str, Any]] = []
    aligned_by_part_id = {id(part): tok for part, tok in zip(word_parts, aligned)}
    for part in parts:
        if part.get("type") not in ("WORD", "NON_LT"):
            append_token_records(
                records,
                str(part.get("string", "")),
                str(part.get("string", "")),
                None,
                False,
                None,
            )
            continue

        tok = aligned_by_part_id.get(id(part))
        original = str(part.get("string", ""))
        word = lower_plain(original)
        accent_type = part.get("accentType")
        ambiguous = accent_type == "MULTIPLE_MEANING"
        mi: str | None = None

        if part.get("type") == "NON_LT" or accent_type == "NONE":
            selected = original
        elif ambiguous:
            variants = variant_cache.get(word, [])
            selected, _how = accent_text.pick_variant(original, variants, tok)
            if selected is None:
                selected = str(part.get("accented", original))
            mi = selected_mi(selected, variants, tok)
        else:
            selected = str(part.get("accented", original))

        append_token_records(records, original, selected, mi, ambiguous, tok)
    return records


def append_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def spot_records(path: Path, limit: int = 5) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    fallback: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if has_stress(record.get("accented", "")):
                fallback.append(record)
                if record.get("mi"):
                    selected.append(record)
                    if len(selected) >= limit:
                        return selected
    return selected or fallback[:limit]


async def build(input_path: Path, output_path: Path) -> int:
    text = input_path.read_text(encoding="utf-8")
    chunks = accent_text.chunk_text(text, accent_text.CHUNK_LIMIT)
    expected_total = sum(token_count(chunk) for chunk in chunks)
    existing_lines = count_jsonl_lines(output_path)
    if existing_lines > expected_total:
        raise RuntimeError(
            f"{output_path} has {existing_lines} lines, but {input_path} only has "
            f"{expected_total} word tokens"
        )

    print(f"chunks: {len(chunks)}")
    print(f"expected JSONL lines: {expected_total:,}")
    if existing_lines:
        print(f"resuming from existing JSONL lines: {existing_lines:,}")
    if existing_lines == expected_total:
        print("output already complete; no external accenting calls needed")
        print(f"JSONL lines: {existing_lines:,}")
        print("spot lines:")
        for record in spot_records(output_path):
            print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        return 0

    remaining_existing = existing_lines
    variant_cache: dict[str, list[dict[str, Any]]] = {}
    async with httpx.AsyncClient(headers={"User-Agent": "accent-script/1.0"}) as client:
        throttle = AsyncThrottle()
        nonce: str | None = None
        for index, chunk in enumerate(chunks, start=1):
            chunk_tokens = token_count(chunk)
            if remaining_existing >= chunk_tokens:
                remaining_existing -= chunk_tokens
                print(f"skip chunk {index}/{len(chunks)} ({chunk_tokens:,} tokens already written)")
                continue

            if nonce is None:
                nonce = await throttle.call(lambda: accent_text.get_nonce(client))
            already_written = remaining_existing
            remaining_existing = 0
            records = await records_for_chunk(client, throttle, nonce, chunk, variant_cache)
            if len(records) != chunk_tokens:
                print(
                    f"[warning] chunk {index} tokenizer/VDU count mismatch: "
                    f"{chunk_tokens} expected vs {len(records)} records",
                    file=sys.stderr,
                )
            to_write = records[already_written:]
            append_records(output_path, to_write)
            print(
                f"wrote chunk {index}/{len(chunks)}: {len(to_write):,} records "
                f"({count_jsonl_lines(output_path):,} total)"
            )

    total = count_jsonl_lines(output_path)
    print(f"JSONL lines: {total:,}")
    print("spot lines:")
    for record in spot_records(output_path):
        print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.input.exists():
        parser.error(f"missing input: {args.input}")
    return asyncio.run(build(args.input, args.output))


if __name__ == "__main__":
    raise SystemExit(main())
