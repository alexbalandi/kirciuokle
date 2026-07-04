# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Prepare MATAS + ALKSNIS for combined UPOS|FEATS token classification."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from metrics import combined_label


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = BASE_DIR / "data" / "raw"
DEFAULT_OUT_DIR = BASE_DIR / "data" / "combined"
DEFAULT_SEED = 13

ALKSNIS_FILES = {
    "train": "lt_alksnis-ud-train.conllu",
    "dev": "lt_alksnis-ud-dev.conllu",
    "test": "lt_alksnis-ud-test.conllu",
}
MATAS_FILE = "MATAS3.conllu"
VALID_SOURCES = {"matas", "alksnis"}


def parse_sources(value: str) -> list[str]:
    sources = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not sources:
        raise argparse.ArgumentTypeError("at least one source is required")
    unknown = sorted(set(sources) - VALID_SOURCES)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown source(s): {', '.join(unknown)}")
    return list(dict.fromkeys(sources))


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run local/tagger-hf/fetch_corpora.py first"
        )


def normalized_text(row: dict) -> str:
    text = row.get("text") or " ".join(row["tokens"])
    return " ".join(str(text).casefold().split())


def read_conllu(path: Path, sentence_prefix: str) -> list[dict]:
    require_file(path)
    sentences: list[dict] = []
    text = ""
    tokens: list[str] = []
    labels: list[str] = []

    def flush() -> None:
        nonlocal text, tokens, labels
        if tokens:
            sentences.append(
                {
                    "id": f"{sentence_prefix}-{len(sentences) + 1}",
                    "text": text or " ".join(tokens),
                    "tokens": tokens,
                    "labels": labels,
                }
            )
        text = ""
        tokens = []
        labels = []

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip("\n")
        if not line:
            flush()
            continue
        if line.startswith("# text = "):
            text = line[len("# text = ") :]
            continue
        if line.startswith("#"):
            continue

        columns = line.split("\t")
        if len(columns) < 6 or not columns[0].isdigit():
            continue
        tokens.append(columns[1])
        labels.append(combined_label(columns[3], columns[5]))

    flush()
    return sentences


def dedupe_sentences(rows: Iterable[dict]) -> tuple[list[dict], int]:
    seen: set[str] = set()
    deduped: list[dict] = []
    dropped = 0
    for row in rows:
        key = normalized_text(row)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, dropped


def drop_leaks(rows: Iterable[dict], heldout_keys: set[str]) -> tuple[list[dict], int]:
    kept: list[dict] = []
    dropped = 0
    for row in rows:
        if normalized_text(row) in heldout_keys:
            dropped += 1
            continue
        kept.append(row)
    return kept, dropped


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_labels(path: Path, labels: Iterable[str]) -> None:
    label_list = sorted(set(labels))
    payload = {
        "labels": label_list,
        "label2id": {label: index for index, label in enumerate(label_list)},
        "id2label": {str(index): label for index, label in enumerate(label_list)},
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def token_count(rows: Iterable[dict]) -> int:
    return sum(len(row["tokens"]) for row in rows)


def labels_in(rows: Iterable[dict]) -> list[str]:
    return [label for row in rows for label in row["labels"]]


def oov_label_stats(rows: Iterable[dict], train_labels: set[str]) -> tuple[int, int, float]:
    labels = labels_in(rows)
    oov = sum(1 for label in labels if label not in train_labels)
    total = len(labels)
    return oov, total, (oov / total if total else 0.0)


def print_split_stats(name: str, rows: list[dict]) -> None:
    print(f"{name}: {len(rows):,} sentences / {token_count(rows):,} tokens")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        type=parse_sources,
        default=parse_sources("matas,alksnis"),
        help="comma-separated training sources: matas,alksnis",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="source corpus cache directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output dataset directory",
    )
    parser.add_argument(
        "--max-train-sentences",
        type=int,
        help="deterministic smoke limit after shuffling",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.max_train_sentences is not None and args.max_train_sentences < 1:
        parser.error("--max-train-sentences must be positive")

    alksnis_dev = read_conllu(args.raw_dir / ALKSNIS_FILES["dev"], "alksnis-dev")
    alksnis_test = read_conllu(args.raw_dir / ALKSNIS_FILES["test"], "alksnis-test")
    heldout_keys = {normalized_text(row) for row in alksnis_dev + alksnis_test}

    train_rows: list[dict] = []
    matas_deduped_dropped = 0
    if "matas" in args.sources:
        matas_rows = read_conllu(args.raw_dir / MATAS_FILE, "matas")
        matas_rows, matas_deduped_dropped = dedupe_sentences(matas_rows)
        train_rows.extend(matas_rows)

    if "alksnis" in args.sources:
        train_rows.extend(
            read_conllu(args.raw_dir / ALKSNIS_FILES["train"], "alksnis-train")
        )

    train_rows, leaked_dropped = drop_leaks(train_rows, heldout_keys)
    random.Random(args.seed).shuffle(train_rows)
    if args.max_train_sentences is not None:
        train_rows = train_rows[: args.max_train_sentences]

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "dev.jsonl", alksnis_dev)
    write_jsonl(args.out / "test.jsonl", alksnis_test)

    all_labels = labels_in(train_rows) + labels_in(alksnis_dev) + labels_in(alksnis_test)
    write_labels(args.out / "labels.json", all_labels)

    train_label_set = set(labels_in(train_rows))
    dev_oov, dev_total, dev_rate = oov_label_stats(alksnis_dev, train_label_set)
    test_oov, test_total, test_rate = oov_label_stats(alksnis_test, train_label_set)

    print(f"sources: {','.join(args.sources)}")
    if "matas" in args.sources:
        print(f"matas duplicate sentences dropped: {matas_deduped_dropped:,}")
    print(f"leakage guard dropped training sentences: {leaked_dropped:,}")
    print_split_stats("train", train_rows)
    print_split_stats("dev", alksnis_dev)
    print_split_stats("test", alksnis_test)
    print(f"label set: {len(set(all_labels)):,}")
    print(f"dev OOV-label rate: {dev_oov:,}/{dev_total:,} ({dev_rate:.2%})")
    print(f"test OOV-label rate: {test_oov:,}/{test_total:,} ({test_rate:.2%})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
