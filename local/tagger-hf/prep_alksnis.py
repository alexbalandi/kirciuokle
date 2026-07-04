# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Prepare UD_Lithuanian-ALKSNIS for combined UPOS|FEATS classification."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Iterable


BASE_URL = (
    "https://raw.githubusercontent.com/UniversalDependencies/"
    "UD_Lithuanian-ALKSNIS/master"
)
SPLITS = {
    "train": "lt_alksnis-ud-train.conllu",
    "dev": "lt_alksnis-ud-dev.conllu",
    "test": "lt_alksnis-ud-test.conllu",
}


def combined_label(upos: str, feats: str) -> str:
    return f"{upos}|{feats or '_'}"


def download(url: str, path: Path, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "tagger-hf-prep/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        path.write_bytes(response.read())


def read_conllu(path: Path) -> list[dict]:
    sentences: list[dict] = []
    text = ""
    tokens: list[str] = []
    labels: list[str] = []

    def flush() -> None:
        nonlocal text, tokens, labels
        if tokens:
            sentences.append(
                {
                    "id": f"{path.stem}-{len(sentences) + 1}",
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
        labels.append(combined_label(columns[3], columns[5] or "_"))

    flush()
    return sentences


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
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


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "alksnis",
        help="output dataset directory",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download ALKSNIS files even if cached",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    raw_dir = args.out / "raw"
    args.out.mkdir(parents=True, exist_ok=True)
    all_labels: list[str] = []

    for split, filename in SPLITS.items():
        source_path = raw_dir / filename
        download(f"{BASE_URL}/{filename}", source_path, args.force)
        rows = read_conllu(source_path)
        write_jsonl(args.out / f"{split}.jsonl", rows)
        all_labels.extend(label for row in rows for label in row["labels"])
        print(f"{split}: {len(rows)} sentences -> {args.out / f'{split}.jsonl'}")

    write_labels(args.out / "labels.json", all_labels)
    print(f"labels: {len(set(all_labels))} -> {args.out / 'labels.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
