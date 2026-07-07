# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch",
#   "transformers<5",
# ]
# ///
"""Prune unused joint_v3 tokenizer pieces and remap the checkpoint embedding.

The SPEC48 prose describes the upstream tokenizer as Unigram, but the local
litlat artifact is BPE. For BPE, retained output pieces also need the merge
support pieces required by tokenizers' model validator; those support pieces are
recorded separately in the report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ACCENTUATOR_DIR = SCRIPT_DIR.parent
LOCAL_DIR = ACCENTUATOR_DIR.parent
REPO_ROOT = LOCAL_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
from joint_lib import safe_relative, word_key  # noqa: E402


DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "joint_v3.best.pt"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "pruned"
DEFAULT_DICTIONARY = ACCENTUATOR_DIR / "data" / "generated.sqlite"
TOKENIZER_FILES = ("tokenizer_config.json", "special_tokens_map.json", "config.json")
TEXT_CORPORA = (
    ACCENTUATOR_DIR / "data" / "eval" / "lrt-corpus.txt",
    ACCENTUATOR_DIR / "data" / "eval" / "wikipedia-corpus.txt",
    ACCENTUATOR_DIR / "data" / "eval" / "literary-corpus.txt",
    ACCENTUATOR_DIR / "data" / "eval" / "literary-corpus-2.txt",
    ACCENTUATOR_DIR / "data" / "eval" / "chrestomatija-plain.txt",
)
JSONL_CORPORA = (
    SCRIPT_DIR / "data" / "train.jsonl",
    SCRIPT_DIR / "data" / "dev.jsonl",
    SCRIPT_DIR / "data" / "alksnis_dev.jsonl",
    SCRIPT_DIR / "data" / "alksnis_test.jsonl",
    SCRIPT_DIR / "data-round2" / "train.jsonl",
    SCRIPT_DIR / "data-round2" / "dev.jsonl",
)
SPECIAL_WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack",
)


def format_duration(seconds: float) -> str:
    return f"{seconds:.1f}s"


def require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"not a joint checkpoint: {path}")
    return payload


def tokenizer_source_from_checkpoint(checkpoint: dict[str, Any]) -> Path:
    source = checkpoint.get("encoder_source") or checkpoint.get("base_model")
    if not source:
        raise ValueError("checkpoint does not record encoder_source/base_model")
    path = Path(str(source))
    if not path.exists():
        raise FileNotFoundError(
            f"checkpoint encoder source must be local for pruning: {source}"
        )
    if not (path / "tokenizer.json").exists():
        raise FileNotFoundError(path / "tokenizer.json")
    return path


def load_tokenizer(source: Path) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("joint tokenizer pruning requires the fast tokenizer")
    return tokenizer


def piece_len_for_small_set(piece: str) -> int:
    return len(piece[1:] if piece.startswith("▁") else piece)


def batched(items: Iterable[str], size: int) -> Iterator[list[str]]:
    batch: list[str] = []
    for item in items:
        if not item:
            continue
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def add_ids_from_encodings(retained_ids: set[int], encodings: Any) -> int:
    added = 0
    for ids in encodings["input_ids"]:
        for raw_id in ids:
            item = int(raw_id)
            if item not in retained_ids:
                retained_ids.add(item)
                added += 1
    return added


def observe_words(
    tokenizer: Any,
    retained_ids: set[int],
    words: Iterable[str],
    source_name: str,
    batch_size: int,
) -> dict[str, int | str]:
    started = time.perf_counter()
    total = 0
    before = len(retained_ids)
    for batch in batched(words, batch_size):
        total += len(batch)
        encodings = tokenizer(batch, add_special_tokens=False)
        add_ids_from_encodings(retained_ids, encodings)
        if total and total % 100_000 == 0:
            print(f"  {source_name}: tokenized {total:,} forms")
    return {
        "source": source_name,
        "items": total,
        "new_piece_ids": len(retained_ids) - before,
        "elapsed": format_duration(time.perf_counter() - started),
    }


def observe_split_sentences(
    tokenizer: Any,
    retained_ids: set[int],
    sentences: Iterable[list[str]],
    source_name: str,
    batch_size: int,
) -> dict[str, int | str]:
    started = time.perf_counter()
    total_sentences = 0
    total_tokens = 0
    before = len(retained_ids)
    batch: list[list[str]] = []
    for words in sentences:
        if not words:
            continue
        batch.append(words)
        total_sentences += 1
        total_tokens += len(words)
        if len(batch) >= batch_size:
            encodings = tokenizer(
                batch,
                is_split_into_words=True,
                add_special_tokens=False,
            )
            add_ids_from_encodings(retained_ids, encodings)
            batch = []
        if total_sentences and total_sentences % 25_000 == 0:
            print(
                f"  {source_name}: tokenized {total_sentences:,} rows "
                f"({total_tokens:,} tokens)"
            )
    if batch:
        encodings = tokenizer(
            batch,
            is_split_into_words=True,
            add_special_tokens=False,
        )
        add_ids_from_encodings(retained_ids, encodings)
    return {
        "source": source_name,
        "sentences": total_sentences,
        "tokens": total_tokens,
        "new_piece_ids": len(retained_ids) - before,
        "elapsed": format_duration(time.perf_counter() - started),
    }


def dictionary_forms(path: Path) -> Iterator[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    db = sqlite3.connect(path)
    try:
        for word, variants_json, default_form in db.execute(
            "SELECT word, variants, default_form FROM words"
        ):
            for value in (word, default_form):
                key = word_key(value)
                if key:
                    yield key
            try:
                variants = json.loads(variants_json or "[]")
            except json.JSONDecodeError:
                variants = []
            if not isinstance(variants, list):
                continue
            for variant in variants:
                if not isinstance(variant, dict):
                    continue
                key = word_key(variant.get("form"))
                if key:
                    yield key
    finally:
        db.close()


def text_lines(path: Path) -> Iterator[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def jsonl_sentences(path: Path) -> Iterator[list[str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            tokens = raw.get("tokens") or []
            if tokens and isinstance(tokens[0], dict):
                words = [str(token.get("word") or "") for token in tokens]
            else:
                words = [str(token) for token in tokens]
            if not isinstance(words, list):
                raise ValueError(f"bad tokens at {path}:{line_number}")
            yield [word for word in words if word]


def special_pieces(tokenizer_json: dict[str, Any], tokenizer: Any) -> set[str]:
    pieces = set(str(item) for item in getattr(tokenizer, "all_special_tokens", []) or [])
    for token in tokenizer_json.get("added_tokens") or []:
        if token.get("special"):
            pieces.add(str(token.get("content")))
    return {piece for piece in pieces if piece}


def add_special_and_short_pieces(
    retained_ids: set[int],
    piece_to_old_id: dict[str, int],
    specials: set[str],
) -> dict[str, int]:
    before = len(retained_ids)
    for piece in specials:
        if piece in piece_to_old_id:
            retained_ids.add(piece_to_old_id[piece])

    small_bases: set[str] = set()
    for piece, old_id in piece_to_old_id.items():
        if piece_len_for_small_set(piece) <= 2:
            retained_ids.add(old_id)
            small_bases.add(piece[1:] if piece.startswith("▁") else piece)

    boundary_added = 0
    for base in small_bases:
        if not base:
            continue
        boundary = f"▁{base}"
        old_id = piece_to_old_id.get(boundary)
        if old_id is not None and old_id not in retained_ids:
            retained_ids.add(old_id)
            boundary_added += 1

    return {
        "special_pieces": len(specials),
        "short_and_boundary_piece_ids_added": len(retained_ids) - before,
        "boundary_family_added": boundary_added,
    }


def bpe_merge_closure(
    retained_pieces: set[str],
    merges: list[list[str]],
) -> tuple[set[str], int]:
    before = len(retained_pieces)
    changed = True
    merge_pairs = [(str(item[0]), str(item[1])) for item in merges]
    while changed:
        changed = False
        for left, right in merge_pairs:
            if left + right not in retained_pieces:
                continue
            if left not in retained_pieces:
                retained_pieces.add(left)
                changed = True
            if right not in retained_pieces:
                retained_pieces.add(right)
                changed = True
    return retained_pieces, len(retained_pieces) - before


def sorted_vocab_items(model: dict[str, Any]) -> list[tuple[str, int, float | None]]:
    vocab = model.get("vocab")
    if isinstance(vocab, dict):
        return [(str(piece), int(old_id), None) for piece, old_id in vocab.items()]
    if isinstance(vocab, list):
        items: list[tuple[str, int, float | None]] = []
        for old_id, entry in enumerate(vocab):
            if isinstance(entry, list | tuple) and entry:
                score = float(entry[1]) if len(entry) > 1 else None
                items.append((str(entry[0]), old_id, score))
            else:
                items.append((str(entry), old_id, None))
        return items
    raise ValueError("unsupported tokenizer model vocab shape")


def rewrite_tokenizer_json(
    source_json: dict[str, Any],
    retained_pieces: set[str],
) -> tuple[dict[str, Any], list[int], dict[str, int], list[dict[str, Any]]]:
    data = json.loads(json.dumps(source_json))
    model = data["model"]
    model_type = str(model.get("type") or "")
    original_items = sorted(sorted_vocab_items(model), key=lambda item: item[1])
    retained_items = [item for item in original_items if item[0] in retained_pieces]
    new_to_old = [old_id for _piece, old_id, _score in retained_items]
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(new_to_old)}

    if model_type == "BPE":
        new_vocab = {
            piece: old_to_new[old_id]
            for piece, old_id, _score in retained_items
        }
        merges = [
            [str(item[0]), str(item[1])]
            for item in model.get("merges", [])
            if (
                str(item[0]) in new_vocab
                and str(item[1]) in new_vocab
                and str(item[0]) + str(item[1]) in new_vocab
            )
        ]
        model["vocab"] = new_vocab
        model["merges"] = merges
    elif model_type == "Unigram":
        model["vocab"] = [
            [piece, score if score is not None else 0.0]
            for piece, _old_id, score in retained_items
        ]
    else:
        raise ValueError(f"unsupported tokenizer model type for pruning: {model_type}")

    added_tokens = []
    for token in data.get("added_tokens") or []:
        content = str(token.get("content") or "")
        new_id = old_to_new.get(next((old for piece, old, _ in original_items if piece == content), -1))
        if new_id is None:
            continue
        copied = dict(token)
        copied["id"] = new_id
        added_tokens.append(copied)
    data["added_tokens"] = added_tokens

    dropped = [
        {
            "piece": piece,
            "old_id": old_id,
            "score": score,
            "bpe_priority": old_id,
        }
        for piece, old_id, score in original_items
        if piece not in retained_pieces
    ]
    return data, new_to_old, old_to_new, dropped


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_tokenizer_sidecars(
    source: Path,
    tokenizer_dir: Path,
    old_to_new: dict[int, int],
    piece_to_old_id: dict[str, int],
    new_vocab_size: int,
) -> None:
    for name in TOKENIZER_FILES:
        source_path = source / name
        if not source_path.exists():
            continue
        target = tokenizer_dir / name
        if name == "tokenizer_config.json":
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            decoder = payload.get("added_tokens_decoder")
            if isinstance(decoder, dict):
                next_decoder = {}
                for token in decoder.values():
                    if not isinstance(token, dict):
                        continue
                    old_id = piece_to_old_id.get(str(token.get("content") or ""))
                    new_id = old_to_new.get(old_id if old_id is not None else -1)
                    if new_id is not None:
                        next_decoder[str(new_id)] = token
                payload["added_tokens_decoder"] = next_decoder
            write_json(target, payload)
            continue
        if name == "config.json":
            payload = json.loads(source_path.read_text(encoding="utf-8"))
            payload["vocab_size"] = new_vocab_size
            write_json(target, payload)
            continue
        shutil.copy2(source_path, target)


def slice_checkpoint(
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    tokenizer_dir: Path,
    output_checkpoint: Path,
    new_to_old: list[int],
    old_tokenizer_vocab_size: int,
    dropped_count: int,
    support_added_count: int,
) -> None:
    state = dict(checkpoint["model_state"])
    embedding_name = "encoder.embeddings.word_embeddings.weight"
    old_embedding = state.get(embedding_name)
    if not torch.is_tensor(old_embedding):
        raise KeyError(embedding_name)
    index = torch.tensor(new_to_old, dtype=torch.long)
    state[embedding_name] = old_embedding.index_select(0, index).contiguous()

    pruned = dict(checkpoint)
    pruned["model_state"] = state
    pruned["encoder_source"] = str(tokenizer_dir.resolve())
    pruned["pruned_vocab"] = {
        "source_checkpoint": safe_relative(checkpoint_path),
        "source_encoder": str(checkpoint.get("encoder_source") or checkpoint.get("base_model")),
        "tokenizer_dir": safe_relative(tokenizer_dir),
        "old_tokenizer_vocab_size": old_tokenizer_vocab_size,
        "old_embedding_rows": int(old_embedding.shape[0]),
        "new_vocab_size": len(new_to_old),
        "dropped_tokenizer_rows": dropped_count,
        "bpe_support_pieces_added": support_added_count,
        "new_to_old": new_to_old,
        "old_to_new": {str(old_id): new_id for new_id, old_id in enumerate(new_to_old)},
    }
    torch.save(pruned, output_checkpoint)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dictionary", type=Path, default=DEFAULT_DICTIONARY)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    started = time.perf_counter()
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    output_dir = args.output_dir
    tokenizer_dir = output_dir / "tokenizer"
    output_checkpoint = output_dir / "joint_v3.pruned.pt"
    report_path = output_dir / "prune_report.json"

    for path in (output_checkpoint, report_path, tokenizer_dir / "tokenizer.json"):
        require_absent(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading checkpoint: {safe_relative(args.checkpoint)}")
    checkpoint = load_checkpoint(args.checkpoint)
    if len(checkpoint.get("labels") or []) != 804:
        raise RuntimeError(f"label count changed before pruning: {len(checkpoint.get('labels') or [])}")
    tokenizer_source = tokenizer_source_from_checkpoint(checkpoint)
    tokenizer = load_tokenizer(tokenizer_source)
    tokenizer_json = json.loads((tokenizer_source / "tokenizer.json").read_text(encoding="utf-8"))
    model = tokenizer_json["model"]
    model_type = str(model.get("type") or "")
    original_items = sorted(sorted_vocab_items(model), key=lambda item: item[1])
    piece_to_old_id = {piece: old_id for piece, old_id, _score in original_items}
    old_tokenizer_vocab_size = len(original_items)

    print(
        "tokenizer: "
        f"type={model_type} pieces={old_tokenizer_vocab_size:,} "
        f"source={safe_relative(tokenizer_source)}"
    )
    retained_ids: set[int] = set()
    source_reports: list[dict[str, int | str]] = []

    print("retention source: generated dictionary plain forms")
    source_reports.append(
        observe_words(
            tokenizer,
            retained_ids,
            dictionary_forms(args.dictionary),
            "dictionary",
            args.batch_size,
        )
    )
    for path in TEXT_CORPORA:
        print(f"retention source: {safe_relative(path)}")
        source_reports.append(
            observe_words(
                tokenizer,
                retained_ids,
                text_lines(path),
                safe_relative(path),
                max(1, args.batch_size // 8),
            )
        )
    for path in JSONL_CORPORA:
        print(f"retention source: {safe_relative(path)}")
        source_reports.append(
            observe_split_sentences(
                tokenizer,
                retained_ids,
                jsonl_sentences(path),
                safe_relative(path),
                max(1, args.batch_size // 8),
            )
        )

    observed_piece_count = len(retained_ids)
    special_short_report = add_special_and_short_pieces(
        retained_ids,
        piece_to_old_id,
        special_pieces(tokenizer_json, tokenizer),
    )
    retained_pieces = {
        piece
        for piece, old_id, _score in original_items
        if old_id in retained_ids
    }
    if model_type == "BPE":
        retained_pieces, support_added_count = bpe_merge_closure(
            retained_pieces,
            model.get("merges") or [],
        )
    else:
        support_added_count = 0

    pruned_json, new_to_old, old_to_new, dropped = rewrite_tokenizer_json(
        tokenizer_json,
        retained_pieces,
    )
    dropped_top20 = dropped[:20]
    new_vocab_size = len(new_to_old)

    print(
        "retention summary: "
        f"observed={observed_piece_count:,}; retained={new_vocab_size:,}; "
        f"dropped={old_tokenizer_vocab_size - new_vocab_size:,}; "
        f"bpe_support_added={support_added_count:,}"
    )
    (tokenizer_dir / "tokenizer.json").write_text(
        json.dumps(pruned_json, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    write_tokenizer_sidecars(
        tokenizer_source,
        tokenizer_dir,
        old_to_new,
        piece_to_old_id,
        new_vocab_size,
    )
    slice_checkpoint(
        checkpoint=checkpoint,
        checkpoint_path=args.checkpoint,
        tokenizer_dir=tokenizer_dir,
        output_checkpoint=output_checkpoint,
        new_to_old=new_to_old,
        old_tokenizer_vocab_size=old_tokenizer_vocab_size,
        dropped_count=old_tokenizer_vocab_size - new_vocab_size,
        support_added_count=support_added_count,
    )

    old_embedding_rows = int(
        checkpoint["model_state"]["encoder.embeddings.word_embeddings.weight"].shape[0]
    )
    retained_share = new_vocab_size / old_embedding_rows
    report = {
        "checkpoint": safe_relative(args.checkpoint),
        "output_checkpoint": safe_relative(output_checkpoint),
        "tokenizer_source": safe_relative(tokenizer_source),
        "tokenizer_dir": safe_relative(tokenizer_dir),
        "tokenizer_model_type": model_type,
        "old_tokenizer_vocab_size": old_tokenizer_vocab_size,
        "old_embedding_rows": old_embedding_rows,
        "retained_count": new_vocab_size,
        "dropped_count": old_tokenizer_vocab_size - new_vocab_size,
        "retained_embedding_param_share": retained_share,
        "observed_piece_count": observed_piece_count,
        "special_short_report": special_short_report,
        "bpe_support_pieces_added": support_added_count,
        "source_reports": source_reports,
        "top20_dropped_by_bpe_priority": dropped_top20,
        "elapsed": format_duration(time.perf_counter() - started),
    }
    write_json(report_path, report)

    print("20 highest-priority dropped pieces (BPE original id order):")
    for item in dropped_top20:
        print(f"  old_id={item['old_id']:>6} piece={item['piece']!r}")
    print(
        "embedding rows: "
        f"old={old_embedding_rows:,} new={new_vocab_size:,} "
        f"retained_share={retained_share:.2%}"
    )
    print(f"checkpoint written: {safe_relative(output_checkpoint)}")
    print(f"tokenizer written: {safe_relative(tokenizer_dir)}")
    print(f"report written: {safe_relative(report_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
