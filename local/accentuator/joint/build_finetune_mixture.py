"""Build the literary fine-tune dataset with a MATAS rehearsal slice.

The intended full fine-tune invocation is human-run, not launched here:

.venv-train/Scripts/python.exe local/accentuator/joint/train_joint.py \
  --init-checkpoint local/accentuator/joint/checkpoints/joint_v1_polish.best.pt \
  --data-dir local/accentuator/joint/data-literary \
  --epochs 2 --lr-scale 0.1 --schedule constant \
  --checkpoint local/accentuator/joint/checkpoints/joint_v2_literary.pt
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Iterable

from joint_lib import (
    DEFAULT_DATA_DIR,
    JOINT_DIR,
    load_joint_checkpoint,
    load_labels,
    read_jsonl,
    safe_relative,
    summarize_joint_rows,
    write_json,
    write_jsonl,
)


DEFAULT_OUT_DIR = JOINT_DIR / "data-literary"
DEFAULT_INIT_CHECKPOINT = JOINT_DIR / "checkpoints" / "joint_v1_polish.best.pt"
DEFAULT_SEED = 20260705


def token_count(row: dict[str, Any]) -> int:
    return len(row.get("tokens", []))


def stress_supervised_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        for token in row.get("tokens", [])
        if isinstance(token, dict) and token.get("stress") is not None
    )


def split_literary(
    rows: list[dict[str, Any]],
    dev_share: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1 or dev_share <= 0:
        return shuffled, []
    dev_count = max(1, int(len(shuffled) * dev_share))
    dev_count = min(dev_count, len(shuffled) - 1)
    return shuffled[dev_count:], shuffled[:dev_count]


def sample_rehearsal(
    rows: list[dict[str, Any]],
    target_tokens: int,
    seed: int,
) -> list[dict[str, Any]]:
    if target_tokens <= 0:
        return []
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    selected: list[dict[str, Any]] = []
    selected_tokens = 0
    for row in shuffled:
        selected.append(row)
        selected_tokens += token_count(row)
        if selected_tokens >= target_tokens:
            break
    return selected


def assert_labels_match_checkpoint(
    labels_path: Path,
    checkpoint_path: Path,
    parser: argparse.ArgumentParser,
) -> list[str]:
    labels = load_labels(labels_path)
    checkpoint = load_joint_checkpoint(checkpoint_path, map_location="cpu")
    checkpoint_labels = [str(label) for label in checkpoint.get("labels", [])]
    if labels != checkpoint_labels:
        parser.error(
            "MATAS labels.json does not match init checkpoint labels "
            f"(labels.json={len(labels):,}, checkpoint={len(checkpoint_labels):,})"
        )
    return labels


def stats_payload(
    *,
    literary_rows: list[dict[str, Any]],
    literary_train: list[dict[str, Any]],
    literary_dev: list[dict[str, Any]],
    matas_rehearsal: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    labels: list[str],
    args: argparse.Namespace,
    target_matas_tokens: int,
) -> dict[str, Any]:
    literary_train_tokens = sum(token_count(row) for row in literary_train)
    matas_tokens = sum(token_count(row) for row in matas_rehearsal)
    actual_ratio = matas_tokens / literary_train_tokens if literary_train_tokens else 0.0
    return {
        "seed": args.seed,
        "literary": {
            "path": str(args.literary),
            "input": summarize_joint_rows(literary_rows),
            "train": summarize_joint_rows(literary_train),
            "dev": summarize_joint_rows(literary_dev),
            "stress_supervised_train_tokens": stress_supervised_count(literary_train),
        },
        "matas_rehearsal": {
            "source_train": str(args.matas_dir / "train.jsonl"),
            "target_tokens": target_matas_tokens,
            "actual_ratio": actual_ratio,
            **summarize_joint_rows(matas_rehearsal),
        },
        "mixture": {
            "train": summarize_joint_rows(train_rows),
            "dev": summarize_joint_rows(literary_dev),
        },
        "requested": {
            "rehearsal_ratio": args.rehearsal_ratio,
            "dev_share": args.dev_share,
            "max_literary_sentences": args.max_literary_sentences,
        },
        "labels": {
            "source": str(args.matas_dir / "labels.json"),
            "init_checkpoint": str(args.init_checkpoint),
            "label_set_size": len(labels),
            "identity_assertion": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--literary", type=Path, required=True)
    parser.add_argument("--matas-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rehearsal-ratio", type=float, default=0.25)
    parser.add_argument("--dev-share", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument(
        "--max-literary-sentences",
        type=int,
        default=None,
        help="Teacher-labeled literary sentence cap for smoke builds.",
    )
    parser.add_argument(
        "--allow-benchmark-smoke",
        action="store_true",
        help="Allow chrestomatija-named inputs for plumbing smoke tests only.",
    )
    args = parser.parse_args(argv)

    if args.rehearsal_ratio < 0:
        parser.error("--rehearsal-ratio must be non-negative")
    if not 0 <= args.dev_share < 1:
        parser.error("--dev-share must be in [0, 1)")
    if args.max_literary_sentences is not None and args.max_literary_sentences <= 0:
        parser.error("--max-literary-sentences must be positive")
    if "chrestomatija" in args.literary.name.casefold() and not args.allow_benchmark_smoke:
        parser.error(
            "refusing chrestomatija-named literary input without "
            "--allow-benchmark-smoke"
        )

    train_path = args.matas_dir / "train.jsonl"
    labels_path = args.matas_dir / "labels.json"
    for path in (args.literary, train_path, labels_path, args.init_checkpoint):
        if not path.exists():
            parser.error(f"missing required input: {path}")
    if args.out.resolve() == args.matas_dir.resolve():
        parser.error("--out must differ from --matas-dir")

    labels = assert_labels_match_checkpoint(labels_path, args.init_checkpoint, parser)
    literary_rows = read_jsonl(args.literary, limit=args.max_literary_sentences)
    if not literary_rows:
        parser.error(f"no literary rows loaded from {args.literary}")

    literary_train, literary_dev = split_literary(
        literary_rows,
        dev_share=args.dev_share,
        seed=args.seed,
    )
    literary_train_tokens = sum(token_count(row) for row in literary_train)
    target_matas_tokens = round(literary_train_tokens * args.rehearsal_ratio)

    matas_rows = read_jsonl(train_path)
    matas_rehearsal = sample_rehearsal(
        matas_rows,
        target_tokens=target_matas_tokens,
        seed=args.seed,
    )
    train_rows = list(literary_train) + list(matas_rehearsal)
    random.Random(args.seed + 1).shuffle(train_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "dev.jsonl", literary_dev)
    shutil.copy2(labels_path, args.out / "labels.json")

    payload = stats_payload(
        literary_rows=literary_rows,
        literary_train=literary_train,
        literary_dev=literary_dev,
        matas_rehearsal=matas_rehearsal,
        train_rows=train_rows,
        labels=labels,
        args=args,
        target_matas_tokens=target_matas_tokens,
    )
    write_json(args.out / "stats.json", payload)

    literary_stats = payload["literary"]["train"]
    matas_stats = payload["matas_rehearsal"]
    dev_stats = payload["literary"]["dev"]
    print(f"wrote {safe_relative(args.out)}")
    print(
        "labels assertion: "
        f"{len(labels):,} labels match {safe_relative(args.init_checkpoint)}"
    )
    print(
        "literary train tokens: "
        f"{literary_stats['tokens']:,} "
        f"(stress-supervised {payload['literary']['stress_supervised_train_tokens']:,})"
    )
    print(f"MATAS rehearsal tokens: {matas_stats['tokens']:,}")
    print(
        "actual rehearsal ratio: "
        f"{matas_stats['actual_ratio']:.4f} "
        f"(requested {args.rehearsal_ratio:g})"
    )
    print(
        "literary dev: "
        f"{dev_stats['sentences']:,} sentences, {dev_stats['tokens']:,} tokens"
    )
    print("dataset stats JSON:")
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
