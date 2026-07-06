from __future__ import annotations

import argparse
import json
from pathlib import Path

from joint_lib import (
    DEFAULT_DATA_DIR,
    DEFAULT_GENERATED,
    DEFAULT_SOURCE_DATA_DIR,
    ProjectionStats,
    collect_word_keys,
    deterministic_split,
    labels_from_joint,
    load_dictionary,
    print_stats_block,
    read_jsonl,
    safe_relative,
    source_row_to_joint,
    summarize_joint_rows,
    write_json,
    write_jsonl,
    write_labels,
)


def project_split(
    name: str,
    rows: list[dict],
    entries: dict,
    mi_cache: dict,
) -> tuple[list[dict], ProjectionStats]:
    stats = ProjectionStats()
    projected = [
        source_row_to_joint(row, entries, stats, mi_cache, source_name=name)
        for row in rows
    ]
    return projected, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data-dir", type=Path, default=DEFAULT_SOURCE_DATA_DIR)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help="MATAS training sentence cap for smoke builds.",
    )
    parser.add_argument(
        "--max-dev-sentences",
        type=int,
        default=1000,
        help="MATAS dev sentence count split from the prepared MATAS training rows.",
    )
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args(argv)

    train_path = args.source_data_dir / "train.jsonl"
    alksnis_dev_path = args.source_data_dir / "dev.jsonl"
    alksnis_test_path = args.source_data_dir / "test.jsonl"
    for path in (train_path, alksnis_dev_path, alksnis_test_path, args.generated):
        if not path.exists():
            parser.error(f"missing required input: {path}")

    source_train = read_jsonl(train_path)
    train_source, matas_dev_source = deterministic_split(
        source_train,
        train_size=args.max_sentences,
        dev_size=args.max_dev_sentences,
        seed=args.seed,
    )
    alksnis_dev_source = read_jsonl(alksnis_dev_path)
    alksnis_test_source = read_jsonl(alksnis_test_path)

    all_sources = train_source + matas_dev_source + alksnis_dev_source + alksnis_test_source
    target_words = collect_word_keys(all_sources)
    print(
        f"loading generated.sqlite rows: {len(target_words):,} target word keys "
        f"from {safe_relative(args.generated)}"
    )
    entries = load_dictionary(args.generated, target_words)
    print(f"dictionary rows loaded: {len(entries):,}")

    mi_cache: dict = {}
    train_rows, train_projection = project_split("matas-train", train_source, entries, mi_cache)
    dev_rows, dev_projection = project_split("matas-dev", matas_dev_source, entries, mi_cache)
    alksnis_dev_rows, alksnis_dev_projection = project_split(
        "alksnis-dev",
        alksnis_dev_source,
        entries,
        mi_cache,
    )
    alksnis_test_rows, alksnis_test_projection = project_split(
        "alksnis-test",
        alksnis_test_source,
        entries,
        mi_cache,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "dev.jsonl", dev_rows)
    write_jsonl(args.out / "alksnis_dev.jsonl", alksnis_dev_rows)
    write_jsonl(args.out / "alksnis_test.jsonl", alksnis_test_rows)

    labels = write_labels(
        args.out / "labels.json",
        labels_from_joint(train_rows)
        + labels_from_joint(dev_rows)
        + labels_from_joint(alksnis_dev_rows)
        + labels_from_joint(alksnis_test_rows),
    )

    split_stats = {
        "train": summarize_joint_rows(train_rows),
        "dev": summarize_joint_rows(dev_rows),
        "alksnis_dev": summarize_joint_rows(alksnis_dev_rows),
        "alksnis_test": summarize_joint_rows(alksnis_test_rows),
    }
    projection_stats = {
        "train": train_projection.as_dict(),
        "dev": dev_projection.as_dict(),
        "alksnis_dev": alksnis_dev_projection.as_dict(),
        "alksnis_test": alksnis_test_projection.as_dict(),
    }
    total_tokens = sum(item["tokens"] for item in split_stats.values())
    total_letter_tokens = sum(item["letter_tokens"] for item in split_stats.values())
    total_supervised = sum(item["stress_supervised"] for item in split_stats.values())
    total_supervised_letter = sum(
        item["stress_supervised_letter"] for item in split_stats.values()
    )
    homograph_tokens = sum(item["homograph_tokens"] for item in projection_stats.values())
    homograph_resolved = sum(item["homograph_resolved"] for item in projection_stats.values())
    stats_payload = {
        "source_data_dir": str(args.source_data_dir),
        "generated": str(args.generated),
        "max_sentences": args.max_sentences,
        "label_set_size": len(labels),
        "dictionary_target_words": len(target_words),
        "dictionary_rows_loaded": len(entries),
        "splits": split_stats,
        "projection": projection_stats,
        "total": {
            "tokens": total_tokens,
            "letter_tokens": total_letter_tokens,
            "stress_supervised": total_supervised,
            "stress_supervised_letter": total_supervised_letter,
            "stress_supervision_share": total_supervised / total_tokens if total_tokens else 0.0,
            "stress_supervision_share_letter_tokens": (
                total_supervised_letter / total_letter_tokens if total_letter_tokens else 0.0
            ),
            "homograph_tokens": homograph_tokens,
            "homograph_resolved": homograph_resolved,
            "homograph_resolved_share": (
                homograph_resolved / homograph_tokens if homograph_tokens else 0.0
            ),
        },
    }
    write_json(args.out / "stats.json", stats_payload)

    print(f"wrote {safe_relative(args.out)}")
    for split_name, stats in split_stats.items():
        print_stats_block(split_name, stats)
    print(f"label-set size: {len(labels):,}")
    print(
        "stress-supervision share (all tokens): "
        f"{total_supervised:,}/{total_tokens:,} "
        f"({100 * total_supervised / (total_tokens or 1):.2f}%)"
    )
    print(
        "stress-supervision share (letter tokens): "
        f"{total_supervised_letter:,}/{total_letter_tokens:,} "
        f"({100 * total_supervised_letter / (total_letter_tokens or 1):.2f}%)"
    )
    print(
        "homograph-resolved share: "
        f"{homograph_resolved:,}/{homograph_tokens:,} "
        f"({100 * homograph_resolved / (homograph_tokens or 1):.2f}%)"
    )
    print("dataset stats JSON:")
    print(json.dumps(stats_payload["total"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
