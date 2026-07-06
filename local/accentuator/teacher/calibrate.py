from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ACCENTUATOR_DIR = SCRIPT_DIR.parent
JOINT_DIR = ACCENTUATOR_DIR / "joint"
LOCAL_DIR = ACCENTUATOR_DIR.parent
TAGGER_DIR = LOCAL_DIR / "tagger-hf"
APP_DIR = LOCAL_DIR / "app"

for _path in (SCRIPT_DIR, ACCENTUATOR_DIR, JOINT_DIR, TAGGER_DIR, APP_DIR):
    sys.path.insert(0, str(_path))

from _common import safe_relative  # noqa: E402
import collect_layers as collect  # noqa: E402
import eval_chrestomatija as chrest  # noqa: E402
from metrics import token_tags_for_label  # noqa: E402


DEFAULT_ACCENT_GOLD = ACCENTUATOR_DIR / "data" / "eval" / "chrestomatija-gold.jsonl"
DEFAULT_ACCENT_CORPUS = ACCENTUATOR_DIR / "data" / "eval" / "chrestomatija-plain.txt"
DEFAULT_ACCENT_SILVER = ACCENTUATOR_DIR / "data" / "eval" / "chrestomatija-vdu-silver.jsonl"
DEFAULT_ACCENT_LAYERS = ACCENTUATOR_DIR / "data" / "teacher" / "chrestomatija-plain.layers.jsonl"
DEFAULT_POS_GOLD = TAGGER_DIR / "data" / "gen2" / "test.jsonl"
DEFAULT_POS_LAYERS = ACCENTUATOR_DIR / "data" / "teacher" / "alksnis-test.layers.jsonl"
DEFAULT_ACCENT_STRATA = ACCENTUATOR_DIR / "data" / "teacher" / "accent-strata.json"
DEFAULT_POS_STRATA = ACCENTUATOR_DIR / "data" / "teacher" / "pos-strata.json"

ACCENT_LAYER_ORDER = ("vdu", "joint", "liepa", "dict")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def accent_group_pattern(token: dict[str, Any]) -> tuple[str, str | None]:
    layers = token.get("layers") or {}
    groups: dict[str, list[str]] = {}
    for name in ACCENT_LAYER_ORDER:
        form = collect.layer_form(layers.get(name))
        if form:
            groups.setdefault(form, []).append(name)
    if not groups:
        return "none", None
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            -len(item[1]),
            min(ACCENT_LAYER_ORDER.index(name) for name in item[1]),
            item[0],
        ),
    )
    pieces = []
    for _form, names in ordered:
        ordered_names = [name for name in ACCENT_LAYER_ORDER if name in names]
        if len(ordered_names) == 1 and len(ordered) == 1:
            pieces.append(f"{ordered_names[0]}-only")
        else:
            pieces.append("+".join(ordered_names))
    return " vs ".join(pieces), ordered[0][0]


def pos_pattern(token: dict[str, Any]) -> tuple[str, str | None]:
    layers = token.get("layers") or {}
    joint = collect.layer_pos(layers.get("joint"))
    tagger = collect.layer_pos(layers.get("tagger"))
    if joint and tagger:
        return ("joint=tagger" if joint == tagger else "joint!=tagger"), joint
    if joint:
        return "joint-only", joint
    if tagger:
        return "tagger-only", None
    return "none", None


def align_layer_tokens_to_gold(
    gold_tokens: list[chrest.GoldToken],
    layer_tokens: list[dict[str, Any]],
) -> list[tuple[chrest.GoldToken, dict[str, Any]]]:
    aligned, skipped_gold, skipped_layers = collect.align_by_key(
        [token.word for token in gold_tokens],
        layer_tokens,
        lambda item: str(item.get("key") or collect.word_key(item.get("word"))),
    )
    if skipped_gold or skipped_layers:
        print(
            f"accent row alignment skipped: gold={skipped_gold:,} layers={skipped_layers:,}"
        )
    return [
        (gold, layer)
        for gold, layer in zip(gold_tokens, aligned)
        if layer is not None
    ]


def calibrate_accents(gold_path: Path, layers_path: Path, out_path: Path) -> dict[str, Any]:
    gold_rows = chrest.load_gold(gold_path)
    layer_rows = collect.load_layer_rows(layers_path)
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"count": 0, "correct": 0})

    total = 0
    for gold_row, layer_row in zip(gold_rows, layer_rows):
        pairs = align_layer_tokens_to_gold(gold_row.tokens, list(layer_row.get("tokens") or []))
        for gold_token, layer_token in pairs:
            pattern, form = accent_group_pattern(layer_token)
            stats[pattern]["count"] += 1
            stats[pattern]["correct"] += int(
                form is not None and chrest.token_exact(gold_token, form)
            )
            total += 1

    patterns = {}
    for pattern, row in stats.items():
        count = int(row["count"])
        correct = int(row["correct"])
        patterns[pattern] = {
            "count": count,
            "correct": correct,
            "accuracy": correct / count if count else 0.0,
            "share": count / total if total else 0.0,
        }
    payload = {
        "kind": "accent",
        "gold": str(gold_path),
        "layers": str(layers_path),
        "total_tokens": total,
        "patterns": patterns,
    }
    write_json(out_path, payload)
    return payload


def load_pos_gold(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in collect.read_jsonl(path)}


def calibrate_pos(gold_path: Path, layers_path: Path, out_path: Path) -> dict[str, Any]:
    gold_by_id = load_pos_gold(gold_path)
    layer_rows = collect.load_layer_rows(layers_path)
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "correct": 0, "slot_correct": 0}
    )
    total = 0

    for layer_row in layer_rows:
        gold_row = gold_by_id.get(str(layer_row.get("id")))
        if not gold_row:
            continue
        labels = list(gold_row.get("labels") or [])
        for token in layer_row.get("tokens") or []:
            source_index = token.get("source_token_index")
            if source_index is None or int(source_index) >= len(labels):
                continue
            gold_label = str(labels[int(source_index)])
            pattern, predicted = pos_pattern(token)
            stats[pattern]["count"] += 1
            if predicted:
                stats[pattern]["correct"] += int(predicted == gold_label)
                stats[pattern]["slot_correct"] += int(
                    token_tags_for_label(predicted) == token_tags_for_label(gold_label)
                )
            total += 1

    patterns = {}
    for pattern, row in stats.items():
        count = int(row["count"])
        correct = int(row["correct"])
        slot_correct = int(row["slot_correct"])
        patterns[pattern] = {
            "count": count,
            "correct": correct,
            "slot_correct": slot_correct,
            "accuracy": correct / count if count else 0.0,
            "slot_accuracy": slot_correct / count if count else 0.0,
            "share": count / total if total else 0.0,
        }
    payload = {
        "kind": "pos",
        "gold": str(gold_path),
        "layers": str(layers_path),
        "total_tokens": total,
        "patterns": patterns,
    }
    write_json(out_path, payload)
    return payload


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def print_accent_table(payload: dict[str, Any]) -> None:
    total = int(payload.get("total_tokens") or 0)
    rows = sorted(
        payload["patterns"].items(),
        key=lambda item: (-float(item[1]["accuracy"]), -int(item[1]["count"]), item[0]),
    )
    print("\nACCENT STRATA")
    print("| pattern | tokens | share | accuracy |")
    print("| --- | ---: | ---: | ---: |")
    for pattern, row in rows:
        print(
            f"| {pattern} | {int(row['count']):,}/{total:,} "
            f"| {pct(float(row['share']))} | {pct(float(row['accuracy']))} |"
        )


def print_pos_table(payload: dict[str, Any]) -> None:
    total = int(payload.get("total_tokens") or 0)
    rows = sorted(
        payload["patterns"].items(),
        key=lambda item: (
            -float(item[1]["accuracy"]),
            -float(item[1]["slot_accuracy"]),
            -int(item[1]["count"]),
            item[0],
        ),
    )
    print("\nPOS STRATA")
    print("| pattern | tokens | share | full-label accuracy | slot accuracy |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for pattern, row in rows:
        print(
            f"| {pattern} | {int(row['count']):,}/{total:,} "
            f"| {pct(float(row['share']))} | {pct(float(row['accuracy']))} "
            f"| {pct(float(row['slot_accuracy']))} |"
        )


def ensure_layers(args: argparse.Namespace) -> None:
    if not args.accent_layers.exists():
        if not args.collect_missing:
            raise FileNotFoundError(args.accent_layers)
        print("accent layers missing; collecting chrestomatija layers")
        collect.collect_rows(
            collect.rows_from_corpus(args.accent_corpus),
            args.accent_layers,
            vdu_silver=args.accent_silver,
            checkpoint=args.checkpoint,
            generated=args.generated,
            batch_size=args.batch_size,
            force_cpu=args.cpu,
            cuda_memory_threshold_mib=args.cuda_memory_threshold_mib,
            tagger_url_override=args.tagger_url,
            tagger_timeout=args.tagger_timeout,
            request_timeout=args.request_timeout,
        )

    if not args.pos_layers.exists():
        if not args.collect_missing:
            raise FileNotFoundError(args.pos_layers)
        print("POS layers missing; collecting ALKSNIS test layers")
        collect.collect_rows(
            collect.rows_from_pos_jsonl(args.pos_gold),
            args.pos_layers,
            vdu_silver=None,
            checkpoint=args.checkpoint,
            generated=args.generated,
            batch_size=args.batch_size,
            force_cpu=args.cpu,
            cuda_memory_threshold_mib=args.cuda_memory_threshold_mib,
            tagger_url_override=args.tagger_url,
            tagger_timeout=args.tagger_timeout,
            request_timeout=args.request_timeout,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accent-gold", type=Path, default=DEFAULT_ACCENT_GOLD)
    parser.add_argument("--accent-corpus", type=Path, default=DEFAULT_ACCENT_CORPUS)
    parser.add_argument("--accent-silver", type=Path, default=DEFAULT_ACCENT_SILVER)
    parser.add_argument("--accent-layers", type=Path, default=DEFAULT_ACCENT_LAYERS)
    parser.add_argument("--pos-gold", type=Path, default=DEFAULT_POS_GOLD)
    parser.add_argument("--pos-layers", type=Path, default=DEFAULT_POS_LAYERS)
    parser.add_argument("--accent-strata", type=Path, default=DEFAULT_ACCENT_STRATA)
    parser.add_argument("--pos-strata", type=Path, default=DEFAULT_POS_STRATA)
    parser.add_argument("--checkpoint", type=Path, default=collect.DEFAULT_CHECKPOINT)
    parser.add_argument("--generated", type=Path, default=collect.DEFAULT_GENERATED)
    parser.add_argument("--batch-size", type=int, default=collect.DEFAULT_BATCH_SIZE)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cuda-memory-threshold-mib", type=int, default=6144)
    parser.add_argument("--tagger-url")
    parser.add_argument("--tagger-timeout", type=float, default=180.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--collect-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    for name in (
        "accent_gold",
        "accent_corpus",
        "accent_silver",
        "pos_gold",
        "checkpoint",
        "generated",
    ):
        value = collect.resolve_input_path(getattr(args, name))
        setattr(args, name, value)
        if not value.exists():
            parser.error(f"missing {name.replace('_', ' ')}: {value}")
    args.accent_layers = collect.resolve_output_path(args.accent_layers)
    args.pos_layers = collect.resolve_output_path(args.pos_layers)
    args.accent_strata = collect.resolve_output_path(args.accent_strata)
    args.pos_strata = collect.resolve_output_path(args.pos_strata)

    ensure_layers(args)
    accent_payload = calibrate_accents(args.accent_gold, args.accent_layers, args.accent_strata)
    pos_payload = calibrate_pos(args.pos_gold, args.pos_layers, args.pos_strata)
    print_accent_table(accent_payload)
    print_pos_table(pos_payload)
    print(f"\naccent strata written: {safe_relative(args.accent_strata)}")
    print(f"POS strata written: {safe_relative(args.pos_strata)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
