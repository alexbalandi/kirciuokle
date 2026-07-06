from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ACCENTUATOR_DIR = SCRIPT_DIR.parent
JOINT_DIR = ACCENTUATOR_DIR / "joint"
LOCAL_DIR = ACCENTUATOR_DIR.parent
REPO_ROOT = LOCAL_DIR.parent
TAGGER_DIR = LOCAL_DIR / "tagger-hf"
APP_DIR = LOCAL_DIR / "app"
SCRIPTS_DIR = REPO_ROOT / "scripts"

for _path in (SCRIPT_DIR, ACCENTUATOR_DIR, JOINT_DIR, TAGGER_DIR, APP_DIR, SCRIPTS_DIR):
    sys.path.insert(0, str(_path))

from _common import count_stress_marks, safe_relative  # noqa: E402
import accent_text  # noqa: E402
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
DEFAULT_POS_UDPIPE_CACHE = ACCENTUATOR_DIR / "data" / "teacher" / "alksnis-udpipe.jsonl"

ACCENT_LAYER_ORDER = ("vdu", "joint", "liepa", "dict")
POS_LAYER_ORDER = ("joint", "tagger", "udpipe")
MIN_UDPIPE_INTERVAL = 1.05


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
    no_stress_form = None
    key = str(token.get("key") or collect.word_key(token.get("word")))
    for form, names in groups.items():
        if (
            "vdu" in names
            and "joint" in names
            and collect.word_key(form) == key
            and count_stress_marks(form) == 0
        ):
            no_stress_form = form
            break
    for _form, names in ordered:
        ordered_names = [name for name in ACCENT_LAYER_ORDER if name in names]
        suffix = ":no-stress" if _form == no_stress_form else ""
        if len(ordered_names) == 1 and len(ordered) == 1:
            pieces.append(f"{ordered_names[0]}-only{suffix}")
        else:
            pieces.append("+".join(ordered_names) + suffix)
    return " vs ".join(pieces), ordered[0][0]


def normalize_slots(slots: dict[str, Any] | None) -> dict[str, str] | None:
    if not isinstance(slots, dict) or not slots:
        return None
    return {str(key): str(value) for key, value in slots.items() if value is not None}


def udpipe_slots_for_token(token: dict[str, Any]) -> dict[str, str] | None:
    slots = normalize_slots(token.get("udpipe_slots"))
    if slots:
        return slots
    layers = token.get("layers") or {}
    udpipe_layer = layers.get("udpipe") if isinstance(layers, dict) else None
    if isinstance(udpipe_layer, dict):
        return normalize_slots(udpipe_layer.get("slots"))
    return None


def label_slots(label: str | None) -> dict[str, str] | None:
    if not label:
        return None
    return normalize_slots(token_tags_for_label(label))


def slots_key(slots: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(slots.items()))


def pos_slots_pattern(opinions: dict[str, dict[str, str] | None]) -> str:
    present = {name: slots for name, slots in opinions.items() if slots}
    if not present:
        return "none"

    groups: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for name in POS_LAYER_ORDER:
        slots = present.get(name)
        if slots:
            groups.setdefault(slots_key(slots), []).append(name)

    ordered = sorted(
        groups.values(),
        key=lambda names: (
            -len(names),
            min(POS_LAYER_ORDER.index(name) for name in names),
            "+".join(names),
        ),
    )
    pieces: list[str] = []
    for names in ordered:
        ordered_names = [name for name in POS_LAYER_ORDER if name in names]
        if len(ordered_names) == 1 and len(ordered) == 1:
            pieces.append(f"{ordered_names[0]}-only")
        else:
            pieces.append("=".join(ordered_names))
    missing = [name for name in POS_LAYER_ORDER if name not in present]
    pattern = " vs ".join(pieces)
    if missing:
        pattern += f"; missing {'+'.join(missing)}"
    return pattern


def pos_pattern(
    token: dict[str, Any],
    udpipe_slots: dict[str, str] | None = None,
) -> tuple[str, str | None]:
    layers = token.get("layers") or {}
    joint = collect.layer_pos(layers.get("joint"))
    tagger = collect.layer_pos(layers.get("tagger"))
    opinions = {
        "joint": label_slots(joint),
        "tagger": label_slots(tagger),
        "udpipe": normalize_slots(udpipe_slots) or udpipe_slots_for_token(token),
    }
    return pos_slots_pattern(opinions), joint


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


def post_lindat_udpipe(text: str, timeout: float) -> str:
    payload = urllib.parse.urlencode(
        {
            "tokenizer": "",
            "tagger": "",
            "model": accent_text.UDPIPE_MODEL,
            "data": text,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        accent_text.UDPIPE_URL,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "teacher-calibrate/1.0",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    result = decoded.get("result")
    if not isinstance(result, str):
        raise RuntimeError(f"UDPipe response did not include result: {decoded!r}")
    return result


def parse_udpipe_conllu(conllu: str) -> list[dict[str, str]]:
    tokens: list[dict[str, str]] = []
    for line in conllu.splitlines():
        if not line or line.startswith("#"):
            continue
        columns = line.split("\t")
        if len(columns) < 6 or not columns[0].isdigit():
            continue
        tokens.append(
            {
                "form": columns[1] or "",
                "lemma": columns[2] or "",
                "upos": columns[3] or "X",
                "feats": columns[5] or "_",
            }
        )
    return tokens


def load_udpipe_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id") or "")
            if not row_id:
                raise ValueError(f"UDPipe cache row missing id at {path}:{line_number}")
            rows[row_id] = row
    return rows


def append_udpipe_cache_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
        handle.write("\n")
        handle.flush()


def ensure_alksnis_udpipe_cache(
    gold_rows: list[dict[str, Any]],
    cache_path: Path,
    *,
    min_interval: float = MIN_UDPIPE_INTERVAL,
    request_timeout: float = 180.0,
) -> dict[str, dict[str, Any]]:
    cache_path = collect.resolve_output_path(cache_path)
    cache = load_udpipe_cache(cache_path)
    pending = [row for row in gold_rows if str(row.get("id") or "") not in cache]
    if not pending:
        print(
            f"ALKSNIS UDPipe cache complete: {len(cache):,}/{len(gold_rows):,} "
            f"rows at {safe_relative(cache_path)}"
        )
        return cache

    interval = max(1.0, float(min_interval))
    print(
        f"building ALKSNIS UDPipe cache: {len(cache):,}/{len(gold_rows):,} "
        f"already present; {len(pending):,} to fetch; throttle {interval:.2f}s"
    )
    last_start = 0.0
    fetched = 0
    for row in gold_rows:
        row_id = str(row.get("id") or "")
        if not row_id or row_id in cache:
            continue
        now = time.monotonic()
        if last_start:
            delay = interval - (now - last_start)
            if delay > 0:
                time.sleep(delay)
        last_start = time.monotonic()
        conllu = post_lindat_udpipe(str(row.get("text") or ""), timeout=request_timeout)
        payload = {
            "id": row_id,
            "text": str(row.get("text") or ""),
            "tokens": parse_udpipe_conllu(conllu),
        }
        append_udpipe_cache_row(cache_path, payload)
        cache[row_id] = payload
        fetched += 1
        if fetched <= 3 or fetched % 25 == 0 or len(cache) == len(gold_rows):
            print(f"UDPipe cached {len(cache):,}/{len(gold_rows):,}: {row_id}")
    return cache


def udpipe_slots_by_source_index(
    gold_row: dict[str, Any],
    cache_row: dict[str, Any] | None,
) -> tuple[dict[int, dict[str, str]], int, int]:
    if not cache_row:
        return {}, 0, 0
    source_tokens = [
        (index, str(token))
        for index, token in enumerate(gold_row.get("tokens") or [])
        if collect.nodict.has_letter(str(token))
    ]
    udpipe_tokens = [
        token
        for token in cache_row.get("tokens") or []
        if isinstance(token, dict) and collect.nodict.has_letter(str(token.get("form") or ""))
    ]
    aligned, skipped_source, skipped_udpipe = collect.align_by_key(
        [collect.word_key(word) for _index, word in source_tokens],
        udpipe_tokens,
        lambda item: collect.word_key(str(item.get("form") or "")),
    )
    by_source: dict[int, dict[str, str]] = {}
    for (source_index, word), udpipe_token in zip(source_tokens, aligned):
        if udpipe_token is None:
            continue
        slots = collect.udpipe_slots(
            {
                "upos": str(udpipe_token.get("upos") or "X"),
                "feats": str(udpipe_token.get("feats") or "_"),
            },
            word,
        )
        if slots:
            by_source[source_index] = slots
    return by_source, skipped_source, skipped_udpipe


def calibrate_pos(
    gold_path: Path,
    layers_path: Path,
    out_path: Path,
    udpipe_cache_path: Path,
    *,
    udpipe_throttle: float = MIN_UDPIPE_INTERVAL,
    udpipe_timeout: float = 180.0,
) -> dict[str, Any]:
    gold_by_id = load_pos_gold(gold_path)
    gold_rows = list(gold_by_id.values())
    udpipe_cache = ensure_alksnis_udpipe_cache(
        gold_rows,
        udpipe_cache_path,
        min_interval=udpipe_throttle,
        request_timeout=udpipe_timeout,
    )
    layer_rows = collect.load_layer_rows(layers_path)
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "slot_correct": 0, "full_label_correct": 0}
    )
    total = 0
    skipped_source_total = 0
    skipped_udpipe_total = 0

    for layer_row in layer_rows:
        gold_row = gold_by_id.get(str(layer_row.get("id")))
        if not gold_row:
            continue
        udpipe_by_source, skipped_source, skipped_udpipe = udpipe_slots_by_source_index(
            gold_row,
            udpipe_cache.get(str(layer_row.get("id"))),
        )
        skipped_source_total += skipped_source
        skipped_udpipe_total += skipped_udpipe
        labels = list(gold_row.get("labels") or [])
        for token in layer_row.get("tokens") or []:
            source_index = token.get("source_token_index")
            if source_index is None or int(source_index) >= len(labels):
                continue
            gold_label = str(labels[int(source_index)])
            gold_slots = token_tags_for_label(gold_label)
            pattern, predicted = pos_pattern(
                token,
                udpipe_slots=udpipe_by_source.get(int(source_index)),
            )
            stats[pattern]["count"] += 1
            if predicted:
                stats[pattern]["slot_correct"] += int(
                    token_tags_for_label(predicted) == gold_slots
                )
                stats[pattern]["full_label_correct"] += int(predicted == gold_label)
            total += 1
    if skipped_source_total or skipped_udpipe_total:
        print(
            f"UDPipe ALKSNIS alignment: skipped_source={skipped_source_total:,} "
            f"skipped_udpipe={skipped_udpipe_total:,}"
        )

    patterns = {}
    for pattern, row in stats.items():
        count = int(row["count"])
        slot_correct = int(row["slot_correct"])
        full_label_correct = int(row["full_label_correct"])
        patterns[pattern] = {
            "count": count,
            "correct": slot_correct,
            "slot_correct": slot_correct,
            "full_label_correct": full_label_correct,
            "accuracy": slot_correct / count if count else 0.0,
            "slot_accuracy": slot_correct / count if count else 0.0,
            "full_label_accuracy": full_label_correct / count if count else 0.0,
            "share": count / total if total else 0.0,
        }
    payload = {
        "kind": "pos",
        "gold": str(gold_path),
        "layers": str(layers_path),
        "udpipe_cache": str(collect.resolve_output_path(udpipe_cache_path)),
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
            -float(item[1]["slot_accuracy"]),
            -float(item[1]["full_label_accuracy"]),
            -int(item[1]["count"]),
            item[0],
        ),
    )
    print("\nPOS STRATA")
    print("| pattern | tokens | share | slot accuracy | full-label accuracy |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for pattern, row in rows:
        print(
            f"| {pattern} | {int(row['count']):,}/{total:,} "
            f"| {pct(float(row['share']))} | {pct(float(row['slot_accuracy']))} "
            f"| {pct(float(row['full_label_accuracy']))} |"
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
    parser.add_argument("--pos-udpipe-cache", type=Path, default=DEFAULT_POS_UDPIPE_CACHE)
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
    parser.add_argument("--udpipe-throttle", type=float, default=MIN_UDPIPE_INTERVAL)
    parser.add_argument("--udpipe-timeout", type=float, default=180.0)
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
    args.pos_udpipe_cache = collect.resolve_output_path(args.pos_udpipe_cache)
    args.accent_strata = collect.resolve_output_path(args.accent_strata)
    args.pos_strata = collect.resolve_output_path(args.pos_strata)

    ensure_layers(args)
    accent_payload = calibrate_accents(args.accent_gold, args.accent_layers, args.accent_strata)
    pos_payload = calibrate_pos(
        args.pos_gold,
        args.pos_layers,
        args.pos_strata,
        args.pos_udpipe_cache,
        udpipe_throttle=args.udpipe_throttle,
        udpipe_timeout=args.udpipe_timeout,
    )
    print_accent_table(accent_payload)
    print_pos_table(pos_payload)
    print(f"\naccent strata written: {safe_relative(args.accent_strata)}")
    print(f"POS strata written: {safe_relative(args.pos_strata)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
