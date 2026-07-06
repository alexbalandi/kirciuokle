from __future__ import annotations

import argparse
import json
import sys
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
import calibrate  # noqa: E402
from joint_lib import stress_target_for_form  # noqa: E402


DEFAULT_ACCENT_STRATA = ACCENTUATOR_DIR / "data" / "teacher" / "accent-strata.json"
DEFAULT_POS_STRATA = ACCENTUATOR_DIR / "data" / "teacher" / "pos-strata.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def output_path_for_layers(layers: Path) -> Path:
    name = layers.name
    if name.endswith(".layers.jsonl"):
        name = name[: -len(".layers.jsonl")] + ".labeled.jsonl"
    else:
        name = layers.stem + ".labeled.jsonl"
    return layers.parent / name


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def pattern_stats(strata: dict[str, Any], pattern: str) -> dict[str, Any] | None:
    patterns = strata.get("patterns") or {}
    item = patterns.get(pattern)
    return item if isinstance(item, dict) else None


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def label_layers(
    layers_path: Path,
    accent_strata_path: Path,
    pos_strata_path: Path,
    output_path: Path,
    min_accent_accuracy: float,
    min_pos_accuracy: float,
) -> dict[str, Any]:
    layer_rows = collect.load_layer_rows(layers_path)
    accent_strata = read_json(accent_strata_path)
    pos_strata = read_json(pos_strata_path)

    total = 0
    accent_accepted = 0
    pos_accepted = 0
    both_accepted = 0
    accent_accuracy_sum = 0.0
    pos_accuracy_sum = 0.0
    both_accuracy_sum = 0.0
    out_rows: list[dict[str, Any]] = []

    for row in layer_rows:
        out_tokens: list[dict[str, Any]] = []
        for token in row.get("tokens") or []:
            total += 1
            accent_pattern, accent_form = calibrate.accent_group_pattern(token)
            pos_pattern, joint_label = calibrate.pos_pattern(token)
            accent_stat = pattern_stats(accent_strata, accent_pattern)
            pos_stat = pattern_stats(pos_strata, pos_pattern)

            stress_target = None
            accepted_accent_accuracy = None
            if (
                accent_form
                and accent_stat
                and float(accent_stat.get("accuracy") or 0.0) >= min_accent_accuracy
            ):
                stress_target = stress_target_for_form(str(token.get("key") or ""), accent_form)
                if stress_target is not None:
                    accepted_accent_accuracy = float(accent_stat.get("accuracy") or 0.0)
                    accent_accepted += 1
                    accent_accuracy_sum += accepted_accent_accuracy

            accepted_label = None
            accepted_pos_accuracy = None
            if (
                joint_label
                and pos_stat
                and float(pos_stat.get("accuracy") or 0.0) >= min_pos_accuracy
            ):
                accepted_label = joint_label
                accepted_pos_accuracy = float(pos_stat.get("accuracy") or 0.0)
                pos_accepted += 1
                pos_accuracy_sum += accepted_pos_accuracy

            if accepted_accent_accuracy is not None and accepted_pos_accuracy is not None:
                both_accepted += 1
                both_accuracy_sum += accepted_accent_accuracy * accepted_pos_accuracy

            out_tokens.append(
                {
                    "word": str(token.get("word") or ""),
                    "pos_label": accepted_label,
                    "stress": stress_target,
                }
            )

        out_rows.append(
            {
                "id": row.get("id"),
                "source": f"teacher:{layers_path.stem}",
                "text": row.get("text") or "",
                "tokens": out_tokens,
            }
        )

    write_jsonl(output_path, out_rows)
    stats = {
        "tokens": total,
        "accent_accepted": accent_accepted,
        "pos_accepted": pos_accepted,
        "both_accepted": both_accepted,
        "accent_coverage": accent_accepted / total if total else 0.0,
        "pos_coverage": pos_accepted / total if total else 0.0,
        "both_coverage": both_accepted / total if total else 0.0,
        "accent_purity_estimate": (
            accent_accuracy_sum / accent_accepted if accent_accepted else 0.0
        ),
        "pos_purity_estimate": pos_accuracy_sum / pos_accepted if pos_accepted else 0.0,
        "both_purity_estimate": both_accuracy_sum / both_accepted if both_accepted else 0.0,
        "output": str(output_path),
    }
    return stats


def print_stats(stats: dict[str, Any]) -> None:
    total = int(stats["tokens"])
    print(f"labeled tokens: {total:,}")
    print(
        "accent coverage: "
        f"{int(stats['accent_accepted']):,}/{total:,} ({pct(float(stats['accent_coverage']))}); "
        f"purity estimate {pct(float(stats['accent_purity_estimate']))}"
    )
    print(
        "POS coverage: "
        f"{int(stats['pos_accepted']):,}/{total:,} ({pct(float(stats['pos_coverage']))}); "
        f"purity estimate {pct(float(stats['pos_purity_estimate']))}"
    )
    print(
        "both coverage: "
        f"{int(stats['both_accepted']):,}/{total:,} ({pct(float(stats['both_coverage']))}); "
        f"joint purity estimate {pct(float(stats['both_purity_estimate']))}"
    )
    print(f"labeled JSONL written: {safe_relative(Path(str(stats['output'])))}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=Path, required=True)
    parser.add_argument("--accent-strata", type=Path, default=DEFAULT_ACCENT_STRATA)
    parser.add_argument("--pos-strata", type=Path, default=DEFAULT_POS_STRATA)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-accent-accuracy", type=float, default=0.98)
    parser.add_argument("--min-pos-accuracy", type=float, default=0.95)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    layers = collect.resolve_input_path(args.layers)
    accent_strata = collect.resolve_input_path(args.accent_strata)
    pos_strata = collect.resolve_input_path(args.pos_strata)
    output = collect.resolve_output_path(args.out) if args.out else output_path_for_layers(layers)
    for path, label in (
        (layers, "layers JSONL"),
        (accent_strata, "accent strata JSON"),
        (pos_strata, "POS strata JSON"),
    ):
        if not path.exists():
            parser.error(f"missing {label}: {path}")

    stats = label_layers(
        layers,
        accent_strata,
        pos_strata,
        output,
        min_accent_accuracy=args.min_accent_accuracy,
        min_pos_accuracy=args.min_pos_accuracy,
    )
    print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
