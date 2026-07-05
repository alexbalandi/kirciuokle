# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Compare the generated open dictionary against the local VDU cache."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    from ._common import (
        DEFAULT_GENERATED,
        DEFAULT_PARITY_REPORT,
        DEFAULT_VDU_SQLITE,
        DEFAULT_VETOES,
        normalize_lt,
        safe_relative,
    )
except ImportError:  # pragma: no cover
    from _common import (
        DEFAULT_GENERATED,
        DEFAULT_PARITY_REPORT,
        DEFAULT_VDU_SQLITE,
        DEFAULT_VETOES,
        normalize_lt,
        safe_relative,
    )


def variant_set(raw: str) -> set[str]:
    try:
        variants = json.loads(raw)
    except json.JSONDecodeError:
        return set()
    forms: set[str] = set()
    for variant in variants or []:
        form = normalize_lt(str((variant or {}).get("form") or "")).lower()
        if form:
            forms.add(form)
    return forms


def load_generated(path: Path) -> dict[str, dict[str, Any]]:
    db = sqlite3.connect(path)
    try:
        rows = db.execute("SELECT word, variants, default_form, provenance FROM words").fetchall()
    finally:
        db.close()
    return {
        word: {
            "variants": variant_set(variants),
            "default": normalize_lt(default_form or "").lower() if default_form else None,
            "provenance": provenance,
        }
        for word, variants, default_form, provenance in rows
    }


def load_vdu(path: Path) -> list[tuple[str, set[str], str | None, str]]:
    db = sqlite3.connect(path)
    try:
        rows = db.execute(
            """
            SELECT word, variants, default_form
            FROM words
            WHERE accent_type IS NOT NULL
              AND accent_type != 'NONE'
              AND variants != '[]'
            ORDER BY word
            """
        ).fetchall()
    finally:
        db.close()
    return [
        (
            word,
            variant_set(variants),
            normalize_lt(default_form or "").lower() if default_form else None,
            variants,
        )
        for word, variants, default_form in rows
    ]


def bucket_entry(vdu_forms: set[str], vdu_default: str | None, gen_forms: set[str], gen_default: str | None) -> str:
    if vdu_forms == gen_forms and vdu_default == gen_default:
        return "EXACT"
    if vdu_default and gen_default and vdu_default == gen_default:
        return "DEFAULT-MATCH"
    if vdu_forms & gen_forms:
        return "OVERLAP"
    return "DISJOINT"


def write_report(
    *,
    vdu_path: Path = DEFAULT_VDU_SQLITE,
    generated_path: Path = DEFAULT_GENERATED,
    output: Path = DEFAULT_PARITY_REPORT,
) -> dict[str, int]:
    generated = load_generated(generated_path)
    vdu_rows = load_vdu(vdu_path)
    norm_deltas: dict[str, str] = {}
    if DEFAULT_VETOES.exists():
        norm_deltas = dict(
            json.loads(DEFAULT_VETOES.read_text(encoding="utf-8")).get("norm_deltas") or {}
        )
    counts: Counter[str] = Counter()
    disjoint_samples: list[dict[str, Any]] = []

    for word, vdu_forms, vdu_default, _raw_variants in vdu_rows:
        gen = generated.get(word)
        if not gen:
            counts["UNCOVERED"] += 1
            continue
        bucket = bucket_entry(vdu_forms, vdu_default, gen["variants"], gen["default"])
        if bucket == "DISJOINT" and word in norm_deltas:
            # documented divergence where VLKK (the normative authority)
            # backs our form against the VDU cache
            bucket = "NORM-DELTA"
        if bucket == "DISJOINT" and "vlkk-rec" in str(gen["provenance"]) and ":headword" in str(gen["provenance"]):
            # the word carries a form quoted verbatim from VLKK's
            # recommended-stress list — the K-nn id in the provenance is the
            # citation; table-derived cells (no ":headword") stay DISJOINT
            bucket = "NORM-DELTA"
        counts[bucket] += 1
        if bucket == "DISJOINT" and len(disjoint_samples) < 120:
            disjoint_samples.append(
                {
                    "word": word,
                    "vdu": sorted(vdu_forms),
                    "generated": sorted(gen["variants"]),
                    "provenance": gen["provenance"],
                }
            )

    total = len(vdu_rows)
    covered = total - counts["UNCOVERED"]
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# VDU Parity Report",
        "",
        f"- VDU positives: {total:,}",
        f"- Generated covered: {covered:,} ({covered / total:.1%})" if total else "- Generated covered: 0",
        "",
        "| bucket | count | percent |",
        "|---|---:|---:|",
    ]
    for bucket in ("EXACT", "DEFAULT-MATCH", "OVERLAP", "NORM-DELTA", "DISJOINT", "UNCOVERED"):
        count = counts[bucket]
        pct = count / total if total else 0
        lines.append(f"| {bucket} | {count:,} | {pct:.1%} |")

    lines.extend(["", "## DISJOINT Samples", ""])
    if not disjoint_samples:
        lines.append("_No DISJOINT samples._")
    else:
        for sample in disjoint_samples:
            lines.append(f"### {sample['word']}")
            lines.append(f"- VDU: {', '.join(sample['vdu'])}")
            lines.append(f"- Generated: {', '.join(sample['generated'])}")
            lines.append(f"- Provenance: `{sample['provenance']}`")
            lines.append("")

    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return {
        "total": total,
        "covered": covered,
        **{bucket.lower().replace("-", "_"): counts[bucket] for bucket in counts},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write VDU parity report for generated open dictionary.")
    parser.add_argument("--vdu", type=Path, default=DEFAULT_VDU_SQLITE, help="Input local/data/words.sqlite path.")
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED, help="Input generated.sqlite path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_PARITY_REPORT, help="Output markdown report path.")
    parser.add_argument("--quiet", action="store_true", help="Suppress summary output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = write_report(vdu_path=args.vdu, generated_path=args.generated, output=args.output)
    if not args.quiet:
        for key, value in summary.items():
            print(f"{key}: {value:,}")
        print(f"wrote {safe_relative(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
