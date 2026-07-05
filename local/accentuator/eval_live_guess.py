"""Evaluate live out-of-dictionary stress guessing against silver LRT truth."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from _common import DEFAULT_GENERATED, DATA_DIR, REPORTS_DIR, normalize_lt, strip_accents  # noqa: E402
from guess_uncovered import (  # noqa: E402
    AgreementBackend,
    BackendLoadError,
    LiepaBackend,
    NNBackend,
    build_backends,
    run_cascade,
)
from train_guesser import stress_of  # noqa: E402

DEFAULT_SILVER = DATA_DIR / "eval" / "lrt-silver.jsonl"
DEFAULT_GUESSES = DATA_DIR / "guesses.sqlite"
DEFAULT_REPORT = REPORTS_DIR / "live-guess-eval.md"
TIERS = ("dict", "precomputed-guess", "live-guess", "unanswered")
LIVE_BACKEND_SPEC = "nn&liepa+liepa"


@dataclass(frozen=True)
class SilverToken:
    word: str
    accented: str
    mi: str | None
    ambiguous: bool


@dataclass(frozen=True)
class DbEntry:
    variants: list[dict[str, Any]]
    default_form: str | None


@dataclass(frozen=True)
class EvalToken:
    word: str
    silver: str
    mi: str | None
    tier: str
    predicted: str | None


def norm_form(text: str | None) -> str:
    return normalize_lt(text or "").lower()


def word_key(text: str | None) -> str:
    return strip_accents(normalize_lt(text or "")).lower()


def load_silver(path: Path) -> list[SilverToken]:
    tokens: list[SilverToken] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            word = word_key(raw.get("word"))
            accented = norm_form(raw.get("accented"))
            if not word or not accented:
                raise ValueError(f"bad silver row at {path}:{line_number}")
            tokens.append(
                SilverToken(
                    word=word,
                    accented=accented,
                    mi=raw.get("mi"),
                    ambiguous=bool(raw.get("ambiguous")),
                )
            )
    return tokens


def batched(values: list[str], size: int = 800) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def load_db_entries(path: Path, words: set[str]) -> dict[str, DbEntry]:
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, DbEntry] = {}
    ordered = sorted(words)
    db = sqlite3.connect(path)
    try:
        for batch in batched(ordered):
            placeholders = ",".join("?" for _ in batch)
            rows = db.execute(
                f"SELECT word, variants, default_form FROM words WHERE word IN ({placeholders})",
                batch,
            ).fetchall()
            for word, variants_json, default_form in rows:
                try:
                    variants = json.loads(variants_json or "[]")
                except json.JSONDecodeError:
                    variants = []
                out[word] = DbEntry(variants=variants, default_form=default_form)
    finally:
        db.close()
    return out


def default_form(entry: DbEntry) -> str | None:
    if entry.default_form:
        return norm_form(entry.default_form)
    for variant in entry.variants:
        form = norm_form(variant.get("form"))
        if form:
            return form
    return None


def pick_dict_form(entry: DbEntry, silver_mi: str | None) -> str | None:
    if silver_mi:
        for variant in entry.variants:
            if silver_mi in (variant.get("mi") or []):
                form = norm_form(variant.get("form"))
                if form:
                    return form
    return default_form(entry)


def pick_guess_form(entry: DbEntry) -> str | None:
    return default_form(entry)


def build_live_backend(min_confidence: float) -> tuple[list[Any], str, str | None]:
    try:
        return build_backends(LIVE_BACKEND_SPEC, min_confidence), LIVE_BACKEND_SPEC, None
    except BackendLoadError as exc:
        warning = f"WARNING: {exc}; falling back to liepa-only"
        print(warning, file=sys.stderr)
        return build_backends("liepa", min_confidence), "liepa", warning


def live_predictions(words: set[str], min_confidence: float) -> tuple[dict[str, str | None], str, str | None]:
    if not words:
        _backends, backend_label, warning = build_live_backend(min_confidence)
        return {}, backend_label, warning
    backends, backend_label, warning = build_live_backend(min_confidence)
    ordered = sorted(words)
    predictions: dict[str, str | None] = {}
    for word, result in zip(ordered, run_cascade(backends, ordered)):
        if result is None:
            predictions[word] = None
            continue
        _backend_name, form, _confidence = result
        predictions[word] = norm_form(form)
    return predictions, backend_label, warning


def classify_tokens(
    silver: list[SilverToken],
    generated: dict[str, DbEntry],
    guesses: dict[str, DbEntry],
    live: dict[str, str | None],
) -> list[EvalToken]:
    rows: list[EvalToken] = []
    for token in silver:
        if token.word in generated:
            rows.append(
                EvalToken(
                    word=token.word,
                    silver=token.accented,
                    mi=token.mi,
                    tier="dict",
                    predicted=pick_dict_form(generated[token.word], token.mi),
                )
            )
        elif token.word in guesses:
            rows.append(
                EvalToken(
                    word=token.word,
                    silver=token.accented,
                    mi=token.mi,
                    tier="precomputed-guess",
                    predicted=pick_guess_form(guesses[token.word]),
                )
            )
        else:
            predicted = live.get(token.word)
            rows.append(
                EvalToken(
                    word=token.word,
                    silver=token.accented,
                    mi=token.mi,
                    tier="live-guess" if predicted else "unanswered",
                    predicted=predicted,
                )
            )
    return rows


def exact_match(predicted: str | None, silver: str) -> bool:
    return predicted is not None and norm_form(predicted) == norm_form(silver)


def position_match(predicted: str | None, silver: str) -> bool:
    if predicted is None:
        return False
    predicted_stress = stress_of(predicted)
    silver_stress = stress_of(silver)
    return predicted_stress is not None and silver_stress is not None and predicted_stress[0] == silver_stress[0]


def score_rows(rows: list[EvalToken]) -> dict[str, dict[str, int]]:
    stats = {
        tier: {
            "tokens": 0,
            "types": 0,
            "token_exact": 0,
            "token_position": 0,
            "type_exact": 0,
            "type_position": 0,
        }
        for tier in TIERS
    }
    seen_types = {tier: set() for tier in TIERS}
    for row in rows:
        tier_stats = stats[row.tier]
        tier_stats["tokens"] += 1
        if exact_match(row.predicted, row.silver):
            tier_stats["token_exact"] += 1
        if position_match(row.predicted, row.silver):
            tier_stats["token_position"] += 1
        if row.word not in seen_types[row.tier]:
            seen_types[row.tier].add(row.word)
            tier_stats["types"] += 1
            if exact_match(row.predicted, row.silver):
                tier_stats["type_exact"] += 1
            if position_match(row.predicted, row.silver):
                tier_stats["type_position"] += 1
    return stats


def pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


def count_cell(count: int, total: int) -> str:
    return f"{count:,} ({pct(count, total)})"


def metric_cell(success: int, total: int, tier: str) -> str:
    if tier == "unanswered" or total == 0:
        return "n/a"
    return f"{success:,}/{total:,} ({pct(success, total)})"


def live_disagreements(rows: list[EvalToken], limit: int = 20) -> list[str]:
    samples: list[str] = []
    seen: set[tuple[str, str | None, str]] = set()
    for row in rows:
        if row.tier != "live-guess" or exact_match(row.predicted, row.silver):
            continue
        key = (row.word, row.predicted, row.silver)
        if key in seen:
            continue
        seen.add(key)
        samples.append(f"{row.word}: live={row.predicted} silver={row.silver}")
        if len(samples) >= limit:
            break
    return samples


def format_report(
    silver: list[SilverToken],
    rows: list[EvalToken],
    stats: dict[str, dict[str, int]],
    backend_label: str,
    warning: str | None,
    generated_path: Path,
    guesses_path: Path,
) -> str:
    total_tokens = len(silver)
    total_types = len({token.word for token in silver})
    ambiguous_tokens = sum(1 for token in silver if token.ambiguous)
    dict_tokens = stats["dict"]["tokens"]
    dict_types = stats["dict"]["types"]
    lines = [
        "# Live Guess Evaluation",
        "",
        "## Corpus",
        f"- silver tokens: {total_tokens:,}",
        f"- silver word types: {total_types:,}",
        f"- ambiguous silver tokens: {ambiguous_tokens:,}",
        f"- dictionary OOV tokens: {total_tokens - dict_tokens:,} ({pct(total_tokens - dict_tokens, total_tokens)})",
        f"- dictionary OOV types: {total_types - dict_types:,} ({pct(total_types - dict_types, total_types)})",
        f"- live backend cascade: `{backend_label}`",
        f"- generated DB: `{generated_path}`",
        f"- guesses DB: `{guesses_path}`",
    ]
    if warning:
        lines.append(f"- backend warning: {warning}")
    lines.extend(
        [
            "",
            "## Tiers",
            "| tier | tokens | types | token exact | token position | type exact | type position |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for tier in TIERS:
        tier_stats = stats[tier]
        lines.append(
            "| "
            + " | ".join(
                [
                    tier,
                    count_cell(tier_stats["tokens"], total_tokens),
                    count_cell(tier_stats["types"], total_types),
                    metric_cell(tier_stats["token_exact"], tier_stats["tokens"], tier),
                    metric_cell(tier_stats["token_position"], tier_stats["tokens"], tier),
                    metric_cell(tier_stats["type_exact"], tier_stats["types"], tier),
                    metric_cell(tier_stats["type_position"], tier_stats["types"], tier),
                ]
            )
            + " |"
        )

    disagreements = live_disagreements(rows)
    lines.extend(["", "## Live-Guess Disagreements"])
    if disagreements:
        lines.extend(f"- {sample}" for sample in disagreements)
    else:
        lines.append("- none")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--guesses", type=Path, default=DEFAULT_GUESSES)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    args = parser.parse_args()

    if not args.silver.exists():
        parser.error(f"missing silver JSONL: {args.silver}")
    silver = load_silver(args.silver)
    words = {token.word for token in silver}
    generated = load_db_entries(args.generated, words)
    guesses = load_db_entries(args.guesses, words - generated.keys())
    live_words = words - generated.keys() - guesses.keys()
    live, backend_label, warning = live_predictions(live_words, args.min_confidence)
    rows = classify_tokens(silver, generated, guesses, live)
    stats = score_rows(rows)
    report = format_report(
        silver=silver,
        rows=rows,
        stats=stats,
        backend_label=backend_label,
        warning=warning,
        generated_path=args.generated,
        guesses_path=args.guesses,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report + "\n", encoding="utf-8", newline="\n")
    print(report)
    print(f"\nreport written: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
