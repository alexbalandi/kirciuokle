"""Evaluate accentuation systems on the Chrestomatija gold extraction."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
JOINT_DIR = SCRIPT_DIR / "joint"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(JOINT_DIR))

from _common import (  # noqa: E402
    DEFAULT_GENERATED,
    REPORTS_DIR,
    count_stress_marks,
    normalize_lt,
    safe_relative,
    strip_accents,
)
import eval_nodict_pipeline as nodict  # noqa: E402


DEFAULT_GOLD = SCRIPT_DIR / "data" / "eval" / "chrestomatija-gold.jsonl"
DEFAULT_CHECKPOINT = JOINT_DIR / "checkpoints" / "joint_v1_polish.best.pt"
DEFAULT_REPORT = REPORTS_DIR / "chrestomatija-eval.md"
WORD_RE = re.compile(r"(?:[^\W\d_][\u0300-\u036f]*)+", re.UNICODE)
MISSING = object()


@dataclass(frozen=True)
class GoldToken:
    word: str
    accented: str
    raw: str


@dataclass(frozen=True)
class GoldSentence:
    text: str
    page: int
    tokens: list[GoldToken]


@dataclass
class SystemResult:
    system: str
    status: str
    predictions: list[list[str | None | object]]
    elapsed_seconds: float
    error: str | None = None
    skipped_gold_tokens: int = 0
    skipped_model_tokens: int = 0


@dataclass(frozen=True)
class ChrestomatijaMetrics:
    system: str
    status: str
    total_tokens: int
    answered_tokens: int
    token_exact: int
    token_position: int
    total_sentences: int
    sentence_exact: int
    elapsed_seconds: float
    error: str | None = None
    skipped_gold_tokens: int = 0
    skipped_model_tokens: int = 0


def norm_form(text: str | None) -> str:
    return normalize_lt(text or "").casefold()


def word_key(text: str | None) -> str:
    return strip_accents(normalize_lt(text or "")).casefold()


def tokenized_words(text: str) -> list[str]:
    return [token.group(0) for token in WORD_RE.finditer(text) if nodict.has_letter(token.group(0))]


def gold_tokens_from_sentence(text: str) -> list[GoldToken]:
    tokens: list[GoldToken] = []
    for raw in tokenized_words(text):
        word = word_key(raw)
        accented = norm_form(raw)
        if word and accented:
            tokens.append(GoldToken(word=word, accented=accented, raw=raw))
    return tokens


def load_gold(path: Path, limit: int | None = None) -> list[GoldSentence]:
    rows: list[GoldSentence] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            text = normalize_lt(str(raw.get("text") or ""))
            page = int(raw.get("page") or 0)
            tokens = gold_tokens_from_sentence(text)
            if not text or page <= 0 or not tokens:
                raise ValueError(f"bad gold row at {path}:{line_number}")
            rows.append(GoldSentence(text=text, page=page, tokens=tokens))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def flatten_tokens(sentences: list[GoldSentence]) -> list[GoldToken]:
    return [token for sentence in sentences for token in sentence.tokens]


def gold_plain_sentence(sentence: GoldSentence) -> str:
    return strip_accents(sentence.text)


def nvidia_memory_used_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    values = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(int(line))
        except ValueError:
            return None
    return max(values) if values else None


def choose_device(force_cpu: bool, threshold_mib: int) -> Any:
    import torch

    if force_cpu:
        print("device: cpu (--cpu)")
        return torch.device("cpu")
    used = nvidia_memory_used_mib()
    if used is None:
        print("nvidia-smi: unavailable or unparsable; CUDA will be used only if torch allows it")
    else:
        print(f"nvidia-smi memory.used: {used:,} MiB")
        if used > threshold_mib:
            print(f"device: cpu (GPU memory in use is above {threshold_mib:,} MiB)")
            return torch.device("cpu")
    cuda_allowed = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() not in {"", "-1"}
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        cuda_allowed = True
    device = torch.device("cuda" if cuda_allowed and torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    return device


def run_joint(
    sentences: list[GoldSentence],
    checkpoint: Path,
    batch_size: int,
    force_cpu: bool,
    cuda_memory_threshold_mib: int,
) -> SystemResult:
    started = time.perf_counter()
    try:
        import torch
        from torch.utils.data import DataLoader

        import eval_joint as joint_eval

        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        device = choose_device(force_cpu, cuda_memory_threshold_mib)
        model, tokenizer, checkpoint_payload = joint_eval.instantiate_from_checkpoint(
            checkpoint,
            device=device,
        )
        char_vocab = checkpoint_payload["char_vocab"]
        plain_rows = joint_eval.rows_from_plain_sentences(
            gold_plain_sentence(sentence) for sentence in sentences
        )
        collator = joint_eval.JointCollator(tokenizer, model.labels, char_vocab)
        loader = DataLoader(
            joint_eval.JointDataset(plain_rows),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        predictions = joint_eval.predict_batches(model, loader, device)
        torch.cuda.empty_cache() if device.type == "cuda" else None

        by_sentence: list[list[str | None | object]] = []
        skipped_gold_total = 0
        skipped_model_total = 0
        for sentence, predicted_row in zip(sentences, predictions):
            pred_tokens = []
            for token in predicted_row.get("tokens", []):
                simple = joint_eval.SimpleToken(str(token.get("word") or ""))
                simple.stress = token.get("stress")
                pred_tokens.append(simple)
            aligned, skipped_gold, skipped_model = nodict.align_tagger_tokens(
                sentence.tokens,
                pred_tokens,
            )
            skipped_gold_total += skipped_gold
            skipped_model_total += skipped_model
            token_predictions: list[str | None | object] = [MISSING] * len(sentence.tokens)
            index_by_id = {id(token): index for index, token in enumerate(sentence.tokens)}
            for gold_token, model_token in aligned:
                token_predictions[index_by_id[id(gold_token)]] = getattr(model_token, "stress", None)
            by_sentence.append(token_predictions)
        return SystemResult(
            system="joint",
            status="ok",
            predictions=by_sentence,
            elapsed_seconds=time.perf_counter() - started,
            skipped_gold_tokens=skipped_gold_total,
            skipped_model_tokens=skipped_model_total,
        )
    except Exception as exc:  # graceful optional-system skip
        return skipped_result("joint", sentences, started, exc)


def batched(values: list[str], size: int = 800) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def default_form_from_entry(variants_json: str | None, default_form: str | None) -> str | None:
    entry = nodict.DbEntry(variants=[], default_form=default_form)
    if variants_json:
        try:
            variants = json.loads(variants_json)
        except json.JSONDecodeError:
            variants = []
        entry = nodict.DbEntry(
            variants=variants if isinstance(variants, list) else [],
            default_form=default_form,
        )
    return nodict.default_form(entry)


def run_dict_default(sentences: list[GoldSentence], generated: Path) -> SystemResult:
    started = time.perf_counter()
    try:
        if not generated.exists():
            raise FileNotFoundError(generated)
        target_words = sorted({token.word for token in flatten_tokens(sentences)})
        defaults: dict[str, str | None] = {word: None for word in target_words}
        db = sqlite3.connect(generated)
        try:
            for batch in batched(target_words):
                placeholders = ",".join("?" for _ in batch)
                query = (
                    "SELECT word, variants, default_form FROM words "
                    f"WHERE word IN ({placeholders})"
                )
                for word, variants_json, default_form in db.execute(query, batch):
                    defaults[str(word)] = default_form_from_entry(variants_json, default_form)
        finally:
            db.close()
        predictions = [
            [defaults.get(token.word) for token in sentence.tokens]
            for sentence in sentences
        ]
        return SystemResult(
            system="dict-default",
            status="ok",
            predictions=predictions,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return skipped_result("dict-default", sentences, started, exc)


def run_liepa(sentences: list[GoldSentence]) -> SystemResult:
    started = time.perf_counter()
    try:
        words = [token.word for token in flatten_tokens(sentences)]
        flat_predictions = nodict.liepa_predictions(words)
        predictions: list[list[str | None | object]] = []
        offset = 0
        for sentence in sentences:
            end = offset + len(sentence.tokens)
            predictions.append(flat_predictions[offset:end])
            offset = end
        return SystemResult(
            system="liepa",
            status="ok",
            predictions=predictions,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as exc:
        return skipped_result("liepa", sentences, started, exc)


def skipped_result(
    system: str,
    sentences: list[GoldSentence],
    started: float,
    exc: Exception,
) -> SystemResult:
    print(f"{system}: skipped ({type(exc).__name__}: {exc})")
    return SystemResult(
        system=system,
        status="skipped",
        predictions=[[MISSING] * len(sentence.tokens) for sentence in sentences],
        elapsed_seconds=time.perf_counter() - started,
        error=f"{type(exc).__name__}: {exc}",
    )


def prediction_answer_form(word: str, predicted: str | None | object) -> str | None | object:
    if predicted is MISSING:
        return MISSING
    return nodict.prediction_answer_form(word, predicted if isinstance(predicted, str) else None)


def gold_has_stress(token: GoldToken) -> bool:
    return count_stress_marks(token.accented) > 0


def token_exact(token: GoldToken, predicted: str | None | object) -> bool:
    if predicted is MISSING:
        return False
    if not gold_has_stress(token):
        return nodict.prediction_unmarked_or_abstained(
            token.word,
            predicted if isinstance(predicted, str) else None,
        )
    answer = prediction_answer_form(token.word, predicted)
    return isinstance(answer, str) and nodict.exact_match(answer, token.accented)


def token_position(token: GoldToken, predicted: str | None | object) -> bool:
    if predicted is MISSING:
        return False
    answer = prediction_answer_form(token.word, predicted)
    return isinstance(answer, str) and nodict.position_match(answer, token.accented)


def score_result(sentences: list[GoldSentence], result: SystemResult) -> ChrestomatijaMetrics:
    total_tokens = answered = exact = position = 0
    sentence_exact = 0
    for sentence, predictions in zip(sentences, result.predictions):
        if len(predictions) != len(sentence.tokens):
            predictions = [MISSING] * len(sentence.tokens)
        sentence_ok = True
        for token, predicted in zip(sentence.tokens, predictions):
            total_tokens += 1
            if predicted is not MISSING and predicted is not None:
                answered += 1
            is_exact = token_exact(token, predicted)
            exact += int(is_exact)
            position += int(token_position(token, predicted))
            sentence_ok = sentence_ok and is_exact
        sentence_exact += int(sentence_ok)
    return ChrestomatijaMetrics(
        system=result.system,
        status=result.status,
        total_tokens=total_tokens,
        answered_tokens=answered,
        token_exact=exact,
        token_position=position,
        total_sentences=len(sentences),
        sentence_exact=sentence_exact,
        elapsed_seconds=result.elapsed_seconds,
        error=result.error,
        skipped_gold_tokens=result.skipped_gold_tokens,
        skipped_model_tokens=result.skipped_model_tokens,
    )


def pct(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{100 * numerator / denominator:.1f}%"


def count_pct(numerator: int, denominator: int) -> str:
    return f"{numerator:,}/{denominator:,} ({pct(numerator, denominator)})"


def sequence_cell(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "n/a"
    return f"{numerator:,}/{denominator:,} ({numerator / denominator:.3f}; {pct(numerator, denominator)})"


def metric_rows(metrics: list[ChrestomatijaMetrics]) -> list[str]:
    lines = [
        "| system | status | answered | token exact | token position | sentence sequence | time |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in metrics:
        if item.status != "ok":
            token_total = item.total_tokens
            lines.append(
                f"| {item.system} | skipped: {escape_cell(item.error or 'unavailable')} "
                f"| 0/{token_total:,} (0.0%) | n/a | n/a | n/a | {item.elapsed_seconds:.1f}s |"
            )
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    item.system,
                    item.status,
                    count_pct(item.answered_tokens, item.total_tokens),
                    count_pct(item.token_exact, item.total_tokens),
                    count_pct(item.token_position, item.answered_tokens),
                    sequence_cell(item.sentence_exact, item.total_sentences),
                    f"{item.elapsed_seconds:.1f}s",
                ]
            )
            + " |"
        )
    return lines


def extraction_stats(sentences: list[GoldSentence]) -> dict[str, int]:
    tokens = flatten_tokens(sentences)
    pages = {sentence.page for sentence in sentences}
    return {
        "sentences": len(sentences),
        "tokens": len(tokens),
        "types": len({token.word for token in tokens}),
        "stress_marks": sum(count_stress_marks(sentence.text) for sentence in sentences),
        "pages": len(pages),
        "first_page": min(pages) if pages else 0,
        "last_page": max(pages) if pages else 0,
    }


def escape_cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def excerpt(text: str, max_len: int = 120) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 1].rstrip() + "…"


def sample_disagreements(
    sentences: list[GoldSentence],
    results: list[SystemResult],
    limit: int = 15,
) -> list[tuple[str, int, str, str, str, str]]:
    samples: list[tuple[str, int, str, str, str, str]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for result in results:
        if result.status != "ok":
            continue
        for sentence, predictions in zip(sentences, result.predictions):
            if len(predictions) != len(sentence.tokens):
                continue
            for token, predicted in zip(sentence.tokens, predictions):
                if token_exact(token, predicted):
                    continue
                answer = prediction_answer_form(token.word, predicted)
                if answer is MISSING:
                    pred_text = "(missing)"
                elif answer is None:
                    pred_text = "(abstain)"
                else:
                    pred_text = str(answer)
                key = (result.system, token.word, token.accented, pred_text)
                if key in seen:
                    continue
                seen.add(key)
                samples.append(
                    (
                        result.system,
                        sentence.page,
                        token.word,
                        token.accented,
                        pred_text,
                        excerpt(sentence.text),
                    )
                )
                if len(samples) >= limit:
                    return samples
    return samples


def disagreement_rows(samples: list[tuple[str, int, str, str, str, str]]) -> list[str]:
    lines = [
        "| system | page | word | gold | predicted | sentence excerpt |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for system, page, word, gold, predicted, sentence in samples:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_cell(system),
                    str(page),
                    escape_cell(word),
                    escape_cell(gold),
                    escape_cell(predicted),
                    escape_cell(sentence),
                ]
            )
            + " |"
        )
    return lines


def format_report(
    gold_path: Path,
    generated_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    sentences: list[GoldSentence],
    metrics: list[ChrestomatijaMetrics],
    disagreements: list[tuple[str, int, str, str, str, str]],
    limit: int | None,
) -> str:
    stats = extraction_stats(sentences)
    lines = [
        "# Chrestomatija Gold Evaluation",
        "",
        "## Corpus",
        f"- gold: `{safe_relative(gold_path)}`",
        f"- generated DB: `{safe_relative(generated_path)}`",
        f"- joint checkpoint: `{safe_relative(checkpoint_path)}`",
        f"- sentence cap: {limit:,}" if limit is not None else "- sentence cap: none",
        f"- extracted sentences scored: {stats['sentences']:,}",
        f"- word tokens: {stats['tokens']:,}",
        f"- word types: {stats['types']:,}",
        f"- stress marks: {stats['stress_marks']:,}",
        f"- pages: {stats['pages']:,} ({stats['first_page']}-{stats['last_page']})",
        "",
        "## Metrics",
        "",
        "Token exact is measured over all gold word tokens; an unmarked gold token counts exact when the system leaves it unmarked or abstains. Token position is measured over answered tokens. Sentence sequence accuracy is exact only when every word token in the sentence is exact.",
        "",
        *metric_rows(metrics),
        "",
        "Thesis context: the 2026 VU thesis reports sentence-level sequence accuracy 0.711 for its transformer and 0.702 for VDU Kirciuoklis on 2,303 Chrestomatija samples. Tokenization and normalization protocols may differ from this reimplementation, so the cross-paper comparison is indicative.",
        "",
        "## Alignment Diagnostics",
        "",
        "| system | skipped gold tokens | skipped model tokens |",
        "| --- | ---: | ---: |",
    ]
    for item in metrics:
        lines.append(
            f"| {item.system} | {item.skipped_gold_tokens:,} | {item.skipped_model_tokens:,} |"
        )
    lines.extend(["", "## Sample Disagreements", ""])
    if disagreements:
        lines.extend(disagreement_rows(disagreements))
    else:
        lines.append("No disagreements sampled.")
    lines.extend(["", f"Report path: `{safe_relative(report_path)}`"])
    return "\n".join(lines) + "\n"


def print_table(metrics: list[ChrestomatijaMetrics]) -> None:
    for line in metric_rows(metrics):
        print(line)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--limit", type=int, default=None, help="Sentence cap.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true", help="Force joint model to CPU.")
    parser.add_argument("--skip-joint", action="store_true")
    parser.add_argument("--skip-dict", action="store_true")
    parser.add_argument("--skip-liepa", action="store_true")
    parser.add_argument("--cuda-memory-threshold-mib", type=int, default=6144)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not args.gold.exists():
        parser.error(f"missing gold JSONL: {args.gold}")

    sentences = load_gold(args.gold, limit=args.limit)
    print(
        f"loaded gold: {len(sentences):,} sentences, "
        f"{len(flatten_tokens(sentences)):,} word tokens"
    )

    results: list[SystemResult] = []
    if args.skip_joint:
        results.append(skipped_result("joint", sentences, time.perf_counter(), RuntimeError("--skip-joint")))
    else:
        print("running joint")
        results.append(
            run_joint(
                sentences=sentences,
                checkpoint=args.checkpoint,
                batch_size=args.batch_size,
                force_cpu=args.cpu,
                cuda_memory_threshold_mib=args.cuda_memory_threshold_mib,
            )
        )
    if args.skip_dict:
        results.append(skipped_result("dict-default", sentences, time.perf_counter(), RuntimeError("--skip-dict")))
    else:
        print("running dict-default")
        results.append(run_dict_default(sentences, args.generated))
    if args.skip_liepa:
        results.append(skipped_result("liepa", sentences, time.perf_counter(), RuntimeError("--skip-liepa")))
    else:
        print("running liepa")
        results.append(run_liepa(sentences))

    metrics = [score_result(sentences, result) for result in results]
    print_table(metrics)
    disagreements = sample_disagreements(sentences, results, limit=15)
    report = format_report(
        gold_path=args.gold,
        generated_path=args.generated,
        checkpoint_path=args.checkpoint,
        report_path=args.report,
        sentences=sentences,
        metrics=metrics,
        disagreements=disagreements,
        limit=args.limit,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8", newline="\n")
    print(f"report written: {safe_relative(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
