# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Train and compare encoder candidates for Lithuanian token tagging."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from metrics import evaluate_label_pairs
from train import derive_run_name, main as train_main


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "combined"
DEFAULT_RUNS_DIR = BASE_DIR / "runs"
DEFAULT_MODELS = (
    "VSSA-SDSA/LT-MLKM-modernBERT,"
    "EMBEDDIA/litlat-bert,"
    "xlm-roberta-base"
)
SMOKE_MODELS = "distilbert-base-multilingual-cased"


def parse_models(value: str) -> list[str]:
    models = [part.strip() for part in value.split(",") if part.strip()]
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    return models


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_label_list(data_dir: Path) -> list[str]:
    payload = json.loads((data_dir / "labels.json").read_text(encoding="utf-8"))
    return list(payload["labels"])


def batched(rows: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def evaluate_cpu(
    model_dir: Path,
    data_dir: Path,
    max_length: int,
    batch_size: int,
) -> tuple[dict[str, float | int], float]:
    import torch  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForTokenClassification,
        AutoTokenizer,
    )

    labels = load_label_list(data_dir)
    rows = read_jsonl(data_dir / "test.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("first-subword alignment requires a fast tokenizer")

    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.to("cpu")
    model.eval()

    predicted_labels: list[str] = []
    gold_labels: list[str] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in batched(rows, batch_size):
            encoded = tokenizer(
                [row["tokens"] for row in batch],
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            outputs = model(**{key: value.to("cpu") for key, value in encoded.items()})
            pred_ids = outputs.logits.argmax(dim=-1).cpu().tolist()

            for batch_index, row in enumerate(batch):
                row_predictions = ["_|_"] * len(row["tokens"])
                seen_word_ids: set[int] = set()
                for token_index, word_id in enumerate(
                    encoded.word_ids(batch_index=batch_index)
                ):
                    if word_id is None or word_id in seen_word_ids:
                        continue
                    seen_word_ids.add(word_id)
                    if word_id >= len(row_predictions):
                        continue
                    label_id = pred_ids[batch_index][token_index]
                    row_predictions[word_id] = (
                        labels[label_id] if 0 <= label_id < len(labels) else "_|_"
                    )
                predicted_labels.extend(row_predictions)
                gold_labels.extend(row["labels"])

    elapsed = time.perf_counter() - started
    metrics = evaluate_label_pairs(predicted_labels, gold_labels)
    tokens_per_second = len(gold_labels) / elapsed if elapsed > 0 else 0.0
    return metrics, tokens_per_second


def markdown_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def format_percent(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    return f"{100 * float(value):.2f}%"


def append_comparison_row(
    path: Path,
    model_name: str,
    run_name: str,
    status: str,
    metrics: dict[str, float | int] | None,
    tokens_per_second: float | None,
    notes: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "| time | model | run | status | upos | feats-exact | slots | aux/verb | tok/s | notes |\n"
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n",
            encoding="utf-8",
        )

    row = {
        "upos": None if metrics is None else metrics.get("upos_accuracy"),
        "feats": None if metrics is None else metrics.get("feats_exact_accuracy"),
        "slots": None if metrics is None else metrics.get("slot_accuracy"),
        "aux": None if metrics is None else metrics.get("aux_verb_accuracy"),
    }
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    tok_s = "n/a" if tokens_per_second is None else f"{tokens_per_second:.1f}"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"| {timestamp} "
            f"| {markdown_escape(model_name)} "
            f"| {markdown_escape(run_name)} "
            f"| {markdown_escape(status)} "
            f"| {format_percent(row['upos'])} "
            f"| {format_percent(row['feats'])} "
            f"| {format_percent(row['slots'])} "
            f"| {format_percent(row['aux'])} "
            f"| {tok_s} "
            f"| {markdown_escape(notes)} |\n"
        )


def train_args_for_model(
    args: argparse.Namespace,
    model_name: str,
    run_name: str,
    max_train_sentences: int | None,
    max_steps: int,
) -> list[str]:
    train_args = [
        "--data-dir",
        str(args.data_dir),
        "--model-name",
        model_name,
        "--fallback-model",
        args.fallback_model,
        "--run-name",
        run_name,
        "--runs-dir",
        str(args.runs_dir),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--weight-decay",
        str(args.weight_decay),
        "--warmup-ratio",
        str(args.warmup_ratio),
        "--max-length",
        str(args.max_length),
        "--max-steps",
        str(max_steps),
        "--logging-steps",
        str(args.logging_steps),
        "--save-total-limit",
        str(args.save_total_limit),
        "--seed",
        str(args.seed),
    ]
    if max_train_sentences is not None:
        train_args.extend(["--max-train-sentences", str(max_train_sentences)])
    if args.fp16:
        train_args.append("--fp16")
    if args.bf16:
        train_args.append("--bf16")
    return train_args


def clear_torch_cache() -> None:
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--models",
        help="comma-separated Hugging Face model names",
    )
    parser.add_argument("--fallback-model", default="")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-sentences", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--inference-batch-size", type=int, default=16)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    model_arg = args.models or (SMOKE_MODELS if args.smoke else DEFAULT_MODELS)
    models = parse_models(model_arg)
    max_train_sentences = args.max_train_sentences
    max_steps = args.max_steps if args.max_steps is not None else -1
    if args.smoke:
        if max_train_sentences is None:
            max_train_sentences = 400
        if args.max_steps is None:
            max_steps = 60

    comparison_path = args.runs_dir / "comparison.md"
    failures = 0
    for model_name in models:
        run_name = derive_run_name(model_name)
        print(f"training {model_name} -> {run_name}", file=sys.stderr)
        try:
            code = train_main(
                train_args_for_model(
                    args,
                    model_name,
                    run_name,
                    max_train_sentences,
                    max_steps,
                )
            )
            if code:
                raise RuntimeError(f"train.py exited with {code}")

            best_dir = args.runs_dir / run_name / "best"
            metrics, tokens_per_second = evaluate_cpu(
                best_dir,
                args.data_dir,
                args.max_length,
                args.inference_batch_size,
            )
            append_comparison_row(
                comparison_path,
                model_name,
                run_name,
                "ok",
                metrics,
                tokens_per_second,
                "best checkpoint on ALKSNIS test",
            )
        except Exception as exc:
            failures += 1
            traceback.print_exc()
            append_comparison_row(
                comparison_path,
                model_name,
                run_name,
                "failed",
                None,
                None,
                str(exc)[:240],
            )
        finally:
            clear_torch_cache()

    print(f"comparison table: {comparison_path}")
    return 1 if failures == len(models) else 0


if __name__ == "__main__":
    raise SystemExit(main())
