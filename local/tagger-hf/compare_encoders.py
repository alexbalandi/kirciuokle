# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Train and compare encoder/head/pooling candidates for Lithuanian tagging."""

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

from head_config import VALID_HEADS, VALID_POOLINGS, derive_run_name, load_head_config
from inference_utils import outputs_to_labels
from metrics import evaluate_label_pairs
from train import main as train_main


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = BASE_DIR / "data" / "combined"
DEFAULT_RUNS_DIR = BASE_DIR / "runs"
DEFAULT_MODELS = (
    "VSSA-SDSA/LT-MLKM-modernBERT,"
    "EMBEDDIA/litlat-bert,"
    "xlm-roberta-base"
)
SMOKE_MODEL = "distilbert-base-multilingual-cased"
RECOMMENDED_CELLS = (
    ("VSSA-SDSA/LT-MLKM-modernBERT", "combined", "first"),
    ("VSSA-SDSA/LT-MLKM-modernBERT", "combined", "last"),
    ("VSSA-SDSA/LT-MLKM-modernBERT", "combined", "first_last"),
    ("VSSA-SDSA/LT-MLKM-modernBERT", "factored", "last"),
    ("EMBEDDIA/litlat-bert", "combined", "last"),
    ("xlm-roberta-base", "combined", "first"),
)
SMOKE_CELLS = (
    (SMOKE_MODEL, "combined", "first"),
    (SMOKE_MODEL, "factored", "last"),
)
COMPARISON_HEADER = (
    "| time | model | head | pooling | run | status | upos | feats-exact | "
    "slots | aux/verb | tok/s | notes |\n"
    "| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
)


def parse_models(value: str) -> list[str]:
    models = [part.strip() for part in value.split(",") if part.strip()]
    if not models:
        raise argparse.ArgumentTypeError("at least one model is required")
    return models


def parse_choices(value: str, valid: Iterable[str], label: str) -> list[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"at least one {label} is required")
    unknown = sorted(set(items) - set(valid))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {label}(s): {', '.join(unknown)}"
        )
    return list(dict.fromkeys(items))


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def batched(rows: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def evaluate_cpu(
    model_dir: Path,
    data_dir: Path,
    batch_size: int,
) -> tuple[dict[str, float | int], float]:
    import torch  # type: ignore[import-not-found]
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    from export_onnx import load_torch_runner, run_torch_model

    head_config = load_head_config(model_dir)
    rows = read_jsonl(data_dir / "test.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("subword alignment requires a fast tokenizer")

    torch_runner, output_names = load_torch_runner(model_dir, head_config)
    if hasattr(torch_runner, "to"):
        torch_runner.to("cpu")
    torch_runner.eval()

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
                max_length=head_config["max_length"],
                return_tensors="pt",
            )
            outputs = run_torch_model(torch_runner, encoded, output_names)
            for batch_index, row in enumerate(batch):
                predicted_labels.extend(
                    outputs_to_labels(
                        outputs=outputs,
                        word_ids=encoded.word_ids(batch_index=batch_index),
                        word_count=len(row["tokens"]),
                        head_config=head_config,
                        batch_index=batch_index,
                    )
                )
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


def ensure_comparison_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(COMPARISON_HEADER, encoding="utf-8")
        return

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or " head " in lines[0]:
        return
    body = "\n".join(lines[2:])
    path.write_text(
        COMPARISON_HEADER + (body + "\n" if body else ""),
        encoding="utf-8",
    )


def append_comparison_row(
    path: Path,
    model_name: str,
    head: str,
    pooling: str,
    run_name: str,
    status: str,
    metrics: dict[str, float | int] | None,
    tokens_per_second: float | None,
    notes: str,
) -> None:
    ensure_comparison_header(path)
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
            f"| {markdown_escape(head)} "
            f"| {markdown_escape(pooling)} "
            f"| {markdown_escape(run_name)} "
            f"| {markdown_escape(status)} "
            f"| {format_percent(row['upos'])} "
            f"| {format_percent(row['feats'])} "
            f"| {format_percent(row['slots'])} "
            f"| {format_percent(row['aux'])} "
            f"| {tok_s} "
            f"| {markdown_escape(notes)} |\n"
        )


def train_args_for_cell(
    args: argparse.Namespace,
    model_name: str,
    head: str,
    pooling: str,
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
        "--head",
        head,
        "--subword-pooling",
        pooling,
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


def cells_from_args(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if args.recommended:
        if args.models or args.heads or args.poolings:
            raise SystemExit("--recommended cannot be combined with --models, --heads, or --poolings")
        return list(RECOMMENDED_CELLS)

    if args.smoke:
        return list(SMOKE_CELLS)

    models = parse_models(args.models or DEFAULT_MODELS)
    heads = parse_choices(args.heads or "combined", VALID_HEADS, "head")
    poolings = parse_choices(args.poolings or "first", VALID_POOLINGS, "pooling")
    return [(model, head, pooling) for model in models for head in heads for pooling in poolings]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--models",
        help="comma-separated Hugging Face model names",
    )
    parser.add_argument("--heads", help="comma-separated heads: combined,factored")
    parser.add_argument(
        "--poolings",
        help="comma-separated subword poolings: first,last,first_last",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--smoke", action="store_true")
    mode_group.add_argument("--recommended", action="store_true")
    parser.add_argument("--fallback-model", default="")
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

    cells = cells_from_args(args)
    max_train_sentences = args.max_train_sentences
    max_steps = args.max_steps if args.max_steps is not None else -1
    if args.smoke:
        max_train_sentences = 400
        max_steps = 60

    comparison_path = args.runs_dir / "comparison.md"
    failures = 0
    for model_name, head, pooling in cells:
        run_name = derive_run_name(model_name, head, pooling)
        print(
            f"training {model_name} / {head} / {pooling} -> {run_name}",
            file=sys.stderr,
        )
        try:
            code = train_main(
                train_args_for_cell(
                    args,
                    model_name,
                    head,
                    pooling,
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
                args.inference_batch_size,
            )
            append_comparison_row(
                comparison_path,
                model_name,
                head,
                pooling,
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
                head,
                pooling,
                run_name,
                "failed",
                None,
                None,
                str(exc)[:240],
            )
        finally:
            clear_torch_cache()

    print(f"comparison table: {comparison_path}")
    return 1 if failures == len(cells) else 0


if __name__ == "__main__":
    raise SystemExit(main())
