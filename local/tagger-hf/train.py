# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Fine-tune a transformer for Lithuanian UPOS|FEATS token classification."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path
from typing import Iterable

from head_config import (
    DEFAULT_LABEL,
    VALID_HEADS,
    VALID_POOLINGS,
    assemble_label_from_ids,
    build_head_config,
    build_slots_from_labels,
    derive_run_name,
    label_token_positions,
    labels_from_file,
    represented_word_indices,
    slot_ids_for_label,
    word_piece_spans,
    write_head_config,
)
from lemma_scripts import apply_lemma_script
from metrics import evaluate_label_pairs


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "VSSA-SDSA/LT-MLKM-modernBERT"
DEFAULT_FALLBACK_MODEL = "xlm-roberta-base"
DEFAULT_DATA_DIR = BASE_DIR / "data" / "combined"
DEFAULT_RUNS_DIR = BASE_DIR / "runs"


def load_labels(path: Path) -> tuple[list[str], dict[str, int], dict[int, str]]:
    labels = labels_from_file(path)
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for index, label in enumerate(labels)}
    return labels, label2id, id2label


def load_lemma_scripts(path: Path) -> tuple[list[str], dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scripts = [str(script) for script in payload["lemma_scripts"]]
    return scripts, {script: index for index, script in enumerate(scripts)}


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    run_name = args.run_name or derive_run_name(
        args.model_name,
        args.head,
        args.subword_pooling,
    )
    return args.runs_dir / run_name


def training_arguments_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "output_dir": str(args.trainer_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "slot_accuracy",
        "greater_is_better": True,
        "report_to": [],
        "seed": args.seed,
        "fp16": args.fp16,
        "bf16": args.bf16,
    }

    parameters = inspect.signature(args.training_arguments_class.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"
    if "label_names" in parameters:
        kwargs["label_names"] = (
            ["labels", "lemma_labels"] if args.lemma_head else ["labels"]
        )
    kwargs["save_strategy"] = "epoch"
    return kwargs


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def strip_metric_prefix(metrics: dict[str, float], prefix: str) -> dict[str, float]:
    stripped: dict[str, float] = {}
    marker = f"{prefix}_"
    for key, value in metrics.items():
        if key.startswith(marker):
            stripped[key[len(marker) :]] = float(value)
    return stripped


def labels_in_rows(rows: Iterable[dict]) -> list[str]:
    return [label for row in rows for label in row["labels"]]


def uses_custom_model(args: argparse.Namespace) -> bool:
    return args.lemma_head or args.head == "factored" or args.subword_pooling == "first_last"


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL)
    parser.add_argument(
        "--run-name",
        help=(
            "run directory name under --runs-dir; defaults to "
            "<model-short>__<head>__<pooling>"
        ),
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="compatibility alias for the full run directory",
    )
    parser.add_argument("--head", choices=VALID_HEADS, default="combined")
    parser.add_argument(
        "--lemma-head",
        action="store_true",
        help="train an auxiliary FORM→LEMMA edit-script classifier head",
    )
    parser.add_argument(
        "--subword-pooling",
        choices=VALID_POOLINGS,
        default="first",
    )
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--max-train-samples",
        "--max-train-sentences",
        dest="max_train_samples",
        type=int,
    )
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = resolve_run_dir(args)
    args.run_name = args.run_name or run_dir.name
    args.trainer_dir = run_dir / "checkpoints"
    best_dir = run_dir / "best"
    metrics_path = run_dir / "metrics.json"
    final_path = run_dir / "final.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        metrics_path,
        {
            "run_name": args.run_name,
            "head": args.head,
            "lemma_head": args.lemma_head,
            "pooling": args.subword_pooling,
            "evaluations": [],
        },
    )

    from datasets import load_dataset  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainerCallback,
        TrainingArguments,
        set_seed,
    )

    from head_modeling import (  # type: ignore[import-not-found]
        PooledDataCollator,
        create_custom_model,
        hidden_size_from_config,
        save_custom_model,
    )

    args.training_arguments_class = TrainingArguments
    set_seed(args.seed)

    labels_path = args.data_dir / "labels.json"
    labels, label2id, id2label = load_labels(labels_path)
    lemma_scripts: list[str] = []
    lemma_script2id: dict[str, int] = {}
    if args.lemma_head:
        lemma_scripts, lemma_script2id = load_lemma_scripts(
            args.data_dir / "lemma_scripts.json"
        )

    data_files = {
        "train": str(args.data_dir / "train.jsonl"),
        "dev": str(args.data_dir / "dev.jsonl"),
        "test": str(args.data_dir / "test.jsonl"),
    }
    dataset = load_dataset("json", data_files=data_files)
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(
            range(min(args.max_train_samples, len(dataset["train"])))
        )
    if args.max_dev_samples:
        dataset["dev"] = dataset["dev"].select(
            range(min(args.max_dev_samples, len(dataset["dev"])))
        )
    if args.max_test_samples:
        dataset["test"] = dataset["test"].select(
            range(min(args.max_test_samples, len(dataset["test"])))
        )

    slots = (
        build_slots_from_labels(labels_in_rows(dataset["train"]))
        if args.head == "factored"
        else None
    )
    slot_count = len(slots or {})

    model_name = args.model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if uses_custom_model(args):
            # A previously saved custom checkpoint must be restored through
            # load_custom_model — AutoModel.from_pretrained cannot map the
            # wrapper's weights and would silently reinitialize them.
            candidate = Path(model_name)
            if (candidate / "head_config.json").exists() and (
                candidate / "pytorch_model.bin"
            ).exists():
                from head_config import load_head_config
                from head_modeling import load_custom_model

                checkpoint_config = load_head_config(candidate)
                if args.head == "combined" and checkpoint_config.get("labels") != labels:
                    raise RuntimeError(
                        "checkpoint label inventory differs from the dataset's; "
                        "align labels.json before continued fine-tuning"
                    )
                if args.lemma_head and checkpoint_config.get("lemma_scripts") != lemma_scripts:
                    raise RuntimeError(
                        "checkpoint lemma-script inventory differs from the "
                        "dataset's; align lemma_scripts.json before continued "
                        "fine-tuning"
                    )
                model = load_custom_model(candidate, checkpoint_config)
            else:
                model = create_custom_model(
                    model_name=model_name,
                    head=args.head,
                    pooling=args.subword_pooling,
                    labels=labels if args.head == "combined" else None,
                    slots=slots,
                    lemma_scripts=lemma_scripts if args.lemma_head else None,
                )
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(labels),
                label2id=label2id,
                id2label=id2label,
            )
    except Exception:
        if not args.fallback_model or args.fallback_model == model_name:
            raise
        print(f"could not load {model_name}; falling back to {args.fallback_model}")
        model_name = args.fallback_model
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if uses_custom_model(args):
            model = create_custom_model(
                model_name=model_name,
                head=args.head,
                pooling=args.subword_pooling,
                labels=labels if args.head == "combined" else None,
                slots=slots,
                lemma_scripts=lemma_scripts if args.lemma_head else None,
            )
        else:
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(labels),
                label2id=label2id,
                id2label=id2label,
            )

    if not tokenizer.is_fast:
        raise RuntimeError("subword alignment requires a fast tokenizer")

    def encode_combined_label(label: str) -> int:
        return label2id.get(label, -100)

    def tokenize_and_align(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
        )
        aligned_labels: list[list[int]] = []
        aligned_slot_labels: list[list[list[int]]] = []
        aligned_lemma_labels: list[list[int]] = []
        first_indices_batch: list[list[int]] = []
        last_indices_batch: list[list[int]] = []

        for batch_index, sentence_labels in enumerate(batch["labels"]):
            sentence_scripts = (
                batch["lemma_scripts"][batch_index] if args.lemma_head else []
            )
            word_ids = list(tokenized.word_ids(batch_index=batch_index))
            word_count = len(sentence_labels)

            if args.subword_pooling == "first_last":
                first, last = word_piece_spans(word_ids, word_count)
                sentence_ids: list[int] = []
                sentence_slot_ids: list[list[int]] = []
                sentence_lemma_ids: list[int] = []
                first_indices: list[int] = []
                last_indices: list[int] = []
                for word_index, label in enumerate(sentence_labels):
                    if first[word_index] == -1 or last[word_index] == -1:
                        continue
                    first_indices.append(first[word_index])
                    last_indices.append(last[word_index])
                    sentence_ids.append(encode_combined_label(label))
                    if args.lemma_head:
                        sentence_lemma_ids.append(
                            lemma_script2id.get(sentence_scripts[word_index], -100)
                        )
                    if slots is not None:
                        sentence_slot_ids.append(slot_ids_for_label(label, slots))
                aligned_labels.append(sentence_ids)
                first_indices_batch.append(first_indices)
                last_indices_batch.append(last_indices)
                if args.lemma_head:
                    aligned_lemma_labels.append(sentence_lemma_ids)
                if slots is not None:
                    aligned_slot_labels.append(sentence_slot_ids)
                continue

            sentence_ids = [-100] * len(word_ids)
            sentence_slot_ids = [[-100] * slot_count for _ in word_ids]
            sentence_lemma_ids = [-100] * len(word_ids)
            positions = label_token_positions(
                word_ids,
                word_count,
                args.subword_pooling,
            )
            for word_index, token_index in enumerate(positions):
                if token_index == -1:
                    continue
                label = sentence_labels[word_index]
                sentence_ids[token_index] = encode_combined_label(label)
                if args.lemma_head:
                    sentence_lemma_ids[token_index] = lemma_script2id.get(
                        sentence_scripts[word_index],
                        -100,
                    )
                if slots is not None:
                    sentence_slot_ids[token_index] = slot_ids_for_label(label, slots)
            aligned_labels.append(sentence_ids)
            if args.lemma_head:
                aligned_lemma_labels.append(sentence_lemma_ids)
            if slots is not None:
                aligned_slot_labels.append(sentence_slot_ids)

        tokenized["labels"] = aligned_labels
        if args.subword_pooling == "first_last":
            tokenized["first_subword_indices"] = first_indices_batch
            tokenized["last_subword_indices"] = last_indices_batch
        if slots is not None:
            tokenized["slot_labels"] = aligned_slot_labels
        if args.lemma_head:
            tokenized["lemma_labels"] = aligned_lemma_labels
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    def metric_lemma_context(split: str) -> list[tuple[str, str]]:
        if not args.lemma_head:
            return []
        context: list[tuple[str, str]] = []
        for row in dataset[split]:
            encoded = tokenizer(
                row["tokens"],
                is_split_into_words=True,
                truncation=True,
                max_length=args.max_length,
            )
            word_ids = list(encoded.word_ids(batch_index=0))
            word_count = len(row["tokens"])
            if args.subword_pooling == "first_last":
                word_indexes = represented_word_indices(word_ids, word_count)
            else:
                word_indexes = [
                    index
                    for index, token_index in enumerate(
                        label_token_positions(
                            word_ids,
                            word_count,
                            args.subword_pooling,
                        )
                    )
                    if token_index != -1
                ]
            for word_index in word_indexes:
                context.append((row["tokens"][word_index], row["lemmas"][word_index]))
        return context

    metric_lemma_contexts: dict[tuple[int, int, int], list[tuple[str, str]]] = {}
    if args.lemma_head:
        for split in ("dev", "test"):
            context = metric_lemma_context(split)
            label_lengths = [len(row) for row in tokenized_dataset[split]["labels"]]
            signature = (
                len(label_lengths),
                max(label_lengths, default=0),
                len(context),
            )
            metric_lemma_contexts.setdefault(signature, context)

    def compute_metrics(eval_prediction: object) -> dict[str, float]:
        import numpy as np

        predictions = getattr(eval_prediction, "predictions")
        label_ids = getattr(eval_prediction, "label_ids")
        lemma_label_ids = None
        if isinstance(label_ids, tuple):
            if args.lemma_head and len(label_ids) > 1:
                lemma_label_ids = label_ids[1]
            label_ids = label_ids[0]

        pred_labels: list[str] = []
        gold_labels: list[str] = []
        lemma_predictions = None

        if args.head == "combined":
            if isinstance(predictions, tuple):
                if args.lemma_head and len(predictions) > 1:
                    lemma_predictions = predictions[1]
                predictions = predictions[0]
            pred_ids = np.argmax(predictions, axis=-1)
            mask = label_ids != -100
            for pred_id, gold_id in zip(
                pred_ids[mask].tolist(),
                label_ids[mask].tolist(),
            ):
                pred_labels.append(labels[pred_id] if pred_id < len(labels) else DEFAULT_LABEL)
                gold_labels.append(labels[gold_id] if gold_id < len(labels) else DEFAULT_LABEL)
        else:
            if not isinstance(predictions, (tuple, list)):
                raise RuntimeError("factored model predictions must be per-slot tensors")
            assert slots is not None
            slot_prediction_tensors = list(predictions)
            if args.lemma_head:
                lemma_predictions = slot_prediction_tensors[-1]
                slot_prediction_tensors = slot_prediction_tensors[:-1]
            slot_predictions = [np.argmax(item, axis=-1) for item in slot_prediction_tensors]
            active_positions = np.argwhere(label_ids != -100)
            for position in active_positions:
                index = tuple(position.tolist())
                pred_ids = [int(slot_prediction[index]) for slot_prediction in slot_predictions]
                gold_id = int(label_ids[index])
                pred_labels.append(assemble_label_from_ids(pred_ids, slots))
                gold_labels.append(labels[gold_id] if gold_id < len(labels) else DEFAULT_LABEL)

        scored = evaluate_label_pairs(pred_labels, gold_labels)
        metrics = {
            "label_accuracy": float(scored["label_accuracy"]),
            "slot_accuracy": float(scored["slot_accuracy"]),
            "upos_accuracy": float(scored["upos_accuracy"]),
            "feats_exact_accuracy": float(scored["feats_exact_accuracy"]),
            "aux_verb_accuracy": float(scored["aux_verb_accuracy"]),
        }
        if args.lemma_head and lemma_predictions is not None:
            lemma_pred_ids = np.argmax(lemma_predictions, axis=-1)
            lemma_mask = label_ids != -100
            if lemma_label_ids is not None:
                lemma_mask = lemma_mask & (lemma_label_ids != -100)
            signature = (
                int(label_ids.shape[0]),
                int(label_ids.shape[1]) if len(label_ids.shape) > 1 else 0,
                int(lemma_mask.sum()),
            )
            contexts = metric_lemma_contexts.get(signature)
            lemma_ok = 0
            lemma_total = 0
            if contexts is not None:
                for (form, gold_lemma), script_id in zip(
                    contexts,
                    lemma_pred_ids[lemma_mask].tolist(),
                ):
                    if 0 <= script_id < len(lemma_scripts):
                        predicted_lemma = apply_lemma_script(form, lemma_scripts[script_id])
                    else:
                        predicted_lemma = form.lower()
                    lemma_ok += int(predicted_lemma == gold_lemma)
                    lemma_total += 1
            metrics["lemma_accuracy"] = lemma_ok / lemma_total if lemma_total else 0.0
        return metrics

    eval_records: list[dict[str, float | int | str | None]] = []

    class MetricsRecorder(TrainerCallback):
        def on_evaluate(self, args_obj, state, control, metrics=None, **kwargs):
            if not metrics or "eval_label_accuracy" not in metrics:
                return
            record = {
                "epoch": float(metrics.get("epoch", state.epoch or 0.0)),
                "step": int(state.global_step),
                "head": args.head,
                "pooling": args.subword_pooling,
                "label_accuracy": float(metrics.get("eval_label_accuracy", 0.0)),
                "slot_accuracy": float(metrics.get("eval_slot_accuracy", 0.0)),
                "upos_accuracy": float(metrics.get("eval_upos_accuracy", 0.0)),
                "feats_exact_accuracy": float(
                    metrics.get("eval_feats_exact_accuracy", 0.0)
                ),
                "aux_verb_accuracy": float(metrics.get("eval_aux_verb_accuracy", 0.0)),
                "lemma_accuracy": float(metrics.get("eval_lemma_accuracy", 0.0)),
            }
            eval_records.append(record)
            write_json(
                metrics_path,
                {
                    "run_name": args.run_name,
                    "head": args.head,
                    "lemma_head": args.lemma_head,
                    "pooling": args.subword_pooling,
                    "evaluations": eval_records,
                },
            )

    training_args = TrainingArguments(**training_arguments_kwargs(args))
    data_collator = (
        PooledDataCollator(tokenizer=tokenizer, slot_count=slot_count)
        if uses_custom_model(args)
        else DataCollatorForTokenClassification(tokenizer)
    )
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MetricsRecorder()],
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_dataset["test"],
        metric_key_prefix="test",
    )

    if uses_custom_model(args):
        save_custom_model(model, best_dir)
    else:
        trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    shutil.copy2(labels_path, best_dir / "labels.json")
    if args.lemma_head:
        shutil.copy2(args.data_dir / "lemma_scripts.json", best_dir / "lemma_scripts.json")

    hidden_size = (
        int(model.hidden_size)
        if uses_custom_model(args)
        else hidden_size_from_config(model.config)
    )
    head_config = build_head_config(
        head=args.head,
        pooling=args.subword_pooling,
        base_model=model_name,
        hidden_size=hidden_size,
        max_length=args.max_length,
        labels=labels if args.head == "combined" else None,
        slots=slots if args.head == "factored" else None,
        lemma_scripts=lemma_scripts if args.lemma_head else None,
    )
    write_head_config(best_dir / "head_config.json", head_config)
    write_json(
        best_dir / "training_meta.json",
        {
            "base_model": model_name,
            "requested_model": args.model_name,
            "head": args.head,
            "pooling": args.subword_pooling,
            "max_length": args.max_length,
            "label_count": len(labels),
            "lemma_head": args.lemma_head,
            "lemma_script_count": len(lemma_scripts),
            "slot_count": slot_count,
        },
    )

    best_dev = max(eval_records, key=lambda item: float(item["slot_accuracy"]), default={})
    final_payload = {
        "run_name": args.run_name,
        "requested_model": args.model_name,
        "base_model": model_name,
        "head": args.head,
        "lemma_head": args.lemma_head,
        "pooling": args.subword_pooling,
        "data_dir": str(args.data_dir),
        "best_model_dir": str(best_dir),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_dev": best_dev,
        "test": strip_metric_prefix(test_metrics, "test"),
    }
    write_json(final_path, final_payload)
    print(f"best model: {best_dir}")
    print(f"final metrics: {final_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
