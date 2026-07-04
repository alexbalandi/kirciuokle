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


DEFAULT_MODEL = "VSSA-SDSA/LT-MLKM-modernBERT"
DEFAULT_FALLBACK_MODEL = "xlm-roberta-base"
SCORING_SLOTS = ("case", "gender", "number", "tense", "person", "voice", "degree")


def parse_feats(raw: str) -> dict[str, str]:
    if not raw or raw == "_":
        return {}
    feats: dict[str, str] = {}
    for feature in raw.split("|"):
        separator = feature.find("=")
        if separator <= 0:
            continue
        feats[feature[:separator]] = feature[separator + 1 :]
    return feats


def split_label(label: str) -> tuple[str, dict[str, str]]:
    if "|" not in label:
        return label, {}
    upos, feats = label.split("|", 1)
    return upos, parse_feats(feats)


def token_tags(upos: str, feats: dict[str, str]) -> dict[str, str]:
    """Faithful port of tokenTags in src/worker/disambiguation.ts."""
    tags: dict[str, str] = {}

    if upos in ("VERB", "AUX"):
        tags["pos"] = "PART_VERB" if feats.get("VerbForm") == "Part" else "VERB"
    elif upos in ("NOUN", "PROPN"):
        tags["pos"] = "NOUN"
    elif upos in ("CCONJ", "SCONJ"):
        tags["pos"] = "CCONJ"
    else:
        tags["pos"] = upos

    for slot, feature in (
        ("gender", "Gender"),
        ("number", "Number"),
        ("case", "Case"),
        ("tense", "Tense"),
        ("person", "Person"),
        ("voice", "Voice"),
    ):
        value = feats.get(feature)
        if value:
            tags[slot] = value

    degree = feats.get("Degree")
    if degree and degree != "Pos":
        tags["degree"] = degree

    return tags


def load_labels(path: Path) -> tuple[list[str], dict[str, int], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    labels = list(payload["labels"])
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for index, label in enumerate(labels)}
    return labels, label2id, id2label


def training_arguments_kwargs(args: argparse.Namespace) -> dict:
    kwargs = {
        "output_dir": str(args.output_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
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
    kwargs["save_strategy"] = "epoch"
    return kwargs


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "alksnis",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs" / "modernbert-alksnis",
    )
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-dev-samples", type=int)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from datasets import load_dataset  # type: ignore[import-not-found]
    from transformers import (  # type: ignore[import-not-found]
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    args.training_arguments_class = TrainingArguments
    set_seed(args.seed)

    labels_path = args.data_dir / "labels.json"
    labels, label2id, id2label = load_labels(labels_path)
    label_tags = [token_tags(*split_label(label)) for label in labels]

    data_files = {
        "train": str(args.data_dir / "train.jsonl"),
        "dev": str(args.data_dir / "dev.jsonl"),
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

    model_name = args.model_name
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
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
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(labels),
            label2id=label2id,
            id2label=id2label,
        )

    if not tokenizer.is_fast:
        raise RuntimeError("first-subword alignment requires a fast tokenizer")

    def tokenize_and_align(batch: dict) -> dict:
        tokenized = tokenizer(
            batch["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=args.max_length,
        )
        aligned_labels: list[list[int]] = []
        for batch_index, sentence_labels in enumerate(batch["labels"]):
            word_ids = tokenized.word_ids(batch_index=batch_index)
            previous_word_id = None
            sentence_ids: list[int] = []
            for word_id in word_ids:
                if word_id is None:
                    sentence_ids.append(-100)
                elif word_id != previous_word_id:
                    sentence_ids.append(label2id[sentence_labels[word_id]])
                else:
                    sentence_ids.append(-100)
                previous_word_id = word_id
            aligned_labels.append(sentence_ids)
        tokenized["labels"] = aligned_labels
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_and_align,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    def compute_metrics(eval_prediction: object) -> dict[str, float]:
        import numpy as np

        predictions = getattr(eval_prediction, "predictions")
        label_ids = getattr(eval_prediction, "label_ids")
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pred_ids = np.argmax(predictions, axis=-1)
        mask = label_ids != -100
        total = int(mask.sum())
        if total == 0:
            return {"label_accuracy": 0.0, "slot_accuracy": 0.0}

        label_ok = int((pred_ids[mask] == label_ids[mask]).sum())
        slot_ok = 0
        for pred_id, gold_id in zip(pred_ids[mask].tolist(), label_ids[mask].tolist()):
            if label_tags[pred_id] == label_tags[gold_id]:
                slot_ok += 1
        return {
            "label_accuracy": label_ok / total,
            "slot_accuracy": slot_ok / total,
        }

    training_args = TrainingArguments(**training_arguments_kwargs(args))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["dev"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    shutil.copy2(labels_path, args.output_dir / "labels.json")
    (args.output_dir / "training_meta.json").write_text(
        json.dumps(
            {
                "base_model": model_name,
                "max_length": args.max_length,
                "label_count": len(labels),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
