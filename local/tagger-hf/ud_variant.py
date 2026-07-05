# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Create the strict-UD variant from a normalized VDU checkpoint.

This script performs only classifier-head surgery. The follow-up ALKSNIS
fine-tune is intentionally external and should be run as:

uv run --with transformers --with datasets --with accelerate --with torch local/tagger-hf/train.py --model-name <surgery-out> --data-dir <ud-dataset> --learning-rate 1e-5 --epochs 6 --fallback-model ""
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from coverage_diff import SLOT_FEATS_KEYS
from head_config import labels_from_file, load_head_config, write_head_config
from metrics import canonicalize_feats, combined_label, feats_string, parse_feats


FOLDED_UPOS = {
    "DET": "PRON",
    "AUX": "VERB",
}


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_unique(labels: list[str], source: Path) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for label in labels:
        if label in seen:
            duplicates.append(label)
        seen.add(label)
    if duplicates:
        sample = ", ".join(repr(label) for label in duplicates[:5])
        raise ValueError(f"{source} contains duplicate labels: {sample}")


def resolve_base_checkpoint(base_run: Path) -> Path:
    best_dir = base_run / "best"
    if best_dir.is_dir():
        return best_dir
    if (base_run / "head_config.json").is_file():
        return base_run
    raise FileNotFoundError(f"missing base checkpoint directory: {best_dir}")


def folded_twin_label(label: str) -> str:
    upos, separator, feats = label.partition("|")
    folded_upos = FOLDED_UPOS.get(upos, upos)
    return f"{folded_upos}|{canonicalize_feats(feats if separator else '_')}"


def stripped_folded_twin_label(label: str) -> str:
    upos, separator, feats = label.partition("|")
    folded_upos = FOLDED_UPOS.get(upos, upos)
    parsed = parse_feats(feats if separator else "_")
    slot_keys = set(SLOT_FEATS_KEYS)
    stripped = {key: value for key, value in parsed.items() if key in slot_keys}
    return combined_label(folded_upos, feats_string(stripped))


def classifier_module(model: object):
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only at runtime.
        raise RuntimeError("ud_variant.py requires torch") from exc

    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, torch.nn.Linear):
        return classifier
    raise TypeError(
        "expected the base model to expose a linear .classifier head; "
        f"got {type(classifier).__name__}"
    )


def set_classifier_module(model: object, classifier: object) -> None:
    setattr(model, "classifier", classifier)


def surgery(
    *,
    base_checkpoint: Path,
    target_labels_path: Path,
    out_dir: Path,
) -> tuple[int, int, int, int, list[str]]:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    import torch

    head_config = load_head_config(base_checkpoint)
    if head_config["head"] != "combined":
        raise ValueError("UD variant surgery requires a combined classifier head")
    if head_config["pooling"] not in {"first", "last"}:
        raise ValueError(
            "UD variant surgery requires a standard HF first/last pooling checkpoint"
        )

    target_labels = labels_from_file(target_labels_path)
    require_unique(target_labels, target_labels_path)

    model = AutoModelForTokenClassification.from_pretrained(base_checkpoint)
    old_classifier = classifier_module(model)
    base_labels = [str(label) for label in head_config["labels"]]
    require_unique(base_labels, base_checkpoint / "head_config.json")
    if len(base_labels) != old_classifier.out_features:
        raise ValueError(
            "base head_config label count does not match classifier rows: "
            f"{len(base_labels)} labels vs {old_classifier.out_features} rows"
        )

    base_label2id = {label: index for index, label in enumerate(base_labels)}
    new_classifier = torch.nn.Linear(
        old_classifier.in_features,
        len(target_labels),
        bias=old_classifier.bias is not None,
    )
    new_classifier.to(
        device=old_classifier.weight.device,
        dtype=old_classifier.weight.dtype,
    )

    initializer_range = float(getattr(model.config, "initializer_range", 0.02) or 0.02)
    copied = 0
    twinned = 0
    stripped_twinned = 0
    fresh = 0
    fresh_labels: list[str] = []

    with torch.no_grad():
        new_classifier.weight.normal_(mean=0.0, std=initializer_range)
        if new_classifier.bias is not None:
            new_classifier.bias.zero_()

        for target_index, label in enumerate(target_labels):
            source_index = base_label2id.get(label)
            if source_index is not None:
                copied += 1
            else:
                twin = folded_twin_label(label)
                source_index = base_label2id.get(twin)
                if source_index is not None:
                    twinned += 1
                else:
                    stripped_twin = stripped_folded_twin_label(label)
                    source_index = base_label2id.get(stripped_twin)
                    if source_index is not None:
                        stripped_twinned += 1
                    else:
                        fresh += 1
                        fresh_labels.append(label)
                        continue

            new_classifier.weight[target_index].copy_(old_classifier.weight[source_index])
            if new_classifier.bias is not None and old_classifier.bias is not None:
                new_classifier.bias[target_index].copy_(old_classifier.bias[source_index])

    label2id = {label: index for index, label in enumerate(target_labels)}
    id2label = {index: label for index, label in enumerate(target_labels)}
    model.config.num_labels = len(target_labels)
    model.config.label2id = label2id
    model.config.id2label = id2label
    if hasattr(model, "num_labels"):
        model.num_labels = len(target_labels)
    set_classifier_module(model, new_classifier)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint, use_fast=True)
    tokenizer.save_pretrained(out_dir)
    write_json(out_dir / "labels.json", {"labels": target_labels})

    updated_head_config = dict(head_config)
    updated_head_config["labels"] = target_labels
    write_head_config(out_dir / "head_config.json", updated_head_config)

    return copied, twinned, stripped_twinned, fresh, fresh_labels


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extend a normalized VDU token-classifier checkpoint to a strict-UD "
            "label set using folded-twin head initialization."
        )
    )
    parser.add_argument(
        "--base-run",
        type=Path,
        required=True,
        help="base run directory, normally local/tagger-hf/runs/<name>",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="target UD labels.json containing DET/AUX labels",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output Hugging Face checkpoint directory",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    base_checkpoint = resolve_base_checkpoint(args.base_run)
    copied, twinned, stripped_twinned, fresh, fresh_labels = surgery(
        base_checkpoint=base_checkpoint,
        target_labels_path=args.labels,
        out_dir=args.out,
    )

    print(f"base checkpoint: {base_checkpoint}")
    print(f"output checkpoint: {args.out}")
    print(f"copied rows: {copied}")
    print(f"twinned rows: {twinned}")
    print(f"stripped-twinned rows: {stripped_twinned}")
    print(f"fresh rows: {fresh}")
    if fresh_labels:
        print("fresh labels:")
        for label in fresh_labels:
            print(f"  {label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
