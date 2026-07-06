from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from joint_lib import (
    ACCENTUATOR_DIR,
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
    JointCollator,
    JointDataset,
    SimpleToken,
    count_parameters,
    has_letter,
    instantiate_from_checkpoint,
    load_joint_rows,
    predict_batches,
    rows_from_plain_sentences,
    safe_relative,
    tokens_from_text_sentence,
    tokens_per_second,
)


sys.path.insert(0, str(ACCENTUATOR_DIR))
import eval_nodict_pipeline as nodict  # noqa: E402


DEFAULT_CORPUS = ACCENTUATOR_DIR / "data" / "eval" / "lrt-smoke.txt"
DEFAULT_SILVER = ACCENTUATOR_DIR / "data" / "eval" / "lrt-smoke-silver.jsonl"
DEFAULT_AUDIT = ACCENTUATOR_DIR / "data" / "eval" / "lrt-silver-audit.json"


def eval_pos_split(
    name: str,
    path: Path,
    model,
    tokenizer,
    char_vocab: dict[str, int],
    device: torch.device,
    limit: int | None,
    batch_size: int,
) -> dict[str, float | int | str]:
    if not path.exists():
        return {"split": name, "missing": str(path)}
    rows = load_joint_rows(path, limit=limit)
    collator = JointCollator(tokenizer, model.labels, char_vocab)
    loader = DataLoader(
        JointDataset(rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    token_count = sum(len(row.get("tokens", [])) for row in rows)
    started = time.perf_counter()
    predictions = predict_batches(model, loader, device)
    tps = tokens_per_second(token_count, started)
    total = correct = upos_correct = 0
    for row in predictions:
        for token in row["tokens"]:
            gold = token["gold_pos"]
            pred = token["pos"]
            total += 1
            correct += int(pred == gold)
            upos_correct += int(pred.split("|", 1)[0] == gold.split("|", 1)[0])
    return {
        "split": name,
        "sentences": len(rows),
        "tokens": total,
        "label_accuracy": correct / total if total else 0.0,
        "upos_accuracy": upos_correct / total if total else 0.0,
        "tokens_per_second": tps,
    }


def print_pos_result(result: dict[str, float | int | str]) -> None:
    if "missing" in result:
        print(f"POS {result['split']}: missing {result['missing']}")
        return
    print(
        f"POS {result['split']}: "
        f"sentences={result['sentences']:,} tokens={result['tokens']:,} "
        f"label={100 * float(result['label_accuracy']):.2f}% "
        f"upos={100 * float(result['upos_accuracy']):.2f}% "
        f"throughput={float(result['tokens_per_second']):.1f} tok/s"
    )


def limited_sentences_for_silver(corpus_text: str, limit: int | None) -> list[str]:
    sentences = nodict.split_sentences(corpus_text)
    if limit is None:
        return sentences
    selected: list[str] = []
    letter_tokens = 0
    for sentence in sentences:
        selected.append(sentence)
        letter_tokens += sum(1 for token in tokens_from_text_sentence(sentence) if has_letter(token))
        if letter_tokens >= limit * 2:
            break
    return selected


def eval_stress_lrt(
    model,
    tokenizer,
    char_vocab: dict[str, int],
    device: torch.device,
    corpus_path: Path,
    silver_path: Path,
    audit_path: Path,
    limit: int | None,
    batch_size: int,
) -> dict[str, object]:
    silver = nodict.load_silver(silver_path)
    if limit is not None:
        silver = silver[:limit]
    audit = nodict.load_audit(audit_path)
    observed_forms = nodict.observed_silver_forms(silver)
    corpus_text = corpus_path.read_text(encoding="utf-8")
    plain_rows = rows_from_plain_sentences(limited_sentences_for_silver(corpus_text, limit))
    collator = JointCollator(tokenizer, model.labels, char_vocab)
    loader = DataLoader(
        JointDataset(plain_rows),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    predictions = predict_batches(model, loader, device)
    model_tokens = []
    for row in predictions:
        for pred in row["tokens"]:
            token = SimpleToken(pred["word"])
            token.stress = pred["stress"]
            token.pos_label = pred["pos"]
            model_tokens.append(token)

    aligned, skipped_silver, skipped_model = nodict.align_tagger_tokens(silver, model_tokens)
    rows = [
        nodict.EvalRow(
            word=silver_token.word,
            silver=silver_token.accented,
            token=token,
            label=getattr(token, "pos_label", ""),
        )
        for silver_token, token in aligned
    ]
    stress_predictions = [getattr(token, "stress", None) for _silver_token, token in aligned]
    raw = nodict.score_predictions("joint", "all", rows, stress_predictions)
    audited, audit_summary = nodict.score_predictions_with_audit(
        "joint",
        "all",
        rows,
        stress_predictions,
        audit,
        observed_forms,
    )
    return {
        "silver_tokens": len(silver),
        "aligned": len(aligned),
        "skipped_silver": skipped_silver,
        "skipped_model": skipped_model,
        "raw": raw,
        "audited": audited,
        "audit_summary": audit_summary,
    }


def print_stress_result(result: dict[str, object]) -> None:
    print(
        "Stress LRT alignment: "
        f"silver={result['silver_tokens']:,} aligned={result['aligned']:,} "
        f"skipped_silver={result['skipped_silver']:,} "
        f"skipped_model={result['skipped_model']:,}"
    )
    print("Stress raw silver:")
    nodict.print_metrics([result["raw"]])
    print("Stress audited silver (reference nodict audited exact: 81.1%):")
    nodict.print_metrics([result["audited"]])
    summary = result["audit_summary"]
    print(
        "Audit overlay: "
        f"excluded={summary.excluded_tokens:,} "
        f"foreign-unmarked={summary.foreign_unmarked_tokens:,} "
        f"foreign-ok={summary.foreign_unmarked_ok:,}"
    )


def print_parameter_report(model) -> None:
    encoder_params = count_parameters(model.encoder)
    pos_params = count_parameters(model.pos_head)
    stress_params = count_parameters(model.stress_head)
    joint_params = encoder_params + pos_params + stress_params
    two_stack_params = encoder_params * 2 + pos_params + stress_params
    print(
        "Parameters: "
        f"joint={joint_params:,} "
        f"(encoder={encoder_params:,}, pos={pos_params:,}, stress={stress_params:,}); "
        f"two-model stack≈{two_stack_params:,}; "
        f"saved={two_stack_params - joint_params:,}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--silver", type=Path, default=DEFAULT_SILVER)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    args = parser.parse_args(argv)

    if not args.checkpoint.exists():
        parser.error(f"missing checkpoint: {args.checkpoint}")
    for path in (args.corpus, args.silver):
        if not path.exists():
            parser.error(f"missing stress eval input: {path}")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_allowed = cuda_visible is None or cuda_visible.strip() not in {"", "-1"}
    device = torch.device("cuda" if cuda_allowed and torch.cuda.is_available() else "cpu")
    model, tokenizer, checkpoint = instantiate_from_checkpoint(args.checkpoint, device=device)
    char_vocab = checkpoint["char_vocab"]
    print(f"device: {device}")
    print(f"checkpoint: {safe_relative(args.checkpoint)}")
    print_parameter_report(model)

    pos_results = [
        eval_pos_split(
            "matas-dev",
            args.data_dir / "dev.jsonl",
            model,
            tokenizer,
            char_vocab,
            device,
            args.limit,
            args.batch_size,
        ),
        eval_pos_split(
            "alksnis-test",
            args.data_dir / "alksnis_test.jsonl",
            model,
            tokenizer,
            char_vocab,
            device,
            args.limit,
            args.batch_size,
        ),
    ]
    print("POS reference: released taggers report 86-89% slot accuracy on ALKSNIS.")
    for result in pos_results:
        print_pos_result(result)

    if device.type == "cuda":
        gpu_tps = max(float(result.get("tokens_per_second", 0.0)) for result in pos_results)
        print(f"single-pass GPU throughput: {gpu_tps:.1f} tok/s")
    else:
        print("single-pass GPU throughput: n/a (CUDA unavailable/hidden for smoke run)")

    stress_result = eval_stress_lrt(
        model,
        tokenizer,
        char_vocab,
        device,
        args.corpus,
        args.silver,
        args.audit,
        args.limit,
        args.batch_size,
    )
    print_stress_result(stress_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
