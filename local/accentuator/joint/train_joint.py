"""Train the joint POS + stress model.

Polish an existing checkpoint with a low-LR constant schedule:
  train_joint.py --init-checkpoint checkpoints/joint_v1.pt --epochs 1 --lr-scale 0.1 --schedule constant --checkpoint checkpoints/joint_v1_polish.pt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from joint_lib import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
    ENCODER,
    LABEL_PAD_ID,
    MAX_CHARS,
    JointCollator,
    JointDataset,
    JointModel,
    batch_to_device,
    count_parameters,
    default_char_vocab,
    find_encoder_checkpoint,
    load_encoder_and_tokenizer,
    load_joint_rows,
    load_joint_checkpoint,
    load_labels,
    load_stress_char_vocab,
    pick_stress_checkpoint,
    safe_relative,
    step_schedule,
)


def load_encoder_with_warm_start() -> tuple[object, object, str, dict[str, str]]:
    warm: dict[str, str] = {}
    checkpoint = find_encoder_checkpoint()
    if checkpoint is not None:
        source = checkpoint.parent
        try:
            encoder, tokenizer, source_text = load_encoder_and_tokenizer(source)
            warm["encoder"] = safe_relative(checkpoint)
            return encoder, tokenizer, source_text, warm
        except Exception as exc:
            warm["encoder_failed"] = f"{safe_relative(checkpoint)}: {exc}"
    encoder, tokenizer, source_text = load_encoder_and_tokenizer(ENCODER)
    warm["encoder"] = f"{ENCODER} (fresh pretrained)"
    return encoder, tokenizer, source_text, warm


def warm_start_stress_head(model: JointModel) -> dict[str, object]:
    checkpoint_path = pick_stress_checkpoint()
    if checkpoint_path is None:
        return {"stress_head": "fresh", "loaded_keys": 0, "skipped_keys": 0}

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = checkpoint.get("state_dict") or {}
    current = model.stress_head.state_dict()
    matched = {}
    skipped = 0
    for key, value in source_state.items():
        if not key.startswith("head."):
            continue
        target_key = key[len("head.") :]
        if target_key in current and tuple(current[target_key].shape) == tuple(value.shape):
            matched[target_key] = value
        else:
            skipped += 1
    current.update(matched)
    model.stress_head.load_state_dict(current)
    return {
        "stress_head": safe_relative(checkpoint_path),
        "loaded_keys": len(matched),
        "skipped_keys": skipped,
    }


def best_checkpoint_path(checkpoint: Path) -> Path:
    return checkpoint.with_name(f"{checkpoint.stem}.best.pt")


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    shutil.copy2(source, tmp)
    tmp.replace(target)


def checkpoint_payload(
    model: JointModel,
    labels: list[str],
    char_vocab: dict[str, int],
    encoder_source: str,
    args: argparse.Namespace,
    warm: dict[str, object],
    epoch: int,
    global_step: int,
    dev_metrics: dict[str, int | float],
) -> dict[str, Any]:
    return {
        "model_state": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "labels": labels,
        "char_vocab": char_vocab,
        "base_model": ENCODER,
        "encoder_source": encoder_source,
        "max_chars": model.max_chars,
        "stress_weight": args.stress_weight,
        "data_dir": str(args.data_dir),
        "warm_start": warm,
        "train_args": vars(args),
        "epoch": epoch,
        "global_step": global_step,
        "dev_metrics": dev_metrics,
    }


def format_dev_metrics(metrics: dict[str, int | float]) -> str:
    return (
        f"sentences={int(metrics['sentences']):,} "
        f"pos_acc={100 * float(metrics['pos_acc']):.2f}% "
        f"({int(metrics['pos_correct']):,}/{int(metrics['pos_total']):,}) "
        f"stress_row_exact={100 * float(metrics['stress_acc']):.2f}% "
        f"({int(metrics['stress_correct']):,}/{int(metrics['stress_total']):,}) "
        f"combined={100 * float(metrics['combined']):.2f}%"
    )


@torch.no_grad()
def evaluate_dev(
    model: JointModel,
    loader: DataLoader,
    device: torch.device,
    sentence_count: int,
) -> dict[str, int | float]:
    model.eval()
    pos_correct = 0
    pos_total = 0
    stress_correct = 0
    stress_total = 0

    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch.get("token_type_ids"),
            first_subword=batch["first_subword"],
            last_subword=batch["last_subword"],
            word_mask=batch["word_mask"],
            char_ids=batch["char_ids"],
            char_valid=batch["char_valid"],
            char_mask=batch["char_mask"],
        )

        pos_gold = batch["pos_labels"]
        pos_mask = pos_gold != LABEL_PAD_ID
        pos_pred = out["pos_logits"].argmax(-1)
        pos_correct += int(((pos_pred == pos_gold) & pos_mask).sum().item())
        pos_total += int(pos_mask.sum().item())

        positions = out["stress_word_positions"]
        if positions.numel() == 0:
            continue
        stress_gold = batch["stress_targets"][positions[:, 0], positions[:, 1]]
        stress_mask = stress_gold != LABEL_PAD_ID
        if not bool(stress_mask.any()):
            continue
        stress_pred = out["stress_logits"].argmax(-1)
        stress_correct += int(((stress_pred == stress_gold) & stress_mask).sum().item())
        stress_total += int(stress_mask.sum().item())

    pos_acc = pos_correct / pos_total if pos_total else 0.0
    stress_acc = stress_correct / stress_total if stress_total else 0.0
    return {
        "sentences": sentence_count,
        "pos_correct": pos_correct,
        "pos_total": pos_total,
        "pos_acc": pos_acc,
        "stress_correct": stress_correct,
        "stress_total": stress_total,
        "stress_acc": stress_acc,
        "combined": (pos_acc + stress_acc) / 2,
    }


def vocab_mismatch_message(
    checkpoint_vocab: dict[str, int],
    dataset_vocab: dict[str, int],
) -> str:
    checkpoint_keys = set(checkpoint_vocab)
    dataset_keys = set(dataset_vocab)
    missing = sorted(dataset_keys - checkpoint_keys)[:10]
    extra = sorted(checkpoint_keys - dataset_keys)[:10]
    changed = sorted(
        key
        for key in checkpoint_keys & dataset_keys
        if checkpoint_vocab[key] != dataset_vocab[key]
    )[:10]
    parts = [
        "init checkpoint char vocab does not match the dataset char vocab",
        f"checkpoint={len(checkpoint_vocab):,}",
        f"dataset={len(dataset_vocab):,}",
    ]
    if missing:
        parts.append(f"missing={missing}")
    if extra:
        parts.append(f"extra={extra}")
    if changed:
        parts.append(f"changed_ids={changed}")
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-sentences", type=int, default=120000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    parser.add_argument("--stress-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dev-eval-sentences", type=int, default=1000)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args(argv)
    if args.lr_scale <= 0:
        parser.error("--lr-scale must be positive")
    if args.dev_eval_sentences <= 0:
        parser.error("--dev-eval-sentences must be positive")

    train_path = args.data_dir / "train.jsonl"
    dev_path = args.data_dir / "dev.jsonl"
    labels_path = args.data_dir / "labels.json"
    if not train_path.exists() or not dev_path.exists() or not labels_path.exists():
        parser.error(f"missing joint dataset under {args.data_dir}; run build_joint_dataset.py")
    if args.init_checkpoint is not None and not args.init_checkpoint.exists():
        parser.error(f"missing init checkpoint: {args.init_checkpoint}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = load_joint_rows(train_path, limit=args.max_sentences)
    dataset_labels = load_labels(labels_path)
    stress_vocab, stress_vocab_source = load_stress_char_vocab()
    dataset_char_vocab = stress_vocab or default_char_vocab(rows)

    init_checkpoint: dict[str, Any] | None = None
    if args.init_checkpoint is not None:
        init_checkpoint = load_joint_checkpoint(args.init_checkpoint, map_location="cpu")
        labels = [str(label) for label in init_checkpoint.get("labels", [])]
        if labels != dataset_labels:
            parser.error(
                "init checkpoint labels do not match labels.json "
                f"(checkpoint={len(labels):,}, dataset={len(dataset_labels):,})"
            )
        raw_char_vocab = init_checkpoint.get("char_vocab")
        if not isinstance(raw_char_vocab, dict):
            parser.error("init checkpoint is missing a char_vocab dict")
        char_vocab = {str(key): int(value) for key, value in raw_char_vocab.items()}
        if char_vocab != dataset_char_vocab:
            parser.error(vocab_mismatch_message(char_vocab, dataset_char_vocab))
        encoder_source_hint = (
            init_checkpoint.get("encoder_source")
            or init_checkpoint.get("base_model")
            or ENCODER
        )
        encoder, tokenizer, encoder_source = load_encoder_and_tokenizer(encoder_source_hint)
        model = JointModel(
            encoder=encoder,
            labels=labels,
            n_chars=max(char_vocab.values(), default=1) + 1,
            max_chars=int(init_checkpoint.get("max_chars", MAX_CHARS)),
            stress_weight=args.stress_weight,
        )
        model.load_state_dict(init_checkpoint["model_state"], strict=True)
        warm: dict[str, object] = {
            "init_checkpoint": safe_relative(args.init_checkpoint),
            "encoder": encoder_source,
            "stress_head": "loaded from init checkpoint",
            "warm_start": "skipped due to --init-checkpoint",
        }
    else:
        labels = dataset_labels
        char_vocab = dataset_char_vocab
        encoder, tokenizer, encoder_source, warm = load_encoder_with_warm_start()
        model = JointModel(
            encoder=encoder,
            labels=labels,
            n_chars=max(char_vocab.values(), default=1) + 1,
            stress_weight=args.stress_weight,
        )
        warm.update(warm_start_stress_head(model))

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cuda_allowed = cuda_visible is None or cuda_visible.strip() not in {"", "-1"}
    device = torch.device("cuda" if cuda_allowed and torch.cuda.is_available() else "cpu")
    model.to(device)
    dev_rows = load_joint_rows(dev_path, limit=args.dev_eval_sentences)
    print(f"device: {device}")
    print(f"train rows: {len(rows):,}; labels: {len(labels):,}; chars: {len(char_vocab):,}")
    print(f"dev rows: {len(dev_rows):,} (cap={args.dev_eval_sentences:,})")
    if args.init_checkpoint is not None:
        source_note = safe_relative(args.init_checkpoint)
        if stress_vocab_source is not None:
            source_note += f" (matches {safe_relative(stress_vocab_source)})"
        print(f"char vocab: {source_note}")
    elif stress_vocab_source is not None:
        print(f"char vocab: {safe_relative(stress_vocab_source)}")
    print("warm start:")
    print(json.dumps(warm, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"parameters: {count_parameters(model):,}")
    print(
        "learning rates: "
        f"encoder={args.encoder_lr * args.lr_scale:g} "
        f"head={args.head_lr * args.lr_scale:g} "
        f"schedule={args.schedule}"
    )

    dataset = JointDataset(rows)
    dev_dataset = JointDataset(dev_rows)
    collator = JointCollator(tokenizer, labels, char_vocab, max_chars=model.max_chars)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr * args.lr_scale},
            {
                "params": list(model.pos_head.parameters())
                + list(model.stress_head.parameters()),
                "lr": args.head_lr * args.lr_scale,
            },
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(loader) * args.epochs)
    warmup_steps = 50 if args.schedule == "constant" else 500
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        step_schedule(total_steps, warmup_steps=warmup_steps, mode=args.schedule),
    )

    init_dev_metrics: dict[str, int | float] | None = None
    if args.init_checkpoint is not None:
        init_dev_metrics = evaluate_dev(model, dev_loader, device, len(dev_rows))
        print(
            "init dev baseline (baseline to beat): "
            f"{format_dev_metrics(init_dev_metrics)}",
            flush=True,
        )

    latest_path = args.checkpoint
    best_path = best_checkpoint_path(args.checkpoint)
    best_score = float("-inf")
    best_epoch: int | None = None
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_total = 0.0
        running_pos = 0.0
        running_stress = 0.0
        running_items = 0
        for batch in loader:
            batch = batch_to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch.get("token_type_ids"),
                    first_subword=batch["first_subword"],
                    last_subword=batch["last_subword"],
                    word_mask=batch["word_mask"],
                    char_ids=batch["char_ids"],
                    char_valid=batch["char_valid"],
                    char_mask=batch["char_mask"],
                    pos_labels=batch["pos_labels"],
                    stress_targets=batch["stress_targets"],
                )
                loss = out["loss"]
            if loss is None:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_total += float(loss.detach().cpu())
            running_pos += float(out["pos_loss"].detach().cpu())
            running_stress += float(out["stress_loss"].detach().cpu())
            running_items += 1
            if global_step == 1 or global_step % args.log_steps == 0:
                denom = max(1, running_items)
                print(
                    f"epoch {epoch + 1}/{args.epochs} "
                    f"step {global_step}/{total_steps} "
                    f"loss={running_total / denom:.4f} "
                    f"pos={running_pos / denom:.4f} "
                    f"stress={running_stress / denom:.4f}",
                    flush=True,
                )
                running_total = running_pos = running_stress = 0.0
                running_items = 0

        dev_metrics = evaluate_dev(model, dev_loader, device, len(dev_rows))
        beat_init = ""
        if init_dev_metrics is not None:
            delta = float(dev_metrics["combined"]) - float(init_dev_metrics["combined"])
            beat_init = f" beat_init={'yes' if delta > 0 else 'no'} delta={delta:+.4f}"
        print(
            f"epoch {epoch + 1} dev: {format_dev_metrics(dev_metrics)}{beat_init}",
            flush=True,
        )

        payload = checkpoint_payload(
            model=model,
            labels=labels,
            char_vocab=char_vocab,
            encoder_source=encoder_source,
            args=args,
            warm=warm,
            epoch=epoch + 1,
            global_step=global_step,
            dev_metrics=dev_metrics,
        )
        atomic_torch_save(payload, latest_path)
        print(f"epoch {epoch + 1} checkpoint saved: {safe_relative(latest_path)}", flush=True)

        combined = float(dev_metrics["combined"])
        if combined > best_score:
            best_score = combined
            best_epoch = epoch + 1
            atomic_copy(latest_path, best_path)
            print(
                f"epoch {epoch + 1} new best checkpoint: {safe_relative(best_path)}",
                flush=True,
            )

    if best_epoch is None:
        print("no epochs ran; no checkpoint saved")
    else:
        print(
            f"best epoch: {best_epoch} combined={100 * best_score:.2f}%; "
            f"latest={safe_relative(latest_path)}; best={safe_relative(best_path)}"
        )
        if init_dev_metrics is not None:
            init_score = float(init_dev_metrics["combined"])
            delta = best_score - init_score
            print(
                "polish comparison: "
                f"best_epoch={best_epoch} combined={100 * best_score:.2f}% "
                f"vs init={100 * init_score:.2f}%; "
                f"beat_init={'yes' if delta > 0 else 'no'} delta={delta:+.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
