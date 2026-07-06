from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from joint_lib import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
    ENCODER,
    JointCollator,
    JointDataset,
    JointModel,
    batch_to_device,
    count_parameters,
    default_char_vocab,
    find_encoder_checkpoint,
    load_encoder_and_tokenizer,
    load_joint_rows,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--max-sentences", type=int, default=120000)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--stress-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260705)
    args = parser.parse_args(argv)

    train_path = args.data_dir / "train.jsonl"
    labels_path = args.data_dir / "labels.json"
    if not train_path.exists() or not labels_path.exists():
        parser.error(f"missing joint dataset under {args.data_dir}; run build_joint_dataset.py")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    rows = load_joint_rows(train_path, limit=args.max_sentences)
    labels = load_labels(labels_path)
    stress_vocab, stress_vocab_source = load_stress_char_vocab()
    char_vocab = stress_vocab or default_char_vocab(rows)

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
    print(f"device: {device}")
    print(f"train rows: {len(rows):,}; labels: {len(labels):,}; chars: {len(char_vocab):,}")
    if stress_vocab_source is not None:
        print(f"char vocab: {safe_relative(stress_vocab_source)}")
    print("warm start:")
    print(json.dumps(warm, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"parameters: {count_parameters(model):,}")

    dataset = JointDataset(rows)
    collator = JointCollator(tokenizer, labels, char_vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {
                "params": list(model.pos_head.parameters())
                + list(model.stress_head.parameters()),
                "lr": args.head_lr,
            },
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(loader) * args.epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        step_schedule(total_steps),
    )

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

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.cpu().state_dict(),
            "labels": labels,
            "char_vocab": char_vocab,
            "base_model": ENCODER,
            "encoder_source": encoder_source,
            "max_chars": model.max_chars,
            "stress_weight": args.stress_weight,
            "data_dir": str(args.data_dir),
            "warm_start": warm,
            "train_args": vars(args),
        },
        args.checkpoint,
    )
    print(f"saved checkpoint: {safe_relative(args.checkpoint)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
