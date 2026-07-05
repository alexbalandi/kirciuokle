"""Neural stress guesser: litlat-bert + hierarchical char-placement head.

Same encoder as our released POS taggers (EMBEDDIA/litlat-bert). The
tokenizer's subwords do not align with stress positions, so the head is
hierarchical: one learned query per CHARACTER of the word (char embedding +
position embedding) cross-attends into the subword hidden states, and a
linear layer scores the three accent marks per character. Training is a
single masked softmax over the flattened (character, mark) grid — the mask
keeps only linguistically valid targets (vowels; non-acute sonorants as the
second element of a mixed diphthong). Softmax confidence doubles as an
abstention knob, mirroring the Anbinderis rules' leave-unstressed behavior.

Run with the CUDA training venv, from the repo root:
  .venv-train/Scripts/python.exe local/accentuator/train_stress_nn.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sqlite3
import sys
import unicodedata
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DEFAULT_GENERATED, DEFAULT_VDU_SQLITE  # noqa: E402
from train_guesser import apply_stress, load_training, load_vdu_eval, stress_of, valid_target  # noqa: E402

MARKS = ["̀", "́", "̃"]  # grave, acute, tilde
MAX_CHARS = 30
ENCODER = "EMBEDDIA/litlat-bert"
OUT_DIR = Path(__file__).resolve().parent / "data" / "stress_nn"


class StressHead(nn.Module):
    def __init__(self, hidden: int, n_chars: int):
        super().__init__()
        self.char_emb = nn.Embedding(n_chars, hidden)
        self.pos_emb = nn.Embedding(MAX_CHARS, hidden)
        self.q_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, 8, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, hidden)
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, len(MARKS))

    def forward(self, char_ids, subword_states, subword_pad_mask):
        pos = torch.arange(char_ids.size(1), device=char_ids.device)
        q = self.q_norm(self.char_emb(char_ids) + self.pos_emb(pos)[None])
        attended, _w = self.attn(
            q, subword_states, subword_states, key_padding_mask=subword_pad_mask
        )
        x = self.attn_norm(q + attended)
        x = self.ffn_norm(x + self.ffn(x))
        return self.out(x)  # (batch, chars, marks)


class StressModel(nn.Module):
    def __init__(self, encoder, n_chars: int):
        super().__init__()
        self.encoder = encoder
        self.head = StressHead(encoder.config.hidden_size, n_chars)

    def forward(self, input_ids, attention_mask, char_ids):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(char_ids, states, attention_mask == 0)


class WordDataset(Dataset):
    def __init__(self, pairs, char_vocab):
        self.items = [
            (w, p, MARKS.index(m)) for w, p, m in pairs if len(w) <= MAX_CHARS
        ]
        self.char_vocab = char_vocab

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def make_collate(tokenizer, char_vocab):
    def collate(batch):
        words = [w for w, _p, _m in batch]
        enc = tokenizer(words, padding=True, truncation=True, max_length=24, return_tensors="pt")
        n = max(len(w) for w in words)
        char_ids = torch.zeros(len(words), n, dtype=torch.long)
        valid = torch.zeros(len(words), n, len(MARKS), dtype=torch.bool)
        target = torch.zeros(len(words), dtype=torch.long)
        for i, (w, p, m) in enumerate(batch):
            for j, ch in enumerate(w):
                char_ids[i, j] = char_vocab.get(ch, 1)
                for k, mark in enumerate(MARKS):
                    valid[i, j, k] = valid_target(w, j, mark)
            target[i] = p * len(MARKS) + m
        return enc["input_ids"], enc["attention_mask"], char_ids, valid, target, words

    return collate


@torch.no_grad()
def batch_predict(model, tokenizer, char_vocab, words, device, batch_size=256):
    """Return [(form, confidence) | None per word]."""
    model.eval()
    out = []
    usable = [(w, 0, 0) for w in words]
    collate = make_collate(tokenizer, char_vocab)
    for lo in range(0, len(usable), batch_size):
        chunk = usable[lo : lo + batch_size]
        input_ids, attention_mask, char_ids, valid, _t, chunk_words = collate(chunk)
        logits = model(
            input_ids.to(device), attention_mask.to(device), char_ids.to(device)
        ).float()
        logits = logits.masked_fill(~valid.to(device), -1e9)
        flat = logits.flatten(1).softmax(-1)
        conf, idx = flat.max(-1)
        for w, c, i in zip(chunk_words, conf.tolist(), idx.tolist()):
            p, m = divmod(i, len(MARKS))
            if p >= len(w) or not valid_target(w, p, MARKS[m]):
                out.append(None)
            else:
                out.append((apply_stress(w, p, MARKS[m]), c))
    return out


def evaluate(preds, rows, label, thresholds=(0.0, 0.5, 0.7, 0.9, 0.95)):
    results = {}
    for thr in thresholds:
        answered = exact = position = 0
        for pred, (word, forms) in zip(preds, rows):
            if pred is None or pred[1] < thr:
                continue
            answered += 1
            norm = [unicodedata.normalize("NFC", f) for f in forms]
            if unicodedata.normalize("NFC", pred[0]) in norm:
                exact += 1
            gold = {(stress_of(f) or (None,))[0] for f in norm}
            if (stress_of(pred[0]) or (None,))[0] in gold:
                position += 1
        a = answered or 1
        results[thr] = (answered, exact, position)
        print(
            f"{label} @conf>={thr}: answered={answered / (len(rows) or 1):.1%} "
            f"exact={exact / a:.1%} position={position / a:.1%} (of answered)"
        )
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=None, help="Training-pair cap for smoke runs.")
    args = parser.parse_args(argv)

    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    pairs = load_training(DEFAULT_GENERATED)
    rng = random.Random(20260705)  # SAME split as the tree experiments
    rng.shuffle(pairs)
    cut = int(len(pairs) * args.holdout)
    held, train_pairs = pairs[:cut], pairs[cut:]
    if args.limit:
        train_pairs = train_pairs[: args.limit]

    chars = sorted({ch for w, _p, _m in train_pairs for ch in w})
    char_vocab = {ch: i + 2 for i, ch in enumerate(chars)}  # 0 pad, 1 unk
    print(f"train={len(train_pairs):,} held-out={len(held):,} chars={len(chars)}")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    model = StressModel(AutoModel.from_pretrained(ENCODER), len(char_vocab) + 2).to(device)

    dataset = WordDataset(train_pairs, char_vocab)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, char_vocab),
        num_workers=0,
    )
    optim = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.encoder_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )
    total_steps = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: min((s + 1) / 500, 0.5 * (1 + math.cos(math.pi * s / total_steps))),
    )

    step = 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for input_ids, attention_mask, char_ids, valid, target, _w in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(
                    input_ids.to(device), attention_mask.to(device), char_ids.to(device)
                )
                logits = logits.masked_fill(~valid.to(device), -1e9)
                loss = nn.functional.cross_entropy(logits.flatten(1).float(), target.to(device))
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
            step += 1
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {running / 200:.4f}", flush=True)
                running = 0.0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "char_vocab": char_vocab, "encoder": ENCODER},
        OUT_DIR / "stress_nn.pt",
    )
    print(f"saved {OUT_DIR / 'stress_nn.pt'}")

    held_rows = [(w, [apply_stress(w, p, m)]) for w, p, m in held if len(w) <= MAX_CHARS]
    preds = batch_predict(model, tokenizer, char_vocab, [w for w, _f in held_rows], device)
    evaluate(preds, held_rows, "in-domain held-out")

    db = sqlite3.connect(DEFAULT_GENERATED)
    covered = {w for (w,) in db.execute("SELECT word FROM words")}
    vdu_rows = [
        (w, forms) for w, forms in load_vdu_eval(DEFAULT_VDU_SQLITE, covered) if len(w) <= MAX_CHARS
    ]
    preds = batch_predict(model, tokenizer, char_vocab, [w for w, _f in vdu_rows], device)
    results = evaluate(preds, vdu_rows, "VDU-uncovered slice")
    (OUT_DIR / "eval.json").write_text(json.dumps({str(k): v for k, v in results.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
