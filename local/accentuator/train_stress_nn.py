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
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DEFAULT_GENERATED, DEFAULT_VDU_SQLITE, strip_accents  # noqa: E402
from train_guesser import apply_stress, load_training, load_vdu_eval, stress_of, valid_target  # noqa: E402

MARKS = ["̀", "́", "̃"]  # grave, acute, tilde
NO_STRESS = -1
MAX_CHARS = 30
ENCODER = "EMBEDDIA/litlat-bert"
OUT_DIR = Path(__file__).resolve().parent / "data" / "stress_nn"
OUT_DIR_V2 = Path(__file__).resolve().parent / "data" / "stress_nn2"
OUT_DIR_V3 = Path(__file__).resolve().parent / "data" / "stress_nn3"
DEFAULT_WORDLIST = Path(__file__).resolve().parent / "data" / "lt_50k.txt"
LITHUANIAN_ALPHABET = set("aąbcčdeęėfghiįyjklmnoprsštuųūvzž")


class StressHead(nn.Module):
    def __init__(self, hidden: int, n_chars: int, no_stress: bool = False):
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
        self.no_stress = nn.Linear(hidden, 1) if no_stress else None

    def representations(self, char_ids, subword_states, subword_pad_mask):
        pos = torch.arange(char_ids.size(1), device=char_ids.device)
        q = self.q_norm(self.char_emb(char_ids) + self.pos_emb(pos)[None])
        attended, _w = self.attn(
            q, subword_states, subword_states, key_padding_mask=subword_pad_mask
        )
        x = self.attn_norm(q + attended)
        return self.ffn_norm(x + self.ffn(x))

    def forward(self, char_ids, subword_states, subword_pad_mask):
        x = self.representations(char_ids, subword_states, subword_pad_mask)
        return self.out(x)  # (batch, chars, marks)

    def forward_with_no_stress(self, char_ids, subword_states, subword_pad_mask, char_mask):
        if self.no_stress is None:
            raise RuntimeError("no-stress head was not initialized")
        x = self.representations(char_ids, subword_states, subword_pad_mask)
        weights = char_mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return self.out(x), self.no_stress(pooled).squeeze(-1)


class StressModel(nn.Module):
    def __init__(self, encoder, n_chars: int, no_stress: bool = False):
        super().__init__()
        self.encoder = encoder
        self.head = StressHead(encoder.config.hidden_size, n_chars, no_stress=no_stress)

    def forward(self, input_ids, attention_mask, char_ids, include_no_stress: bool = False):
        states = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if include_no_stress:
            return self.head.forward_with_no_stress(
                char_ids, states, attention_mask == 0, char_ids != 0
            )
        return self.head(char_ids, states, attention_mask == 0)


class WordDataset(Dataset):
    def __init__(self, pairs, char_vocab, labeled: bool = False):
        if labeled:
            self.items = []
            for w, label, p, m in pairs:
                if len(w) > MAX_CHARS:
                    continue
                mark = NO_STRESS if p == NO_STRESS else MARKS.index(m)
                self.items.append((w, label, p, mark))
        else:
            self.items = [
                (w, "", p, MARKS.index(m)) for w, p, m in pairs if len(w) <= MAX_CHARS
            ]
        self.char_vocab = char_vocab

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def tokenize_words(tokenizer, words, labels=None):
    if labels is None:
        return tokenizer(words, padding=True, truncation=True, max_length=24, return_tensors="pt")
    if len(words) != len(labels):
        raise ValueError("labels length must match words length")
    labels = [label or "" for label in labels]
    if all(labels):
        return tokenizer(
            words, labels, padding=True, truncation=True, max_length=48, return_tensors="pt"
        )
    encoded = [
        tokenizer(word, label, truncation=True, max_length=48)
        if label
        else tokenizer(word, truncation=True, max_length=48)
        for word, label in zip(words, labels)
    ]
    return tokenizer.pad(encoded, padding=True, return_tensors="pt")


def make_collate(tokenizer, char_vocab, labeled: bool = False):
    def collate(batch):
        words = [w for w, _label, _p, _m in batch]
        labels = [label for _w, label, _p, _m in batch]
        enc = tokenize_words(tokenizer, words, labels if labeled else None)
        n = max(len(w) for w in words)
        char_ids = torch.zeros(len(words), n, dtype=torch.long)
        valid = torch.zeros(len(words), n, len(MARKS), dtype=torch.bool)
        target = torch.zeros(len(words), dtype=torch.long)
        for i, (w, _label, p, m) in enumerate(batch):
            for j, ch in enumerate(w):
                char_ids[i, j] = char_vocab.get(ch, 1)
                for k, mark in enumerate(MARKS):
                    valid[i, j, k] = valid_target(w, j, mark)
            target[i] = n * len(MARKS) if p == NO_STRESS else p * len(MARKS) + m
        return enc["input_ids"], enc["attention_mask"], char_ids, valid, target, words

    return collate


def _add_labeled_row(rows, seen, word: str, label: str, form: str | None) -> None:
    if not word.isalpha() or not form:
        return
    form = unicodedata.normalize("NFC", form)
    parsed = stress_of(form)
    if parsed is None:
        return
    pos, mark = parsed
    if strip_accents(form) != word or not valid_target(word, pos, mark):
        return
    key = (word, label, form)
    if key in seen:
        return
    seen.add(key)
    rows.append((word, label, pos, mark))


def load_labeled_training(path: Path) -> list[tuple[str, str, int, str]]:
    db = sqlite3.connect(path)
    rows: list[tuple[str, str, int, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for word, default_form, variants in db.execute("SELECT word, default_form, variants FROM words"):
        _add_labeled_row(rows, seen, word, "", default_form)
        for variant in json.loads(variants or "[]"):
            form = variant.get("form")
            mi = variant.get("mi") or []
            labels = mi if isinstance(mi, list) else [mi]
            if not labels and variant.get("info"):
                labels = [variant["info"]]
            for raw_label in labels:
                if raw_label is None:
                    continue
                label = str(raw_label)
                if label:
                    _add_labeled_row(rows, seen, word, label, form)
    return rows


def _add_no_stress_row(rows, seen, word: str, label: str) -> bool:
    if not word.isalpha() or len(word) > MAX_CHARS:
        return False
    key = (word, label)
    if key in seen:
        return False
    seen.add(key)
    rows.append((word, label, NO_STRESS, ""))
    return True


def _no_stress_labels_for(word: str, rows, seen) -> int:
    return int(_add_no_stress_row(rows, seen, word, "")) + int(
        _add_no_stress_row(rows, seen, word, "dkt. tikr.")
    )


def load_no_stress_rows(
    vdu_path: Path = DEFAULT_VDU_SQLITE,
    wordlist: Path = DEFAULT_WORDLIST,
) -> tuple[list[tuple[str, str, int, str]], dict[str, int]]:
    rows: list[tuple[str, str, int, str]] = []
    seen: set[tuple[str, str]] = set()
    counts = {"vdu": 0, "foreign": 0}

    db = sqlite3.connect(vdu_path)
    try:
        for word, default_form, variants in db.execute("SELECT word, default_form, variants FROM words"):
            forms = []
            if default_form:
                forms.append(str(default_form))
            for variant in json.loads(variants or "[]"):
                if isinstance(variant, dict) and variant.get("form"):
                    forms.append(str(variant["form"]))
            if all(stress_of(form) is None for form in forms):
                counts["vdu"] += _no_stress_labels_for(str(word), rows, seen)
    finally:
        db.close()

    if wordlist.exists():
        for line in wordlist.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            word = line.split()[0]
            if word.isalpha() and any(ch.lower() not in LITHUANIAN_ALPHABET for ch in word):
                counts["foreign"] += _no_stress_labels_for(word, rows, seen)
    return rows, counts


def split_labeled_rows(rows, holdout):
    by_word = defaultdict(list)
    for row in rows:
        by_word[row[0]].append(row)
    keys = list(by_word)
    rng = random.Random(20260705)
    rng.shuffle(keys)
    cut = int(len(keys) * holdout)
    held_keys = set(keys[:cut])
    held = [row for key in keys[:cut] for row in by_word[key]]
    train = [row for key in keys[cut:] for row in by_word[key] if key not in held_keys]
    return held, train


def homograph_word_count(rows) -> int:
    forms_by_word = defaultdict(set)
    for word, _label, pos, mark in rows:
        if pos == NO_STRESS:
            continue
        forms_by_word[word].add(apply_stress(word, pos, mark))
    return sum(1 for forms in forms_by_word.values() if len(forms) >= 2)


def load_vdu_labeled_eval(path: Path, covered: set[str]) -> list[tuple[str, str, str]]:
    db = sqlite3.connect(path)
    rows: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for word, variants in db.execute(
        "SELECT word, variants FROM words WHERE variants IS NOT NULL AND variants != '[]'"
    ):
        if not word.isalpha() or word in covered or len(word) > MAX_CHARS:
            continue
        for variant in json.loads(variants):
            form = variant.get("form")
            if not form:
                continue
            label = str(variant.get("info") or "")
            key = (word, label, unicodedata.normalize("NFC", form))
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    return rows


def _has_no_stress_head(model) -> bool:
    return getattr(model.head, "no_stress", None) is not None


@torch.no_grad()
def batch_predict(model, tokenizer, char_vocab, words, device, batch_size=256, labels=None):
    """Return [(form, confidence) | None per word]."""
    model.eval()
    out = []
    use_labels = labels is not None
    if labels is None:
        labels = [""] * len(words)
    elif len(labels) != len(words):
        raise ValueError("labels length must match words length")
    usable = [(w, label or "", 0, 0) for w, label in zip(words, labels)]
    collate = make_collate(tokenizer, char_vocab, labeled=use_labels)
    for lo in range(0, len(usable), batch_size):
        chunk = usable[lo : lo + batch_size]
        input_ids, attention_mask, char_ids, valid, _t, chunk_words = collate(chunk)
        include_no_stress = _has_no_stress_head(model)
        if include_no_stress:
            logits, no_stress_logits = model(
                input_ids.to(device),
                attention_mask.to(device),
                char_ids.to(device),
                include_no_stress=True,
            )
        else:
            logits = model(
                input_ids.to(device), attention_mask.to(device), char_ids.to(device)
            )
        logits = logits.masked_fill(~valid.to(device), -1e9)
        flat_logits = logits.flatten(1)
        if include_no_stress:
            flat_logits = torch.cat([flat_logits, no_stress_logits[:, None].float()], dim=1)
        flat = flat_logits.float().softmax(-1)
        conf, idx = flat.max(-1)
        no_stress_idx = logits.size(1) * len(MARKS)
        for w, c, i in zip(chunk_words, conf.tolist(), idx.tolist()):
            if include_no_stress and i == no_stress_idx:
                out.append(("", c))
                continue
            p, m = divmod(i, len(MARKS))
            if p >= len(w) or not valid_target(w, p, MARKS[m]):
                out.append(None)
            else:
                out.append((apply_stress(w, p, MARKS[m]), c))
    return out


def prediction_form(word: str, pred) -> str | None:
    if pred is None:
        return None
    return word if pred[0] == "" else pred[0]


def evaluate(preds, rows, label, thresholds=(0.0, 0.5, 0.7, 0.9, 0.95)):
    results = {}
    for thr in thresholds:
        answered = exact = position = 0
        for pred, (word, forms) in zip(preds, rows):
            if pred is None or pred[1] < thr:
                continue
            answered += 1
            form = prediction_form(word, pred)
            norm = [unicodedata.normalize("NFC", f) for f in forms]
            if form is not None and unicodedata.normalize("NFC", form) in norm:
                exact += 1
            gold = {(stress_of(f) or (None,))[0] for f in norm}
            if form is not None and (stress_of(form) or (None,))[0] in gold:
                position += 1
        a = answered or 1
        results[thr] = (answered, exact, position)
        print(
            f"{label} @conf>={thr}: answered={answered / (len(rows) or 1):.1%} "
            f"exact={exact / a:.1%} position={position / a:.1%} (of answered)"
        )
    return results


def evaluate_homograph_switch(model, tokenizer, char_vocab, held, device):
    grouped = defaultdict(list)
    for word, label, pos, mark in held:
        if pos != NO_STRESS and len(word) <= MAX_CHARS:
            grouped[word].append((label, apply_stress(word, pos, mark)))
    homographs = {
        word: rows for word, rows in grouped.items() if len({form for _label, form in rows}) >= 2
    }
    eval_rows = [
        (word, label, form)
        for word, rows in homographs.items()
        for label, form in rows
    ]
    preds = batch_predict(
        model,
        tokenizer,
        char_vocab,
        [word for word, _label, _form in eval_rows],
        device,
        labels=[label for _word, label, _form in eval_rows],
    )
    word_ok = {word: True for word in homographs}
    answered = exact = 0
    for pred, (word, _label, form) in zip(preds, eval_rows):
        predicted = prediction_form(word, pred)
        ok = predicted is not None and unicodedata.normalize("NFC", predicted) == form
        if pred is not None:
            answered += 1
        if ok:
            exact += 1
        word_ok[word] = word_ok[word] and ok
    n_words = len(homographs)
    n_rows = len(eval_rows)
    correct_words = sum(1 for ok in word_ok.values() if ok)
    print(
        f"homograph switch: words={n_words:,} rows={n_rows:,} "
        f"answered={answered / (n_rows or 1):.1%} row-exact={exact / (n_rows or 1):.1%} "
        f"word-all-exact={correct_words / (n_words or 1):.1%}"
    )
    return {"words": n_words, "rows": n_rows, "correct_words": correct_words, "row_exact": exact}


def evaluate_unconditioned_regression(model, tokenizer, char_vocab, held, device):
    rows = []
    seen = set()
    for word, label, pos, mark in held:
        if pos == NO_STRESS or label or len(word) > MAX_CHARS or word in seen:
            continue
        seen.add(word)
        rows.append((word, [apply_stress(word, pos, mark)]))
    preds = batch_predict(
        model,
        tokenizer,
        char_vocab,
        [word for word, _forms in rows],
        device,
        labels=[""] * len(rows),
    )
    answered = exact = position = 0
    for pred, (word, forms) in zip(preds, rows):
        if pred is None:
            continue
        answered += 1
        form = prediction_form(word, pred)
        norm = [unicodedata.normalize("NFC", form) for form in forms]
        if form is not None and unicodedata.normalize("NFC", form) in norm:
            exact += 1
        gold = {(stress_of(form) or (None,))[0] for form in norm}
        if form is not None and (stress_of(form) or (None,))[0] in gold:
            position += 1
    n = len(rows)
    a = answered or 1
    print(
        f"unconditioned regression: n={n:,} answered={answered / (n or 1):.1%} "
        f"exact={exact / a:.1%} position={position / a:.1%} (of answered; "
        f"over all exact={exact / (n or 1):.1%}; v1 known exact=97.9%)"
    )
    return {"n": n, "answered": answered, "exact": exact, "position": position}


def evaluate_vdu_labeled(model, tokenizer, char_vocab, vdu_variant_rows, device):
    preds = batch_predict(
        model,
        tokenizer,
        char_vocab,
        [word for word, _label, _form in vdu_variant_rows],
        device,
        labels=[label for _word, label, _form in vdu_variant_rows],
    )
    answered = exact = 0
    for pred, (_word, _label, form) in zip(preds, vdu_variant_rows):
        if pred is None:
            continue
        answered += 1
        predicted = prediction_form(_word, pred)
        if predicted is not None and unicodedata.normalize("NFC", predicted) == form:
            exact += 1
    n = len(vdu_variant_rows)
    print(
        f"VDU gap slice (labeled): variants={n:,} answered={answered / (n or 1):.1%} "
        f"exact={exact / (answered or 1):.1%} (of answered; over all exact={exact / (n or 1):.1%})"
    )
    return {"variants": n, "answered": answered, "exact": exact}


def evaluate_no_stress_heldout(model, tokenizer, char_vocab, held, device):
    rows = [(word, label) for word, label, pos, _mark in held if pos == NO_STRESS]
    preds = batch_predict(
        model,
        tokenizer,
        char_vocab,
        [word for word, _label in rows],
        device,
        labels=[label for _word, label in rows],
    )
    predicted_no_stress = wrongly_accented = 0
    for pred in preds:
        if pred is None:
            continue
        if pred[0] == "":
            predicted_no_stress += 1
        else:
            wrongly_accented += 1
    n = len(rows)
    word_count = len({word for word, _label in rows})
    print(
        f"no-stress held-out: rows={n:,} words={word_count:,} "
        f"predicted-no-stress={predicted_no_stress / (n or 1):.1%} "
        f"wrongly-accented={wrongly_accented / (n or 1):.1%}"
    )
    return {
        "rows": n,
        "words": word_count,
        "predicted_no_stress": predicted_no_stress,
        "wrongly_accented": wrongly_accented,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--encoder-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--holdout", type=float, default=0.02)
    parser.add_argument("--limit", type=int, default=None, help="Training-pair cap for smoke runs.")
    parser.add_argument("--labels", action="store_true", help="Train on (word, morphology-label, form) rows.")
    parser.add_argument("--v3", action="store_true", help="Train the learned no-stress v3 model.")
    parser.add_argument(
        "--no-stress-rows",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include learned no-stress rows when training v3 labels.",
    )
    args = parser.parse_args(argv)
    if args.v3 and not args.labels:
        parser.error("--v3 requires --labels")
    if args.epochs is None:
        args.epochs = 4 if args.v3 else 3
    use_no_stress_rows = args.v3 and args.labels and (
        args.no_stress_rows if args.no_stress_rows is not None else True
    )

    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    if args.labels:
        pairs = load_labeled_training(DEFAULT_GENERATED)
        if use_no_stress_rows:
            no_stress_pairs, no_stress_counts = load_no_stress_rows()
            pairs.extend(no_stress_pairs)
            print(
                f"no-stress rows: {len(no_stress_pairs):,} "
                f"(VDU-cache={no_stress_counts['vdu']:,}, "
                f"foreign-lt_50k={no_stress_counts['foreign']:,})"
            )
        words = {word for word, _label, _pos, _mark in pairs}
        print(
            f"labeled dataset: rows={len(pairs):,} words={len(words):,} "
            f"homograph words={homograph_word_count(pairs):,}"
        )
        held, train_pairs = split_labeled_rows(pairs, args.holdout)
        if args.limit:
            train_pairs = train_pairs[: args.limit]
        chars = sorted({ch for w, _label, _p, _m in train_pairs for ch in w})
    else:
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
    model = StressModel(
        AutoModel.from_pretrained(ENCODER),
        len(char_vocab) + 2,
        no_stress=args.v3,
    ).to(device)

    dataset = WordDataset(train_pairs, char_vocab, labeled=args.labels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer, char_vocab, labeled=args.labels),
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
                if args.v3:
                    logits, no_stress_logits = model(
                        input_ids.to(device),
                        attention_mask.to(device),
                        char_ids.to(device),
                        include_no_stress=True,
                    )
                    logits = logits.masked_fill(~valid.to(device), -1e9)
                    flat_logits = torch.cat(
                        [logits.flatten(1), no_stress_logits[:, None].float()], dim=1
                    )
                    loss = nn.functional.cross_entropy(flat_logits.float(), target.to(device))
                else:
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

    if args.v3:
        out_dir = OUT_DIR_V3
        ckpt_name = "stress_nn3.pt"
    else:
        out_dir = OUT_DIR_V2 if args.labels else OUT_DIR
        ckpt_name = "stress_nn2.pt" if args.labels else "stress_nn.pt"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {"state_dict": model.state_dict(), "char_vocab": char_vocab, "encoder": ENCODER}
    if args.labels:
        ckpt["labeled"] = True
    if args.v3:
        ckpt["no_stress"] = True
    torch.save(
        ckpt,
        out_dir / ckpt_name,
    )
    print(f"saved {out_dir / ckpt_name}")

    db = sqlite3.connect(DEFAULT_GENERATED)
    covered = {w for (w,) in db.execute("SELECT word FROM words")}
    if args.labels:
        held_rows = [
            (w, [apply_stress(w, p, m)])
            for w, _label, p, m in held
            if p != NO_STRESS and len(w) <= MAX_CHARS
        ]
        held_labels = [
            label for w, label, p, _m in held if p != NO_STRESS and len(w) <= MAX_CHARS
        ]
        preds = batch_predict(
            model,
            tokenizer,
            char_vocab,
            [w for w, _forms in held_rows],
            device,
            labels=held_labels,
        )
        in_domain = evaluate(preds, held_rows, "in-domain held-out (labeled)", thresholds=(0.0, 0.9))
        switch = evaluate_homograph_switch(model, tokenizer, char_vocab, held, device)
        regression = evaluate_unconditioned_regression(model, tokenizer, char_vocab, held, device)
        no_stress_heldout = (
            evaluate_no_stress_heldout(model, tokenizer, char_vocab, held, device)
            if args.v3
            else None
        )
        vdu_rows = [
            (w, forms)
            for w, forms in load_vdu_eval(DEFAULT_VDU_SQLITE, covered)
            if len(w) <= MAX_CHARS
        ]
        preds = batch_predict(
            model,
            tokenizer,
            char_vocab,
            [w for w, _forms in vdu_rows],
            device,
            labels=[""] * len(vdu_rows),
        )
        vdu_unconditioned = evaluate(
            preds, vdu_rows, "VDU gap slice (unconditioned fallback)", thresholds=(0.0, 0.9)
        )
        vdu_labeled = evaluate_vdu_labeled(
            model, tokenizer, char_vocab, load_vdu_labeled_eval(DEFAULT_VDU_SQLITE, covered), device
        )
        (out_dir / "eval.json").write_text(
            json.dumps(
                {
                    "in_domain_labeled": {str(k): v for k, v in in_domain.items()},
                    "homograph_switch": switch,
                    "unconditioned_regression": regression,
                    "no_stress_heldout": no_stress_heldout,
                    "vdu_unconditioned": {str(k): v for k, v in vdu_unconditioned.items()},
                    "vdu_labeled": vdu_labeled,
                },
                ensure_ascii=False,
            )
        )
    else:
        held_rows = [(w, [apply_stress(w, p, m)]) for w, p, m in held if len(w) <= MAX_CHARS]
        preds = batch_predict(model, tokenizer, char_vocab, [w for w, _f in held_rows], device)
        evaluate(preds, held_rows, "in-domain held-out")

        vdu_rows = [
            (w, forms)
            for w, forms in load_vdu_eval(DEFAULT_VDU_SQLITE, covered)
            if len(w) <= MAX_CHARS
        ]
        preds = batch_predict(model, tokenizer, char_vocab, [w for w, _f in vdu_rows], device)
        results = evaluate(preds, vdu_rows, "VDU-uncovered slice")
        (OUT_DIR / "eval.json").write_text(json.dumps({str(k): v for k, v in results.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
