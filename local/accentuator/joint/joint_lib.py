from __future__ import annotations

import json
import math
import random
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


JOINT_DIR = Path(__file__).resolve().parent
ACCENTUATOR_DIR = JOINT_DIR.parent
LOCAL_DIR = ACCENTUATOR_DIR.parent
REPO_ROOT = LOCAL_DIR.parent
TAGGER_DIR = LOCAL_DIR / "tagger-hf"
APP_DIR = LOCAL_DIR / "app"

for path in (ACCENTUATOR_DIR, TAGGER_DIR, APP_DIR):
    sys.path.insert(0, str(path))

from _common import DEFAULT_GENERATED, normalize_lt, safe_relative, strip_accents  # noqa: E402
from kirciuokle import disambiguate as disamb  # noqa: E402
from train_guesser import apply_stress, stress_of, valid_target  # noqa: E402


ENCODER = "EMBEDDIA/litlat-bert"
DEFAULT_SOURCE_DATA_DIR = TAGGER_DIR / "data" / "gen2"
DEFAULT_DATA_DIR = JOINT_DIR / "data"
DEFAULT_CHECKPOINT = JOINT_DIR / "checkpoints" / "joint_v1.pt"
DEFAULT_STRESS_NN3 = ACCENTUATOR_DIR / "data" / "stress_nn3" / "stress_nn3.pt"
DEFAULT_STRESS_NN2 = ACCENTUATOR_DIR / "data" / "stress_nn2" / "stress_nn2.pt"
MARKS = ["\u0300", "\u0301", "\u0303"]
MARK_TO_ID = {mark: index for index, mark in enumerate(MARKS)}
MAX_CHARS = 30
MAX_SUBWORDS = 128
LABEL_PAD_ID = -100
LITHUANIAN_ALPHABET = set("aąbcčdeęėfghiįyjklmnoprsštuųūvzž")
WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class DictionaryEntry:
    variants: list[dict[str, Any]]
    default_form: str | None


@dataclass
class ProjectionStats:
    tokens: int = 0
    supervised: int = 0
    stressed: int = 0
    no_stress: int = 0
    masked: int = 0
    dictionary_hits: int = 0
    positive_unique: int = 0
    positive_same_stress_ties: int = 0
    positive_different_stress_ties: int = 0
    nonpositive: int = 0
    foreign_no_stress: int = 0
    invalid_or_long: int = 0
    homograph_tokens: int = 0
    homograph_resolved: int = 0

    def update_supervision(self, stress: list[Any] | str | None) -> None:
        self.tokens += 1
        if stress is None:
            self.masked += 1
            return
        self.supervised += 1
        if stress == "none":
            self.no_stress += 1
        else:
            self.stressed += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "tokens": self.tokens,
            "supervised": self.supervised,
            "stress_supervision_share": self.supervised / self.tokens if self.tokens else 0.0,
            "stressed_targets": self.stressed,
            "no_stress_targets": self.no_stress,
            "masked_tokens": self.masked,
            "dictionary_hits": self.dictionary_hits,
            "positive_unique": self.positive_unique,
            "positive_same_stress_ties": self.positive_same_stress_ties,
            "positive_different_stress_ties": self.positive_different_stress_ties,
            "nonpositive_or_missing": self.nonpositive,
            "foreign_no_stress": self.foreign_no_stress,
            "invalid_or_long_stress": self.invalid_or_long,
            "homograph_tokens": self.homograph_tokens,
            "homograph_resolved": self.homograph_resolved,
            "homograph_resolved_share": (
                self.homograph_resolved / self.homograph_tokens
                if self.homograph_tokens
                else 0.0
            ),
        }


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def load_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(label) for label in payload["labels"]]


def write_labels(path: Path, labels: Iterable[str]) -> list[str]:
    label_list = sorted(set(str(label) for label in labels))
    write_json(
        path,
        {
            "labels": label_list,
            "label2id": {label: index for index, label in enumerate(label_list)},
            "id2label": {str(index): label for index, label in enumerate(label_list)},
        },
    )
    return label_list


def parse_combined_label(label: str) -> tuple[str, dict[str, str]]:
    parts = str(label).split("|")
    upos = parts[0] if parts and parts[0] else "X"
    feats: dict[str, str] = {}
    for item in parts[1:]:
        if not item or item == "_" or "=" not in item:
            continue
        key, value = item.split("=", 1)
        if key and value:
            feats[key] = value
    return upos, feats


def token_from_label(word: str, label: str) -> Any:
    upos, feats = parse_combined_label(label)
    return disamb.Token(form=word, lemma=word.casefold(), upos=upos, xpos="_", feats=feats)


def variant_labels(variant: dict[str, Any]) -> list[str]:
    raw_mi = variant.get("mi")
    labels: list[str] = []
    if isinstance(raw_mi, list):
        labels.extend(str(item).strip() for item in raw_mi if str(item).strip())
    elif raw_mi:
        labels.append(str(raw_mi).strip())
    if not labels and variant.get("info"):
        labels.append(str(variant["info"]).strip())
    return [label for label in labels if label]


def word_key(text: str | None) -> str:
    return strip_accents(normalize_lt(text or "")).casefold()


def is_alpha_key(text: str) -> bool:
    return bool(text) and text.isalpha()


def has_letter(text: str) -> bool:
    return any(unicodedata.category(ch).startswith("L") for ch in text)


def is_foreign_letter_token(text: str) -> bool:
    letters = [ch.casefold() for ch in word_key(text) if ch.isalpha()]
    return bool(letters) and any(ch not in LITHUANIAN_ALPHABET for ch in letters)


def stress_target_for_form(word: str, form: str | None) -> list[Any] | str | None:
    if not form:
        return None
    normalized = normalize_lt(form).casefold()
    if word_key(normalized) != word:
        return None
    parsed = stress_of(normalized)
    if parsed is None:
        return "none"
    pos, mark = parsed
    if pos >= MAX_CHARS:
        return None
    if mark not in MARK_TO_ID or not valid_target(word, pos, mark):
        return None
    return [int(pos), mark]


def target_key(target: list[Any] | str | None) -> tuple[int, str]:
    if target == "none":
        return (-1, "")
    if isinstance(target, list) and len(target) == 2:
        return (int(target[0]), str(target[1]))
    return (-999, "")


def entry_stress_options(word: str, entry: DictionaryEntry | None) -> set[tuple[int, str]]:
    if entry is None:
        return set()
    options: set[tuple[int, str]] = set()
    for variant in entry.variants:
        target = stress_target_for_form(word, variant.get("form"))
        if target is not None:
            options.add(target_key(target))
    return options


def score_variant(
    variant: dict[str, Any],
    context_tags: dict[str, str],
    mi_cache: dict[str, dict[str, str]],
) -> int:
    labels = variant_labels(variant)
    if not labels:
        return 0
    best: int | None = None
    for label in labels:
        if label not in mi_cache:
            mi_cache[label] = disamb.parse_mi(label)
        score = disamb.score_tags(mi_cache[label], context_tags)
        best = score if best is None else max(best, score)
    return int(best or 0)


def project_stress(
    word: str,
    pos_label: str,
    entry: DictionaryEntry | None,
    stats: ProjectionStats,
    mi_cache: dict[str, dict[str, str]],
) -> list[Any] | str | None:
    key = word_key(word)
    if is_foreign_letter_token(word):
        stats.foreign_no_stress += 1
        return "none"
    if not is_alpha_key(key):
        return None
    if entry is None:
        stats.nonpositive += 1
        return None

    stats.dictionary_hits += 1
    options = entry_stress_options(key, entry)
    is_homograph = len(options) > 1
    if is_homograph:
        stats.homograph_tokens += 1

    context_tags = disamb.token_tags(token_from_label(word, pos_label))
    scored: list[tuple[int, dict[str, Any], list[Any] | str]] = []
    for variant in entry.variants:
        target = stress_target_for_form(key, variant.get("form"))
        if target is None:
            continue
        scored.append((score_variant(variant, context_tags, mi_cache), variant, target))

    if not scored:
        stats.invalid_or_long += 1
        return None

    best_score = max(score for score, _variant, _target in scored)
    if best_score <= 0:
        stats.nonpositive += 1
        return None

    top = [(variant, target) for score, variant, target in scored if score == best_score]
    top_targets = {target_key(target) for _variant, target in top}
    if len(top_targets) > 1:
        stats.positive_different_stress_ties += 1
        return None

    target = top[0][1]
    if len(top) == 1:
        stats.positive_unique += 1
    else:
        stats.positive_same_stress_ties += 1
    if is_homograph:
        stats.homograph_resolved += 1
    return target


def collect_word_keys(rows: Iterable[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in rows:
        tokens = row.get("tokens") or []
        if tokens and isinstance(tokens[0], dict):
            words = [str(token.get("word") or "") for token in tokens]
        else:
            words = [str(token) for token in tokens]
        for word in words:
            key = word_key(word)
            if is_alpha_key(key):
                keys.add(key)
    return keys


def load_dictionary(path: Path, target_words: set[str]) -> dict[str, DictionaryEntry]:
    entries: dict[str, DictionaryEntry] = {}
    if not target_words:
        return entries
    db = sqlite3.connect(path)
    try:
        words = sorted(target_words)
        for offset in range(0, len(words), 800):
            chunk = words[offset : offset + 800]
            placeholders = ",".join("?" for _ in chunk)
            query = (
                "SELECT word, variants, default_form FROM words "
                f"WHERE word IN ({placeholders})"
            )
            for word, variants_json, default_form in db.execute(query, chunk):
                try:
                    variants = json.loads(variants_json or "[]")
                except json.JSONDecodeError:
                    variants = []
                entries[str(word)] = DictionaryEntry(
                    variants=variants if isinstance(variants, list) else [],
                    default_form=str(default_form) if default_form else None,
                )
    finally:
        db.close()
    return entries


def source_row_to_joint(
    row: dict[str, Any],
    entries: dict[str, DictionaryEntry],
    stats: ProjectionStats,
    mi_cache: dict[str, dict[str, str]],
    source_name: str,
) -> dict[str, Any]:
    words = [str(word) for word in row["tokens"]]
    labels = [str(label) for label in row["labels"]]
    tokens: list[dict[str, Any]] = []
    for word, label in zip(words, labels):
        key = word_key(word)
        stress = project_stress(word, label, entries.get(key), stats, mi_cache)
        stats.update_supervision(stress)
        tokens.append({"word": word, "pos_label": label, "stress": stress})
    return {
        "id": row.get("id"),
        "source": source_name,
        "text": row.get("text") or " ".join(words),
        "tokens": tokens,
    }


def load_joint_rows(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    return read_jsonl(path, limit=limit)


def labels_from_joint(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [
        str(token["pos_label"])
        for row in rows
        for token in row.get("tokens", [])
        if isinstance(token, dict)
    ]


def default_char_vocab(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    chars = sorted(
        {
            ch
            for row in rows
            for token in row.get("tokens", [])
            for ch in word_key(str(token.get("word") or ""))
        }
    )
    return {ch: index + 2 for index, ch in enumerate(chars)}


def pick_stress_checkpoint() -> Path | None:
    for path in (DEFAULT_STRESS_NN3, DEFAULT_STRESS_NN2):
        if path.exists():
            return path
    return None


def load_stress_char_vocab() -> tuple[dict[str, int] | None, Path | None]:
    checkpoint = pick_stress_checkpoint()
    if checkpoint is None:
        return None, None
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    raw_vocab = payload.get("char_vocab")
    if not isinstance(raw_vocab, dict):
        return None, checkpoint
    return {str(key): int(value) for key, value in raw_vocab.items()}, checkpoint


class JointDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class JointCollator:
    def __init__(
        self,
        tokenizer: Any,
        labels: list[str],
        char_vocab: dict[str, int],
        max_subwords: int = MAX_SUBWORDS,
        max_chars: int = MAX_CHARS,
    ) -> None:
        self.tokenizer = tokenizer
        self.labels = labels
        self.label2id = {label: index for index, label in enumerate(labels)}
        self.char_vocab = char_vocab
        self.max_subwords = max_subwords
        self.max_chars = max_chars

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        sentence_words = [
            [str(token["word"]) for token in row.get("tokens", [])] for row in rows
        ]
        encoded = self.tokenizer(
            sentence_words,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=self.max_subwords,
            return_tensors="pt",
        )
        batch_size = len(rows)
        max_words = max((len(words) for words in sentence_words), default=0)
        pos_labels = torch.full((batch_size, max_words), LABEL_PAD_ID, dtype=torch.long)
        stress_targets = torch.full((batch_size, max_words), LABEL_PAD_ID, dtype=torch.long)
        first_subword = torch.full((batch_size, max_words), -1, dtype=torch.long)
        last_subword = torch.full((batch_size, max_words), -1, dtype=torch.long)
        word_mask = torch.zeros((batch_size, max_words), dtype=torch.bool)
        char_ids = torch.zeros((batch_size, max_words, self.max_chars), dtype=torch.long)
        char_valid = torch.zeros(
            (batch_size, max_words, self.max_chars, len(MARKS)),
            dtype=torch.bool,
        )
        char_mask = torch.zeros((batch_size, max_words, self.max_chars), dtype=torch.bool)

        for batch_index, row in enumerate(rows):
            tokens = row.get("tokens", [])
            word_ids = list(encoded.word_ids(batch_index=batch_index))
            first = [-1] * len(tokens)
            last = [-1] * len(tokens)
            for subword_index, word_id in enumerate(word_ids):
                if word_id is None or word_id < 0 or word_id >= len(tokens):
                    continue
                if first[word_id] == -1:
                    first[word_id] = subword_index
                last[word_id] = subword_index
            for word_index, token in enumerate(tokens):
                if first[word_index] == -1:
                    continue
                word_mask[batch_index, word_index] = True
                first_subword[batch_index, word_index] = first[word_index]
                last_subword[batch_index, word_index] = last[word_index]
                label = str(token["pos_label"])
                pos_labels[batch_index, word_index] = self.label2id.get(
                    label,
                    LABEL_PAD_ID,
                )
                chars = word_key(str(token["word"]))
                for char_index, ch in enumerate(chars[: self.max_chars]):
                    char_mask[batch_index, word_index, char_index] = True
                    char_ids[batch_index, word_index, char_index] = self.char_vocab.get(ch, 1)
                    for mark_index, mark in enumerate(MARKS):
                        char_valid[batch_index, word_index, char_index, mark_index] = (
                            valid_target(chars, char_index, mark)
                        )
                target = token.get("stress")
                if target is None:
                    continue
                if target == "none":
                    stress_targets[batch_index, word_index] = self.max_chars * len(MARKS)
                elif isinstance(target, list) and len(target) == 2:
                    pos = int(target[0])
                    mark = str(target[1])
                    if 0 <= pos < self.max_chars and mark in MARK_TO_ID:
                        stress_targets[batch_index, word_index] = (
                            pos * len(MARKS) + MARK_TO_ID[mark]
                        )

        batch: dict[str, Any] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "pos_labels": pos_labels,
            "stress_targets": stress_targets,
            "first_subword": first_subword,
            "last_subword": last_subword,
            "word_mask": word_mask,
            "char_ids": char_ids,
            "char_valid": char_valid,
            "char_mask": char_mask,
            "rows": rows,
        }
        if "token_type_ids" in encoded:
            batch["token_type_ids"] = encoded["token_type_ids"]
        return batch


class StressHead(nn.Module):
    def __init__(self, hidden: int, n_chars: int, max_chars: int = MAX_CHARS) -> None:
        super().__init__()
        self.max_chars = max_chars
        self.char_emb = nn.Embedding(n_chars, hidden)
        self.pos_emb = nn.Embedding(max_chars, hidden)
        self.q_norm = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, 8, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.ffn_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, len(MARKS))
        self.no_stress = nn.Linear(hidden, 1)

    def representations(
        self,
        char_ids: torch.Tensor,
        subword_states: torch.Tensor,
        subword_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        pos = torch.arange(char_ids.size(1), device=char_ids.device)
        q = self.q_norm(self.char_emb(char_ids) + self.pos_emb(pos)[None])
        attended, _weights = self.attn(
            q,
            subword_states,
            subword_states,
            key_padding_mask=subword_pad_mask,
        )
        x = self.attn_norm(q + attended)
        return self.ffn_norm(x + self.ffn(x))

    def forward(
        self,
        char_ids: torch.Tensor,
        subword_states: torch.Tensor,
        subword_pad_mask: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.representations(char_ids, subword_states, subword_pad_mask)
        weights = char_mask.to(x.dtype).unsqueeze(-1)
        pooled = (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        return self.out(x), self.no_stress(pooled).squeeze(-1)


class JointModel(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        labels: list[str],
        n_chars: int,
        max_chars: int = MAX_CHARS,
        stress_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.labels = labels
        self.max_chars = max_chars
        self.stress_weight = float(stress_weight)
        hidden = int(getattr(encoder.config, "hidden_size"))
        self.pos_head = nn.Linear(hidden, len(labels))
        self.stress_head = StressHead(hidden, n_chars, max_chars=max_chars)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        first_subword: torch.Tensor,
        last_subword: torch.Tensor,
        word_mask: torch.Tensor,
        char_ids: torch.Tensor,
        char_valid: torch.Tensor,
        char_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        pos_labels: torch.Tensor | None = None,
        stress_targets: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        hidden = self.encoder(**kwargs, return_dict=True).last_hidden_state
        word_reps = gather_word_reps(hidden, first_subword)
        pos_logits = self.pos_head(word_reps)

        flat_mask = word_mask & (first_subword >= 0) & (last_subword >= first_subword)
        flat_positions = flat_mask.nonzero(as_tuple=False)
        stress_logits: torch.Tensor
        stress_word_positions = flat_positions
        if flat_positions.numel() == 0:
            stress_logits = hidden.new_empty((0, self.max_chars * len(MARKS) + 1))
        else:
            subword_states, subword_pad_mask = gather_subword_spans(
                hidden,
                flat_positions,
                first_subword,
                last_subword,
            )
            flat_char_ids = char_ids[flat_mask]
            flat_char_mask = char_mask[flat_mask]
            char_logits, no_stress_logits = self.stress_head(
                flat_char_ids,
                subword_states,
                subword_pad_mask,
                flat_char_mask,
            )
            valid = char_valid[flat_mask]
            char_logits = char_logits.masked_fill(~valid, -1e9)
            stress_logits = torch.cat(
                [char_logits.flatten(1), no_stress_logits[:, None].float()],
                dim=1,
            )

        losses: dict[str, torch.Tensor] = {}
        total_loss: torch.Tensor | None = None
        if pos_labels is not None:
            flat_pos_labels = pos_labels.reshape(-1)
            if torch.any(flat_pos_labels != LABEL_PAD_ID):
                pos_loss = F.cross_entropy(
                    pos_logits.reshape(-1, pos_logits.shape[-1]),
                    flat_pos_labels,
                    ignore_index=LABEL_PAD_ID,
                )
            else:
                pos_loss = pos_logits.sum() * 0.0
            losses["pos_loss"] = pos_loss
            total_loss = pos_loss
        if stress_targets is not None:
            if stress_logits.numel() == 0:
                stress_loss = pos_logits.sum() * 0.0
            else:
                flat_targets = stress_targets[flat_mask]
                if torch.any(flat_targets != LABEL_PAD_ID):
                    stress_loss = F.cross_entropy(
                        stress_logits.float(),
                        flat_targets,
                        ignore_index=LABEL_PAD_ID,
                    )
                else:
                    stress_loss = stress_logits.sum() * 0.0
            losses["stress_loss"] = stress_loss
            total_loss = (
                stress_loss * self.stress_weight
                if total_loss is None
                else total_loss + stress_loss * self.stress_weight
            )

        return {
            "loss": total_loss,
            "pos_logits": pos_logits,
            "stress_logits": stress_logits,
            "stress_word_positions": stress_word_positions,
            **losses,
        }


def gather_word_reps(hidden: torch.Tensor, first_subword: torch.Tensor) -> torch.Tensor:
    safe = first_subword.clamp(min=0, max=hidden.shape[1] - 1)
    expanded = safe.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
    return torch.gather(hidden, 1, expanded)


def gather_subword_spans(
    hidden: torch.Tensor,
    flat_positions: torch.Tensor,
    first_subword: torch.Tensor,
    last_subword: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    sequences: list[torch.Tensor] = []
    lengths: list[int] = []
    for batch_index, word_index in flat_positions.tolist():
        start = int(first_subword[batch_index, word_index].item())
        end = int(last_subword[batch_index, word_index].item())
        seq = hidden[batch_index, start : end + 1]
        sequences.append(seq)
        lengths.append(max(1, end - start + 1))
    padded = pad_sequence(sequences, batch_first=True)
    max_len = padded.shape[1]
    lens = torch.tensor(lengths, device=hidden.device)
    arange = torch.arange(max_len, device=hidden.device)[None, :]
    pad_mask = arange >= lens[:, None]
    return padded, pad_mask


def batch_to_device(batch: dict[str, Any], device: torch.device | str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def decode_stress_prediction(word: str, pred_id: int, max_chars: int = MAX_CHARS) -> str | None:
    no_stress_id = max_chars * len(MARKS)
    key = word_key(word)
    if pred_id == no_stress_id:
        return ""
    pos, mark_id = divmod(int(pred_id), len(MARKS))
    if pos >= len(key) or mark_id >= len(MARKS):
        return None
    mark = MARKS[mark_id]
    if not valid_target(key, pos, mark):
        return None
    return apply_stress(key, pos, mark)


@torch.no_grad()
def predict_batches(
    model: JointModel,
    loader: Iterable[dict[str, Any]],
    device: torch.device | str,
) -> list[dict[str, Any]]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        rows = batch["rows"]
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
        pos_ids = out["pos_logits"].argmax(-1).cpu()
        stress_ids = out["stress_logits"].argmax(-1).cpu()
        stress_positions = out["stress_word_positions"].cpu().tolist()
        stress_by_position = {
            (int(batch_index), int(word_index)): int(stress_id)
            for (batch_index, word_index), stress_id in zip(stress_positions, stress_ids.tolist())
        }
        word_mask = batch["word_mask"].cpu()
        for batch_index, row in enumerate(rows):
            row_predictions = []
            for word_index, token in enumerate(row.get("tokens", [])):
                if word_index >= word_mask.shape[1] or not bool(word_mask[batch_index, word_index]):
                    continue
                pos_id = int(pos_ids[batch_index, word_index].item())
                pos_label = model.labels[pos_id] if 0 <= pos_id < len(model.labels) else ""
                stress_id = stress_by_position.get((batch_index, word_index))
                stress_form = (
                    decode_stress_prediction(str(token["word"]), stress_id, model.max_chars)
                    if stress_id is not None
                    else None
                )
                row_predictions.append(
                    {
                        "word": str(token["word"]),
                        "gold_pos": str(token.get("pos_label") or ""),
                        "pos": pos_label,
                        "stress": stress_form,
                    }
                )
            predictions.append({"row": row, "tokens": row_predictions})
    return predictions


def count_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def format_pct(value: float) -> str:
    return f"{value:.2%}"


def find_encoder_checkpoint() -> Path | None:
    roots = [
        TAGGER_DIR / "release",
        TAGGER_DIR / "artifacts",
        TAGGER_DIR / "runs",
    ]
    preferred: list[Path] = []
    for root in roots:
        if root.exists():
            preferred.extend(sorted(root.rglob("pytorch_model.bin")))
            preferred.extend(sorted(root.rglob("model.safetensors")))
    scored: list[tuple[int, Path]] = []
    for path in preferred:
        parent = path.parent
        score = 0
        text = str(parent).lower()
        if "hf-vdu" in text or "gen2__vdu" in text:
            score -= 20
        if "release" in text:
            score -= 10
        if "runs" in text:
            score += 5
        if (parent / "config.json").exists():
            score -= 2
        if (parent / "tokenizer.json").exists():
            score -= 1
        scored.append((score, path))
    if not scored:
        return None
    return sorted(scored, key=lambda item: (item[0], str(item[1])))[0][1]


def load_encoder_and_tokenizer(encoder_source: str | Path | None = None) -> tuple[Any, Any, str]:
    from transformers import AutoModel, AutoTokenizer

    source = str(encoder_source or ENCODER)
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("joint subword alignment requires a fast tokenizer")
    encoder = AutoModel.from_pretrained(source)
    return encoder, tokenizer, source


def load_joint_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def instantiate_from_checkpoint(path: Path, device: torch.device | str = "cpu") -> tuple[JointModel, Any, dict[str, Any]]:
    checkpoint = load_joint_checkpoint(path, map_location="cpu")
    labels = [str(label) for label in checkpoint["labels"]]
    char_vocab = {str(k): int(v) for k, v in checkpoint["char_vocab"].items()}
    encoder_source = checkpoint.get("encoder_source") or checkpoint.get("base_model") or ENCODER
    encoder, tokenizer, _source = load_encoder_and_tokenizer(encoder_source)
    model = JointModel(
        encoder=encoder,
        labels=labels,
        n_chars=max(char_vocab.values(), default=1) + 1,
        max_chars=int(checkpoint.get("max_chars", MAX_CHARS)),
        stress_weight=float(checkpoint.get("stress_weight", 1.0)),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    checkpoint["char_vocab"] = char_vocab
    return model, tokenizer, checkpoint


def tokens_from_text_sentence(sentence: str) -> list[str]:
    return [match.group(0) for match in WORD_RE.finditer(sentence)]


def rows_from_plain_sentences(sentences: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for index, sentence in enumerate(sentences, start=1):
        words = tokens_from_text_sentence(sentence)
        if not words:
            continue
        rows.append(
            {
                "id": f"plain-{index}",
                "text": sentence,
                "tokens": [
                    {"word": word, "pos_label": "X|_", "stress": None}
                    for word in words
                ],
            }
        )
    return rows


def deterministic_split(
    rows: list[dict[str, Any]],
    train_size: int | None,
    dev_size: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    copied = list(rows)
    random.Random(seed).shuffle(copied)
    if train_size is None:
        train_size = max(1, int(len(copied) * 0.95))
    train = copied[:train_size]
    dev = copied[train_size : train_size + dev_size]
    if not dev:
        dev = copied[: min(dev_size, len(copied))]
    return train, dev


def step_schedule(total_steps: int, warmup_steps: int = 500, mode: str = "cosine"):
    total_steps = max(1, total_steps)
    if mode not in {"cosine", "constant"}:
        raise ValueError(f"unknown schedule mode: {mode}")
    if mode == "constant":
        warmup_steps = max(1, warmup_steps)
    else:
        warmup_steps = max(1, min(warmup_steps, total_steps))

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if mode == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return schedule


def tokens_per_second(token_count: int, started: float) -> float:
    elapsed = max(1e-9, time.perf_counter() - started)
    return token_count / elapsed


def summarize_joint_rows(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    sentence_count = 0
    token_count = 0
    letter_tokens = 0
    stress_supervised = 0
    stress_supervised_letter = 0
    stress_none = 0
    for row in rows:
        sentence_count += 1
        for token in row.get("tokens", []):
            token_count += 1
            is_letter = has_letter(str(token.get("word") or ""))
            if is_letter:
                letter_tokens += 1
            if token.get("stress") is not None:
                stress_supervised += 1
                if is_letter:
                    stress_supervised_letter += 1
                if token.get("stress") == "none":
                    stress_none += 1
    return {
        "sentences": sentence_count,
        "tokens": token_count,
        "letter_tokens": letter_tokens,
        "stress_supervised": stress_supervised,
        "stress_supervised_letter": stress_supervised_letter,
        "stress_none": stress_none,
    }


def print_stats_block(label: str, stats: dict[str, Any]) -> None:
    print(f"{label}:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f} ({value:.2%})")
        else:
            print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")


class SimpleToken:
    def __init__(self, form: str) -> None:
        self.form = form


def prediction_label_counts(predictions: Iterable[dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for row in predictions:
        for token in row.get("tokens", []):
            counter[str(token.get("pos") or "")] += 1
    return counter
