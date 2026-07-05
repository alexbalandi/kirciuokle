# /// script
# requires-python = ">=3.11"
# dependencies = ["phonology_engine"]
# ///
"""Guesser tier: stress guesses for words no open source covers.

Builds guesses.sqlite (words schema) for every wordlist/VDU-key word the
main artifact does not cover. Selectable backends (SPEC18), each recorded
in per-word provenance; cascades try stages in order and the first stage
that answers wins:

- `liepa` (default): BSD-licensed phonology_engine wrapping the LIEPA
  synthesizer components — 88.1% exact on the dictionary-gap slice.
- `anbinderis`: Anbinderis & Kasparaitis 2010 letter rules trained on our
  own dictionary (fully open provenance; abstains on ambiguity).
- `nn`: litlat-bert + char cross-attention head (train_stress_nn.py
  checkpoint; needs the .venv-train interpreter), `--min-confidence` gated.
- `nn&liepa`: high-trust agreement between nn and LIEPA.

Numbers per candidate: reports/guesser-bench.md. This tier is deliberately
separate from generated.sqlite, which holds a zero-disagreement gate.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

try:  # pragma: no cover
    from ._common import (
        DATA_DIR,
        DEFAULT_GENERATED,
        DEFAULT_VDU_SQLITE,
        normalize_lt,
        normalize_notation,
        strip_accents,
    )
except ImportError:  # pragma: no cover
    from _common import (
        DATA_DIR,
        DEFAULT_GENERATED,
        DEFAULT_VDU_SQLITE,
        normalize_lt,
        normalize_notation,
        strip_accents,
    )

MARKS = {"0": "̀", "1": "́", "2": "̃"}
DEFAULT_GUESSES = DATA_DIR / "guesses.sqlite"
BACKENDS = (
    "liepa",
    "anbinderis",
    "nn",
    "anbinderis+liepa",
    "nn+liepa",
    "anbinderis+nn+liepa",
    "nn&liepa",
    "nn&liepa+liepa",
    "nn&liepa+nn+liepa",
)


class BackendLoadError(Exception):
    pass


def engine_accent(pe, word: str) -> str | None:
    try:
        raw = pe.process_and_collapse(word, "word_with_all_numeric_stresses")
    except Exception:
        return None
    out = [MARKS.get(ch, ch) for ch in raw.lower()]
    form = normalize_notation(normalize_lt(unicodedata.normalize("NFC", "".join(out))))
    return form if strip_accents(form) == word else None


class LiepaBackend:
    name = "liepa"

    def __init__(self) -> None:
        from phonology_engine import PhonologyEngine

        self.pe = PhonologyEngine()

    def predict_many(self, words: list[str]) -> list[tuple[str, float | None] | None]:
        return [
            (form, None) if form else None
            for form in (engine_accent(self.pe, word) for word in words)
        ]


class AnbinderisBackend:
    name = "anbinderis"

    def __init__(self) -> None:
        try:  # pragma: no cover
            from .anbinderis_rules import AnbinderisModel
            from .train_guesser import load_training
        except ImportError:  # pragma: no cover
            from anbinderis_rules import AnbinderisModel
            from train_guesser import load_training

        self.model = AnbinderisModel(load_training(DEFAULT_GENERATED))

    def predict_many(self, words: list[str]) -> list[tuple[str, float | None] | None]:
        return [
            (form, None) if form else None
            for form in (self.model.predict_form(word) for word in words)
        ]


class NNBackend:
    name = "nn"

    def __init__(self, min_confidence: float) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError as exc:
            raise BackendLoadError(
                "nn backend requires torch/transformers; run with .venv-train/Scripts/python.exe"
            ) from exc
        try:  # pragma: no cover
            from .train_stress_nn import ENCODER, MAX_CHARS, OUT_DIR, StressModel, batch_predict
        except ImportError:  # pragma: no cover
            from train_stress_nn import ENCODER, MAX_CHARS, OUT_DIR, StressModel, batch_predict

        ckpt = torch.load(OUT_DIR / "stress_nn.pt", map_location="cpu", weights_only=False)
        tokenizer = AutoTokenizer.from_pretrained(ckpt.get("encoder", ENCODER))
        model = StressModel(AutoModel.from_pretrained(ckpt.get("encoder", ENCODER)),
                            len(ckpt["char_vocab"]) + 2)
        model.load_state_dict(ckpt["state_dict"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.char_vocab = ckpt["char_vocab"]
        self.batch_predict = batch_predict
        self.max_chars = MAX_CHARS
        self.min_confidence = min_confidence

    def predict_many(self, words: list[str]) -> list[tuple[str, float | None] | None]:
        out: list[tuple[str, float | None] | None] = [None] * len(words)
        positions = [i for i, word in enumerate(words) if len(word) <= self.max_chars]
        eligible = [words[i] for i in positions]
        preds = self.batch_predict(self.model, self.tokenizer, self.char_vocab, eligible, self.device)
        for i, pred in zip(positions, preds):
            if pred is None:
                continue
            form, conf = pred
            if conf >= self.min_confidence:
                out[i] = (form, conf)
        return out


class AgreementBackend:
    name = "agree-nn-liepa"

    def __init__(self, nn_backend: NNBackend, liepa_backend: LiepaBackend) -> None:
        self.nn_backend = nn_backend
        self.liepa_backend = liepa_backend

    def predict_many(self, words: list[str]) -> list[tuple[str, float | None] | None]:
        nn_preds = self.nn_backend.predict_many(words)
        liepa_preds = self.liepa_backend.predict_many(words)
        out: list[tuple[str, float | None] | None] = []
        for nn_pred, liepa_pred in zip(nn_preds, liepa_preds):
            if nn_pred is None or liepa_pred is None:
                out.append(None)
                continue
            nn_form, nn_conf = nn_pred
            liepa_form, _liepa_conf = liepa_pred
            if unicodedata.normalize("NFC", nn_form) == unicodedata.normalize("NFC", liepa_form):
                out.append((liepa_form, nn_conf))
            else:
                out.append(None)
        return out


def build_backends(spec: str, min_confidence: float):
    cache = {}
    stages = []

    def get_backend(name: str):
        if name not in cache:
            if name == "liepa":
                cache[name] = LiepaBackend()
            elif name == "anbinderis":
                cache[name] = AnbinderisBackend()
            elif name == "nn":
                cache[name] = NNBackend(min_confidence)
        return cache[name]

    for name in spec.split("+"):
        if name == "liepa":
            stages.append(get_backend("liepa"))
        elif name == "anbinderis":
            stages.append(get_backend("anbinderis"))
        elif name == "nn":
            stages.append(get_backend("nn"))
        elif name == "nn&liepa":
            stages.append(AgreementBackend(get_backend("nn"), get_backend("liepa")))
    return stages


def run_cascade(backends, words: list[str]):
    results = [None] * len(words)
    remaining = list(range(len(words)))
    for backend in backends:
        preds = backend.predict_many([words[i] for i in remaining])
        next_remaining = []
        for i, pred in zip(remaining, preds):
            if pred is None:
                next_remaining.append(i)
            else:
                form, conf = pred
                results[i] = (backend.name, form, conf)
        remaining = next_remaining
        if not remaining:
            break
    return results


def provenance(name: str, word: str, conf: float | None) -> str:
    if name == "nn":
        return f"open-accentuator:nn-guess:{word}:conf={conf:.3f}"
    if name == "agree-nn-liepa":
        return f"open-accentuator:agree-nn-liepa:{word}:conf={conf:.3f}"
    return f"open-accentuator:{name}-guess:{word}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_GUESSES)
    parser.add_argument("--wordlist", type=Path, default=DATA_DIR / "lt_50k.txt")
    parser.add_argument("--backend", choices=BACKENDS, default="liepa")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    covered = set()
    if DEFAULT_GENERATED.exists():
        db = sqlite3.connect(DEFAULT_GENERATED)
        covered = {w for (w,) in db.execute("SELECT word FROM words")}
    candidates: set[str] = set()
    if args.wordlist.exists():
        candidates |= {
            line.split()[0]
            for line in args.wordlist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if DEFAULT_VDU_SQLITE.exists():
        vdu = sqlite3.connect(DEFAULT_VDU_SQLITE)
        candidates |= {w for (w,) in vdu.execute("SELECT word FROM words")}
    todo = sorted(w for w in candidates if w not in covered and w.isalpha())
    if args.limit is not None:
        todo = todo[: args.limit]

    try:
        backends = build_backends(args.backend, args.min_confidence)
    except BackendLoadError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    counts = {backend.name: 0 for backend in backends}
    now = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    out = sqlite3.connect(args.output)
    out.executescript(
        """
        CREATE TABLE words (
          word TEXT PRIMARY KEY, variants TEXT NOT NULL, fetched_at TEXT NOT NULL,
          negative_until TEXT, default_form TEXT, accent_type TEXT,
          default_form_title TEXT, accent_type_title TEXT, provenance TEXT NOT NULL
        );
        """
    )
    rows = []
    for word, pred in zip(todo, run_cascade(backends, todo)):
        if pred is None:
            continue
        backend_name, form, conf = pred
        counts[backend_name] += 1
        variants = json.dumps(
            [{"form": form, "info": "spėjimas", "mi": []}],
            ensure_ascii=False, separators=(",", ":"),
        )
        rows.append((word, variants, now, None, form, "ONE",
                     form[:1].upper() + form[1:], "ONE",
                     provenance(backend_name, word, conf)))
    out.executemany("INSERT OR IGNORE INTO words VALUES (?,?,?,?,?,?,?,?,?)", rows)
    out.commit()
    if len(backends) > 1:
        by_backend = " + ".join(f"{backend.name} {counts[backend.name]:,}" for backend in backends)
        print(f"guessed {len(rows):,} ({by_backend}) of {len(todo):,} uncovered candidates -> {args.output}")
    else:
        print(f"guessed {len(rows):,} of {len(todo):,} uncovered candidates -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
