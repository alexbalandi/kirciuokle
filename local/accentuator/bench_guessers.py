"""Unified benchmark for out-of-dictionary stress guessers.

Scores every available candidate on identical slices and metrics:

- slice `held`: 2% in-domain held-out from generated.sqlite (same seed as
  every guesser experiment, so numbers are comparable across runs);
- slice `gap`: VDU-cache words the dictionary does NOT cover — the words a
  guesser tier actually exists for (gold = VDU variant sets).

Metrics: answered (share of the slice the candidate answers), exact /
position (of answered), and exact-over-all (answered x exact — the number
that matters when abstentions cascade to a lower tier).

Candidates (each skipped gracefully when its runtime is missing):
- `trie`       naive longest-suffix majority vote (train_guesser.py)
- `anbinderis` faithful A&K 2010 end-bgn rules (anbinderis_rules.py)
- `liepa`      phonology_engine, BSD-wrapped LIEPA components
- `agree(nn,liepa)` agreement between nn@0 and LIEPA
- `agree->liepa` agreement first, then LIEPA fallback
- `nn`         litlat-bert + char cross-attention head (train_stress_nn.py
               checkpoint; reported at several confidence thresholds)

Run with the training venv for the full table:
  .venv-train/Scripts/python.exe local/accentuator/bench_guessers.py
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DEFAULT_GENERATED, DEFAULT_VDU_SQLITE, safe_relative  # noqa: E402
from train_guesser import SuffixModel, apply_stress, load_training, load_vdu_eval, stress_of  # noqa: E402

REPORT = Path(__file__).resolve().parent / "reports" / "guesser-bench.md"


def score_rows(predictions, rows):
    answered = exact = position = 0
    for pred, (_word, forms) in zip(predictions, rows):
        if pred is None:
            continue
        answered += 1
        norm = [unicodedata.normalize("NFC", f) for f in forms]
        if unicodedata.normalize("NFC", pred) in norm:
            exact += 1
        gold = {(stress_of(f) or (None,))[0] for f in norm}
        if (stress_of(pred) or (None,))[0] in gold:
            position += 1
    n = len(rows) or 1
    a = answered or 1
    return {
        "answered": answered / n,
        "exact": exact / a,
        "position": position / a,
        "exact_all": exact / n,
    }


def candidate_trie(train_pairs):
    model = SuffixModel()
    model.train(train_pairs)

    def predict(word):
        hit = model.predict(word)
        return apply_stress(word, *hit) if hit else None

    return predict


def candidate_anbinderis(train_pairs):
    from anbinderis_rules import AnbinderisModel

    model = AnbinderisModel(train_pairs)
    return model.predict_form


def candidate_liepa(_train_pairs):
    from guess_uncovered import engine_accent
    from phonology_engine import PhonologyEngine

    pe = PhonologyEngine()
    return lambda word: engine_accent(pe, word)


def candidate_nn(_train_pairs, threshold=0.0):
    import torch

    from train_stress_nn import ENCODER, MAX_CHARS, OUT_DIR, StressModel, batch_predict

    from transformers import AutoModel, AutoTokenizer

    ckpt = torch.load(OUT_DIR / "stress_nn.pt", map_location="cpu", weights_only=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(ckpt.get("encoder", ENCODER))
    model = StressModel(
        AutoModel.from_pretrained(ckpt.get("encoder", ENCODER)), len(ckpt["char_vocab"]) + 2
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    def predict_many(words):
        preds = batch_predict(model, tokenizer, ckpt["char_vocab"], words, device)
        return [
            (p[0] if p is not None and p[1] >= threshold and len(w) <= MAX_CHARS else None)
            for p, w in zip(preds, words)
        ]

    predict_many.batched = True
    return predict_many


def _candidate_agreement(train_pairs, fallback_to_liepa=False):
    nn_predict = candidate_nn(train_pairs, 0.0)
    liepa_predict = candidate_liepa(train_pairs)

    def predict_many(words):
        nn_preds = nn_predict(words)
        liepa_preds = [liepa_predict(w) for w in words]
        out = []
        for nn_form, liepa_form in zip(nn_preds, liepa_preds):
            agreed = (
                nn_form is not None
                and liepa_form is not None
                and unicodedata.normalize("NFC", nn_form) == unicodedata.normalize("NFC", liepa_form)
            )
            if agreed or fallback_to_liepa:
                out.append(liepa_form)
            else:
                out.append(None)
        return out

    predict_many.batched = True
    return predict_many


def candidate_agree(train_pairs):
    return _candidate_agreement(train_pairs)


def candidate_agree_then_liepa(train_pairs):
    return _candidate_agreement(train_pairs, fallback_to_liepa=True)


def run_candidate(predict, rows):
    words = [w for w, _f in rows]
    if getattr(predict, "batched", False):
        preds = predict(words)
    else:
        preds = [predict(w) for w in words]
    return score_rows(preds, rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=float, default=0.02)
    parser.add_argument(
        "--nn-thresholds", type=str, default="0,0.5,0.9",
        help="Comma-separated confidence cutoffs for the nn candidate.",
    )
    args = parser.parse_args(argv)

    pairs = load_training(DEFAULT_GENERATED)
    rng = random.Random(20260705)
    rng.shuffle(pairs)
    cut = int(len(pairs) * args.holdout)
    held, train_pairs = pairs[:cut], pairs[cut:]
    held_rows = [(w, [apply_stress(w, p, m)]) for w, p, m in held]

    db = sqlite3.connect(DEFAULT_GENERATED)
    covered = {w for (w,) in db.execute("SELECT word FROM words")}
    gap_rows = load_vdu_eval(DEFAULT_VDU_SQLITE, covered)
    print(f"train={len(train_pairs):,} held={len(held_rows):,} gap={len(gap_rows):,}")

    candidates = [
        ("trie", candidate_trie),
        ("anbinderis", candidate_anbinderis),
        ("liepa", candidate_liepa),
        ("agree(nn,liepa)", candidate_agree),
        ("agree->liepa", candidate_agree_then_liepa),
    ]
    for thr in [float(t) for t in args.nn_thresholds.split(",")]:
        candidates.append((f"nn@{thr:g}", lambda tp, thr=thr: candidate_nn(tp, thr)))

    table = []
    for name, factory in candidates:
        try:
            predict = factory(train_pairs)
        except Exception as exc:  # missing runtime, missing checkpoint
            print(f"{name}: skipped ({type(exc).__name__}: {exc})")
            continue
        row = {"name": name}
        for slice_name, rows in (("held", held_rows), ("gap", gap_rows)):
            row[slice_name] = run_candidate(predict, rows)
            s = row[slice_name]
            print(
                f"{name:12} {slice_name:4} answered={s['answered']:6.1%} "
                f"exact={s['exact']:6.1%} position={s['position']:6.1%} "
                f"exact-over-all={s['exact_all']:6.1%}"
            )
        table.append(row)

    lines = [
        "# Out-of-dictionary stress guesser benchmark",
        "",
        f"- training types: {len(train_pairs):,} (generated.sqlite, default forms)",
        f"- `held`: {len(held_rows):,} in-domain held-out types (seed 20260705)",
        f"- `gap`: {len(gap_rows):,} VDU-cache words the dictionary does not cover",
        "- metrics of answered; `exact-over-all` = answered x exact",
        "",
        "| candidate | slice | answered | exact | position | exact-over-all |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in table:
        for slice_name in ("held", "gap"):
            s = row[slice_name]
            lines.append(
                f"| {row['name']} | {slice_name} | {s['answered']:.1%} "
                f"| {s['exact']:.1%} | {s['position']:.1%} | {s['exact_all']:.1%} |"
            )
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {safe_relative(REPORT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
