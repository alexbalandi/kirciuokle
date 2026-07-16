"""Quick stress panel for a joint checkpoint — the SPEC59 interjection gate.

Runs the checkpoint on sentences containing the formerly-masked interjections
plus supervised controls, decodes stress exactly like the browser (validity
mask + no-stress baseline), and prints chosen form + top logits.

Usage: .venv/Scripts/python.exe panel_intj.py --checkpoint checkpoints/joint_v4.best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from joint_lib import (  # noqa: E402
    MARKS,
    JointCollator,
    instantiate_from_checkpoint,
)
from train_guesser import apply_stress, valid_target  # noqa: E402

SENTENCES = [
    # formerly-masked interjections (the SPEC59 fix targets)
    ("Prašom užeiti.", {"Prašom": "Prãšom"}),
    ("prašom", {"prašom": "prãšom"}),
    ("Labai prašom, sėskitės.", {"prašom": "prãšom"}),
    ("Ačiū labai.", {"Ačiū": "Ãčiū"}),
    ("Deja, nieko nebus.", {"Deja": "Dejà"}),
    ("Labas rytas.", {"Labas": "Lãbas"}),
    ("Dėkui už pagalbą.", {"Dėkui": "Dė̃kui"}),
    # supervised controls that must NOT regress
    ("Prašome užeiti.", {"Prašome": "Prãšome"}),
    ("Jis buvo prašomas užpildyti anketą.", {"prašomas": "prãšomas"}),
    ("Šiandien gražus oras.", {"gražus": "gražùs", "oras": "óras"}),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, checkpoint = instantiate_from_checkpoint(args.checkpoint, device=device)
    labels = [str(label) for label in checkpoint["labels"]]
    char_vocab = checkpoint["char_vocab"]
    collator = JointCollator(tokenizer, labels, char_vocab)

    failures = 0
    checks = 0
    for text, expected in SENTENCES:
        words = text.replace(",", " ").replace(".", " ").split()
        rows = [{"tokens": [{"word": w, "pos_label": labels[0], "stress": None} for w in words]}]
        batch = collator(rows)
        batch_t = {
            k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)
        }
        with torch.no_grad():
            out = model(
                input_ids=batch_t["input_ids"],
                attention_mask=batch_t["attention_mask"],
                first_subword=batch_t["first_subword"],
                last_subword=batch_t["last_subword"],
                word_mask=batch_t["word_mask"],
                char_ids=batch_t["char_ids"],
                char_valid=batch_t["char_valid"],
                char_mask=batch_t["char_mask"],
                token_type_ids=batch_t.get("token_type_ids"),
            )
        stress_logits = out["stress_logits"].float().cpu()
        positions = out["stress_word_positions"].cpu()
        print(f"\n>> {text}")
        for row_idx in range(positions.shape[0]):
            word_index = int(positions[row_idx, 1])
            word = words[word_index]
            logits = stress_logits[row_idx]
            n_marks = len(MARKS)
            no_stress = float(logits[-1])
            chars = list(word.lower())
            best = None
            for pos in range(min(len(chars), 30)):
                for mark_index, mark in enumerate(MARKS):
                    value = float(logits[pos * n_marks + mark_index])
                    if value > (best[0] if best else no_stress):
                        if valid_target(word.lower(), pos, mark):
                            best = (value, pos, mark)
            accented = apply_stress(word, best[1], best[2]) if best else word
            marker = ""
            if word in expected:
                checks += 1
                ok = accented.lower() == expected[word].lower()
                marker = "  ✓" if ok else f"  ✗ EXPECTED {expected[word]}"
                failures += 0 if ok else 1
            print(f"   {word:12} -> {accented:14}{marker}")

    print(f"\npanel: {checks - failures}/{checks} expected words correct")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
