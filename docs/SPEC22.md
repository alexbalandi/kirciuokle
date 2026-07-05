# SPEC22 — POS-conditioned stress model (accent given morphology)

## Goal

The current neural stresser (`local/accentuator/train_stress_nn.py`)
predicts one stress per word from the default form — it cannot represent
homographs (galvà nom.sg vs gálva instr.sg differ by case). But every
variant in our dictionary carries a morphology label (`mi`), and at serving
time the tagger produces exactly such a label per token in context. Upgrade
the training pipeline so the model predicts accentuation GIVEN the
morphology label, using the SAME litlat-bert encoder: conditioning is a
tokenizer text pair — `tokenizer(word, label)` — so the char-query
cross-attention head can read the label subwords without any architecture
change.

Modify ONLY `local/accentuator/train_stress_nn.py`. Keep the current
unconditioned behavior working (flag-gated), because the existing
checkpoint and its consumers (guess_uncovered.py nn backend,
bench_guessers.py) must keep functioning until the v2 checkpoint wins.

## 1. Dataset

Add `load_labeled_training(path) -> list[tuple[str, str, int, str]]`
(word, label, stress_pos, mark):

- Iterate `SELECT word, variants FROM words` on generated.sqlite; parse the
  variants JSON; for each variant take `form` and EACH label string in its
  `mi` list (fall back to `info` when `mi` is empty).
- Reuse the existing filters exactly: `word.isalpha()`, stress parse via
  `stress_of`, `strip_accents(form) == word`, `valid_target` — see
  `load_training` for the pattern.
- ALSO emit one `(word, "", pos, mark)` row per word from the default form
  (empty label = unconditioned fallback; at inference an empty label must
  keep working).
- Dedup identical (word, label, form) triples.

## 2. Training changes (all inside train_stress_nn.py)

- New CLI flag `--labels` switching to the labeled dataset. Holdout split
  MUST group BY WORD KEY, not by row: shuffle the distinct word keys with
  `random.Random(20260705)`, hold out 2% of KEYS, and every row of a
  held-out word goes to eval — otherwise labels of the same word leak
  across the split.
- Collate/tokenize: `tokenizer(words, labels, padding=..., truncation=...,
  max_length=48, return_tensors="pt")` (text-pair mode). Char ids/valid
  mask stay computed from the word only. When the label is "", pass the
  word alone (single-text encoding) so unconditioned inference stays in
  distribution.
- `batch_predict` gains an optional `labels` argument (default None →
  current behavior), threading pairs through tokenization the same way.
- Checkpoint: save to `data/stress_nn2/stress_nn2.pt` when `--labels` is
  on (do not overwrite the v1 checkpoint); store `{"labeled": True}` in the
  checkpoint dict.

## 3. Evaluation (runs automatically after training, like v1)

- `in-domain held-out (labeled)`: predict with the true label; report the
  usual answered/exact/position at thresholds 0/0.9.
- `homograph switch`: the subset of held-out WORDS having ≥2 rows whose
  forms differ. A word counts correct only if ALL its rows are predicted
  exactly right (the model actually switches stress with the label).
  Report count and word-level accuracy — this is the headline metric.
- `unconditioned regression`: same held-out words, empty label, compare
  exact vs the default form — must not collapse (print side by side with
  v1's known 97.9%).
- `VDU gap slice (labeled)`: for each uncovered VDU word, use each VDU
  variant's own `info` label as the condition and count exact if the
  prediction matches THAT variant's form. Report alongside the
  unconditioned gap number.

## Pass criteria

1. `.venv-train/Scripts/python.exe local/accentuator/train_stress_nn.py --labels --limit 3000 --epochs 1 --batch-size 64`
   completes end-to-end (smoke): dataset stats printed (total rows, words,
   homograph words), all four eval sections print, checkpoint written to
   data/stress_nn2/.
2. `.venv-train/Scripts/python.exe local/accentuator/train_stress_nn.py --limit 2000 --epochs 1 --batch-size 64`
   (NO --labels) still works and writes the v1 checkpoint path — no
   regression to the unconditioned path.
3. Report the smoke numbers in the final message. Do NOT launch the full
   training run — the human launches it on the GPU after review.

Do not commit.
