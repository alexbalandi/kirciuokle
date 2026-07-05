# SPEC27 — Stress model v3: learned no-stress class

## Goal

The LRT eval showed the conditioned model accents foreign words it should
leave alone (only 55.5% of foreign-unmarked tokens correctly untouched at
conf 0). Standard Lithuanian text leaves unadapted foreign words
UNACCENTED — that must become a learned prediction, not a
confidence-threshold artifact.

Modify ONLY `local/accentuator/train_stress_nn.py` and
`local/accentuator/eval_nodict_pipeline.py`.

## 1. Architecture: a virtual no-stress cell

`StressHead` output stays (batch, chars, 3). Add a learned no-stress
logit: `self.no_stress = nn.Linear(hidden, 1)` applied to the MEAN of the
char representations (the post-FFN x, masked to real chars). The training
softmax runs over `chars*3 + 1` cells: flatten the grid, append the
no-stress logit. Target index `n*3` (with n = padded char count) — or any
equivalent stable encoding — means "no stress". The validity mask gets an
always-true final cell. `batch_predict` returns `("", confidence)`-style:
introduce an explicit sentinel — return `(None_form, conf)` as the tuple
`("", conf)` when no-stress wins, and every consumer treats `""` as
"leave the word unmarked" (bench_guessers/guess_uncovered treat "" as an
abstention-with-confidence; eval_nodict_pipeline scores it as an
unmarked answer). Keep checkpoint-shape compatibility irrelevant — v3
saves to `data/stress_nn3/stress_nn3.pt` with `{"no_stress": True}`.

## 2. Training data: no-stress rows (NO LRT data — that is the eval set)

New flag `--no-stress-rows` (default on when `--labels` is on):
- VDU-cache unmarked entries: words in `DEFAULT_VDU_SQLITE` whose every
  variant form carries zero stress marks (`stress_of` is None for all) —
  emit `(word, "", NO_STRESS)` and `(word, "dkt. tikr.", NO_STRESS)` rows.
- Attested foreign tokens: words in the lt_50k wordlist containing any
  letter outside the Lithuanian alphabet `aąbcčdeęėfghiįyjklmnoprsštuųūvzž`
  (e.g. w, x, q) — same two rows each. `.isalpha()` must still hold.
- Print the no-stress row count (expect a few thousand).
- CRITICALLY: do not read anything under `data/eval/` for training.

## 3. Training run config

- v3 default epochs 4 (the mobile-paradigm ending errors look like
  underfit; one extra epoch is the cheap lever).
- Everything else (label conditioning, word-key-grouped holdout, masks)
  stays as v2. `--labels` + v3 behavior gated behind `--v3` flag so v2
  reproduction stays possible.

## 4. Evaluation additions (automatic after training)

- `no-stress held-out`: held-out no-stress words — fraction predicted
  no-stress (target metric), fraction wrongly accented.
- Existing suites unchanged (in-domain labeled, homograph switch,
  unconditioned regression, VDU gap) — numbers must not regress by more
  than ~1pp vs v2 (report side by side; the v2 numbers are in the SPEC22
  commit message).
- `eval_nodict_pipeline.py`: add `--checkpoint PATH` (default the v2
  file) so the v3 checkpoint can be evaluated; treat a `""` prediction as
  an unmarked answer (exact vs silver when silver is also unmarked; in
  audited mode it feeds the foreign-unmarked diagnostic as
  "desired unmarked").

## Pass criteria

1. `.venv-train/Scripts/python.exe local/accentuator/train_stress_nn.py --v3 --labels --limit 4000 --epochs 1 --batch-size 64`
   completes end-to-end (smoke): no-stress row count printed, all eval
   sections incl. `no-stress held-out` print, checkpoint written to
   data/stress_nn3/.
2. `.venv-train/Scripts/python.exe local/accentuator/train_stress_nn.py --labels --limit 2000 --epochs 1 --batch-size 64`
   (v2 path, no --v3) still works and writes to data/stress_nn2/ —
   IMPORTANT: this smoke will overwrite the good v2 checkpoint; first
   copy `data/stress_nn2/stress_nn2.pt` to `data/stress_nn2/stress_nn2.full.bak`
   and restore it after the smoke run.
3. `.venv-train/Scripts/python.exe local/accentuator/eval_nodict_pipeline.py --corpus local/accentuator/data/eval/lrt-smoke.txt --silver local/accentuator/data/eval/lrt-smoke-silver.jsonl --checkpoint local/accentuator/data/stress_nn3/stress_nn3.pt`
   runs with the smoke v3 checkpoint (numbers meaningless; code path is
   the test).
4. Report smoke numbers. Do NOT launch the full training (human launches
   on the GPU after review).

Do not commit.
