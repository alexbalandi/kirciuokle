# SPEC19 — Agreement ensemble stage for the guess tier

## Context

Benchmark (reports/guesser-bench.md) on the 2,751-word VDU gap slice:
when the litlat-bert stresser (nn, any confidence) and LIEPA
(phonology_engine) independently produce the SAME accented form — 50.5% of
the slice — that form is 99.5% exact against VDU. Where they disagree,
LIEPA alone is 76.4% exact. The guess tier should therefore expose
agreement as its own high-trust stage with distinct provenance.

Modify ONLY `local/accentuator/guess_uncovered.py` and
`local/accentuator/bench_guessers.py`.

## 1. guess_uncovered.py — new stage syntax `nn&liepa`

- Extend `--backend` choices with: `nn&liepa`, `nn&liepa+liepa`,
  `nn&liepa+nn+liepa`. The `&` token means an agreement stage: run BOTH
  sub-backends on the stage's words; the stage answers a word only when
  both answer and the NFC-normalized forms are equal.
- Composition stays `+`-separated stages processed by the existing
  `run_cascade`; implement the agreement stage as a backend-like object
  (`name`, `predict_many`) wrapping two existing backend instances, so
  `run_cascade` needs no changes. When building `nn&liepa+nn+liepa`,
  construct each backend instance ONCE and share it between stages (do not
  load the checkpoint or the LIEPA engine twice).
- Agreement-stage provenance:
  `open-accentuator:agree-nn-liepa:{word}:conf={nn_conf:.3f}` — implement
  by giving the stage `name = "agree-nn-liepa"` and letting it return the
  nn confidence as the tuple's conf; extend `provenance()` accordingly.
  The variants JSON and info label ("spėjimas") stay unchanged.
- `--min-confidence` keeps applying to the nn sub-backend exactly as today
  (inside NNBackend), including within the agreement stage.
- Per-backend counts in the final print must include the agreement stage
  under its name.

## 2. bench_guessers.py — score the ensemble

- Add candidate `agree(nn,liepa)`: reuse `candidate_nn(_, 0.0)` (batched)
  and `candidate_liepa`; answer = liepa form when NFC-equal to the nn
  form, else None.
- Add candidate `agree->liepa`: agreement answer when available, else the
  plain liepa answer (this is the production cascade `nn&liepa+liepa`; on
  the gap slice it should score ~88% exact-over-all with the agreed half
  at 99.5%).
- Both candidates must be skipped gracefully (same try/except pattern)
  when torch or phonology_engine is unavailable.
- Keep the report format; the two new rows appear in
  reports/guesser-bench.md.

## Verified contracts

Everything needed is already in the two files: `NNBackend`,
`LiepaBackend`, `build_backends`, `run_cascade`, `provenance` in
guess_uncovered.py (see SPEC18 for their shapes); `candidate_nn`
(`predict_many.batched = True`), `candidate_liepa`, `score_rows`,
`run_candidate` in bench_guessers.py. NFC-normalize with
`unicodedata.normalize("NFC", form)` before comparing forms.

## Pass criteria (repo root; GPU is free — CUDA use is fine)

1. `.venv-train/Scripts/python.exe local/accentuator/guess_uncovered.py --backend "nn&liepa" --limit 300 --output local/accentuator/data/guesses-smoke.sqlite`
   → answered < 300 (agreement abstains); ALL provenances start with
   `open-accentuator:agree-nn-liepa:` and end with `:conf=0.xxx`.
2. `.venv-train/Scripts/python.exe local/accentuator/guess_uncovered.py --backend "nn&liepa+liepa" --limit 300 --output local/accentuator/data/guesses-smoke.sqlite`
   → per-backend counts printed for both stages; provenances are a mix of
   `agree-nn-liepa` and `liepa-guess`.
3. `uv run local/accentuator/guess_uncovered.py --limit 100 --output local/accentuator/data/guesses-smoke.sqlite`
   → default liepa path unchanged.
4. `uv run local/accentuator/guess_uncovered.py --backend "nn&liepa" --limit 10 --output local/accentuator/data/guesses-smoke.sqlite`
   → clean one-line error naming `.venv-train/Scripts/python.exe`, nonzero
   exit (torch missing under uv).
5. Delete `local/accentuator/data/guesses-smoke.sqlite`.
6. `.venv-train/Scripts/python.exe local/accentuator/bench_guessers.py --nn-thresholds 0`
   → table includes `agree(nn,liepa)` and `agree->liepa` rows on both
   slices; on `gap`, `agree(nn,liepa)` exact must be ≥ 99% and
   `agree->liepa` exact-over-all ≥ 85%. Report file rewritten.

Do not commit; leave the working tree for review.
