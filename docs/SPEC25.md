# SPEC25 — Label-conditioned live guessing + audited-silver scoring

## Context

The v2 stress model (stress_nn2) is label-conditioned, and the no-dict
evaluator already passes the tagger's bridged label per token. But
`eval_live_guess.py`'s LIVE tier still guesses unconditioned default
forms — wasting the conditioning on exactly the tokens where inflection
matters. Thread labels through the live cascade.

Modify ONLY `local/accentuator/guess_uncovered.py` and
`local/accentuator/eval_live_guess.py`.

## 1. guess_uncovered.py — optional labels through the backends

- `NNBackend.predict_many(words, labels=None)`: thread `labels` into
  `batch_predict` (SPEC22 added its labels parameter; when labels is
  None keep today's behavior). Empty-string labels are valid (trained
  fallback mode).
- `AgreementBackend.predict_many(words, labels=None)`: pass labels to the
  nn side only (LIEPA has no label input).
- `LiepaBackend`/`AnbinderisBackend`: accept and ignore `labels=None`
  keyword for interface uniformity.
- `run_cascade(backends, words, labels=None)`: slice labels alongside
  words for each stage. The artifact-builder `main()` keeps calling
  without labels — no behavior change there.

## 2. eval_live_guess.py — condition the live tier on OUR tagger

- Reuse the tagger subprocess + label bridge from
  `eval_nodict_pipeline.py` (import its helpers; refactor small shared
  functions there into importable form if needed — do NOT duplicate the
  bridge logic).
- New flag `--conditioned-live` (default ON; `--no-conditioned-live` to
  compare): when on, tag the corpus (needs `--corpus` path argument,
  default `data/eval/lrt-corpus.txt`), align tokens to silver exactly the
  way eval_nodict_pipeline does, and pass each live-tier token's bridged
  label through the cascade. Tokens that fail alignment fall back to "".
- Distinct-word deduplication must become distinct-(word, label)
  deduplication for the live tier when conditioning is on.
- Report gains a line: live tier conditioned vs unconditioned (run both
  modes in one invocation is NOT required; the flag switches).

## 3. Both evaluators — score against the AUDITED silver

`local/accentuator/data/eval/lrt-silver-audit.json` (exists) maps word →
`{"action", "accept" (list of NFC lowercase forms), "source", "note"}`.
Three opus audit passes produced it (798 suspect types adjudicated against
e-LKŽ / VLKK / wiktionary). Add `--audit PATH` (default that file, skip
silently if missing) to BOTH `eval_live_guess.py` and
`eval_nodict_pipeline.py`, applied when scoring a token whose lowercase
word has an entry:

- `replace`: the gold form set becomes `accept` (silver's form was wrong).
- `accept-also`: gold = silver form ∪ `accept` (both readings valid).
- `accept-any-observed`: gold = every silver form observed for this word
  anywhere in the corpus.
- `exclude`: token leaves the denominator entirely (silver unreliable) —
  count excluded tokens, report the count.
- `unmarked-ok`: token moves to its own `foreign-unmarked` category,
  excluded from exact/position denominators; additionally report, as a
  separate diagnostic, the fraction of these tokens where the pipeline
  ALSO left the word unmarked or abstained (that is the desired behavior).

Comparison of forms stays NFC + lowercase. Reports must show both raw and
audited numbers for the headline rows (one extra table or column).

## Pass criteria

1. `.venv-train/Scripts/python.exe local/accentuator/eval_live_guess.py --silver local/accentuator/data/eval/lrt-smoke-silver.jsonl --corpus local/accentuator/data/eval/lrt-smoke.txt`
   (conditioned default) completes; tagger subprocess terminated; report
   shows the live tier with labels in use (print a sample of 5 live
   (word, label) pairs).
2. Same command with `--no-conditioned-live` reproduces the old behavior.
3. `uv run local/accentuator/guess_uncovered.py --limit 100 --output local/accentuator/data/guesses-smoke.sqlite` still works unchanged
   (then delete the smoke artifact).
4. Report both smoke live-tier exact numbers (conditioned vs not) in the
   final message.

A chained eval run may still be executing when you start — file edits are
safe (its modules are already imported), but do NOT run your pass criteria
until `local/accentuator/reports/nodict-eval.md` has a fresh timestamp
(the chain's last step); poll its mtime, or if it has not changed within
25 minutes proceed anyway.

Do not commit.
