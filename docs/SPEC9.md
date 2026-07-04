# Phase 9 — best-corpus training: MATAS + multi-encoder comparison pipelines

MATAS v3.0 is acquired and validated: 2,137,281 tokens / 144,026 sentences
of manually checked Lithuanian morphology, native CoNLL-U with UPOS + full
UD FEATS (+ Jablonskis XPOS + Multext MISC), **CC BY 4.0**, from
`https://clarin-repo.lt/server/api/core/bitstreams/c985d423-14b1-408a-ab47-6cb61a69094c/content`
(handle `20.500.11821/61`, zip contains `MATAS3.conllu`, readmes). This
phase turns `local/tagger-hf/` into a complete train/eval/compare bench on
MATAS+ALKSNIS with multiple encoder candidates.

All work stays inside `local/tagger-hf/` (plus its README). Heavy deps
lazily imported; every script must run on Windows via
`uv run --with <deps> local/tagger-hf/<script>.py`.

## 1. `fetch_corpora.py` (new)

Downloads and caches into `local/tagger-hf/data/raw/`:
- ALKSNIS train/dev/test conllu (UD GitHub raw, as prep_alksnis does now);
- MATAS v3.0 zip from the bitstream URL above; verify size ≈ 24,221,501
  bytes; extract `MATAS3.conllu`.
- Skip anything already present (offline-friendly); `--force` re-downloads.

## 2. `prep_corpus.py` (supersedes prep_alksnis.py — keep that file as a
thin deprecated wrapper that calls the new one with `--sources alksnis`)

Builds the unified token-classification dataset:
- `--sources matas,alksnis` (default both), `--out data/combined`.
- Labels: `UPOS|FEATS` with FEATS **canonicalized** (split on `|`, sort
  key=value pairs, rejoin) so MATAS and ALKSNIS produce identical label
  strings for identical analyses.
- **Leakage guard**: drop from training any sentence whose normalized text
  (casefold + whitespace-collapse) appears in ALKSNIS dev or test. Print
  how many were dropped. (MATAS and ALKSNIS may share source texts.)
- Splits: train = MATAS(deduped) + ALKSNIS-train; dev = ALKSNIS-dev;
  test = ALKSNIS-test — the same gold test the tagger benchmark uses, so
  results are directly comparable with the LINDAT/Stanza numbers.
- `--max-train-sentences N` for smoke runs (deterministic head after
  shuffling with a fixed seed).
- Outputs as today (jsonl + labels.json + stats printed: sentences,
  tokens, label-set size, dev/test OOV-label rate).

## 3. `train.py` (extend)

- `--run-name` (default derived from model name) → everything under
  `runs/<run-name>/`; write `metrics.json` per eval (epoch, label accuracy,
  and the **scoring-slot projection accuracy** — factor the projection into
  a shared `metrics.py` module) and `final.json` with best-dev results +
  test results of the best checkpoint.
- Keep existing behavior otherwise (first-subword alignment, fallback
  model, fp32 CPU-safe defaults, `--max-steps` for smoke).

## 4. `compare_encoders.py` (new)

Orchestrates the bake-off:
- `--models` default:
  `VSSA-SDSA/LT-MLKM-modernBERT,EMBEDDIA/litlat-bert,xlm-roberta-base`
  (rationale for litlat-bert: EMBEDDIA's Lithuanian-Latvian-English
  trilingual BERT — the strongest published Baltic-focused encoder).
- For each model: train with identical hyperparameters on the prepared
  dataset, evaluate the best checkpoint on the gold test split with the
  same metric family as `scripts/bench_taggers.py` (upos, feats-exact,
  slots, aux/verb), append a row to `runs/comparison.md` (markdown table,
  includes tokens/sec measured on CPU inference over the test set).
- `--smoke`: `--max-train-sentences 400 --max-steps 60` and models
  defaulting to `distilbert-base-multilingual-cased` only — must complete
  on CPU in minutes and exercise the FULL path (prep assumed done →
  train → eval → comparison table row).
- Failures of one model must not abort the others (log and continue).

## 5. Housekeeping

- `.gitignore`: `local/tagger-hf/data/` and `local/tagger-hf/runs/`.
- README rewrite of the affected sections: corpus table (ALKSNIS numbers +
  MATAS 2.14M tokens CC BY 4.0 with the attribution line "Rimkutė,
  Bielinskienė, Boizou, Dadurkevičius, Kovalevskaitė, Utka — MATAS v3.0,
  CLARIN-LT, hdl:20.500.11821/61"), fetch/prep/train/compare commands,
  smoke command, GPU expectations (ModernBERT-0.2B on ~2M tokens ≈ 1-3 h
  on a single modern GPU; CPU not recommended for the full run), and the
  note that the ONNX export + sidecar (already present) consume
  `runs/<run-name>/best` unchanged.

## Quality bar

- `py_compile` clean; no changes outside `local/tagger-hf/`, README files,
  and `.gitignore`.
- The orchestrator will run: fetch (cached zip already at
  `local/tagger-hf/data/raw/` if you look for it — but handle absence),
  prep (full), and `compare_encoders.py --smoke` end-to-end on CPU.
- `npm run check` still passes (should be untouched).
