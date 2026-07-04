# Phase 10 — tagger head/pooling experiment matrix (build only, no training)

Extends `local/tagger-hf/` with the two architecture axes agreed on top of
the encoder axis. Nothing is trained in this phase beyond the CPU smoke.

## Axes

1. `--head combined|factored` (train.py; default `combined`)
   - `combined`: existing single softmax over `UPOS|FEATS` strings.
   - `factored`: a custom wrapper model (AutoModel encoder + one linear head
     per slot). Slots = UPOS plus every FEATS key observed in the training
     data (Case, Gender, Number, Tense, Person, Voice, Degree, Mood,
     VerbForm, Definite, Reflex, Polarity, PronType, NumType, ...), each
     head over that key's observed values plus a `__none__` class. Loss =
     mean of per-head cross-entropies (ignore index -100 as today).
     Prediction assembles `UPOS|key=val|...` from non-none heads with the
     same canonical FEATS ordering as prep, so the existing metrics and
     comparison path consume it unchanged.
2. `--subword-pooling first|last|first_last` (default `first`)
   - `first`: today's behavior (labels on first subword).
   - `last`: labels aligned to the last subword of each word (data-side
     change).
   - `first_last`: custom wrapper gathers hidden states of first and last
     subword per word and classifies on their concatenation (2×hidden into
     the head(s)). Works for both head types (the factored wrapper and a
     combined-head wrapper share the gathering code).

Lithuanian rationale (document in README): suffixing language — inflection
lives in the ending, so `last`/`first_last` may out-tag `first`,
especially for fragmenting tokenizers (xlm-r) and less so for the native
LT tokenizer; that interaction is part of what the matrix measures.

## Run identity & artifacts

- Default `--run-name` becomes `<model-short>__<head>__<pooling>`.
- `runs/<run>/best/` must contain a `head_config.json`:
  `{head, pooling, base_model, labels: [...] | slots: {key: [values...]},
  hidden_size, max_length}` — the single source of truth for export and
  serving.
- `metrics.json`/`final.json` records include head and pooling.

## export_onnx.py and server.py

- Both become config-driven via `head_config.json`.
- Combined+first stays exactly as today. `last`/`first_last` change the
  gather logic in the server (and the wrapper is what gets exported, so
  the ONNX graph takes subword index tensors or the server computes word
  positions and gathers on the output — choose the simpler: export the
  plain encoder+heads over the full sequence, do first/last gathering in
  numpy in the server).
- Factored export: multiple named outputs (one logits tensor per slot);
  INT8 quantization as today; the torch-vs-ONNX agreement check compares
  assembled label strings.

## compare_encoders.py

- `--heads` and `--poolings` comma lists; runs the full cross with
  `--models`. Default remains the previous single-config behavior
  (combined, first).
- `--recommended` preset (mutually exclusive with the lists): the 6 agreed
  cells —
  1. LT-MLKM-modernBERT / combined / first
  2. LT-MLKM-modernBERT / combined / last
  3. LT-MLKM-modernBERT / combined / first_last
  4. LT-MLKM-modernBERT / factored / last
  5. EMBEDDIA/litlat-bert / combined / last
  6. xlm-roberta-base / combined / first (baseline)
- `runs/comparison.md` gains `head` and `pooling` columns (keep appending;
  a header change is fine — regenerate the header if the file predates it).
- `--smoke` now runs TWO tiny cells to cover both code paths:
  distilbert-base-multilingual-cased with (combined, first) and
  (factored, last), each `--max-train-sentences 400 --max-steps 60`.

## Self-check

- `local/tagger-hf/selfcheck.py`: dependency-light sanity script (plain
  `python selfcheck.py`, no pytest) that round-trips label assembly:
  canonical FEATS ordering, factored assemble/decode inverse on samples
  from labels.json if present else synthetic ones, and pooling index
  computation on a toy tokenization. Exit non-zero on failure.

## Quality bar

- py_compile clean; `selfcheck.py` passes; `--help` works everywhere.
- Do NOT run prep or real training; do NOT run the smoke (the orchestrator
  runs it). No changes outside `local/tagger-hf/` and its README. No git.
- README: matrix table, recommended preset command, and the pooling
  rationale note.
