# Phase 8 — tagger ML exploration: benchmark harness + HF/ONNX scaffold

Goal: measure whether Python ML taggers can match or beat the production
tagger (LINDAT UDPipe 2), and prepare the high-performance CPU path
(fine-tuned Lithuanian transformer exported to ONNX INT8). Everything plugs
into the existing UDPipe-REST sidecar contract from `local/`.

## Part 1 — `scripts/bench_taggers.py` (uv script)

Benchmarks tagger backends against the gold **UD_Lithuanian-ALKSNIS test
set** (download and cache to the scratch dir at runtime:
`https://raw.githubusercontent.com/UniversalDependencies/UD_Lithuanian-ALKSNIS/master/lt_alksnis-ud-test.conllu`).

Protocol:
- Reconstruct each sentence's raw text from the gold tokens (use the
  `# text =` comment). Feed **raw text** to each backend (they tokenize
  themselves, like production does).
- Align backend tokens to gold tokens exactly like the production pipeline
  aligns (case-insensitive form match with an 8-token scan-ahead window,
  after filtering non-letter tokens on the backend side; gold tokens
  filtered the same way). Unaligned gold tokens count against
  `aligned%`.
- Metrics per backend, over aligned tokens:
  - `upos` accuracy;
  - `lemma` accuracy (casefolded);
  - `feats` exact-match accuracy;
  - **`slots` accuracy — the one that matters**: project both gold and
    predicted (UPOS+FEATS) through the production scoring projection
    (port `tokenTags` from `src/worker/disambiguation.ts`: pos family with
    PART_VERB/NOUN-PROPN/CCONJ-SCONJ merges and Degree=Pos dropped; slots
    case, gender, number, tense, person, voice, degree) and count
    exact projection matches;
  - `aux_verb` accuracy: on gold AUX/VERB tokens only, how often the
    backend gets the AUX-vs-VERB distinction right (drives the yra→yrà
    lemma exception);
  - wall-clock tokens/sec (excluding model load).
- Backends (`--backends lindat,stanza,trankit`, `--limit N` sentences,
  default 400):
  - `lindat`: POST to the LINDAT UDPipe REST API (model
    lithuanian-alksnis) — the production reference. Batch sentences into
    few requests; be polite.
  - `stanza`: `stanza` pipeline lt (tokenize,pos,lemma), downloads model on
    first run.
  - `trankit`: `trankit` Pipeline('lithuanian'), if importable — guard with
    a clear "backend unavailable: <reason>" so the script degrades to the
    other backends (trankit's dependency pins may not install on new
    Pythons; requires-python can stay >=3.10 but document `--python 3.11`).
- Output: a compact table to stdout + one line per backend of example
  mismatches (5 samples: form, gold slots, predicted slots).
- Inline script metadata (`# /// script`) like the other scripts; heavy
  deps (stanza, trankit, torch) must NOT be in the script header — import
  lazily inside the backend constructors and print install hints
  (`uv run --with stanza ...`).

## Part 2 — `local/tagger-hf/` scaffold (ready to train, not trained here)

The "better + faster" candidate: fine-tune
`VSSA-SDSA/LT-MLKM-modernBERT` (Apache-2.0 Lithuanian ModernBERT, native LT
tokenizer) for joint morphological tagging on ALKSNIS, then export to ONNX
INT8 for fast CPU serving.

```
local/tagger-hf/
  README.md          # the full recipe: prep -> train -> export -> serve,
                     # expected timings (GPU hours vs CPU), and the lemma
                     # caveat (below)
  prep_alksnis.py    # download UD ALKSNIS train/dev/test; build a token-
                     # classification dataset with combined labels
                     # "UPOS|FEATS" (full feats string); save to disk
                     # (datasets arrow or jsonl)
  train.py           # HF Trainer fine-tune: AutoModelForTokenClassification
                     # on the combined label set; first-subword alignment
                     # (-100 elsewhere); configurable base model (default
                     # LT-MLKM-modernBERT, fallback xlm-roberta-base);
                     # eval on dev each epoch reporting label accuracy AND
                     # the scoring-slot projection accuracy; saves best
  export_onnx.py     # optimum export + INT8 dynamic quantization
                     # (ORTQuantizer); verifies ONNX vs torch predictions
                     # agree on the dev set within a tolerance count
  server.py          # sidecar speaking the UDPipe REST contract
                     # (POST /process -> {"result": conllu}): regex word
                     # tokenizer, ONNX Runtime inference, emits UPOS + FEATS
                     # from the combined label; LEMMA column: emit the
                     # lowercased form EXCEPT emit "būti" for form "yra"
                     # tagged AUX (keeps the production yra-exception
                     # working); XPOS "_"
  Dockerfile         # python-slim + onnxruntime, model mounted/copied
  requirements.txt
```

These scripts must be import-clean and argument-complete (argparse, sane
defaults) but are NOT run in this environment (no GPU; training is a later
job). `server.py` may assume the exported model directory layout produced
by `export_onnx.py`.

**Lemma caveat to document:** token-classification gives no lemmatizer. The
production pipeline needs lemma only for the LEMMA_EXCEPTIONS table (yra:
būti vs irti); the AUX/VERB distinction carries that information, hence the
server-side rule above. If the exceptions table ever grows beyond
AUX-distinguishable pairs, revisit.

## Quality bar

- `uv run scripts/bench_taggers.py --backends lindat --limit 20` must run
  end-to-end (network OK) — the orchestrator runs the full benchmark.
- `python -m py_compile` clean for all new scripts; no modifications to
  `docs/`, other `scripts/`, `src/`, or `local/app`.
- Update `local/README.md` with a short "Tagger backends & benchmarking"
  section linking the harness and the tagger-hf recipe.
- `npm run check` still passes.
