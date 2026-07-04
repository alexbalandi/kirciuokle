# Lithuanian HF/ONNX Tagger Bench

This directory trains, evaluates, compares, exports, and serves a candidate
replacement for the UDPipe-style Lithuanian tagger sidecar. The training task
is configurable over head shape and subword pooling. The default remains the
original joint `UPOS|FEATS` classifier with first-subword alignment. The
runtime contract remains `POST /process -> {"result": "<conllu>"}`.

The default encoder is `VSSA-SDSA/LT-MLKM-modernBERT` (Lithuanian
ModernBERT). The comparison bench also includes `EMBEDDIA/litlat-bert`, a
Lithuanian-Latvian-English trilingual BERT and the strongest published
Baltic-focused encoder candidate here, plus `xlm-roberta-base`.

## Corpora

| Corpus | Role | Size | License / attribution |
| --- | --- | ---: | --- |
| UD_Lithuanian-ALKSNIS | Gold dev/test and optional training source | 2,341 train sentences / 47,641 train tokens, plus about 11.5k dev tokens and 10.8k test tokens | Universal Dependencies ALKSNIS. The test split is the same gold split used by `scripts/bench_taggers.py`. |
| MATAS v3.0 | Main training source | 144,026 sentences / 2,137,281 tokens | CC BY 4.0. Rimkutė, Bielinskienė, Boizou, Dadurkevičius, Kovalevskaitė, Utka — MATAS v3.0, CLARIN-LT, hdl:20.500.11821/61 |

MATAS is fetched from the CLARIN-LT bitstream and cached as
`local/tagger-hf/data/raw/MATAS3.conllu.zip`; `fetch_corpora.py` reuses that
zip when present and extracts `MATAS3.conllu`.

## Fetch

From the repository root:

```sh
uv run local/tagger-hf/fetch_corpora.py
```

Use `--force` only when you intentionally want to re-download cached raw
corpora. The fetcher downloads ALKSNIS train/dev/test CoNLL-U files, validates
the MATAS zip size, and extracts MATAS.

## Prepare

```sh
uv run local/tagger-hf/prep_corpus.py \
  --sources matas,alksnis \
  --out local/tagger-hf/data/combined
```

The prepared split layout is:

- `train`: MATAS with duplicate normalized sentences removed, plus ALKSNIS
  train, after dropping any training sentence whose normalized text appears in
  ALKSNIS dev or test.
- `dev`: ALKSNIS dev.
- `test`: ALKSNIS test, matching the benchmark gold split.

Labels are `UPOS|FEATS`; FEATS are canonicalized by sorting `key=value` pairs
so matching analyses from MATAS and ALKSNIS share one label string. The script
writes `train.jsonl`, `dev.jsonl`, `test.jsonl`, `labels.json`, and prints
sentence/token counts, label-set size, and dev/test OOV-label rates.

For a deterministic small dataset:

```sh
uv run local/tagger-hf/prep_corpus.py --max-train-sentences 400
```

`prep_alksnis.py` remains as a deprecated wrapper for:

```sh
uv run local/tagger-hf/prep_corpus.py --sources alksnis
```

## Train

```sh
uv run --with transformers --with datasets --with accelerate --with torch \
  local/tagger-hf/train.py \
  --data-dir local/tagger-hf/data/combined \
  --head combined \
  --subword-pooling first
```

If `--run-name` is omitted, the run name is
`<model-short>__<head>__<pooling>`, for example
`lt-mlkm-modernbert__combined__first`.

Each run is written under `local/tagger-hf/runs/<run-name>/`:

- `checkpoints/`: Hugging Face trainer checkpoints.
- `metrics.json`: one record per dev evaluation with epoch, label accuracy,
  scoring-slot projection accuracy, head, and pooling.
- `best/`: the best checkpoint saved as a normal Hugging Face model directory.
- `best/head_config.json`: the serving/export source of truth with head,
  pooling, base model, labels or slots, hidden size, and max length.
- `final.json`: best-dev metrics and test metrics for the best checkpoint.

The best checkpoint is selected by scoring-slot projection accuracy, the same
projection used by the production disambiguation path. `--max-steps` and
`--max-train-sentences` are available for smoke runs.

Head/pooling matrix:

| Head | Pooling | What it tests |
| --- | --- | --- |
| `combined` | `first` | Existing single softmax over `UPOS\|FEATS` on the first subword. |
| `combined` | `last` | Same label space, but labels attach to the last subword. |
| `combined` | `first_last` | First and last subword states are concatenated before one softmax. |
| `factored` | `first` | One classifier for UPOS and one per FEATS key, using first subwords. |
| `factored` | `last` | Factored slots with labels attached to last subwords. |
| `factored` | `first_last` | Factored slots over concatenated first and last subword states. |

Lithuanian is suffixing, so inflectional evidence often lives near the word
ending. `last` and `first_last` may therefore beat `first`, especially for
fragmenting tokenizers such as `xlm-roberta-base`; native Lithuanian tokenizers
may show a smaller gain. The matrix is designed to measure that interaction.

Full ModernBERT-0.2B training on roughly 2M tokens is expected to take about
1-3 hours on a single modern GPU, depending on batch size and sequence length.
CPU is not recommended for the full run.

## Compare Encoders

```sh
uv run --with transformers --with datasets --with accelerate --with torch \
  local/tagger-hf/compare_encoders.py
```

The default bake-off trains:

```text
VSSA-SDSA/LT-MLKM-modernBERT,EMBEDDIA/litlat-bert,xlm-roberta-base
```

By default this keeps the previous single configuration:
`--heads combined --poolings first`. Use comma lists to run a full cross:

```sh
uv run --with transformers --with datasets --with accelerate --with torch \
  local/tagger-hf/compare_encoders.py \
  --models VSSA-SDSA/LT-MLKM-modernBERT,EMBEDDIA/litlat-bert \
  --heads combined,factored \
  --poolings first,last,first_last
```

The agreed six-cell preset is:

```sh
uv run --with transformers --with datasets --with accelerate --with torch \
  local/tagger-hf/compare_encoders.py --recommended
```

It runs ModernBERT with `combined/first`, `combined/last`,
`combined/first_last`, and `factored/last`; LitLat BERT with `combined/last`;
and XLM-R with `combined/first` as the baseline. Each cell uses the same
hyperparameters. Failures are logged and do not stop the remaining cells.
Successful runs append a row to `local/tagger-hf/runs/comparison.md` with
model, head, pooling, ALKSNIS-test UPOS, FEATS-exact, slots, AUX/VERB
accuracy, and CPU inference tokens/sec.

CPU smoke path:

```sh
uv run --with transformers --with datasets --with accelerate --with torch \
  local/tagger-hf/compare_encoders.py --smoke
```

Smoke mode assumes the prepared dataset already exists, then trains two tiny
cells: `distilbert-base-multilingual-cased` with `combined/first` and with
`factored/last`. Each uses `--max-train-sentences 400` and `--max-steps 60`,
evaluates the best checkpoint, and appends comparison rows.

## Export ONNX INT8

The ONNX export and sidecar consume `runs/<run-name>/best`, driven by
`head_config.json`:

```sh
uv run --with "optimum[onnxruntime]" --with transformers --with torch \
  local/tagger-hf/export_onnx.py \
  --model-dir local/tagger-hf/runs/lt-mlkm-modernbert__combined__first/best \
  --data-dir local/tagger-hf/data/combined \
  --output-dir local/tagger-hf/artifacts/lt-mlkm-modernbert-onnx
```

The script exports FP32 ONNX under `fp32/`, dynamically quantizes it to INT8
under `int8/`, and compares assembled Torch and ONNX label strings on dev
examples. Factored exports use one named ONNX output per slot.

## Self-check

```sh
uv run local/tagger-hf/selfcheck.py
```

This lightweight check validates canonical FEATS ordering, factored
slot-to-label round trips, and toy first/last subword pooling indices. It does
not run prep, training, export, or smoke training.

## Serve

Build and run with the exported `int8/` directory mounted at `/model`:

```sh
docker build -f local/tagger-hf/Dockerfile -t kirciuokle-tagger-hf .
docker run --rm -p 8001:8001 -v "%cd%/local/tagger-hf/artifacts/lt-mlkm-modernbert-onnx/int8:/model:ro" kirciuokle-tagger-hf
```

Then point the local app at:

```sh
TAGGER_URL=http://127.0.0.1:8001
```

The server tokenizes words with a regex, runs ONNX Runtime, applies the
configured first/last/first-last pooling from `head_config.json`, decodes
combined or factored outputs into `UPOS|FEATS`, and emits CoNLL-U with `XPOS`
set to `_`.

## Lemma Caveat

Token classification does not provide a lemmatizer. The production pipeline
currently needs lemma only for the `LEMMA_EXCEPTIONS` table, specifically
`yra`: `būti` vs `irti`. The server emits the lowercased form as LEMMA except
when the form is `yra` and the predicted UPOS is `AUX`, in which case it emits
`būti`. If the exceptions table grows beyond AUX-distinguishable pairs, add a
lemmatizer or a joint lemma head.
