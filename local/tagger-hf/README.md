# Lithuanian HF/ONNX Tagger Scaffold

This directory is a ready-to-train candidate replacement for the UDPipe-style
tagger sidecar. It fine-tunes a Lithuanian transformer for joint
`UPOS|FEATS` token classification on UD_Lithuanian-ALKSNIS, exports it to
ONNX, quantizes it to INT8 for CPU serving, and exposes the existing
`POST /process -> {"result": "<conllu>"}` contract.

The default base model is `VSSA-SDSA/LT-MLKM-modernBERT` (Apache-2.0,
Lithuanian ModernBERT with a native LT tokenizer). The fallback model is
`xlm-roberta-base`.

## 1. Prepare ALKSNIS

From the repository root:

```sh
uv run local/tagger-hf/prep_alksnis.py --out local/tagger-hf/data/alksnis
```

This downloads UD_Lithuanian-ALKSNIS train/dev/test and writes:

- `train.jsonl`, `dev.jsonl`, `test.jsonl`: one sentence per line with
  `tokens` and combined `labels` like `NOUN|Case=Nom|Gender=Masc|Number=Sing`;
- `labels.json`: the stable label list used by training, export, and serving;
- `raw/*.conllu`: cached source files.

## 2. Fine-Tune

Install the training stack only for the run:

```sh
uv run --with transformers --with datasets --with accelerate --with torch \
  local/tagger-hf/train.py \
  --data-dir local/tagger-hf/data/alksnis \
  --output-dir local/tagger-hf/runs/modernbert-alksnis
```

The trainer uses first-subword alignment: the first subword of each gold token
gets the combined `UPOS|FEATS` label, and later subwords get `-100`.
Evaluation runs each epoch and reports:

- `label_accuracy`: exact combined-label accuracy;
- `slot_accuracy`: the production scoring-slot projection accuracy using the
  same projection as `src/worker/disambiguation.ts`.

The best checkpoint is selected by `slot_accuracy`. Expect roughly 1-3 GPU
hours on a single modern consumer GPU for a first useful run, depending on
batch size and max sequence length. CPU training is possible for smoke tests
with `--max-train-samples`, but a full run is expected to be many hours or
days on CPU and is not the intended path.

If the default model cannot be loaded, pass:

```sh
--model-name xlm-roberta-base
```

or keep the default `--fallback-model xlm-roberta-base`.

## 3. Export ONNX INT8

After training:

```sh
uv run --with "optimum[onnxruntime]" --with transformers --with torch \
  local/tagger-hf/export_onnx.py \
  --model-dir local/tagger-hf/runs/modernbert-alksnis \
  --data-dir local/tagger-hf/data/alksnis \
  --output-dir local/tagger-hf/artifacts/modernbert-alksnis-onnx
```

The script exports a FP32 ONNX model under `fp32/`, dynamically quantizes it
with `ORTQuantizer` under `int8/`, then compares Torch and ONNX argmax labels
on dev examples. Increase `--max-mismatches` only if quantization changes a
small, reviewed number of argmax decisions.

CPU serving with ONNX Runtime INT8 should be materially faster than Python
Stanza after model load, but exact latency depends on CPU vector extensions,
sequence length, and thread settings. Use `scripts/bench_taggers.py` before
changing the production tagger.

## 4. Serve

Build and run with the exported `int8/` directory mounted at `/model`:

```sh
docker build -f local/tagger-hf/Dockerfile -t kirciuokle-tagger-hf .
docker run --rm -p 8001:8001 -v "%cd%/local/tagger-hf/artifacts/modernbert-alksnis-onnx/int8:/model:ro" kirciuokle-tagger-hf
```

Then point the local app at:

```sh
TAGGER_URL=http://127.0.0.1:8001
```

The server tokenizes words with a regex, runs ONNX Runtime token
classification, decodes each combined label into UPOS and FEATS, and emits
CoNLL-U with `XPOS` set to `_`.

## Lemma Caveat

Token classification does not provide a lemmatizer. The production pipeline
currently needs lemma only for the `LEMMA_EXCEPTIONS` table, specifically
`yra`: `būti` vs `irti`. The server therefore emits the lowercased form as
LEMMA except when the form is `yra` and the predicted UPOS is `AUX`, in which
case it emits `būti`. That preserves the current `yra` exception path. If the
exceptions table grows beyond AUX-distinguishable pairs, revisit this design
and add a lemmatizer or a joint lemma head.
