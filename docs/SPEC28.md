# SPEC28 — ONNX export + slim no-dict inference pipeline + browser analysis

## Goal

Production-ready, torch-free inference for the no-dictionary accentuation
pipeline, plus a written feasibility analysis for running it client-side
in the browser. Three deliverables, all new files under
`local/accentuator/onnx/` (create the dir; nothing existing is modified):

## 1. `export_stress_onnx.py`

- Load a stress checkpoint (`--checkpoint`, default
  `data/stress_nn2/stress_nn2.pt`; the v3 file at data/stress_nn3/ may
  appear later — support both shapes, the `no_stress` flag in the ckpt
  dict tells you which). Rebuild StressModel from train_stress_nn.py.
- Export to ONNX with dynamic axes (batch, subword length, char length):
  inputs input_ids, attention_mask, char_ids; output the raw logits
  (chars×3 grid, plus the no-stress logit as a second output when v3).
  Study `local/tagger-hf/export_onnx.py` for the working torch.onnx
  pattern + INT8 dynamic quantization via onnxruntime; reuse the same
  recipe (opset, quantize_dynamic).
- Write `stress.onnx` and `stress.int8.onnx` + a `stress.meta.json`
  (char_vocab, marks, max_chars, encoder tokenizer id, no_stress flag).
- Parity gate: run 200 random dictionary words (+ their labels for the
  labeled path) through torch and through both ONNX files; report
  agreement (fp32 must be ≥99.5% identical argmax; int8 ≥98%).

## 2. `nodict_onnx.py` — the slim pipeline

- Torch-free (onnxruntime + tokenizers/transformers tokenizer only;
  import guard that errors helpfully if torch sneaks in via transformers
  AutoTokenizer — use `tokenizers` directly if cleaner).
- Pipeline: sentence in → existing tagger ONNX (reuse the server-side
  inference utilities from local/tagger-hf/inference_utils.py if
  importable without torch, else the minimal CoNLL-U path from
  eval_nodict_pipeline.py) → label bridge (import parse_mi/token_tags/
  score_tags from local/app kirciuokle.disambiguate — pure python) →
  stress ONNX per (word, label) → accented sentence out.
- CLI: `nodict_onnx.py "lietuviškas sakinys"` prints the accented
  sentence; `--bench` runs a 500-token timing benchmark (tokens/s on
  CPU int8).
- Verification: run it on 30 sentences from data/eval/lrt-smoke.txt and
  compare token outputs against eval_nodict_pipeline's predictions with
  the same checkpoint — ≥98% agreement (small drift from int8 is OK,
  report the number).

## 3. `BROWSER.md` — client-side feasibility analysis

Written analysis (a doc, no code) covering, with real numbers you
measure:
- artifact sizes: tagger ONNX int8 (measure the existing files under
  local/tagger-hf/artifacts and release/), stress fp32/int8 (from step 1),
  tokenizer files; total download for the full no-dict pipeline;
- parameter breakdown of litlat-bert: embedding matrix vs transformer
  body (load the config/count from the checkpoint) — quantify how much a
  Lithuanian-only vocabulary prune could save (count how many of the
  tokenizer's vocab entries actually occur when tokenizing our 575k
  dictionary words + the LRT corpus; the unused-row share of the
  embedding matrix is the prunable mass);
- runtime paths: onnxruntime-web WASM (SIMD+threads) and WebGPU EP —
  expected throughput class for a BERT-base at int8 in browser (cite
  official ort-web docs/benchmarks via web search, be concrete about
  what is measured vs estimated);
- memory ceiling: peak RAM for int8 BERT-base inference in WASM
  (2-4x model size rule of thumb, cite);
- the verdict: is "load weights into the user's session and let Chrome
  do the job" viable today, for which user segment (desktop vs mobile),
  and what the two-model (tagger+stress) vs future shared-encoder
  single-model design means for the download budget;
- ship-shape recommendation: what to build first (e.g. WASM demo with
  the int8 pair behind a "download 200MB once, cached" UX vs waiting
  for the joint model).

## Pass criteria

1. Export runs against the v2 checkpoint on CPU; parity numbers printed.
2. `nodict_onnx.py "Vilnius yra gražus miestas"` prints an accented
   sentence with no torch import (verify with `python -X importtime` or
   sys.modules assertion).
3. Agreement + bench numbers reported.
4. BROWSER.md written with measured sizes and cited runtime claims.
5. GPU may be busy training — everything here is CPU-only.

Do not commit.
