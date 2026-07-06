# SPEC40 — bundled_weights_pilot: in-browser inference site

## Goal

`bundled_weights_pilot/` at the repo root: a self-contained static site
that accentuates + POS-tags Lithuanian text ENTIRELY in the browser
(onnxruntime-web WASM, no server inference). Pilot quality bar: works
smoothly on local for real texts, honest about model size. This is a
separate folder; do not touch the production site.

## Architecture (decided — follow it)

- **Model**: the joint accentuator ONNX
  (local/accentuator/joint/hf_release/joint.int8.onnx + meta). Copy
  into the pilot's `model/` dir via a prep script (not committed —
  gitignore `bundled_weights_pilot/model/`).
- **Tokenizer**: transformers.js (@huggingface/transformers or @xenova
  via CDN) used ONLY for tokenization (XLM-R tokenizer.json from the
  hf_release folder). Inference runs through onnxruntime-web
  (`ort.wasm`, SIMD+threads when crossOriginIsolated, fallback single
  thread).
- **Long texts — batch, don't pack**: the model trained one sentence
  per sequence; NEVER concatenate sentences into one window. Pipeline:
  sentence-split (same [.!?…]+capital heuristic as the repo) →
  tokenize all → sort indices by subword length → fill batches to a
  TOKEN BUDGET (default 2048 padded subwords per batch, tunable) → run
  batches sequentially → restore original order → render
  progressively (each finished batch updates the DOM; progress bar
  with sentence counts).
- **Decode in JS** (port the semantics exactly):
  - stress: per word token, softmax over the (char × 3 marks +
    no-stress) grid MASKED by the validity rules — port valid_target
    from local/accentuator/train_guesser.py faithfully (long-vowel /
    bare-i / mixed-diphthong / sonorant rules); char vocab + marks
    from joint.meta.json; no-stress wins → word stays unmarked.
  - POS: softmax over the 804 labels; keep labels with probability
    > 0.1 (always keep top-1), map to display.
- **UI**: minimal clone of the production look (see index.html /
  src/client/style.css for tone — panels, result area, token
  underlines): input textarea, "Accentuate" button, result panel with
  accented text; every word clickable → popover listing its POS labels
  with probabilities >0.1 (percentages, label strings verbatim);
  status line showing model-load state (download size + cached via
  Cache API), batch progress, tokens/s. Lithuanian-only UI is fine for
  the pilot.

## Quantization / size experiments (tricks allowed — measure, report)

In the prep script (`prepare_model.py`, python, runs from repo root):
1. Baseline: copy the existing int8 (537.6 MB — partial quantization
   scope kept for parity).
2. FULL dynamic int8 quantization of all supported nodes: measure size
   + token-agreement parity vs torch on 200 LRT-smoke sentences
   (the earlier stress-only attempt failed its gate at higher
   strictness; for the pilot report the number — if ≥97% both heads,
   ship it as the default bundle).
3. OPTIONAL stretch (only if time permits and it works cleanly):
   embedding vocab prune — tokenize generated.sqlite words + all
   data/eval corpora, keep used token rows (~17k of 84k measured),
   remap ids in a custom tokenizer.json + sliced embedding; verify
   parity ≥97% on the same sample; report the size win. If risky,
   document as future work in the README instead — do not ship a
   broken prune.
Report a size/parity table for whatever was built.

## Local run + verification

- `bundled_weights_pilot/serve.py` (or npx serve config) — static
  server with the COOP/COEP headers needed for WASM threads.
- Add a `bundled-pilot` entry to .claude/launch.json (port 8788).
- Playwright-or-manual smoke: load page, paste a 3-sentence text,
  verify accented output appears and a word popover lists POS entries;
  paste a LONG text (data/eval/lrt-smoke.txt content) and verify
  progressive rendering + no crash. Print timing (tokens/s) for both.

## README.md (in the folder)

Explain: what this pilot is, the architecture above (incl. WHY
batch-not-pack), the size/parity table, how to run locally, current
limitations (bundle size vs the ~120MB prune target, desktop-only
expectations per onnx/BROWSER.md), and the path to production
(vocab prune, Cache API UX, WebGPU EP).

## Pass criteria

1. prepare_model.py builds the bundle; size/parity table printed.
2. Static server serves; the smoke texts work end to end in a real
   browser context (Playwright headless ok); paste timings.
3. POS popover shows >0.1-probability labels with percentages.
4. README complete. Nothing large committed (model/ gitignored).

Do not commit.
