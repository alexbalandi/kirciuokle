# Bundled Weights Pilot

`bundled_weights_pilot/` is a static, local-only pilot for running the joint
Lithuanian accentuator entirely in the browser. It self-hosts ONNX Runtime Web
WASM and transformers.js under `model/runtime/`; transformers.js is used only
for the XLM-R tokenizer.

## Architecture

- Model files are built into `bundled_weights_pilot/model/` by
  `prepare_model.py` from `local/accentuator/joint/hf_release/`.
- `prepare_model.py` vendors the browser runtime into `model/runtime/`:
  `onnxruntime-web@1.22.0` (`ort.min.mjs` plus the threaded WASM `.mjs`/`.wasm`
  companions) and `@huggingface/transformers@3.7.1`. The runtime files and npm
  tarballs are recorded with sha256s in `model/manifest.json`.
- `prepare_model.py` also writes `model/label_bridge.json`. The bridge uses
  `local/accentuator/eval_nodict_pipeline.py` to collect the closed vocabulary
  of dictionary `mi` labels and `local/app/kirciuokle/disambiguate.py` to parse
  both dictionary `mi` labels and model `UPOS|FEATS` labels into the same slot
  dictionaries. The browser ports `score_tags`, applies the fewest-spurious-slot
  tie-break, caches the best `mi` label for each model label, and merges duplicate
  `mi` strings by summed probability in the popover.
- `prepare_model.py` generates `bundled_weights_pilot/i18n.js` from
  `src/client/i18n.ts`, which remains the source of truth for production UI
  strings, the stress-primer copy, and morphology abbreviation glosses. Only
  pilot-specific runtime strings live in `prepare_model.py`.
- The app imports both runtime libraries from `./model/runtime/`; cold loads do
  not depend on jsDelivr or any other CDN. `ort.env.wasm.wasmPaths` points at
  the same-origin runtime directory.
- `onnxruntime-web` runs the joint ONNX graph in WASM. The server sends
  `Cross-Origin-Opener-Policy: same-origin` and
  `Cross-Origin-Embedder-Policy: require-corp`, so ORT can use threaded WASM
  when `crossOriginIsolated` is available. It falls back to one WASM thread.
  `ort.env.wasm.proxy = true` moves session creation and `session.run` into
  ORT's worker.
- The tokenizer is loaded from the bundled `tokenizer.json` with the local
  transformers.js. The app tokenizes each sentence as split words and stitches
  `<s> word-subwords... </s>` to match the Python fast-tokenizer alignment.
- Long input is sentence-split, tokenized, length-sorted, batched to a padded
  subword token budget, run sequentially, restored to original order, and
  rendered after every finished batch.
- Sentences are never packed into a shared window. The model was trained with
  one sentence per sequence, so packing would create cross-sentence attention
  patterns it never saw and would also make word-level first/last subword spans
  harder to keep faithful.
- Stress decoding ports the Python validity mask: long-vowel, bare `i`, mixed
  diphthong, and sonorant rules are applied before the char-mark grid is
  compared with the no-stress cell. If no-stress wins, the word is unchanged.
- POS decoding softmaxes the 804-label head, keeps every label with probability
  above 0.1, bridges each kept model label to the closed `mi` vocabulary, sums
  duplicate `mi` strings, and sorts the popover rows by summed probability.
  Underlines mirror production: green for one confident context choice, amber
  for multiple readings, dotted for the no-stress/foreign cell.

## Size And Parity

Run `uv run bundled_weights_pilot/prepare_model.py` from the repo root to copy
the bundle, run quantization experiments, and print the current table. The app
loads the default model recorded in `model/manifest.json`.

| artifact | file | size | POS parity | stress parity | default | note |
| --- | --- | ---: | ---: | ---: | --- | --- |
| partial dynamic int8 baseline | `joint.int8.onnx` | 537,586,710 B / 512.7 MiB | 99.83% | 98.96% | yes | Existing hf_release partial quantization scope kept for parity. |
| full dynamic int8 | `joint.full-int8.onnx` | 156,384,275 B / 149.1 MiB | 99.39% | 96.18% | no | Below the 97% stress parity gate, so it remains an experiment. |

The optional vocabulary-prune experiment is not shipped in this pilot. Prior
measurements showed roughly 17k used token rows out of 84k, with a target near
120 MB after a safer custom tokenizer and sliced embedding path. That remains
future work because a bad ID remap would silently corrupt every prediction.

## SPEC43 Checks

- CDN-blocked cold load: Playwright routed network access to localhost only;
  the page loaded and ran inference with `blocked=0`.
- Load-path JS heap: with precise Chromium memory info, the model buffer was
  `534.7 MB` before session handoff and `22.0 MB` after session creation,
  transfer to the proxy worker, explicit reference clearing, and GC
  (`-512.7 MB`).
- Cache API storage runs after the model buffer is handed to ORT and released.
  It uses an independent background re-fetch, stores bounded same-origin chunk
  entries, and never clones the main model response stream.
- Proxy responsiveness: while `session.run` was active, a language-switch DOM
  interaction completed in `20.8 ms`, below the 200 ms gate.
- Throughput on this machine: previous non-proxy SPEC42 repro log averaged
  `143.4 tokens/s`; the 3-run proxy check averaged `161.0 tokens/s`.
- SPEC42 repro after proxy: 3 iterations stayed flat in the main page
  (`0.0 MB` tracked WASM, `23.7-23.8 MB` JS heap). The ORT heap now lives in the
  proxy worker, so the page-level WebAssembly monkey-patch no longer sees the
  old in-main `1306.9 MB` WASM allocation.

## SPEC44 Checks

- Cache writes are quota-gated with `navigator.storage.estimate()` before any
  Cache API write starts. If available headroom is below `modelBytes * 1.2`, the
  app skips the write and reports the existing cache-unavailable state.
- Cache stores use an independent background re-fetch with a 20s no-progress
  abort watchdog, so Cache API backpressure cannot block the model response
  body. The writer stores bounded chunk entries to avoid Chromium's large
  single-entry Cache API rejection. A failed, skipped, or watchdog-cancelled
  write is handled once for the page load and inference continues uncached.
- The model status switches to `Initializing model` as soon as the contiguous
  model buffer is complete; session creation no longer waits on cache
  bookkeeping.

## SPEC45 Checks

- ORT proxy execution remains the first attempt, but session creation is now a
  progressive enhancement: a 45s watchdog falls back once to `proxy=false` on the
  main thread, extending to 180s when ORT WASM runtime fetches show real
  progress.
- The proxy attempt wraps reachable Worker construction and `error` /
  `messageerror` events with `console.warn` diagnostics, and abandoned proxy
  workers are terminated before fallback.
- The model and memory status lines localize worker, fallback, and active
  execution-mode text. Harnesses assert the machine-readable mode through
  `window.__pilotRuntimeConfig`.

## Run Locally

```powershell
uv run bundled_weights_pilot/prepare_model.py
uv run bundled_weights_pilot/serve.py
npx.cmd --yes --package playwright node bundled_weights_pilot\memtest.mjs --no-worker 1
npx.cmd --yes --package playwright node bundled_weights_pilot\memtest.mjs worker-mode 1
npx.cmd --yes --package playwright node bundled_weights_pilot\memtest.mjs normal-cache 1
npx.cmd --yes --package playwright node bundled_weights_pilot\memtest.mjs --quota-stress 1
npx.cmd --yes --package playwright node bundled_weights_pilot\memtest.mjs cold-blocked 1
npx.cmd --yes --package playwright node bundled_weights_pilot\memtest.mjs responsive 1
npx.cmd --yes --package playwright node bundled_weights_pilot\memtest.mjs spec42-flat 3
npx.cmd --yes --package playwright node bundled_weights_pilot\memtest.mjs --no-worker spec42-flat 3
```

Open `http://127.0.0.1:8788/`. The first load downloads the ONNX file and
attempts to store it through Cache API; later loads reuse that local cache when
the browser accepts the large entry. Headless Chromium may reject it because of
temporary-profile storage quota, in which case the status line says so and
inference still runs.

## Limitations

- The baseline model is very large for a web page. The shipped 512.7 MiB ONNX
  still requires a large transient JS buffer while the model response is read.
  In Chromium with precise memory enabled, the retained model buffer measured
  `534.7 MB` just before session creation and dropped to `22.0 MB` after ORT
  accepted it and the app cleared its references. The resident ORT session now
  lives in the proxy worker; browser WASM still has a 32-bit linear-memory
  ceiling, and worker WASM memory can grow without shrinking. The pilot keeps
  one session for the page lifetime, decodes each batch into small JS word rows,
  drops ORT tensors after every run, halves later batch budgets above the 75%
  WASM high-water mark when visible, and shows a reload message if the browser
  refuses another allocation. The vocabulary-prune work should lower both the
  download and the resident WASM/session footprint by cutting the oversized
  embedding tables before they ever reach the browser.
- This is a desktop-class demo. See `local/accentuator/onnx/BROWSER.md` for the
  current browser feasibility notes and mobile risk.
- Sentences longer than the model's 128-subword window are truncated at the
  token-feeding stage; overflow words render unchanged.
- WASM CPU throughput depends heavily on cross-origin isolation, CPU, tabs, and
  thermal state. WebGPU is the likely production acceleration path, but this
  pilot keeps the execution provider to WASM.
- The app is intentionally separate from the production site and does not call
  server inference.

## Path To Production

1. Finish the vocabulary-prune pipeline with tokenizer ID remapping and parity
   gates.
2. Improve Cache API UX with explicit cache clearing and better first-download
   progress.
3. Measure the same joint graph in ORT WebGPU on Chrome/Edge desktop.
4. Revisit mobile only after bundle size and peak RAM are substantially lower.
