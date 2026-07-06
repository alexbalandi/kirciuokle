# SPEC43 — Pilot: load-path memory + self-hosted runtime + responsiveness

## Context (measured)

SPEC42's harness proved there is NO run-path leak: WASM flat at
1,306.9 MB over 25 runs. The real problem is baseline + load-transient
footprint in a real browser: model bytes held in a JS ArrayBuffer,
copied into the WASM heap at session creation, plus a Cache API write —
transient ~2 GB. Also: the "runs fully in your browser" pilot still
loads ort runtime + tokenizer lib from CDN at every cold start, and
inference blocks the main thread.

Only files under `bundled_weights_pilot/`. Do not undo SPEC41/42 work.

## 1. Load-path memory

- After `InferenceSession.create(...)` resolves, IMMEDIATELY drop every
  reference to the model ArrayBuffer (and any copies fetchWithCache
  holds) so GC can reclaim ~513 MB of JS heap. Verify with the memtest
  harness: extend it to log jsHeapMB right after load; before/after
  numbers pasted.
- Cache API write must not double-buffer: stream the response into
  cache via `cache.put(url, response.clone())` at FETCH time (clone
  streams, no second full buffer), never `new Response(buffer)` after
  the fact. If the current code already streams, say so.

## 2. Self-host the runtime (kill CDN dependence)

- prepare_model.py additionally vendors into `model/runtime/`:
  the onnxruntime-web dist files actually used (`ort.min.mjs` +
  `ort-wasm-simd-threaded.wasm/.mjs` and the threaded worker file if
  separate; version-pin 1.22.0) and the transformers.js ESM bundle.
  Download once at prep time (pip/npm cache or direct URLs), record
  sha256s in the manifest.
- index.html/app.js import from `./model/runtime/` instead of
  cdn.jsdelivr.net; `ort.env.wasm.wasmPaths` points at the local dir.
  With COEP require-corp this also removes the CORP dependency on CDNs.
- Remove the no-op `ort.env.wasm.simd = true` line (1.22 dist is
  SIMD-only per upstream docs).

## 3. Responsiveness

- Set `ort.env.wasm.proxy = true` (now safe: same-origin runtime
  files) so session.run happens off the main thread; verify the UI
  stays interactive during a long-text run (Playwright: trigger a run
  on a ~10-sentence text, assert a DOM interaction — e.g. language
  switch — completes with <200ms task blocking while batches run).
- Keep progressive per-batch rendering as is.

## Pass criteria

1. Cold load in Playwright with network access ONLY to localhost
   (block cdn requests via route interception) — page fully works:
   proves self-hosting is complete.
2. Post-load jsHeapMB before vs after the buffer-release fix pasted
   (expect roughly -500 MB).
3. Proxy responsiveness check passes; note tokens/s before/after
   (proxy may cost a little throughput — report it).
4. 3 iterations of the SPEC42 repro still flat on WASM MB.
5. README updated (self-hosted runtime, memory model numbers).

Do not commit.
