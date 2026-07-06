# SPEC45 — Pilot: proxy worker as progressive enhancement

## Bug (live verify, embedded Chromium)

With `ort.env.wasm.proxy = true`, some browser contexts fail the module
worker spawn SILENTLY: no error, no fetch of
`ort-wasm-simd-threaded.mjs/.wasm` ever happens, and
`InferenceSession.create` never settles — page stuck at "Initializing
model", 513MB buffer never released. Works in dev Chromium, fails in
the embedded preview browser; assume real-world browsers vary too.

Fix in `bundled_weights_pilot/app.js` (+README note). Nothing else.

## Required behavior

1. Wrap session creation with a WATCHDOG (45s): if `create` has not
   settled, tear down the attempt, set `ort.env.wasm.proxy = false`,
   and retry ONCE on the main thread. Status line reflects the retry
   ("Initializing model (fallback)…" — add localized strings).
2. Instrument the proxy attempt: hook worker `error`/`messageerror`
   where reachable and `console.warn` any failure so it is never
   silent again.
3. Keep proxy=true as the FIRST attempt (it is faster and keeps the UI
   responsive when it works). The memory status line should state which
   mode ended up active (worker / main thread) — localized.
4. The 45s watchdog must not false-positive on slow machines where
   create is genuinely progressing: if the wasm runtime files HAVE been
   fetched (track via a fetch interceptor on the wasmPaths prefix or
   PerformanceObserver resource entries), extend the deadline to 180s
   before falling back.

## Pass criteria

1. memtest gains a `--no-worker` phase: block Worker construction
   (`page.addInitScript` overriding `window.Worker` to throw) → page
   must still reach Ready via the fallback within the watchdog window
   and accentuate the repro sentence. Paste output.
2. Normal phase still uses the worker (assert wasm fetch happened and
   mode=worker in the status).
3. 3 SPEC42 iterations still flat in both modes.

Do not commit.
