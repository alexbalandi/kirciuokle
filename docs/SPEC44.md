# SPEC44 — Pilot: cache-write backpressure hang (found in live verify)

## Bug (reproduced in the preview browser, quota-constrained profile)

`fetchWithCache` streams the model while `cache.put(url,
response.clone())` writes the clone. Cloned Response branches SHARE
backpressure: when the Cache write stalls mid-stream on a
quota-constrained profile (instead of failing fast), the MAIN body read
blocks at the shared high-water mark — UI freezes at
"Downloading · 512.7 MiB/512.7 MiB", `InferenceSession.create` never
runs, the model buffer is never released. Console shows repeated
"Cache API store failed… [object DOMException] / UnknownError" from
retries that stall the same way.

Fix in `bundled_weights_pilot/app.js` (+ README note). Nothing else.

## Required behavior

1. BEFORE attempting any cache write, check
   `navigator.storage.estimate()`: if `quota - usage <
   modelBytes * 1.2`, skip caching entirely (status shows the existing
   "cache unavailable" state) — never start a write that cannot finish.
2. When a cache write IS attempted: make it non-blocking for the main
   read AND cancellable — wrap `cache.put` with a watchdog (no progress
   / not settled within 20s) that calls `clone.body.cancel()` (or
   aborts via an AbortController-driven re-fetch design) so the main
   branch can never be blocked by a stalled cache branch. On any
   failure/cancel: log once, continue uncached (existing path).
3. No retries of the cache write within the same page load (the retry
   loop multiplies the stall).
4. The download-complete → session-create transition must be provably
   independent of cache state: add a status transition
   ("Initializing model…") the moment the buffer is complete, before
   any cache bookkeeping.

## Pass criteria

1. memtest gains a `--quota-stress` phase: run the cold load in a
   Playwright context with a restricted storage quota (Playwright can
   set `--js-flags` no; instead emulate: pre-fill the origin's Cache
   storage with junk until estimate() headroom < model size, or mock
   `caches.open` to return a stalling put). The page must reach
   "Ready" and accentuate the repro sentence despite the stalled/full
   cache, within a bounded time (≤ 180s on CPU).
2. Normal path (no quota stress) still works and still caches when
   space allows.
3. Paste both phase outputs.

Do not commit.
