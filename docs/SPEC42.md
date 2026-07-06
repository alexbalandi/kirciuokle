# SPEC42 — Pilot: WASM memory-leak hunt + fix (measured)

## Symptom (user repro)

In `bundled_weights_pilot/` (WASM inference), the tab crashes SOME TIME
AFTER accentuating a moderately large text (~70 words); the production
site is unaffected. Classic cumulative WASM-heap growth: the 513MB
model + session workspace already dominates the 32-bit WASM address
space, `WebAssembly.Memory` never shrinks, and per-run transients or
retained outputs ratchet it over the ceiling on a later run.

Only files under `bundled_weights_pilot/`. SPEC41's UI-parity changes
land first — build on them, don't conflict.

## 1. Reproduce with measurement (before touching anything)

Playwright: load the pilot, run the user's paragraph (below) 25×
sequentially; after each run record `performance.memory.usedJSHeapSize`
AND the WASM memory size (`ort` env / `WebAssembly.Memory.buffer
.byteLength` — reach it via the session's wasm memory if exposed, else
track `performance.measureUserAgentSpecificMemory()` where available).
Print the per-iteration curve. A monotonic climb = the leak; flat =
fragmentation/ceiling story (then only mitigations 3b/3c apply).

Repro text: "81-erių vilnietė pardavė butą ir nusikaltėliui atidavė 114
tūkst. eurų. Tuo metu valstybės institucijos, nevyriausybinės
organizacijos ir verslininkai suka galvas, kaip dar padėti žmonėms
nepakliūti į sukčių pinkles. Pavyzdžiui, verslai ant kvitų spausdina
patarimus bei numerį, kuriuo reikėtų skambinti įtarus, kad susiduria su
nusikaltėliu. Vis dėlto ekspertai pabrėžia, kad svarbiausia vadovautis
kritiniu mąstymu ir elgtis atsakingai, jog tokie atvejai
nebepasikartotų."

## 2. Fix the retention

Audit the inference/render path for anything holding per-run data
beyond the render: raw session.run result objects captured in closures
(popover handlers!), arrays of logits kept per word, re-created
sessions, detached DOM with listeners. Required end-state:
- exactly ONE InferenceSession for the page lifetime;
- after each batch, extract ONLY plain-JS per-word results (accented
  string, word-class, top≤5 [labelId, prob] pairs) and let every
  tensor/typed-array from the run go out of scope;
- popovers read from the extracted structures, never from run outputs.

## 3. Mitigations (all three)

a. Token-budget cap per batch stays, but halve it automatically when
   the measured WASM memory exceeds 75% of its current maximum.
b. Status bar gains a memory line (used WASM MB / JS heap MB),
   localized (three languages, keys added consistently with SPEC41's
   i18n module).
c. Graceful ceiling behavior: wrap session.run in try/catch; on an
   allocation failure, show a localized "memory limit reached — reload
   the page" message instead of a dead tab, and stop the batch loop.

## Pass criteria

1. BEFORE-fix curve pasted (25 iterations).
2. AFTER-fix curve pasted: stable within noise after iteration ~3
   (or, if the growth was fragmentation-only, demonstrate the cap +
   graceful-ceiling path by forcing a low budget).
3. The user's repro text runs 25× without crash in Playwright.
4. README's limitations section updated with the memory model
   (bundle size × 2–4 runtime, 32-bit ceiling, what the prune will fix).

Do not commit.
