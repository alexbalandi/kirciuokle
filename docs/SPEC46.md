# SPEC46 — Main site: Web/Local mode switch with in-browser inference

## Goal

Bring the pilot's in-browser inference into the PRODUCTION site as a
first-class, typed, tested mode — behind a user-facing switch. NO
deploy (local vite verification only); model assets served locally in
dev (R2 comes later — keep the base URL a single config constant).

Files: main site only (`index.html`, `src/client/**`, `src/shared/**`,
`vite`/`package.json` config as needed, tests). Do NOT modify
`bundled_weights_pilot/` (reference implementation to port from) or
`src/worker/**` beyond what typing needs.

## Modes

- **Web** (default, current behavior): server pipeline (VDU + UDPipe).
- **Local**: in-browser ONNX inference with the joint model — port the
  pilot's engine (bundled_weights_pilot/app.js is the reference: ort
  session with proxy→main-thread fallback watchdog, cache quota
  preflight + watchdog-cancellable cache writes, length-bucketed
  batching by token budget, adaptive cap, validity-mask stress decode,
  label bridge with fewest-spurious tie-break and probability merging)
  into typed modules under `src/client/local/`:
  `engine.ts` (session lifecycle + modes), `batching.ts`,
  `decode.ts` (validity mask — port EXACTLY; fixtures below),
  `bridge.ts` (label bridge), `assets.ts` (manifest/model fetch +
  Cache API), `types.ts`.
- CRITICAL CONTRACT: the local engine emits the SAME `Part[]` shape
  the web worker returns (src/shared/types.ts — Part with
  accented/variants/tokenTags/chosenMi...) so rendering, legend,
  popovers, and user-choice all work unchanged. Local variants list =
  bridge-merged mi labels with probabilities, each mapped into the
  existing variant structure (probability carried in a new optional
  typed field, e.g. `p?: number`, rendered as a percentage chip only
  when present).

## UI

- Segmented control "Režimas: Internetu / Vietinis" (localized; EN
  Web/Local, RU Онлайн/Локально) next to the accent button; below it
  ONE small explainer line that changes with mode (localized):
  web = text is sent to the accentuation server; local = everything
  runs in your browser after a one-time model download (~size from
  manifest), nothing leaves the device.
- Mode persisted in localStorage; Local assets load LAZILY — and NOT
  automatically: first switch to Local shows an explicit consent
  notice + button, NO download starts until the button is pressed.
  Notice text (localized, size from the manifest): EN "To accentuate
  locally, the site downloads the model once — about X MB of traffic.
  It stays saved in your browser for future visits."; the button says
  "Download model (X MB)" with a cloud-arrow-down icon (a fetch-from-
  cloud glyph — deliberately NOT the tray/save-file download icon, so
  it is not read as saving a file to disk). After consent once,
  subsequent visits with the model cached skip the notice (cache hit →
  straight to ready); if the cache was evicted, the notice reappears.
  Switch back to Web is instant; a loaded local engine stays warm for
  the session.
- Options control "Rodymas: geriausi / visi" (top / all): in Web mode
  it filters the popover (top = chosen reading only + "show all"
  inline link; all = current full list). In Local mode it is FORCED to
  "top" (control disabled with localized tooltip: the model ranks
  labels by probability; only >10% are shown).
- NO percentages in Web mode (match scores are not probabilities — do
  not fake them). Percentages appear only on local-mode readings.
- WASM stats: a small icon button (activity/pulse glyph) in the result
  panel header, ONLY visible in Local mode; opens a compact popover:
  inference mode (worker / main-thread fallback), memory line, last
  run tokens/s + batch count, model version + cache state. Localized.
- Stress-mark primer stays as is.

## Dev asset serving

- `--` model files live in bundled_weights_pilot/model/ already; make
  vite serve them in dev under `/local-model/` (fs.allow + a tiny dev
  server alias or copy step — pick the cleanest; document it).
  `LOCAL_MODEL_BASE` constant in one place; assert manifest sha256s
  after fetch (they are in the pilot manifest).

## Types & tests (vitest, wired into `npm run check`)

- Everything strictly typed (no `any` beyond ort/transformers
  interop shims typed in one `ambient.d.ts`).
- Unit tests:
  1. decode: validity fixtures — the exact cases from
     train_guesser tests: abatija+grave ok / +acute banned;
     slėnio ė grave banned; vyras y grave banned / acute ok;
     pirko i grave ok tilde banned, r tilde ok grave banned;
     vienas i acute ok; namas all marks ok. Plus: no-stress cell wins
     → word unmarked; mark insertion NFC-correct (ė̃ cluster case).
  2. batching: token budget respected, order restored, single
     over-budget sentence isolated, deterministic.
  3. bridge: merges duplicate mi strings summing p; fewest-spurious
     tie-break; caches per label id.
  4. mode/service selection: local engine returns Part[] matching the
     shape contract (type-level test + runtime shape assertion).
- `npm run check` and `npm run build` green.

## Pass criteria

1. check+build green; test count reported.
2. Vite dev + Playwright: flip to Local, wait for ready, accentuate
   the SPEC42 repro paragraph — accented output renders with the SAME
   UI as web mode; popover shows mi labels with % chips; options
   control disabled-on-top in Local, toggling in Web; stats icon only
   in Local and opens the popover; mode persists across reload
   (localStorage) without auto-downloading on load into Web mode.
3. Web mode regression: accentuate the same text via the normal path
   (needs network) — unchanged behavior incl. legend/user choice.
4. Screenshot of Local mode with popover open saved to
   docs-local-mode.png (repo root ok, small).

Do not commit. Do not deploy. Do not touch R2.
