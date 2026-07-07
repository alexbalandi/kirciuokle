# SPEC53b — Client: light/heavy selector + cache-aware redownload

The bundle now ships two int8 tiers; `local-model/manifest.json` has
`"tiers": { "heavy": "joint.int8.partial.onnx", "light":
"joint.int8.full.onnx" }` and a `models` map with per-file `bytes` and
`tier`. Wire the client to let users choose and to recover an
absent/evicted cache. Files: `src/client/**`, `index.html`, tests.
`npm run check` + `build` green. No deploy.

## A. Tier selector

- `src/client/local/assets.ts`: read `manifest.tiers` + `manifest.models`.
  Add a `LocalModelTier = "light" | "heavy"`. Resolve a tier → model
  filename + expected bytes. Keep `joint.int8.partial.onnx` (heavy) the
  default. Everything else (tokenizer, meta, label_bridge, decode) is
  tier-independent.
- Persist the chosen tier in localStorage `accent-local-tier` (default
  "heavy"). Cache is already per-URL, so both tiers cache independently.
- UI: a localized segmented control shown in Local mode near the mode
  explainer — LT "Modelis: Lengvas ({lightSize}) / Tikslus ({heavySize})",
  EN "Model: Light / Accurate", RU "Модель: Лёгкая / Точная", with sizes
  in MB from the manifest. Selecting a tier: if its file is cached, load
  it; else show the consent/download card for that tier's size. Changing
  tier disposes the current ORT session and (re)loads the chosen one.
- Thread the active tier through the engine + status so the WASM-stats
  popover shows which tier is loaded.

## B. Cache-aware (re)download

- The existing consent flow handles first visit. Extend it: on entering
  Local mode (or switching tier), probe the Cache API for the chosen
  tier's file. If ABSENT — whether never downloaded OR evicted after a
  prior ready session — show the consent/download card. When the absence
  is an eviction (there was a prior successful load this browser, e.g. a
  localStorage flag `accent-local-downloaded-<tier>` was set), the card
  copy is the localized "the model is no longer saved in your browser —
  download it again", button "Download again ({size})"; otherwise the
  first-visit copy. Set the downloaded flag on a successful load; the
  redownload path clears+refetches.
- Byte-length check: when loading from cache, compare cached
  content-length to the manifest `bytes`; on mismatch, purge that cache
  entry and show the redownload card.
- Add a small localized "re-download model" text button inside the
  WASM-stats popover that purges the active tier's cache entry and
  re-runs the download — usable even while a model is loaded.

## Localization

Add all new strings to LT/EN/RU consistently (tier labels, "Download
again", the evicted-cache message, the stats re-download control).

## Pass criteria

1. check + build green; test count.
2. Vite dev + Playwright (real file loads use the LIGHT 133 MB tier to
   keep it fast; the heavy path may be asserted via mocked fetch):
   - both tiers show with MB sizes; selecting Light actually loads and
     accentuates a sentence;
   - clear the Cache API entry → the redownload card reappears with the
     "download again" copy; clicking it re-downloads and loads;
   - the stats popover re-download control purges + reloads.
   Screenshot the selector and the redownload card.
3. Web mode unaffected.

Keep the change surgical — reuse the existing consent/card/engine
plumbing; do not rewrite the load pipeline. Do not commit, do not deploy.
